# Filesystem KV tier: capacity limit and eviction

## Decision

The filesystem secondary tier stored blocks until the disk was full. It had no
capacity parameter, no eviction path and no accounting: a long agent run left
~63 GB of block files behind, and once the disk filled up `store_block` raised
`OSError`, the job failed, and every later lookup missed because the file was
never created.

This change makes `max_bytes` a required part of every `type: "fs"` tier
configuration and enforces it inside the engine. No filesystem level means
(separate partition, loop device, project quota) are used: the cache directory
stays an ordinary directory and the limit is held by vLLM code alone.

## Configuration

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "eviction_policy": "lru",
    "secondary_tiers": [
      {
        "type": "fs",
        "root_dir": "/mnt/samsung-970/kvtier",
        "max_bytes": 68719476736,
        "max_overshoot_bytes": 1073741824,
        "n_read_threads": 16,
        "n_write_threads": 16
      }
    ]
  }
}
```

`max_bytes` has no default: construction fails with
`fs tier requires 'max_bytes' in its configuration` when it is missing, because
an unbounded disk cache is not a valid mode. `max_overshoot_bytes` defaults to
1 GiB and bounds the soft overflow described below.

## How the limit is held

* **One counter per directory, not per instance.** `spec.get_manager()` runs in
  the scheduler process and creates one FS tier instance per KV group (9 for a
  hybrid model). `get_dir_capacity()` keys a single `DirCapacity` object on
  `realpath(root_dir)`, so all instances share one `used` and one limit.
* **One writer per directory.** `<root_dir>/.fs_tier.lock` is held with
  `flock(LOCK_EX | LOCK_NB)` for the lifetime of the process. A second engine
  pointed at the same directory fails at startup instead of double-booking it.
  This is the only cross-process guard; the hot path takes no file locks.
* **Real occupancy.** A block file costs `round(st_size, 4 KiB) + 4 KiB`, i.e.
  the file rounded up to an ext4 block plus one inode block. The counter is
  rebuilt by a directory scan at startup, so the limit applies immediately
  after a restart and stale layouts from an earlier cache geometry are counted
  (and evicted as the oldest entries) instead of growing unbounded.
* **Reserve before write.** `submit_store` reserves space for every block of a
  job before queueing any I/O, releases the reservation when a write fails, and
  never lets peak occupancy exceed `max_bytes + max_overshoot_bytes`.
* **Soft overflow.** When no victim can be evicted (everything in flight or
  pinned), a job may push usage above `max_bytes` by at most
  `max_overshoot_bytes` instead of failing. After every finished job the cache
  is trimmed back to `max_bytes`.
* **Two eviction classes.** Blocks that were written and never read are evicted
  first, oldest first (FIFO). Blocks that were read are evicted last, least
  recently read first (LRU). Blocks with an I/O in flight are never evicted.
* **Failure is explicit.** If even the overshoot window is exhausted the tier
  fails the whole job (`DualQueueThreadPool.fail_job`) and reserves nothing; the
  tiering manager releases the pinned blocks and counts `OFFLOAD-STORE-FAIL`, so
  a block that never reached the disk is never treated as stored.

## Mamba state cap

Mamba state files (the `_g<k>` directories of the groups that carry state) are
accounted apart from attention files and are additionally capped by
`state_max_bytes` (default 4 GiB; the deployment uses 8 GiB). When the state cap
is exceeded, a checkpoint is dropped together with its whole block bundle:
attention kept without its checkpoint can never rebuild a prefix from block
zero, so leaving it behind only wastes the tier.

The victim is chosen among checkpoints whose bundle was never read, with FIFO by
commit time as the tie-break; a checkpoint with a reader in flight is never
dropped. FIFO alone dropped the checkpoint that had just been restored and kept a
stale one no session ever read again.

`FS-CAP-SCAN` reports `state_bytes`, `attn_bytes`, `checkpoints` and
`state_max`; `FS-METRICS` adds `state_bytes`, `attn_bytes`, `state_limit`,
`state_evict`, `state_evict_unread` and `state_evict_read`.

## Metrics

`FS-METRICS` (INFO, every 10 s) and `DirCapacity.snapshot()` expose
`fs_bytes_used`, `fs_bytes_limit`, `fs_overshoot_bytes`, `fs_peak_overshoot_bytes`,
`fs_evictions_total`, `fs_store_rejected_total`, `fs_store_failed_total`,
`fs_load_errors_total`, `fs_accounting_drift_bytes`, `fs_instances`, `fs_files`,
`fs_stored_blocks`, `fs_loaded_blocks`, `fs_lookups`, `fs_lookup_hits`,
`fs_inflight`.

`FS-CAP-SCAN` is logged once per directory at startup, `FS-CAP-REJECT` /
`FS-CAP-OVERSHOOT` on the corresponding events, and `FS-DRIFT` every 60 s from a
background thread that compares the committed counter with the real on-disk sum
(tolerance 1% of the limit, an `ERROR` above it).

`VLLM_FS_TIER_STATS=1` additionally logs `FS-LOOK-MISS` for the first lookups
and every 500th miss.

## Measured behaviour

Host: 2× V100-16GB, TP=2, block size 1024, DFlash2, CPU tier 16 GiB, FS tier in
`/mnt/samsung-970/kvtier` with `max_bytes` 64 GiB and 1 GiB overshoot.

* Limit holds. Peak `fs_bytes_used` = 68,718,530,560 B of 68,719,476,736 B
  (99.9986%), `fs_overshoot_bytes` 0, `fs_store_rejected_total` 0,
  `fs_store_failed_total` 0, `fs_load_errors_total` 0.
* Accounting matches the disk. Startup scan reported `used=10,516,144,128,
  files=760`, `du -sb` reported 10,513,030,617 — a 3.1 MB difference, exactly
  the 760 inode blocks the counter adds.
* Eviction is stable under sustained load. Over 22 minutes of an 8-session
  rotation: 1220 stores, 1216 evictions, 4845 → 4849 files, usage pinned at
  99.99% of the limit. Every stored block displaces an older one; that is the
  intended steady state of a full cache.
* Drift stayed inside tolerance during the run (worst sample −54.5 MB, i.e.
  −0.08% of the limit) and returned to 0.
* Storage cost is 42 MiB per 1024 tokens across both ranks, so 64 GiB is about
  1.5M tokens of cache by arithmetic.

## Known limitations

* **A full cache with a rotating working set restores little.** In the same 22
  minute window the tier served 49 blocks out of 131,457 lookups (0.04%). The
  eviction policy is correct — never-read blocks go before read ones — but with
  a working set above the cache size, the blocks a session needs on return are
  the ones evicted first. Reuse in that regime comes from the GPU prefix cache
  and the CPU tier: of 3.43M prompt tokens in the window, 77% were served from
  the local prefix cache, 16% were recomputed and 7% came from the tier.
* **`num_preemptions_total` stays 0** on this configuration because
  `--max-num-seqs 4` caps concurrency well below the pool: pressure shows up as
  a queue (up to 13 waiting) while `kv_cache_usage_perc` peaks at 0.64 against
  the 0.8 ceiling implied by `--watermark 0.2`. Any acceptance criterion phrased
  as "preemptions > 0" is unreachable without raising `--max-num-seqs`.
* **Deferred work still occurs.** The scheduler counters show deferred stores
  and deferred promotions during rotation (`OFFLOAD-DEFER`), with
  `waiting_by_reason{deferred}` remaining 0. Deferral is counted, not removed.
* Read throughput of the tier is bounded by the thread count: on this NVMe the
  configured 16 read threads yield ~2.3 GB/s aggregate with 96 ms per 16 MiB
  block, while 4 threads yield ~2.9 GB/s with 19 ms.
* **Attention blocks without a state bundle cannot be hit.** Attention is stored
  per block in the normal pass, while a Mamba state bundle is captured only at a
  prompt boundary — one per prompt — so attention blocks far outnumber bundles.
  The tier lookup takes the minimum across groups, so a block whose state is
  absent zeroes the whole request. Measured on the test host after a 36-minute
  ten-session run: 85 state bundles against 585 unique block-6 attention files,
  of which only 85 still had a partner; 584 block-7 and 1189 block-8 files.
  Around 85% of the stored block-6 attention could therefore never produce a hit.
  Storing a bundle whenever its attention half is stored, and dropping attention
  that has lost its partner, is a separate change.

