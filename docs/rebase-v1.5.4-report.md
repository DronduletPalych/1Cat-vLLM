# Ребейз форка 1Cat-vLLM на upstream v1.5.4

- **Дата:** 2026-11-20
- **Форк:** `DronduletPalych/1Cat-vLLM`
- **Ветка:** `rebase/v1.5.4` (force-push в `main`)
- **Upstream-база:** v1.5.4 = [`1fa9c438bc`](https://github.com/1CatAI/1Cat-vLLM/commit/1fa9c438bc6e593dbfdd9430b23012a82023bed8) (2026-11-19)
- **Старая точка синхронизации:** [`c98a8d6648`](https://github.com/1CatAI/1Cat-vLLM/commit/c98a8d66482333eb8d8b87be72e8856445ac99ec) (upstream main на 2026-11-07, смержен в [`ac249f195f`](https://github.com/DronduletPalych/1Cat-vLLM/commit/ac249f195f) «sync: merge upstream/main 2026-11-07»)
- **Старый tip форка:** [`1a99f340e6`](https://github.com/DronduletPalych/1Cat-vLLM/commit/1a99f340e612d44ba0c3d6fb244011a49710f0c1) (тег v1.5.25)
- **Новый tip:** [`7684827bab`](https://github.com/DronduletPalych/1Cat-vLLM/commit/7684827bab0c5be71755548ba9ebc8dad82e7f40)

## Метод

`git rebase --onto v1.5.4 c98a8d6648` с `origin/main`. Между точкой синхронизации и v1.5.4 — 53 upstream-коммита; наших правок — 49 коммитов (merge-коммиты синхронизации выкинуты линейным rebase'ом). Конфликты возникли в 6 коммитах — 13 конфликтных блоков, все разрешены вручную. Остальные 43 коммита наехали чисто.

## Соответствие SHA (старый → новый)

| # | Старый | Новый | Коммит |
|---|--------|-------|--------|
| 1 | [`188989f91c`](https://github.com/DronduletPalych/1Cat-vLLM/commit/188989f91c514cdcb9d1b36cd4037a609a2d4c5f) | [`49ab72d6f9`](https://github.com/DronduletPalych/1Cat-vLLM/commit/49ab72d6f9f47b5224d03c68417d84c323dd6dd0) | [Core] Add KV cache watermark admission headroom |
| 2 | [`38671c5d19`](https://github.com/DronduletPalych/1Cat-vLLM/commit/38671c5d19af) | [`53578ffb63`](https://github.com/DronduletPalych/1Cat-vLLM/commit/53578ffb6316) | [Core] Fix GGUF launch regressions on SM70 TP2 |
| 3 | [`50bfc0f511`](https://github.com/DronduletPalych/1Cat-vLLM/commit/50bfc0f5112f) | [`bff9d236df`](https://github.com/DronduletPalych/1Cat-vLLM/commit/bff9d236df3d) | [Core] Let the CPU KV tier serve hybrid Mamba boundary states |
| 4 | [`a91004827b`](https://github.com/DronduletPalych/1Cat-vLLM/commit/a91004827b07) | [`dc3116a8d8`](https://github.com/DronduletPalych/1Cat-vLLM/commit/dc3116a8d880) | Trigger CI |
| 5 | [`0287d1189e`](https://github.com/DronduletPalych/1Cat-vLLM/commit/0287d1189ec6) | [`e221e60e92`](https://github.com/DronduletPalych/1Cat-vLLM/commit/e221e60e9268) | [Core] Load host-tier KV synchronously inside the scheduler step |
| 6 | [`edb464fd15`](https://github.com/DronduletPalych/1Cat-vLLM/commit/edb464fd15f4) | [`68ad1e10c1`](https://github.com/DronduletPalych/1Cat-vLLM/commit/68ad1e10c134) | [Core] Defer host-tier stores while a load is in flight |
| 7 | [`5ea59cc07e`](https://github.com/DronduletPalych/1Cat-vLLM/commit/5ea59cc07ef02165edf97e2fad4baa7b8c5b160f) | [`2fb389e2bf`](https://github.com/DronduletPalych/1Cat-vLLM/commit/2fb389e2bfffc23f6273ed35b9afcb7854bbad03) | [Core] Reserve the full ISL of admitted requests and drop stale recv signals |
| 8 | [`e4492c3e47`](https://github.com/DronduletPalych/1Cat-vLLM/commit/e4492c3e47ce5b06ee86d17bc6d733b6aa084a87) | [`28c57eb846`](https://github.com/DronduletPalych/1Cat-vLLM/commit/28c57eb8469e78e4f57efb0d015b1422b596a366) | [Core] Make the KV watermark reserve prompt plus max output |
| 9 | [`5c6d1c2291`](https://github.com/DronduletPalych/1Cat-vLLM/commit/5c6d1c229196) | [`1aa931de11`](https://github.com/DronduletPalych/1Cat-vLLM/commit/1aa931de11a0) | [Core] Clamp host-tier store keys to the request's live blocks |
| 10 | [`aa34b5f5b0`](https://github.com/DronduletPalych/1Cat-vLLM/commit/aa34b5f5b0ba) | [`2b58a79e9c`](https://github.com/DronduletPalych/1Cat-vLLM/commit/2b58a79e9c50) | [Core] Clamp store blocks to the request's live block IDs too |
| 11 | [`87234ea050`](https://github.com/DronduletPalych/1Cat-vLLM/commit/87234ea0507a19d39bab8a2184028e4adb81179b) | [`22b8cc41ec`](https://github.com/DronduletPalych/1Cat-vLLM/commit/22b8cc41ec5dcf06554e5cf555290385d029b5b1) | [Core] Measure the KV watermark in tokens, not blocks |
| 12 | [`ec91504b97`](https://github.com/DronduletPalych/1Cat-vLLM/commit/ec91504b9747) | [`8a29511e70`](https://github.com/DronduletPalych/1Cat-vLLM/commit/8a29511e709b) | [Core] Apply the token watermark on the step path too |
| 13 | [`7aa0a7db85`](https://github.com/DronduletPalych/1Cat-vLLM/commit/7aa0a7db859e) | [`b5dce1c4ac`](https://github.com/DronduletPalych/1Cat-vLLM/commit/b5dce1c4ac67) | [Core] Cap the watermark output reserve per request |
| 14 | [`b16c95a3e5`](https://github.com/DronduletPalych/1Cat-vLLM/commit/b16c95a3e50c) | [`771c7119cb`](https://github.com/DronduletPalych/1Cat-vLLM/commit/771c7119cb86) | [Core] Default the implicit max output to 32768 tokens |
| 15 | [`4b4b78cb8e`](https://github.com/DronduletPalych/1Cat-vLLM/commit/4b4b78cb8edc) | [`fd57037bf1`](https://github.com/DronduletPalych/1Cat-vLLM/commit/fd57037bf18a) | [Core] Call the watermark reserve helper through self |
| 16 | [`b979fecba7`](https://github.com/DronduletPalych/1Cat-vLLM/commit/b979fecba712) | [`98968e8517`](https://github.com/DronduletPalych/1Cat-vLLM/commit/98968e8517e4) | [Core] Cover the capped watermark output reservation |
| 17 | [`b4eb9fbe6d`](https://github.com/DronduletPalych/1Cat-vLLM/commit/b4eb9fbe6d1d) | [`c0efe17235`](https://github.com/DronduletPalych/1Cat-vLLM/commit/c0efe1723533) | [Core] Treat an in-flight CPU-tier store as a miss, not a deferral |
| 18 | [`2b938bf4d1`](https://github.com/DronduletPalych/1Cat-vLLM/commit/2b938bf4d1cc) | [`5f006d7549`](https://github.com/DronduletPalych/1Cat-vLLM/commit/5f006d75496f) | [Core] Never defer a CPU-tier hit on another request's in-flight load |
| 19 | [`a4dc453dea`](https://github.com/DronduletPalych/1Cat-vLLM/commit/a4dc453deada) | [`61210ed4a8`](https://github.com/DronduletPalych/1Cat-vLLM/commit/61210ed4a8cd) | [Core] Derive the host-tier load span from computed tokens |
| 20 | [`1e03261d25`](https://github.com/DronduletPalych/1Cat-vLLM/commit/1e03261d25bf) | [`40cef0fb4b`](https://github.com/DronduletPalych/1Cat-vLLM/commit/40cef0fb4b39) | [Core] Drop the temporary host-tier diagnostics |
| 21 | [`3354014f40`](https://github.com/DronduletPalych/1Cat-vLLM/commit/3354014f403b) | [`f8285c74bb`](https://github.com/DronduletPalych/1Cat-vLLM/commit/f8285c74bb64) | kv_offload: give Mamba groups private CPU pages with mixed block sizes |
| 22 | [`3aac954942`](https://github.com/DronduletPalych/1Cat-vLLM/commit/3aac95494271) | [`7dafc7ffa2`](https://github.com/DronduletPalych/1Cat-vLLM/commit/7dafc7ffa243) | [Core] Hand off one Mamba boundary state per prompt |
| 23 | [`2a375ba422`](https://github.com/DronduletPalych/1Cat-vLLM/commit/2a375ba422a7) | [`bb5d9c03f6`](https://github.com/DronduletPalych/1Cat-vLLM/commit/bb5d9c03f66c) | [KV Offload] Size tier slots per group, not per token block |
| 24 | [`61aa61cf19`](https://github.com/DronduletPalych/1Cat-vLLM/commit/61aa61cf1945) | [`32af1821c6`](https://github.com/DronduletPalych/1Cat-vLLM/commit/32af1821c600) | [KV Offload] Create the boundary-state store job again |
| 25 | [`ade5f9c688`](https://github.com/DronduletPalych/1Cat-vLLM/commit/ade5f9c68877) | [`3f02981fb0`](https://github.com/DronduletPalych/1Cat-vLLM/commit/3f02981fb0ea) | Load host-tier state before the mamba preprocess reads it |
| 26 | [`60d2413f75`](https://github.com/DronduletPalych/1Cat-vLLM/commit/60d2413f7515) | [`238d693955`](https://github.com/DronduletPalych/1Cat-vLLM/commit/238d693955f6) | Load the host tier before inputs are prepared |
| 27 | [`5f389fe0c8`](https://github.com/DronduletPalych/1Cat-vLLM/commit/5f389fe0c8b5) | [`d3be1a0d2e`](https://github.com/DronduletPalych/1Cat-vLLM/commit/d3be1a0d2e26) | Submit host-tier loads even when the connector is wrapped |
| 28 | [`5c0498c1c0`](https://github.com/DronduletPalych/1Cat-vLLM/commit/5c0498c1c06b) | [`07b446e03b`](https://github.com/DronduletPalych/1Cat-vLLM/commit/07b446e03b8c) | [KV Offload] Load the host tier before the Mamba preprocess in MRV2 |
| 29 | [`98bcc401be`](https://github.com/DronduletPalych/1Cat-vLLM/commit/98bcc401beb1) | [`d83ca76577`](https://github.com/DronduletPalych/1Cat-vLLM/commit/d83ca7657798) | [KV Offload] Account capacity of the filesystem tier directory |
| 30 | [`0259f5baf6`](https://github.com/DronduletPalych/1Cat-vLLM/commit/0259f5baf687) | [`76c7b3c421`](https://github.com/DronduletPalych/1Cat-vLLM/commit/76c7b3c421c1) | [KV Offload] Tell the grouped offload manager which groups hold Mamba state |
| 31 | [`8d40e73b3b`](https://github.com/DronduletPalych/1Cat-vLLM/commit/8d40e73b3be0) | [`192a72ff2a`](https://github.com/DronduletPalych/1Cat-vLLM/commit/192a72ff2a73) | [KV Offload] Cap Mamba checkpoints in the filesystem tier |
| 32 | [`c25034b247`](https://github.com/DronduletPalych/1Cat-vLLM/commit/c25034b24720) | [`128afac70b`](https://github.com/DronduletPalych/1Cat-vLLM/commit/128afac70bdc) | [KV Offload] Store the attention half together with a Mamba boundary state |
| 33 | [`c37fd59c38`](https://github.com/DronduletPalych/1Cat-vLLM/commit/c37fd59c38c0) | [`4915f54c92`](https://github.com/DronduletPalych/1Cat-vLLM/commit/4915f54c92bb) | [KV Offload] Report failed tier stores and per-tier lookup statistics |
| 34 | [`31a99b52b4`](https://github.com/DronduletPalych/1Cat-vLLM/commit/31a99b52b475) | [`061f59bf1d`](https://github.com/DronduletPalych/1Cat-vLLM/commit/061f59bf1d35) | [KV Offload] Verify CPU page contents behind VLLM_KV_OFFLOAD_SELFCHECK |
| 35 | [`87e015b6e1`](https://github.com/DronduletPalych/1Cat-vLLM/commit/87e015b6e145) | [`c0aa84d9e4`](https://github.com/DronduletPalych/1Cat-vLLM/commit/c0aa84d9e4db) | [KV Offload] Test the filesystem tier state cap wiring |
| 36 | [`0dfdafb6f3`](https://github.com/DronduletPalych/1Cat-vLLM/commit/0dfdafb6f3d2) | [`95a9b419ba`](https://github.com/DronduletPalych/1Cat-vLLM/commit/95a9b419baef) | [KV Offload] Admit the leading keys of a store batch when a group pool is full |
| 37 | [`c2f943c6b4`](https://github.com/DronduletPalych/1Cat-vLLM/commit/c2f943c6b466) | [`263ae085bb`](https://github.com/DronduletPalych/1Cat-vLLM/commit/263ae085bb6a) | [KV Offload] Expose the primary pool stats through the tiering manager |
| 38 | [`face7570f9`](https://github.com/DronduletPalych/1Cat-vLLM/commit/face7570f99e) | [`e49286518c`](https://github.com/DronduletPalych/1Cat-vLLM/commit/e49286518cc5) | kv_offload: write-through eviction for the tiered CPU pool |
| 39 | [`0efe400616`](https://github.com/DronduletPalych/1Cat-vLLM/commit/0efe400616ca) | [`b7aa1b67a1`](https://github.com/DronduletPalych/1Cat-vLLM/commit/b7aa1b67a178) | kv_offload: admit into a full primary pool by waiting for the eviction copy |
| 40 | [`06f948a613`](https://github.com/DronduletPalych/1Cat-vLLM/commit/06f948a6135d) | [`088e0b591c`](https://github.com/DronduletPalych/1Cat-vLLM/commit/088e0b591c56) | kv_offload: store a Mamba boundary state together with its attention half |
| 41 | [`1c5b5a268c`](https://github.com/DronduletPalych/1Cat-vLLM/commit/1c5b5a268cf46265e6fa5397e9df1688de3d979f) | [`988e0fe516`](https://github.com/DronduletPalych/1Cat-vLLM/commit/988e0fe5165bfaac15ec605e27de3a9441878dd1) | [KV Offload] Flush deferred stores before MRv2 zeroes reused blocks |
| 42 | [`15dbf1addd`](https://github.com/DronduletPalych/1Cat-vLLM/commit/15dbf1adddc7) | [`dd0ffb8239`](https://github.com/DronduletPalych/1Cat-vLLM/commit/dd0ffb823986) | [KV Offload] Describe the filesystem tier capacity limit |
| 43 | [`c8df9448ec`](https://github.com/DronduletPalych/1Cat-vLLM/commit/c8df9448ec38) | [`6540132c5f`](https://github.com/DronduletPalych/1Cat-vLLM/commit/6540132c5fc3) | GGUF: load Qwen3.5 clip mmproj natively (vision from GGUF, no HF repo) |
| 44 | [`90f57627ad`](https://github.com/DronduletPalych/1Cat-vLLM/commit/90f57627ad90) | [`fd6baafd38`](https://github.com/DronduletPalych/1Cat-vLLM/commit/fd6baafd389c) | fs tier: classify lookups that miss (never stored vs stored then evicted) |
| 45 | [`e729073557`](https://github.com/DronduletPalych/1Cat-vLLM/commit/e729073557ee3c587e23575bd777da4be8e67ea3) | [`4d52bc38e2`](https://github.com/DronduletPalych/1Cat-vLLM/commit/4d52bc38e20b60b16a05383c6254917a09781e16) | warmup: make the extra #1402 profiles skippable and non-fatal |
| 46 | [`7e8480cc76`](https://github.com/DronduletPalych/1Cat-vLLM/commit/7e8480cc7619) | [`4009b1c921`](https://github.com/DronduletPalych/1Cat-vLLM/commit/4009b1c921ca) | fs tier: write a finished request's prompt blocks through to the disk tier |
| 47 | [`bcce5ac0f0`](https://github.com/DronduletPalych/1Cat-vLLM/commit/bcce5ac0f03b) | [`6833dfa730`](https://github.com/DronduletPalych/1Cat-vLLM/commit/6833dfa73055) | docs: requirements for KV cache tiers, metrics and acceptance procedure |
| 48 | [`58460350ee`](https://github.com/DronduletPalych/1Cat-vLLM/commit/58460350ee56) | [`c7a3fd1c7f`](https://github.com/DronduletPalych/1Cat-vLLM/commit/c7a3fd1c7f0b) | docs: label block-probe counts and the FS contribution as an upper bound |
| 49 | [`1a99f340e6`](https://github.com/DronduletPalych/1Cat-vLLM/commit/1a99f340e612d44ba0c3d6fb244011a49710f0c1) | [`7684827bab`](https://github.com/DronduletPalych/1Cat-vLLM/commit/7684827bab0c5be71755548ba9ebc8dad82e7f40) | envs: register VLLM_MRV2_WARMUP_EXTENDED so the switch reaches the workers |

Конфликты были в коммитах **1, 7, 8, 11, 41, 45** таблицы.

## Конфликты и их разрешение

### 1. [`188989f91c`](https://github.com/DronduletPalych/1Cat-vLLM/commit/188989f91c514cdcb9d1b36cd4037a609a2d4c5f) → [`49ab72d6f9`](https://github.com/DronduletPalych/1Cat-vLLM/commit/49ab72d6f9f47b5224d03c68417d84c323dd6dd0) — [Core] Add KV cache watermark admission headroom

Шесть блоков. Upstream v1.5.4 между точками синхронизации переработал admission: добавил partial-prefix cache (`partial_prefix_cache`, `retention_policy`, `KVPartialHitDriftError`, `_reclaim_partial_blocks`), а наш коммит в ту же область вводит watermark-резерв.

| Файл, место | Upstream (v1.5.4) | Наш коммит | Разрешение |
|---|---|---|---|
| `kv_cache_manager.py`, `KVCacheManager.__init__` | 4 новых параметра: `partial_prefix_cache`, `num_speculative_tokens`, `retention_policy`, `enable_decode_ckpt` | параметр `watermark: float` и `self.watermark_blocks` | union: все 5 параметров, обе инициализации |
| `kv_cache_manager.py`, `can_allocate` | расчёт `partial_entry` (partial prefix hit) | расчёт `watermark_blocks` (только WAITING/PREEMPTED при активных запросах) | union: обе переменные — обе нужны ниже |
| `kv_cache_manager.py`, ветка `full_sequence_must_fit` | `return None` при `not _reclaim_partial_blocks(...)` | `return None` при `num_blocks_to_allocate + watermark_blocks > free` | оба условия — независимые причины отказа → объединены в одно через `or` |
| `kv_cache_manager.py`, step-ветка | то же `_reclaim_partial_blocks` | тот же watermark-провер | union через `or`, как выше |
| `sched/scheduler.py`, конструктор `KVCacheManager` | 4 новых аргумента | `watermark=self.scheduler_config.watermark` | union |
| `sched/scheduler.py`, вызов `allocate_slots` | вызов обёрнут в `try/except KVPartialHitDriftError` (их фикс stale partial hit: дроп энтри и requeue) | аргумент `has_scheduled_reqs=bool(self.running)` | сохранены и обёртка, и аргумент |

### 2. [`5ea59cc07e`](https://github.com/DronduletPalych/1Cat-vLLM/commit/5ea59cc07ef02165edf97e2fad4baa7b8c5b160f) → [`2fb389e2bf`](https://github.com/DronduletPalych/1Cat-vLLM/commit/2fb389e2bfffc23f6273ed35b9afcb7854bbad03) — [Core] Reserve the full ISL of admitted requests and drop stale recv signals

Один блок в `full_sequence_must_fit`: в одну точку упали upstream'ов `copy_last`-цикл (частные copy-on-write блоки partial hit, увеличивают `num_blocks_to_allocate`) и наш расчёт `other_reserved` (сумма ISL-резервов других admitted запросов из `self._reserved_blocks`). Оба блока оставлены, порядок: сначала цикл (он меняет `num_blocks_to_allocate`), затем `other_reserved`.

### 3. [`e4492c3e47`](https://github.com/DronduletPalych/1Cat-vLLM/commit/e4492c3e47ce5b06ee86d17bc6d733b6aa084a87) → [`28c57eb846`](https://github.com/DronduletPalych/1Cat-vLLM/commit/28c57eb8469e78e4f57efb0d015b1422b596a366) — [Core] Make the KV watermark reserve prompt plus max output

Один блок. Это эволюция **нашей же** логики из пункта 2: HEAD содержал старую схему (цикл + `other_reserved` + проверка free-блоков + `_reclaim_partial_blocks`), а коммит заменяет схему резервирования: `new_reserved = num_blocks_to_allocate + output_blocks`, `budget = num_gpu_blocks - watermark_blocks`, `used + new_reserved > budget`. Взята новая схема форка (старая `other_reserved` удалена целиком — её заменяет `used`), при этом сохранены upstream'ов `copy_last`-цикл и условие `or not _reclaim_partial_blocks(...)`.

### 4. [`87234ea050`](https://github.com/DronduletPalych/1Cat-vLLM/commit/87234ea0507a19d39bab8a2184028e4adb81179b) → [`22b8cc41ec`](https://github.com/DronduletPalych/1Cat-vLLM/commit/22b8cc41ec5dcf06554e5cf555290385d029b5b1) — [Core] Measure the KV watermark in tokens, not blocks

Три блока:
- `kv_cache_manager.py`: блок-бюджет (пункт 3) заменён токен-бюджетом `kv_cache_tokens - watermark_tokens` с guard'ом `if self.kv_cache_tokens` (пools с неопределённым размером не трогает). Взян наш новый вариант, reclaim-условие сохранено внутри guard'а.
- `sched/scheduler.py`, импорты: upstream импортирует `is_lookahead_block_hashing_enabled`, мы — `get_max_concurrency_for_kv_cache_config` → union в один импорт.
- `sched/scheduler.py`, `__init__`: upstream'ов `self.hash_block_size = hash_block_size` + наш расчёт `kv_cache_tokens` → union.

### 5. [`1c5b5a268c`](https://github.com/DronduletPalych/1Cat-vLLM/commit/1c5b5a268cf46265e6fa5397e9df1688de3d979f) → [`988e0fe516`](https://github.com/DronduletPalych/1Cat-vLLM/commit/988e0fe5165bfaac15ec605e27de3a9441878dd1) — [KV Offload] Flush deferred stores before MRv2 zeroes reused blocks

Один блок: импорты в `tests/v1/worker/test_gpu_model_runner_v2_kv_zero.py`. Upstream добавил `from vllm.config.sm70_moe import Sm70UnquantizedMoEConfig`, наш коммит — `from vllm import envs`. Union: оба импорта.

### 6. [`e729073557`](https://github.com/DronduletPalych/1Cat-vLLM/commit/e729073557ee3c587e23575bd777da4be8e67ea3) → [`4d52bc38e2`](https://github.com/DronduletPalych/1Cat-vLLM/commit/4d52bc38e20b60b16a05383c6254917a09781e16) — warmup: make the extra #1402 profiles skippable and non-fatal

Один блок в цикле warmup-профилей `vllm/v1/worker/gpu/warmup.py`:
- **Upstream** (v1.5.4): профили > 0 используют `profile_sampling_params` (без prompt logprobs — не строят FP32-логиты на 605 MiB на последнем pipeline-ранке).
- **Наш коммит**: профиль 0 остаётся fatal'ным (`_warmup_cycle`, `with_grammar=True`), профили > 0 — non-fatal через `_warmup_cycle_optional`, mixed-prefill цикл под гейтом `extended_warmup` (env `VLLM_MRV2_WARMUP_EXTENDED`).

Разрешение: профиль 0 — `_warmup_cycle` с полными `sampling_params`; профиль > 0 — `_warmup_cycle_optional` **с `profile_sampling_params`** (upstream'ово сбережение памяти сохранено), `with_grammar=False`; mixed-цикл — `if extended_warmup and not model_runner.is_pooling_model`.

## Проверка

1. **Синтаксический гейт:** все изменённые в ребейзе `.py` файлы — `ast.parse` чист.
2. **Маркеры форка на месте:** `VLLM_MRV2_WARMUP_EXTENDED` в `envs.py`, `--watermark` в `arg_utils.py`, `watermark` в `config/scheduler.py`, `vllm/v1/kv_offload/tiering/fs/`.
3. **Тесты затронутых областей** (scheduler, partial prefix, Mamba sparse retention, kv-zero, offloading connector, warmup) прогнаны на обеих ветках:

| | До ребейза (`1a99f340e6`) | После (rebase/v1.5.4) |
|---|---|---|
| падений | 150 | 154 |

- **Починены ребейзом (2):** `test_gpu_model_runner_v2_kv_zero.py::test_v2_initializes_kv_zeroer_from_all_attention_groups`, `::test_v2_warms_qwen38_mtp_moe_prefill_and_decode_shapes`.
- **«Новые» (6):** `test_mamba_sparse_retention.py::test_sparse_retains_both_prompt_replay_boundaries[0-True-800/801]`, `[16000-True-800/801]`, `test_partial_prefix_cache.py::test_sm70_vision_default_unlimited_images_and_vit_sdpa_order`, `test_gpu_warmup_blocks.py::test_kernel_prefill_warmup_runs_default_batch_and_extra_profile`. **Все 6 падают и на чистом upstream v1.5.4 без наших патчей** (mamba-семейство на чистом v1.5.4 валит 33 кейса) — не регрессия ребейза.
- **Базовые 150 падений** — известная offload-defer регрессия, фикс в работе на ветке `fix/offload-defer-regression` (отдельный клон, WIP).

## Открытые вопросы

- Тег следующего релиза форка (по потоку v1.5.16…v1.5.25 → **v1.5.26**) не поставлен.
- Ветка `fix/offload-defer-regression` стоит на старом main — после завершения её фикса нужен ребейз на новый main.
- Деплой на caco-zero-three (nix-пакет, пин на tip) — за пользователем.
