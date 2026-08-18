# CyberChair — архитектурный аудит перед рефакторингом

Дата аудита: 2026-08-18  
Объект: текущее рабочее дерево `main` с пользовательскими незакоммиченными изменениями  
Режим: read-only аудит; исходный код и архитектура не изменялись

## Метод и baseline

Аудит выполнен по фактическому рабочему дереву, а не только по `README.md` или последнему commit. На старте уже были изменены 18 tracked-файлов и добавлены 4 untracked-файла; эти изменения приняты как baseline и не затрагивались.

Проверки:

| Проверка | Результат | Время / замечания |
|---|---:|---|
| `venv/bin/python -m pytest -q` | `318 passed, 57 subtests passed` | 22.97 s pytest; 23.38 s wall |
| compile check (`python -m compileall`) | passed | без вывода и ошибок |
| `git diff --check` | passed | whitespace errors нет |
| warnings pytest | 0 в стандартном выводе | отдельной секции warnings нет |
| lint / format / type check | не настроены | нет `pyproject.toml`, `ruff.toml`, `mypy.ini`, `setup.cfg`, `tox.ini`; соответствующие CLI в окружении не найдены |

Дополнительная диагностика:

- AST inventory модулей, классов и публичных методов;
- статический поиск Telegram/LLM/SQLite call sites;
- read-only анализ реальных SQLite-файлов;
- `EXPLAIN QUERY PLAN` для ключевых запросов;
- синтетическое трассирование SQL на временных БД без сети;
- локальный профиль Pillow-рендера;
- инспекция установленного `pyTelegramBotAPI==4.36.0`.

Никаких архитектурных исправлений, удалений или диагностических правок исходников не делалось.

---

# A. Executive summary

1. Baseline зелёный и тестовый контур заметно сильнее среднего для проекта такого размера: 318 тестов покрывают policy, persona, pending, provider adapters, media, memory и многие single-producer ветки.
2. Архитектура остаётся развиваемой как modular monolith; rewrite и микросервисы не нужны. Итоговый verdict: **C — накопился серьёзный architectural debt, нужен крупный, но staged refactor**.
3. Правило `MAX 1 LLM CALL PER USER EVENT` фактически нарушается: обычное сообщение может синхронно вызвать summary в `ingest()`, а затем conversational LLM в `maybe_reply()`. Диагностический сценарий воспроизвёл `summary_calls=1 + generate_calls=1` для одного event.
4. Direct и pending ветки специально подавляют summary (`refresh_memory=False`) и в основном действительно соблюдают один финальный producer и один conversational LLM-call. Provider retry или переключения на второй provider нет.
5. `LearningService` — фактический god-object не только по размеру (1930 строк), а по 89 методам, 60 public methods, созданию 14+ подсистем и владению routing, provider, memory, media, pending, telemetry, settings и runtime caches.
6. `bot.py` (1363 строки) одновременно Telegram adapter, composition root, event router, command policy, scheduler bootstrap, global user store и часть бизнес-логики. Сам Telegram delivery расположен правильно; priority/arbitration и feature decisions — нет.
7. Decision state фрагментирован между `ConversationDecision`, direct `ResponseDecision`, `MediaDecision`, `AutonomousDecision`, `PendingConversation`, `PersonaSelection`, raw `str | None`, tuples и booleans в `bot.py`. Это усложняет доказательство single-producer и commit-after-delivery.
8. Один direct event делает 22–31 SQLite connections и десятки SQL statements. Причина — повторное построение state/context и каждая repository operation в отдельном connection/transaction. При текущих объёмах это ещё быстро, но coupling и failure surface уже велики.
9. SQLite подходит текущей модели одного процесса: per-chat files, WAL, `synchronous=NORMAL`, короткие transactions и in-process `RLock` дают приемлемую базу. PostgreSQL сейчас не обоснован.
10. Concurrency не ограничена на уровне domain: Telegram создаёт 2 worker threads, scheduler работает отдельным daemon thread, typing создаёт дополнительные threads. Нет global LLM/media semaphore и per-chat event lock; user reply, summary, autonomous и до трёх Pillow pipelines могут пересекаться.
11. Media pipeline ограничивает вход до 48 млн pixels, но RGBA-декодирование может занять около 192 MB на один image до resize. Два Telegram workers плюс scheduler дают реалистичный OOM-класс риска. Типичный template render занял median 108 ms и дал process peak RSS около 50 MB.
12. Persistence происходит до подтверждённой Telegram delivery: generated/routing/media cooldown/pending могут быть записаны до send; pending continuation удаляется до generation. Ошибка send или crash оставляют state, не соответствующий факту доставки.
13. Memory design в целом осмыслен: raw messages и generated capped по 50, stable memory capped по 40, summary cursor транзакционный. Но `memory_candidates`, `daily_summaries` и `llm_calls` не имеют retention; старые daily summaries записываются, но production generation читает только текущий день.
14. Security baseline неудовлетворителен: tracked path `.env.example` в текущем worktree содержит непустые значения, похожие на реальные Telegram/XAI credentials; Docker Compose использует этот файл как `env_file`, а Dockerfile копирует его в image. По существующей Git history regex-признаков непустых ключей не найдено, то есть риск пока локальный/uncommitted, но deployable.
15. Сильные стороны, которые следует сохранить: persona/TrollMode 50/50, provider-neutral request objects, local fallback без retry, per-chat SQLite isolation, bounded raw memory, atomic summary finalize, data-driven media catalog, deterministic media rolls и transport-free policy/media selectors.

---

# B. Current architecture

## B1. Фактическая схема

```text
Telegram Bot API
  -> TeleBot polling (2 worker threads)
     -> bot.py handlers / command priority / delivery
        -> LearningService (composition + orchestration + state facade)
           -> ChatStateAnalyzer -> ConversationPolicy
           -> DirectAddressRouter -> LocalIntentClassifier
           -> PendingConversation helpers
           -> PersonaBuilder + MemeLexicon + LexicalDiversityTracker
           -> configured LLM provider
                -> ResponsesLLMProvider
                   -> GrokProvider | OpenAIGenerator
           -> LocalResponder | LocalGenerator -> MarkovModel
           -> MemoryService -> ChatRepository (per-chat SQLite)
           -> MediaService -> MediaCatalog / MemeSourceSelector
           -> MemeRenderer (Pillow + temp files)

  scheduler.py daemon thread
     -> utility messages / movie quotes / #FREEKUCHER
     -> LearningService.maybe_autonomous()
     -> bot.py autonomous delivery
```

Это modular monolith с одним большим orchestration object, а не layered core с узким adapter boundary.

## B2. Module map

Обозначения в колонках: DB — прямой/косвенный SQLite access; LLM — может инициировать network call; TG — Telegram API.

| Module | Responsibility и public API | Основные dependencies / callers | Owned state | DB | LLM | TG |
|---|---|---|---|:---:|:---:|:---:|
| `bot.py` | handlers, admin UI, priority routing, delivery, downloads, startup | imports `LearningService`, `scheduler`, `utils`; entry point | global bot, known users, locks, identity cache, feature cooldowns | via service | via service | yes |
| `scheduler.py` | 30-sec loop, quotes, workday utilities, autonomous callback | called by `bot.main()` | module-level last-event/quote/random state | via callbacks | via callback | yes |
| `env_loader.py` | OS/.env/.env.example loading | imported before Telegram setup | process environment | no | no | no |
| `messages.py` | static utility/quote/reaction content | bot/scheduler | constants | no | no | no |
| `utils.py` | chair trigger and work calendar/time helpers | bot/scheduler | 2026 calendar constants | no | no | no |
| `learning/service.py` | all orchestration, routing, generation, pending, media, settings, telemetry | called by bot/scheduler/tests; imports nearly every learning module | repositories, models, per-chat decisions, provider clients, cooldown helpers, meme source map | yes | yes | hook only |
| `learning/repository.py` | per-chat SQLite schema and all persistence APIs | service, memory, media, diagnostics | path, chat id, RLock | yes | no | no |
| `learning/settings.py` | immutable global config from environment/code | service and all policies | dataclass only | no | no | no |
| `learning/chat_state.py` | derive activity/type/topic/target from recent dialogue | service | none | via repository | no | no |
| `learning/conversation_policy.py` | reply probability/style/intensity/target | service, autonomous policy | none | no | no | no |
| `learning/direct_address.py` | local intent and P1/P2/P3 producer preference | service | classifier only | no | no | no |
| `learning/pending_conversation.py` | pending data model and parsing/matching functions | service/local responder | none | no | no | no |
| `learning/memory_service.py` | short context, relevant memory, incremental summary | service/chat state | timezone/provider resolver | yes | summary | no |
| `learning/persona.py` | persona/purpose/budget/callback/meme selection, `GenerateRequest` | service, compatibility prompt helper | per-chat recent meme deques | caller fetches | no direct call | no |
| `learning/llm_provider.py` | provider protocol and request/result types | adapters/persona | none | no | interface | no |
| `learning/responses_provider.py` | shared Responses API execution, cleanup, usage parsing | Grok/OpenAI adapters | client via subclass | usage callback | yes | no |
| `learning/grok_provider.py` | xAI client, models, reasoning, cache key, summary schema | factory | cached OpenAI-compatible client | no | yes | no |
| `learning/openai_generator.py` | OpenAI client and request kwargs | factory | cached client | no | yes | no |
| `learning/provider_factory.py` | construct/select known providers | service/tests | none | no | no | no |
| `learning/local_responder.py` | deterministic direct fallback and TrollMode roast | service | lexical tracker ref | reads repository | no | no |
| `learning/generator.py` | local Markov/combine/mutation modes | service | settings/RNG | no | no | no |
| `learning/markov.py` | second-order weighted model | LocalGenerator/service | transition lists per model | no | no | no |
| `learning/meme_lexicon.py` | versioned meme concepts/selection | persona/local/media | immutable entries | no | no | no |
| `learning/lexical_diversity.py` | phrase feature/penalty/scoring | guard/persona/local | configuration only | no | no | no |
| `learning/response_quality.py` | post-generation lexical/incomplete guard | service | tracker ref | no | no | no |
| `learning/filters.py` | generated-output safety/copy/length validation | service/generator | none | no | no | no |
| `learning/preprocessing.py` | normalization, rejection, secret/link filtering | most text modules | regex constants | no | no | no |
| `learning/media_catalog.py` | data-driven asset catalog | service/media/renderer | in-memory catalog | no | no | no |
| `learning/media_service.py` | local media scoring/decision/cooldowns | service | settings/catalog/RNG | yes | no | no |
| `learning/meme_sources.py` | explicit meme caption source ranking/anti-repeat | service | per-chat bounded deques | caller supplies rows | no | no |
| `learning/meme_renderer.py` | Pillow render and generated-file cleanup | service/bot/tests | catalog/output dir/fonts | no | no | no |
| `learning/autonomous_policy.py` | local autonomous eligibility/probability | service | none | no | no | no |
| `learning/chat_action.py` | Telegram typing/upload actions and refresh threads | bot/service hook | active per-chat thread map | no | no | yes |
| `learning/triggers.py` | in-memory cooldown/hour caps/history | service | defaultdicts + RLock | no | no | no |
| `learning/llm_prompts.py` | summary prompt; legacy generation wrapper | memory/tests | none | no | no | no |
| `learning/cost_diagnostics.py` | CLI/report formatting | operator | none | yes | no | no |
| `learning/__init__.py` | broad compatibility re-export surface | bot/tests | none | no | no | no |

## B3. Dependency observations

- `LearningService` imports 20 internal modules and constructs the graph itself.
- `bot.py` depends on the large facade, but bypasses it for utility state, priority routing and some responses.
- Policy, direct router, media selector and provider adapters themselves are reasonably transport-free.
- `ChatStateAnalyzer` depends on `MemoryService` and repository instead of an immutable event context.
- `PersonaBuilder` does **not** query DB itself. The duplicate reads are performed by `LearningService`/`MemoryService` before calling it.
- `MediaService` does query repository through its public methods and independently rebuilds media/history signals.

---

# C. Event lifecycle

## C1. Text message: exact current order

```text
Telegram update
  -> TeleBot worker thread
  -> effective_message_text()
  -> #FREEKUCHER priority (may send and stop)
  -> reject foreign s g m / s g d
  -> exact "с м стул" (plan -> download/render -> photo send -> commit)
  -> exact "с стул" workday response
  -> Sglypa bot route
  -> remember_user() global JSON state
  -> "стул голос" route
  -> fetch/cache bot identity
  -> calculate who/direct/pending candidates in bot.py
  -> ingest message
       direct/pending: refresh_memory=False
       other: refresh_memory=True -> possible synchronous summary LLM
  -> "к кто" command
  -> pending continuation
  -> direct address/reply
  -> activity sampling
  -> creator random lane
  -> rare canned trigger
  -> ordinary maybe_reply()
       ChatStateAnalyzer
       ConversationPolicy
       callbacks + MemeLexicon + MediaService
       one of Media / configured LLM / Markov
  -> send_contextual_response()
  -> Telegram send
```

Important deviations from the conceptual target flow:

- direct flow classifies intent **before** `ConversationPolicy`; ordinary flow has no separate intent stage;
- pending priority is partly in `bot.py`, but replies to CyberChair are initially classified as direct and only redirected to pending inside `maybe_direct_reply()`;
- persistence/telemetry/cooldown generally happens **before** Telegram delivery, not after;
- quality guard runs only for generated text/caption, not all final producers;
- normalization is not an event object: `bot.py`, service, pending and persona independently normalize/extract text;
- `к кто` is handled after `ingest()`, so this utility phrase can enter memory/Markov even though other control commands do not.

## C2. Other update types

- `photo`: caption `с м стул` owns the event; otherwise only image metadata is stored. Ordinary photo captions do not enter text routing.
- image `document`: stores image metadata; GIF MIME additionally stores GIF metadata.
- `animation`: stores Telegram file metadata only.
- `sticker`: stores Telegram file metadata only.
- admin commands: separate decorated handlers; Telegram stops at the first matching handler.
- callback query: admin recheck, per-chat setting mutation, Telegram edit/answer.
- scheduler: independent 30-second loop; can send quote, utility, `#FREEKUCHER`, autonomous response and (legacy callback permitting) random media.

## C3. Duplicate work in a substantive direct event

Typical direct AI path:

1. `ingest()` writes message and reads count.
2. `ChatStateAnalyzer` reads short-term dialogue, latest message and current-day summary.
3. `generate_llm()` reads current-day summary and recent generated text.
4. `_dialogue_context()` reads short-term dialogue again.
5. `short_term_context()` calls `relevant_memory()`, which reads summary and stable memories.
6. `generate_llm()` separately calls `relevant_memory()` again for stable memory.
7. validation reads generated history again.
8. routing, response mode and generated result are persisted in separate transactions.

Thus the same summary can be read three times and short-term dialogue twice for one event. Stable facts can appear once in the context header and again as selected callbacks.

## C4. `bot.py`: adapter boundary audit

| Correctly belongs in Telegram adapter | Potentially belongs in core/orchestrator |
|---|---|
| Telegram handler registration and update-type extraction | relative priority of special commands, pending, direct, activity, creator and ordinary routes |
| bot identity/member lookup and Telegram DTO conversion | repeated direct-address/special-phrase decisions |
| file metadata/download and `send_message`/`send_photo`/`send_animation`/`send_sticker` | response-mode and producer arbitration |
| callback-query acknowledgement and keyboard rendering | pending priority/continuation decisions |
| chat action transport hook | media-versus-text selection and fallback policy |
| polling/reconnect bootstrap | cost/activity gates that decide whether generation happens |
| mapping Telegram delivery errors to a typed result | commit/cooldown/telemetry side effects triggered before delivery |

`bot.py` should remain the composition/transport entry point. The problem is not its awareness of Telegram; it is that the same domain decision is partly made in `bot.py` and partly repeated in `LearningService`.

---

# D. Dependency map

```text
bot.py
  +-> scheduler.py
  +-> LearningService
        +-> ChatRepository
        +-> MemoryService -------> llm_prompts -> SummarizeRequest
        +-> ChatStateAnalyzer ---> MemoryService / repository
        +-> ConversationPolicy
        +-> AutonomousPolicy ----> ConversationPolicy
        +-> DirectAddressRouter
        +-> PersonaBuilder ------> MemeLexicon / GenerateRequest
        +-> LocalResponder ------> MemeLexicon / pending helpers / repository
        +-> LocalGenerator ------> MarkovModel / filters
        +-> ResponseQualityGuard -> LexicalDiversityTracker
        +-> MediaService --------> repository / MediaCatalog
        +-> MemeSourceSelector
        +-> MemeRenderer --------> MediaCatalog / Pillow
        +-> provider factory ----> GrokProvider / OpenAIGenerator
                                  \-> ResponsesLLMProvider -> OpenAI SDK Responses API
```

The high-risk edges are not the pure policy modules; they are orchestration-to-repository fan-out, pre-delivery commits, and transport-level reimplementation of domain priority.

---

# E. State ownership

| State | Actual owner | Persistence / bounds | Observation |
|---|---|---|---|
| raw chat messages | `ChatRepository.messages` | SQLite, newest 50 | bounded; unsummarized overflow can be lost during provider outage |
| generated actions/text | `generated` | SQLite, newest 50 | used as dialogue, cooldown and quality history |
| derived chat state | `ChatStateAnalyzer` + service last-state dict | transient | recalculated repeatedly; cache unbounded by chat count |
| conversation decisions | service `_last_*decision` dicts | transient | diagnostic only; no unified plan |
| memory summary | `daily_summaries` + `summary_state` | unbounded daily rows | only current logical day read in live generation |
| stable memory | `long_memories` | capped at 40 | used by generation/local roast/callbacks |
| candidates | `memory_candidates` | unbounded | promoted candidates remain duplicated here |
| pending | `pending_conversations`, key=user | one row/user, lazy TTL cleanup | cleared before successful continuation; race-prone |
| per-chat settings | `settings` table | small | 7 effective categories; repeated DB reads |
| global settings | `LearningSettings` | process memory | env/code/config precedence is inconsistent for timezone |
| media history | gifs/stickers/chat_images/usage tables | caps 1000/1000/2000/100/500 | metadata only for Telegram media; sensible caps |
| meme anti-repeat | `PersonaBuilder` / `MemeSourceSelector` deques | bounded per chat, in memory | reset on process restart |
| lexical history | derived from `generated` | bounded 40/50 | no separate persistent store |
| routing telemetry | `routing_events` | 31-day retention | no event correlation |
| LLM telemetry | `llm_calls` | unbounded | usage only, no prompts/secrets |
| known users | `bot_state.json` global map | unbounded | not per chat; removal in one chat removes globally |
| cooldowns | TriggerEngine, bot globals, generated/media tables | mixed transient/persistent | multiple independent mechanisms and restart semantics |

---

# F. LLM map

## F1. Network call boundary

There is one physical creation point: `ResponsesLLMProvider._create()` -> `client.responses.create(...)` (`learning/responses_provider.py:23-24`). It is reached through:

- `ResponsesLLMProvider.generate()` for conversational/autonomous/meme requests;
- `ResponsesLLMProvider.summarize()` for memory summary.

No provider retry, validation retry, refusal retry or automatic second-provider fallback exists. Local generation may retry once, but it is not LLM.

## F2. Call-purpose map

| Caller / event | Purpose metadata | Model / reasoning | Default max output | Context | Retry | Fallback |
|---|---|---|---:|---|---|---|
| direct substantive | `reply` / `troll_user` | selected provider; Grok reply model low | 180–480 by purpose | 10/20 messages + memory | none | LocalResponder preserving behavior mode |
| pending continuation | `reply` | reply model low | usually 360 | constructed pending context + history | none | deterministic local continuation |
| ordinary random AI | `random_reply` | reply model low | 120 | 10 or state-based limit | none | none |
| creator | `creator` | reply model low | 120 | targeted context | none | none |
| repeated chair legacy API | `reply` / `stul_cooldown` | reply model low | 120–dynamic | targeted context | none | Markov is preselected alternative, not post-AI retry |
| chair question legacy API | `question` | reply model low | dynamic | targeted context | none | none |
| voice story | `voice_story` | reply model low | 150 hardcoded | no dialogue context | none | none |
| Sglypa | `sglypa` | reply model low | 50 | target + context | none | none |
| autonomous | `autonomous` | reply model low | 90 | 8 messages / 2600 chars | none | none after AI selection |
| manual meme | `meme_caption` / call type `meme` | reply model low | min(64, configured 50) | quote/hint/image caption + memory | none | old/fresh/callback/Markov local caption |
| memory refresh | call type `summary` | Grok summary model, `none`; OpenAI regular model | 240 | prior summary <=1800 chars + fragment <=20000 chars | none | cursor remains pending |
| admin `/generate` | requested purpose | selected provider | dynamic | normal context | none | user-facing failure text |

Provider defaults in code:

- Grok reply: `XAI_REPLY_MODEL` default `grok-4.5`, reasoning `low`;
- Grok summary: `XAI_SUMMARY_MODEL` default `grok-4.3`, reasoning `none`, strict JSON schema;
- OpenAI: `OPENAI_MODEL` default `gpt-5.6-luna`, reasoning `none`, verbosity low;
- timeouts: Grok 60 s, OpenAI 20 s.

Input context is bounded by message count and characters, but `MemoryService.short_term_context()` prepends a memory header up to 2200 chars in addition to the row character budget. Summary can be roughly 21.8k chars plus instructions.

## F3. Correct formulation of the invariant

Recommended future invariant:

> For one Telegram user event, all foreground work together may reserve at most one LLM permit. Summary is not a background call today; if triggered synchronously by `ingest`, it counts against that event. Autonomous scheduler events have their own event id and one-call limit. A future truly queued summary is a background event and must use its own bounded semaphore.

Current status:

- direct/pending/manual meme: at most one LLM call;
- autonomous tick: at most one LLM call;
- summary refresh operation: one LLM call;
- ordinary user event: **can make two calls** (summary + random reply), reproduced diagnostically;
- callbacks, intent, quality and media selection: zero LLM calls.

---

# G. DB map

## G1. Tables and retention

| Table | Purpose | Current main-chat rows | Bound / cleanup |
|---|---|---:|---|
| `messages` | raw human short-term | 50 | hard cap 50 |
| `generated` | bot text/actions | 50 | hard cap 50 |
| `settings` | per-chat overrides | 15 | naturally small |
| `daily_summaries` | JSON per day | 11 | unbounded |
| `summary_state` | cursor/pending time | 1 | singleton |
| `long_memories` | promoted stable facts | 40 | hard cap 40 |
| `memory_candidates` | candidate evidence | 396 | unbounded |
| `gifs` | Telegram metadata | 58 | configurable cap 1000 |
| `stickers` | Telegram metadata | 66 | configurable cap 1000 |
| `chat_images` | Telegram image metadata | 122 | configurable cap 2000 |
| `chat_image_usage` | caption/image anti-repeat | 5 | last 500 |
| `media_metadata` | manual tags | 0 | no orphan cleanup |
| `media_usage` | cooldown/anti-repeat | 44 | last 100 |
| `pending_conversations` | user pending state | 0 | lazy TTL delete |
| `llm_calls` | usage/cost | 431 | unbounded |
| `routing_events` | counters | 150 | 31 days |
| `scheduled_events` | persistent claims | 21 | 14 days |
| `chat_stats` | aggregated counts | 7 | inner JSON caps for words/minutes |

Largest real DB: 768 KiB, 192 pages, 50 freelist pages. All inspected active DBs report WAL mode. There is no schema-version table; migrations are `CREATE IF NOT EXISTS` plus conditional `ALTER TABLE` in repository initialization.

## G2. Connection lifecycle and transaction model

- Every repository method opens a new `sqlite3.connect(timeout=5)`, sets WAL and `synchronous=NORMAL`, then closes it.
- Per-repository `RLock` serializes methods inside one process.
- Multi-step event semantics are not transactional because each call is a separate transaction.
- WAL is appropriate for this single-process/read-heavy model, but setting `journal_mode` on every connection is unnecessary lock/latency surface.

Synthetic per-event SQL trace (fake provider, temp DB, no network):

| Scenario | Connections | SELECT/WITH | Writes | LLM generate | LLM summary | Local wall time |
|---|---:|---:|---:|---:|---:|---:|
| bare `стул` direct | 22 | 22 | 19 | 0 | 0 | 14.0 ms |
| direct social | 31 | 31 | 19 | 0 | 0 | 17.9 ms |
| direct useful | 25 | 24 | 20 | 1 | 0 | 15.4 ms |
| direct troll_user | 25 | 24 | 20 | 1 | 0 | 15.1 ms |
| meme plan | 16 | 15 | 1 | 1 | 0 | 10.3 ms |
| meme commit after hypothetical send | 4 | 2 | 8 | 0 | 0 | 3.6 ms |
| pending origin | 23 | 17 | 15 | 1 | 0 | 13.5 ms |
| pending continuation | 24 | 18 | 16 | 1 | 0 | 16.7 ms |
| summary ingest | 13 | 27 | 13 | 0 | 1 | 10.6 ms |

These timings measure local orchestration only and are not production latency. They demonstrate call amplification.

## G3. Query plans

Real DB median timings at current size:

- recent messages: 0.065 ms;
- meme-source join: 0.098 ms;
- chat-image candidates: 1.20 ms;
- LLM report: 0.079 ms.

`EXPLAIN QUERY PLAN` findings:

- short-term dialogue uses `idx_messages_created` and `idx_generated_created`, then a temp B-tree for union ordering;
- meme-source join relies on an automatic covering index for `reply_to_message_id`;
- `chat_image_candidates` executes two correlated subqueries per candidate and scans messages for each;
- `chat_image_usage(file_unique_id, caption_hash)` is a full scan;
- `generated WHERE created_at >= ? AND kind = ?` uses only created-at index;
- pending TTL cleanup uses `idx_pending_created` correctly.

Useful future indexes, only after measurement at larger data: `messages(reply_to_message_id)`, `chat_image_usage(file_unique_id, caption_hash)`, `generated(kind, created_at)`. At current bounded table sizes they are not urgent.

## G4. SQLite verdict

SQLite is suitable now because the deployment is a single bot process, writes are small, data is per-chat, and WAL is enabled. PostgreSQL becomes justified only if there are multiple bot processes writing the same chat DB, operational analytics over many chats, remote multi-host access, or sustained lock/retention pressure. None is established by current code/data.

---

# H. Concurrency map

```text
main thread
  -> TeleBot polling
      -> worker pool: 2 handler threads (TeleBot defaults)

daemon scheduler thread
  -> every 30 s: claims/sends utilities, evaluates autonomous
  -> may call provider and Pillow delivery concurrently with handlers

per active chat-action
  -> daemon refresh thread every ~4 s

shared process objects
  -> LearningService caches/deques/dicts (mostly no encompassing lock)
  -> provider clients
  -> per-chat ChatRepository RLock (method-level only)
```

There are no executors, asyncio tasks, semaphores or timers. Relevant locks:

- repository `RLock` per chat object;
- LearningService `RLock` only around repository/model cache creation and some cleanup;
- TriggerEngine `RLock`, but not all trigger reads/writes are enclosed as an event transaction;
- ChatActionManager `RLock` for refresh ownership;
- separate bot locks for known-user state and a few feature cooldowns.

Possible same-chat overlap today:

- user reply generation + second user reply generation (2 Telegram workers);
- user reply + scheduler autonomous;
- user reply + synchronous summary in another worker;
- two pending continuations reading the same pending row;
- two media choices passing cooldown before either commits;
- up to two handler Pillow pipelines plus one scheduler Pillow pipeline;
- provider calls from both workers plus scheduler.

Future controls are justified, in this order:

1. global LLM semaphore/counter with explicit call classes;
2. media/Pillow semaphore (likely 1 at current RAM envelope);
3. per-chat orchestration lock or serial queue around plan/commit, not around Telegram polling globally;
4. separate bounded summary permit only after summary becomes a true background event.

---

# I. Fallback graph

```text
DIRECT substantive
  configured provider
    success + quality + validation -> text
    empty/error/incomplete reject/refusal/validation reject
      -> LocalResponder
         behavior_mode=useful_answer -> useful/unavailable local response
         behavior_mode=troll_user    -> roast only

DIRECT social/trivial
  optional GIF/sticker -> final
  else selected route:
    configured provider (long serious reply only) -> LocalResponder on failure
    Markov -> LocalResponder if empty/meta-repeat
    LocalResponder -> final

PENDING
  configured provider -> deterministic local continuation on failure

MANUAL MEME
  AI caption allowed -> local source on failure
  AI cooldown/unavailable -> old quote -> fresh quote -> callback -> Markov -> fresh last resort
  chat image invalid/render failure -> curated background
  curated render/send failure -> no response

ORDINARY RANDOM
  media -> final
  else configured provider OR Markov
  provider failure -> no response

AUTONOMOUS
  media -> final
  else configured provider
  provider failure -> no response

SUMMARY
  provider failure -> cursor unchanged, pending_since retained
```

No fallback cycles and no second LLM retry were found. `useful_answer` and `troll_user` modes are preserved through direct provider fallback. The semantic risk is elsewhere: telemetry labels configured OpenAI calls as `grok`, and branch-specific no-response behavior is inconsistent.

---

# J. Technical debt findings

## ARCH-01

ID: ARCH-01  
Severity: CRITICAL  
Problem: Deployable tracked path `.env.example` currently contains non-empty values matching real Telegram/XAI credential formats. Compose loads it and Docker copies it into the image.  
Evidence: `.env.example`; `docker-compose.yml:5-7`; `Dockerfile:12`; `.dockerignore:1-7`. Existing Git history scan found no matching non-empty secrets, so the exposure appears uncommitted at audit time. Values are intentionally omitted.  
Real impact: token theft, bot takeover, paid API abuse, credentials baked into images/registries.  
Why it exists: `.env.example` is used both as documentation and actual deployment env fallback.  
Recommended change: rotate affected credentials; keep only placeholders in tracked example; Compose should use private `.env`/secret injection; ensure build context excludes all real env files; add secret scan in CI/pre-commit.  
Risk of change: MEDIUM  
Estimated scope: SMALL  
Behaviour change: NONE

## ARCH-02

ID: ARCH-02  
Severity: CRITICAL  
Problem: `MAX 1 LLM CALL PER USER EVENT` is violated for ordinary messages when ingest-triggered summary and random AI reply occur together.  
Evidence: `bot.py:1211-1214`; `learning/service.py:268-305`; `learning/memory_service.py:199-231`; `learning/service.py:1045-1187`. Diagnostic reproduction: one ordinary event produced `summary_calls=1`, `generate_calls=1`.  
Real impact: duplicate spend, concurrent latency in one handler, breach of a stated safety invariant.  
Why it exists: summary is described conceptually as memory maintenance but is implemented synchronously inside foreground ingest; only direct/pending paths disable it.  
Recommended change: R0 add an event-scoped LLM budget/permit and invariant test; later move summary to a distinct queued/background event or defer it when foreground generation is possible.  
Risk of change: MEDIUM  
Estimated scope: MEDIUM  
Behaviour change: NONE

## ARCH-03

ID: ARCH-03  
Severity: CRITICAL  
Problem: No admission control exists for concurrent LLM or Pillow work, and no per-chat orchestration serialization exists.  
Evidence: `bot.py:50,1301-1324`; installed TeleBot defaults `threaded=True,num_threads=2`; `scheduler.py:154-235`; `learning/meme_renderer.py:207-240`; no `Semaphore` call site. Max accepted image is 48M pixels (`settings.py:176-179`), approximately 192 MB as RGBA before resize.  
Real impact: OOM/crash, provider burst spend, out-of-order responses, cooldown/pending races.  
Why it exists: concurrency emerged from Telegram and scheduler independently while service state remained mostly synchronous/in-memory.  
Recommended change: first instrument concurrent counts/RSS; then global LLM semaphore, media semaphore=1, and per-chat plan/commit serialization. Avoid a single global event lock.  
Risk of change: MEDIUM  
Estimated scope: MEDIUM  
Behaviour change: POSSIBLE

## ARCH-04

ID: ARCH-04  
Severity: HIGH  
Problem: Domain state is committed before delivery confirmation; pending is consumed before generation succeeds.  
Evidence: `learning/service.py:713-735` records routing/generated/pending before `bot.py:1020-1066` sends; `service.py:840-890` deletes pending at line 855; media commits at `service.py:1147-1159` before sender; manual meme is the notable safer exception, committing after `send_photo` at `bot.py:548-553`.  
Real impact: phantom pending, consumed cooldowns, false telemetry, lost continuation after crash/provider/send failure, failed guaranteed response.  
Why it exists: service returns final payload but also owns persistence side effects; no delivery receipt/commit phase.  
Recommended change: future `ResponsePlan` should be pure/prepared; delivery returns receipt; one idempotent commit persists success. Pending should be claimed atomically and finalized/restored based on outcome.  
Risk of change: HIGH  
Estimated scope: LARGE  
Behaviour change: NONE

## ARCH-05

ID: ARCH-05  
Severity: HIGH  
Problem: `LearningService` is a god-object with low cohesion and many reasons to change.  
Evidence: `learning/service.py` is 1930 lines, 89 methods (60 public), 45 direct `self.repository(...).method` call sites, and constructs routing, provider, memory, policy, media, renderer, quality, persona and caches in `__init__` (`service.py:89-160`).  
Real impact: changes in one feature affect routing/state/fallback; tests rely on private members; safe concurrency and transaction boundaries are hard to add.  
Why it exists: iterative compatibility facade accumulated each new feature.  
Recommended change: staged extraction after safety baseline: EventOrchestrator, ContextSnapshotBuilder, Generation/Fallback coordinator, MemoryFacade, MediaCoordinator. Keep LearningService as a compatibility facade during migration.  
Risk of change: HIGH  
Estimated scope: LARGE  
Behaviour change: NONE

## ARCH-06

ID: ARCH-06  
Severity: HIGH  
Problem: Event routing and business priority are duplicated between `bot.py` and core.  
Evidence: direct/mention/special detection in `bot.py:1189-1208`, repeated in `service.py:892-945` and `service.py:1045-1060`; special command/pending/activity/creator priority remains in `bot.py:1140-1277`.  
Real impact: accidental priority divergence, hidden summary exceptions, difficult all-branch integration tests.  
Why it exists: bot layer retained legacy features while LearningService added new routing.  
Recommended change: introduce immutable normalized event and move arbitration to one core orchestrator; keep identity lookup, Telegram object conversion and sending in adapter.  
Risk of change: HIGH  
Estimated scope: LARGE  
Behaviour change: NONE

## ARCH-07

ID: ARCH-07  
Severity: HIGH  
Problem: Context and state are rebuilt repeatedly, causing 22–31 DB connections per direct event and duplicate prompt memory.  
Evidence: `chat_state.py:287-342`; `service.py:925-992`; `service.py:548-593`; `memory_service.py:90-162`. SQL trace results in section G. Summary is read up to three times and dialogue twice.  
Real impact: latency/lock amplification, larger failure surface, inconsistent snapshots under concurrency, repeated facts in prompt.  
Why it exists: each component independently requests the repository view it needs.  
Recommended change: future per-event `ContextSnapshot` containing normalized event, recent rows, current summary, relevant stable facts, recent generated/media and resolved settings. Build once under a clear consistency boundary.  
Risk of change: MEDIUM  
Estimated scope: MEDIUM  
Behaviour change: NONE

## ARCH-08

ID: ARCH-08  
Severity: HIGH  
Problem: Several tables grow without retention, and candidate processing worsens with growth.  
Evidence: `memory_candidates=396`, `llm_calls=431` in main DB; no deletes for them; daily summaries also unbounded. `repository.py:817-875` scans all candidates and re-queries the whole table for each new candidate.  
Real impact: indefinite disk growth, increasingly expensive summary finalize, slower cleanup/backup.  
Why it exists: bounded raw history was implemented, but derived/telemetry lifecycle was not completed.  
Recommended change: define retention by semantics: retain aggregated LLM cost after raw rows expire; prune/reconcile promoted/stale candidates; keep only needed daily summaries or roll them up. Measure before deleting production data.  
Risk of change: MEDIUM  
Estimated scope: MEDIUM  
Behaviour change: POSSIBLE

## ARCH-09

ID: ARCH-09  
Severity: HIGH  
Problem: `/forget_chat` does not fully erase data and SQLite free pages remain recoverable.  
Evidence: `repository.py:1178-1199` omits `llm_calls`; it performs DELETEs but no secure deletion/VACUUM. Bot claims memory is destroyed at `bot.py:996-997`.  
Real impact: user/operator expectation mismatch; usage metadata remains, and deleted raw text may remain in free pages until reused/vacuumed.  
Why it exists: new tables were added after clear logic; SQLite logical deletion was treated as physical erasure.  
Recommended change: explicitly define forget semantics; include all intended tables, close/drop cached repository as needed, checkpoint/remove DB file or VACUUM when physical erasure is required; test every schema table.  
Risk of change: MEDIUM  
Estimated scope: SMALL  
Behaviour change: INTENTIONAL

## ARCH-10

ID: ARCH-10  
Severity: HIGH  
Problem: Pending continuation has check/delete races and lazy stale cleanup.  
Evidence: `service.py:796-818` reads pending; bot may read it before route; `service.py:840-855` reads again and deletes in a later transaction. `repository.py:464-473` cleans TTL only when queried.  
Real impact: two worker threads can consume the same pending and both generate; a crash after delete loses context; stale rows for inactive users remain indefinitely.  
Why it exists: pending is modeled as row state but not as an atomic claim state machine.  
Recommended change: atomic `claim_pending(user,event)` transaction with version/status; delete/finalize after delivery; periodic bounded TTL cleanup; retain same-user/reply semantics.  
Risk of change: MEDIUM  
Estimated scope: MEDIUM  
Behaviour change: NONE

## ARCH-11

ID: ARCH-11  
Severity: MEDIUM  
Problem: Provider abstraction leaks provider names and legacy OpenAI naming into core.  
Evidence: direct producer literal `grok` in `direct_address.py:74-86`; configured provider calls logged/recorded as Grok in `service.py:713-724,1162-1185`; APIs named `generate_openai`, `openai_allowed`, `openai_chat_id`, `openai_random_reply_chance`.  
Real impact: OpenAI-selected chats have misleading route telemetry; a third provider requires core edits; operational diagnostics are inaccurate.  
Why it exists: abstraction was added after OpenAI/Grok-specific code and compatibility names were preserved.  
Recommended change: use producer=`llm` plus resolved `provider_key`; isolate entitlement from provider naming; keep deprecated wrapper only at facade boundary.  
Risk of change: MEDIUM  
Estimated scope: MEDIUM  
Behaviour change: NONE

## ARCH-12

ID: ARCH-12  
Severity: MEDIUM  
Problem: Decision state is fragmented and cannot represent plan/commit/fallback uniformly.  
Evidence: `ConversationDecision`, direct `ResponseDecision`, `MediaDecision`, `AutonomousDecision`, `PendingConversation`, `PersonaSelection`, plus `str|None`, tuple media results and booleans in `bot.py`.  
Real impact: producer exclusivity is verified branch-by-branch, delivery errors cannot be fed back cleanly, telemetry fields diverge.  
Why it exists: each feature introduced the minimum local return type.  
Recommended change: after event normalization, introduce `ResponsePlan` with event_id, required, intent, behavior mode, producer, payload/media, budget permit, pending action, fallback policy and post-delivery commit. Do not collapse rich internal policy types prematurely.  
Risk of change: HIGH  
Estimated scope: LARGE  
Behaviour change: NONE

## ARCH-13

ID: ARCH-13  
Severity: MEDIUM  
Problem: Configuration precedence/documentation has drift and some settings are ineffective.  
Evidence: `.env.example` omits 42 consumed settings; `REPLY_MAX_OUTPUT_TOKENS` is configured but unused by `PersonaBuilder.output_budget()` (`persona.py:137-150`); `OPENAI_DAILY_MIN/MAX` are unused; timezone uses `config.txt` before `TIMEZONE` env (`settings.py:41-53`); many policy thresholds remain hardcoded.  
Real impact: operators believe a 100-token reply cap applies while actual purpose caps default up to 480; tuning and incident reproduction are unreliable.  
Why it exists: dynamic budgets and policy matrices evolved faster than deployment docs.  
Recommended change: generate a config inventory/test, mark deprecated keys, document actual precedence, and move only operational knobs—not every scoring constant—to settings.  
Risk of change: LOW  
Estimated scope: MEDIUM  
Behaviour change: POSSIBLE

## ARCH-14

ID: ARCH-14  
Severity: MEDIUM  
Problem: Schema evolution and query/index lifecycle are informal.  
Evidence: repository initialization contains schema plus ad-hoc conditional ALTERs (`repository.py:54-286`); no schema version; correlated image query at `repository.py:1032-1048`; missing composite lookup indexes shown by EXPLAIN.  
Real impact: future migrations become hard to roll back/test; max-size media queries and candidate finalize degrade.  
Why it exists: additive compatibility was sufficient for early DB versions.  
Recommended change: lightweight numbered SQLite migrations in the monolith; add query regression tests and indexes only where bounded-size profiling justifies them.  
Risk of change: MEDIUM  
Estimated scope: MEDIUM  
Behaviour change: NONE

## ARCH-15

ID: ARCH-15  
Severity: MEDIUM  
Problem: Shutdown is abrupt and temp/source cleanup is process-lifetime dependent.  
Evidence: daemon scheduler `bot.py:1301-1324`; endless polling/reconnect `bot.py:1344-1359`; no SIGTERM handler, provider/client close, scheduler stop event or worker drain. Renderer cleans stale output only on initialization (`meme_renderer.py:280-287`); `/tmp/cyberchair_source_*` has no restart scavenger.  
Real impact: in-flight response/usage loss, leftover source files after kill, skipped updates because restart uses `skip_pending=True`. SQLite remains crash-safe but logical event state can be partial.  
Why it exists: development-oriented single-process lifecycle.  
Recommended change: stop event, `bot.stop_polling`, bounded worker drain, client close where supported, scheduled temp scavenging, explicit policy for `skip_pending`.  
Risk of change: MEDIUM  
Estimated scope: MEDIUM  
Behaviour change: POSSIBLE

## ARCH-16

ID: ARCH-16  
Severity: MEDIUM  
Problem: Exception handling preserves uptime but loses root cause and error taxonomy.  
Evidence: broad catches in `responses_provider.py:125-127,162-164`, scheduler `232-233`, polling `bot.py:1354-1359`; Telegram text/GIF/sticker delivery is mostly unwrapped while state is already committed.  
Real impact: provider/auth/timeout/rate-limit/programming bugs look alike; scheduled claim can be lost after send failure; troubleshooting concurrent failures is weak.  
Why it exists: generic fallback-first resilience.  
Recommended change: typed error categories and structured logs with safe exception class/code; programming bugs retain traceback; delivery failure becomes explicit result. Never log prompts or full message text.  
Risk of change: LOW  
Estimated scope: MEDIUM  
Behaviour change: NONE

## ARCH-17

ID: ARCH-17  
Severity: MEDIUM  
Problem: Telemetry lacks event correlation and cannot prove core invariants.  
Evidence: `llm_calls` and `routing_events` have no event/update/message id; logs usually include chat id only (`service.py:954-958`, `responses_provider.py:90-95`).  
Real impact: impossible to query calls-per-event, reconstruct fallback/delivery, or distinguish simultaneous same-chat events.  
Why it exists: counters were added independently.  
Recommended change: generate a non-sensitive correlation id from chat/update/message id, propagate through plan/provider/delivery, and store it with LLM/routing events.  
Risk of change: LOW  
Estimated scope: MEDIUM  
Behaviour change: NONE

## ARCH-18

ID: ARCH-18  
Severity: MEDIUM  
Problem: Scheduled events are claimed before delivery, making transient send errors permanent misses.  
Evidence: `scheduler.py:159-179` calls persistent claim before `send_message`; `repository.py:488-500` commits claim atomically; scheduler catches the later exception and continues.  
Real impact: daily quote/#FREEKUCHER can disappear for that day after a Telegram outage.  
Why it exists: duplicate prevention was prioritized over retry semantics.  
Recommended change: small outbox/claim status (`reserved/sent`) or commit claim after successful delivery with idempotency window.  
Risk of change: MEDIUM  
Estimated scope: SMALL  
Behaviour change: NONE

## ARCH-19

ID: ARCH-19  
Severity: MEDIUM  
Problem: Failed manual meme plans leak `_command_meme_sources` entries in memory.  
Evidence: entry inserted at `service.py:1849`; removed only in `mark_command_meme_sent()` at `1897`; `send_manual_meme()` does not discard it on render/send failure (`bot.py:520-568`).  
Real impact: unbounded small-object growth during persistent renderer/Telegram failures; stale source bookkeeping.  
Why it exists: post-send commit owns the only pop path.  
Recommended change: plan token with explicit abort/finalize or weak/bounded event-key map; clear on every terminal delivery outcome.  
Risk of change: LOW  
Estimated scope: SMALL  
Behaviour change: NONE

## ARCH-20

ID: ARCH-20  
Severity: MEDIUM  
Problem: Summary durability is weaker than the bounded raw-window claim suggests.  
Evidence: messages are capped at 50 on every insert (`repository.py:303-305`); failed summary leaves cursor unchanged (`memory_service.py:229-233`); later unsummarized old rows may be evicted before a successful refresh.  
Real impact: during provider outage/high traffic, some conversation can disappear before being summarized; stable memory may never observe it.  
Why it exists: raw retention and summary batching use the same 50-row window without an unsummarized retention guarantee.  
Recommended change: retain rows at least through summary cursor, or copy pending summary fragments to a bounded queue/table; keep a hard global cap to preserve privacy/storage constraints.  
Risk of change: MEDIUM  
Estimated scope: MEDIUM  
Behaviour change: NONE

## ARCH-21

ID: ARCH-21  
Severity: MEDIUM  
Problem: Global known-user state violates per-chat isolation.  
Evidence: `bot.py:93-99,380-441`; one `bot_state.json` map is used for every chat, and `remove_known_user` removes globally after membership check in one chat.  
Real impact: cross-chat coupling, unbounded map, incorrect forgetting for users who left one chat but remain in another.  
Why it exists: this feature predates per-chat SQLite.  
Recommended change: move known users to per-chat repository or key JSON by chat; define retention.  
Risk of change: MEDIUM  
Estimated scope: SMALL  
Behaviour change: POSSIBLE

## ARCH-22

ID: ARCH-22  
Severity: LOW  
Problem: Legacy/compatibility APIs and settings remain with no live callers.  
Evidence: candidates include `send_startup_quote`, `send_restart_gif`, `maybe_stul_cooldown_reply`, `maybe_question_reply`, `maybe_random_media`, `TriggerEngine.decide_user_reply`, `build_generate_request`, repository `messages_since`, `recent_dialogue`, `mark_gif_used`, `mark_sticker_used`, MediaService `select_chat_image`, and settings `openai_daily_min/max`, `reply_max_output_tokens`. Static references are definition-only or tests-only.  
Real impact: enlarged API surface, misleading docs/tests, risk of editing dead path instead of live route.  
Why it exists: compatibility was intentionally preserved through iterative changes.  
Recommended change: characterize and deprecate first; remove only in final legacy stage after caller audit and one release window.  
Risk of change: LOW  
Estimated scope: MEDIUM  
Behaviour change: POSSIBLE

## Dead-code candidate table

| Symbol | File | References | Safe to remove now? | Reason |
|---|---|---|---|---|
| `build_generate_request` | `learning/llm_prompts.py` | definition only | uncertain | documented compatibility export pattern |
| `messages_since`, `recent_dialogue` | `learning/repository.py` | definition only | likely yes after API check | no runtime/tests caller |
| `mark_gif_used`, `mark_sticker_used` | `learning/repository.py` | definition only | uncertain | replaced by generic media marking |
| `select_chat_image` | `learning/media_service.py` | definition only | likely yes | callers use `score_chat_images` directly |
| `provider_unavailable_reason` | `learning/service.py` | definition only | uncertain | plausible external/UI API |
| `conversation_diagnostics`, `autonomous_diagnostics` | `learning/service.py` | definition only | uncertain | useful debug surface |
| `maybe_stul_cooldown_reply`, `maybe_question_reply` | `learning/service.py` | tests only | no, stage later | legacy behavior tests still characterize them |
| `maybe_random_media` | `learning/service.py` | tests only; scheduler receives `None` | no, stage later | explicit compatibility comment in `bot.main` |
| `send_startup_quote`, `send_restart_gif` | `bot.py` | tests only | uncertain | packaged behavior no longer invoked |
| `is_weekend`, `is_public_holiday`, `next_workday_start` | `utils.py` | definition only | uncertain | public utility API may be external |
| `utc_now` | `pending_conversation.py` | definition only | likely yes | no caller |
| `OPENAI_DAILY_MIN/MAX` | settings | not read elsewhere | yes after env compatibility notice | stale quota concept |
| `REPLY_MAX_OUTPUT_TOKENS` | settings | not used in budget selection | deprecate, not silently remove | deployment currently sets it |

---

# K. Single-producer audit matrix

| Branch | Producer arbitration | Double-send risk inside one handler | Finding |
|---|---|---|---|
| normal reply | media OR LLM OR Markov | low | one final producer; summary can be second LLM side-call |
| direct address | GIF/sticker OR LLM/Markov/local | low | substantive does not select contextual meme; fallback local only after AI failure |
| reply to CyberChair | same as direct, pending recheck inside | low | duplicated routing reads, but one final result |
| pending | LLM then local fallback | medium under concurrency | single thread is safe; same pending can be claimed twice |
| `с м стул` text | one meme | low | early return; no text send |
| photo caption `с м стул` | one meme | low | photo handler owns event |
| GIF/sticker incoming | ingest only | none | no response producer |
| Markov | selected alternative | low | AI and Markov mutually exclusive in live direct/normal routes |
| LocalResponder | terminal fallback | low | behavior mode preserved |
| autonomous | media OR LLM | medium cross-flow | can overlap user event/scheduler utilities |
| provider failure | local only in direct/pending/meme | low double-send | normal/creator/autonomous may produce nothing |
| validation/refusal/incomplete | returns None then branch fallback | low | no second AI |
| budget fallback | social may stay local; P3 still LLM | low | budget is deliberately soft and not global |
| meme AI cooldown | local source chain | low | no AI call |

The scheduler can intentionally send more than one independent scheduled message in one 30-second tick (for example quote plus autonomous, or end-of-day plus weekly summary). This is outside the user-event invariant but should receive separate event ids.

---

# L. Memory audit

## What generation actually uses

- short-term: messages + generated text from last 30 minutes, capped by 50 rows and purpose-specific message limit;
- current logical-day summary: topic analysis, context header, callbacks, autonomous policy;
- stable memory: relevance-filtered prompt facts and callbacks/local troll fallback;
- recent generated: lexical diversity, copy validation, cooldown/meta-joke checks;
- callbacks: derived at runtime from current summary + stable memory;
- meme/old quote: `messages` table only, which is capped at 50 total messages;
- media history: media usage/candidates for media decisions.

## Written but weakly read

- older `daily_summaries`: retained and exposed via `recent_summaries()`, but live generation asks only `summary_for_day(current_day)`;
- `memory_candidates`: used during promotion bookkeeping and tests, not directly in generation;
- `chat_stats.word_mentions`: status/statistics only, not generation;
- `media_metadata`: only useful if tags are explicitly populated; current main DB has none.

## Duplicate representations

- a fact can exist in current summary, promoted stable memory and still-promoted candidate row;
- callbacks are a view over summary/stable but may be repeated next to the same memory in prompt history;
- generated text is both response history and general action/cooldown ledger.

## Summary cursor

`finalize_summary()` atomically writes summary, observes/promotes candidates and advances `last_message_row_id`; failure leaves cursor unchanged. This part is good. Weakness: the raw messages cap may evict unsummarized rows while cursor remains behind.

## PendingConversation detail

| Requirement | Actual behavior | Assessment |
|---|---|---|
| hard / soft | both are persisted; hard continuation is mandatory when matching, soft continuation may yield to a sufficiently clear new topic | semantically intentional and covered by unit tests |
| TTL | default 1200 s; expiry is checked and deleted only when pending is queried | correct for active users, but stale inactive rows remain |
| reply linkage | `reply_to_message_id` can bind continuation to the originating bot question | supported |
| same-user matching | primary key is `user_id`; another user does not consume the row | supported within a chat DB |
| new-topic override | local `looks_like_new_topic()`/expected-type logic decides whether soft pending yields | useful, but depends on a second normalization/classification path |
| expected type | `how_to`, `choice`, `factual`, `open_advice` guide matching/local fallback | used |
| invalid continuation | hard pending still receives deterministic continuation handling; soft mismatch can fall through to normal/direct routing | behavior is explicit but split across bot/service |
| several pending rows | replacement for the same user is last-write-wins; different users coexist | acceptable only if overwrite is intended and observable |
| cleanup | query-time TTL deletion only; no periodic sweep | incomplete lifecycle |
| pending + direct | bot marks reply-to-chair as direct first; service then rechecks pending | final priority usually correct, implementation duplicated |
| pending + special command | early special handlers win before pending | current priority is intentional but needs one characterization matrix |
| concurrency | read, delete and response generation are separate transactions | two workers can both consume/respond; crash after delete loses the pending |

Architecturally the model is useful and should stay. The required future change is an atomic claim/finalize lifecycle, not replacement of hard/soft semantics.

---

# M. RAM and disk lifecycle

## Long-lived RAM

- bounded Markov LRU: max 20 chats;
- unbounded-by-chat service decision/cache dicts, TriggerEngine maps, Persona/MemeSource per-chat maps and known users;
- cached provider HTTP clients (appropriate, but no explicit close);
- `_command_meme_sources` leak on failed delivery;
- ChatAction active map is bounded by concurrent active chats and cleaned in `finally`.

## Media RAM

- Telegram download bytes capped at 20 MB;
- image dimensions capped at 12k and 48M pixels;
- converted RGBA copy can approach 192 MB before thumbnail to 1600px;
- no media semaphore means 2 handler renders + scheduler render can overlap;
- source images use context managers; converted images are released by reference lifetime, but not explicitly closed;
- typical `doomer_wojak` render median was 107.7 ms, peak process RSS around 50 MB in isolated run.

## Disk

- rendered PNGs are deleted in sender `finally`; stale generated PNGs older than one hour are cleaned only when renderer initializes;
- downloaded chat source is deleted in `finally`, but a process kill can leave `/tmp/cyberchair_source_*` indefinitely until OS cleanup;
- renderer deletes partial output on save exception;
- SQLite/data are persistent Docker volume; no backup mechanism in project;
- stdout/stderr logging retention is delegated to Docker/host and not bounded here;
- WAL checkpoint/backup lifecycle is not explicitly managed.

---

# N. Provider architecture verdict

The provider layer itself is sound enough:

- provider-neutral `GenerateRequest`/`SummarizeRequest`;
- shared Responses mechanics;
- persona and chat policy stay outside adapters;
- no prompt/output/secrets in usage telemetry;
- provider-specific model/reasoning/cache schema mostly stays inside adapter.

But core is not truly provider-replaceable because it encodes `grok` as producer/route and retains OpenAI-specific entitlement/method names. Grok can be switched to existing OpenAI without changing core behavior, but adding a third provider or obtaining accurate provider route telemetry requires core edits. Target should be provider-neutral core plus `provider_key` metadata, not a new abstraction hierarchy.

---

# O. Config inventory and precedence

Actual precedence:

```text
existing process env
  > private .env
  > tracked .env.example
  > code default
```

For the small supported per-chat set:

```text
settings table override
  > selected global default (only where implemented)
  > hardcoded DB default "1" / code default
```

Exceptions:

- timezone: `config.txt` > `TIMEZONE` env > `Europe/Moscow`;
- work start/end: `config.txt` > bot hardcoded 09:00–17:30;
- several per-chat toggles default directly to `1` without matching global config;
- policy scoring matrices, autonomous time windows, scheduler random windows and renderer layout constants are code constants.

Not every magic number should become env. Operational limits/cost/concurrency/retention belong in config; semantic persona scoring is better versioned in code with tests.

## O1. Complete setting-key inventory

Values are deliberately not reproduced here. Global keys consumed by the current code are:

| Group | Keys |
|---|---|
| Telegram/process | `TELEGRAM_BOT_TOKEN`, legacy `BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `LOG_LEVEL`, `LEARNING_DATA_DIR`, `TIMEZONE` |
| provider credentials/routing | `XAI_API_KEY`, `OPENAI_API_KEY`, `LLM_PROVIDER`, `OPENAI_ENABLED`, `OPENAI_CHAT_ID`, `OPENAI_MODEL`, `OPENAI_TIMEOUT`, `XAI_MODEL`, `XAI_REPLY_MODEL`, `XAI_SUMMARY_MODEL`, `XAI_REPLY_REASONING_EFFORT`, `XAI_SUMMARY_REASONING_EFFORT`, `XAI_BASE_URL`, `XAI_TIMEOUT`, `XAI_DAILY_CHAT_BUDGET_USD` |
| feature/activity | `LEARNING_ENABLED`, `DEFAULT_ACTIVITY_PERCENT`, `MIN_TRAINING_MESSAGES`, `RANDOM_REPLY_CHANCE`, `ACTIVE_CHAT_REPLY_CHANCE`, `OPENAI_RANDOM_REPLY_CHANCE`, `REPLY_TO_STUL_CHANCE`, `TROLL_USER_PROBABILITY`, `STUL_MARKOV_REPLY_CHANCE`, `FREQUENT_STUL_MARKOV_CHANCE`, `BARE_STUL_REPLY_FACTOR`, `DIRECT_SOCIAL_MARKOV_SHARE`, `CREATOR_USERNAME`, `SGLYPA_REPLY_CHANCE`, `TRIGGER_REACTION_CHANCE`, `TRIGGER_REACTION_COOLDOWN`, `SPECIAL_PHRASES` |
| generation/cooldown | `MAX_GENERATED_WORDS`, `MIN_GENERATED_WORDS`, `GENERATED_MESSAGE_COOLDOWN`, `ADDRESSED_REPLY_COOLDOWN`, `MAX_GENERATED_PER_HOUR`, `ALLOW_USER_MENTIONS`, `MAX_STORED_TEXT_LENGTH`, `MODEL_CACHE_SIZE`, `MARKOV_EXCLUDE_RECENT_MESSAGES`, `MARKOV_MIN_MESSAGE_AGE_SECONDS`, `MARKOV_RECENT_HISTORY_SIZE`, `MODE_MARKOV_WEIGHT`, `MODE_COMBINE_WEIGHT`, `MODE_QUOTE_MUTATION_WEIGHT`, `MODE_CONTEXTUAL_WEIGHT`, `MODE_RANDOM_OLD_PHRASE_WEIGHT` |
| context/memory/pending | `MAX_MESSAGES_PER_CHAT`, `CONTEXT_MESSAGE_LIMIT`, `REPLY_CONTEXT_MESSAGE_LIMIT`, `TARGETED_CONTEXT_MESSAGE_LIMIT`, `COMPLEX_CONTEXT_MESSAGE_LIMIT`, `AUTONOMOUS_CONTEXT_MESSAGE_LIMIT`, `SHORT_MEMORY_MINUTES`, `PENDING_CONVERSATION_TTL_SECONDS`, `SUMMARY_MESSAGE_INTERVAL`, `SUMMARY_TIME_INTERVAL`, `MAX_LONG_MEMORIES` |
| output budgets | legacy `REPLY_MAX_OUTPUT_TOKENS`, `SHORT_MAX_OUTPUT_TOKENS`, `TROLL_USER_MAX_OUTPUT_TOKENS`, `OPINION_MAX_OUTPUT_TOKENS`, `RECOMMENDATION_MAX_OUTPUT_TOKENS`, `USEFUL_MAX_OUTPUT_TOKENS`, `RECIPE_MAX_OUTPUT_TOKENS`, `COMPLEX_MAX_OUTPUT_TOKENS`, `AUTONOMOUS_MAX_OUTPUT_TOKENS`, `MEME_MAX_OUTPUT_TOKENS`, `SUMMARY_MAX_OUTPUT_TOKENS` |
| autonomous | `QUIET_START_HOUR`, `QUIET_END_HOUR`, `AUTONOMOUS_ON_WEEKENDS`, `AUTONOMOUS_MIN_SILENCE`, `AUTONOMOUS_MAX_SILENCE`, `AUTONOMOUS_COOLDOWN`, `AUTONOMOUS_BOT_PAUSE`, `AUTONOMOUS_NO_RESPONSE_COOLDOWN`, `AUTONOMOUS_DAILY_LIMIT`, `AUTONOMOUS_ACTIVE_MESSAGE_COUNT`, `AUTONOMOUS_PROBABILITY_CAP`, `AUTONOMOUS_WORK_HOUR_FACTOR`, `AUTONOMOUS_EVENING_FACTOR` |
| media | `GIF_ENABLED`, `GIF_POST_CHANCE`, `GIF_POST_COOLDOWN`, `MAX_GIFS_PER_CHAT`, `STICKER_ENABLED`, `MAX_STICKERS_PER_CHAT`, `MEDIA_REPLY_CHANCE`, `MEDIA_HUMOR_BONUS`, `MEDIA_ARGUMENT_BONUS`, `MEDIA_MEME_CHANCE`, `MEDIA_GIF_SHARE`, `MEDIA_COOLDOWN`, `MEME_RENDER_COOLDOWN`, `MEDIA_TEMPLATE_COOLDOWN`, `MANUAL_MEME_COOLDOWN`, `CHAT_IMAGE_BACKGROUND_CHANCE`, `MAX_CHAT_IMAGES_PER_CHAT`, `MAX_CHAT_IMAGE_BYTES`, `MAX_CHAT_IMAGE_DIMENSION`, `MAX_CHAT_IMAGE_PIXELS`, `MEDIA_RECENT_LIMIT`, `MEME_QUOTE_MAX_CHARS`, `MEME_QUOTE_HARD_LIMIT` |
| state/policy | `STATE_LOW_MESSAGE_COUNT`, `STATE_LOW_SILENCE_SECONDS`, `STATE_HIGH_MESSAGES_5M`, `STATE_HIGH_MESSAGES_1M`, `STATE_BURST_MESSAGES_1M`, `STATE_BURST_PARTICIPANTS`, `STATE_TOPIC_MIN_OCCURRENCES`, `STATE_TOPIC_MIN_MESSAGES`, `POLICY_BURST_PROBABILITY_CAP` |
| currently stale | `OPENAI_DAILY_MIN`, `OPENAI_DAILY_MAX`; `REPLY_MAX_OUTPUT_TOKENS` is compatibility-only and not the live universal ceiling |

Per-chat SQLite keys are: `learning`, `talk`, `troll_mode`, `autonomous_enabled`, `media_enabled`, `activity_percent`, `llm_provider`; one-shot/internal marker `startup_meme_v1` is also stored in the same generic table.

Runtime/code-only values include voice-story cooldown 600 s; bot Sglypa cooldown 15 s; #FREEKUCHER cooldown 60 s; work hours 09:00–17:30; scheduler initial/random windows 10–30/20–60 minutes; routing/persona/media score matrices; retention literals for routing 31 days and scheduled claims 14 days. `config.txt` can override work hours and timezone. This is the main magic-number inventory; semantic weights should not all be externalized.

## O2. Duplicate-logic inventory

| Concern | Implementations that overlap | Divergence risk |
|---|---|---|
| direct trigger parsing | `bot.py` pre-route, `LearningService.maybe_direct_reply`, `LearningService.maybe_reply`, preprocessing helpers | different priority/summary behavior |
| text normalization | `normalize_spaces`, memory normalization, meme-source `_normalized`, filter-specific cleanup | same text can classify/select differently |
| intent/social classification | DirectAddressRouter classifier plus bot special checks and chat-state topic/type heuristics | no single canonical event intent |
| profanity/persona tone | LocalResponder regex/templates, PersonaBuilder instructions, output filters | local/provider fallbacks can drift stylistically |
| context selection | ChatStateAnalyzer, MemoryService, service dialogue builder, media queries | repeated reads and duplicate facts |
| lexical/copy control | LexicalDiversityTracker, validation recent-generated reads, meme/Markov anti-repeat | multiple windows and thresholds |
| cooldowns | TriggerEngine maps, bot globals, `generated`, `media_usage`, scheduled claims, persona/meme deques | different persistence and restart semantics |
| settings resolution | global dataclass, generic per-chat reads, `config.txt`, bot constants | precedence differs by feature |
| Telegram extraction | text handler, caption/photo/document handlers, direct/reply checks | captions do not share normal text lifecycle |
| media selection | MediaService contextual path, manual meme planner, scheduler callback, direct GIF/sticker path | different fallback/commit ordering |

This is accidental divergence, not merely repeated syntax. Consolidation should follow event/plan/context boundaries so it does not flatten intentional persona differences.

---

# P. Security and logging

Positive findings:

- `.env` is gitignored and Docker-ignored;
- provider repr/errors do not expose keys;
- prompts/output are not stored in `llm_calls`;
- message-content logging is generally avoided;
- ingest/output filters detect Telegram-token and common secret patterns;
- API responses are configured `store=False`.

Risks:

- ARCH-01 credentials in `.env.example`/image/Compose;
- broad `print(exception)` in polling/scheduler may expose provider/Telegram error strings;
- local SQLite contains chat text/usernames without encryption (acceptable if host permissions/threat model are explicit);
- no automated secret scanning;
- no correlation id, making safe debugging harder and encouraging future content logging.

Recommended structured fields: `event_id`, `chat_hash_or_id`, `message_id`, `producer`, `provider_key`, `purpose`, `plan_outcome`, `fallback_reason`, `delivery_outcome`, `latency_bucket`. Do not log message text by default.

## P1. Exception-handling classification

| Class | Current handling | Main defect | Desired boundary behavior |
|---|---|---|---|
| recoverable policy/empty result | usually `None` or local fallback | reason often disappears | typed outcome with fallback reason |
| provider/auth/rate-limit/timeout/refusal | broad catch in Responses adapter; direct/pending can fall back | all failures collapse to type name/empty result; normal/autonomous silently miss | sanitized provider category/code, no retry within event |
| Telegram delivery | mixed; polling outer loop catches, individual sends often do not | state may already be committed; root delivery outcome absent | typed delivery receipt/error, post-success commit |
| SQLite busy/operational | generally propagates to broad handler/polling catch | partial multi-method event state; no transaction-level retry policy | short bounded repository retry only for known busy errors plus atomic unit of work |
| media decode/render/download | local `try/finally` is relatively good | some source-map/temp lifecycle gaps; no resource admission | typed media failure, abort plan, cleanup and optional local media fallback |
| programming bug | frequently indistinguishable from operational failure at broad catch | stack/root cause can be lost and bot silently continues in inconsistent path | log traceback with correlation id; do not reinterpret as provider refusal |

Silent swallowing is concentrated in provider/scheduler resilience paths. The more dangerous pattern is not swallowing alone, but swallowing after a pre-delivery state mutation.

---

# Q. Telemetry

Current useful telemetry:

- provider usage/tokens/cost by call type;
- routing producer/intent/response mode;
- incomplete/truncation reason;
- lexical penalty trigger;
- meme caption source;
- media usage/cooldowns;
- bounded generated history.

Current gaps:

- no event id/calls-per-event;
- no delivery success/failure;
- no phase latency;
- no concurrent LLM/media gauge;
- no summary lag/cursor distance;
- `llm_calls` retention absent;
- configured OpenAI route may still be labelled Grok.

The 10 production metrics worth keeping:

1. accepted updates by event class;
2. response plan rate and required-response miss rate;
3. final producer and behavior mode;
4. LLM calls per event (hard invariant);
5. provider outcome/refusal/incomplete/quality rejection;
6. input/output/reasoning tokens and exact cost;
7. fallback transition;
8. delivery success/failure and end-to-end latency;
9. active/concurrent LLM and media jobs plus peak RSS;
10. summary lag, pending age and DB lock/busy errors.

No external observability platform is required; structured logs plus SQLite counters are enough initially.

---

# R. Performance and structural event cost

| Scenario | DB work | LLM | Context/budget | Media | Dominant bound |
|---|---|---:|---|---|---|
| bare `стул` | 22 conn / 22 reads / 19 writes measured | 0 | state + local recent history | optional GIF/sticker | DB/orchestration + human jitter |
| direct simple social | 31 / 31 / 19 measured | 0 normally | state/callback/media rows | possible GIF/sticker or rare Markov | DB/orchestration |
| useful question | 25 / 24 / 20 measured | 1 | 10–20 messages; 240–480 tokens by purpose | no direct meme | LLM-bound |
| `troll_user` | 25 / 24 / 20 measured | 1 | same context; 180 tokens | no direct meme | LLM-bound |
| manual meme | plan 16/15/1 + commit 4/2/8 | 1 if ready, else 0 | caption 50 tokens | download <=20MB, Pillow, photo upload | media/LLM-bound |
| pending continuation | 24 / 18 / 16 measured | 1 | pending context + normal history, usually useful budget | none | LLM-bound |
| autonomous | many state/summary/media reads; writes only on selection | 0 or 1 | 8 messages, 90 tokens | possible media/Pillow | policy cheap, selected action IO-bound |
| summary | 13 / 27 / 13 measured | 1 | <=50 rows / <=20k chars, 240 tokens | none | LLM-bound |

Telegram delivery is network-bound. SQLite is not currently a raw speed bottleneck; its problem is amplification and non-atomic semantics. Pillow is CPU/RAM-bound and synchronous. LLM dominates selected text latency. Context preparation is CPU/DB-bound but measured in tens of milliseconds locally.

---

# S. Test architecture audit

## Composition

- 318 pytest-discovered unittest-style tests plus 57 subtests;
- pure unit coverage: classifiers, policy, persona, pending parsing, filters, lexicon, quality, Markov;
- local integration: LearningService + temp SQLite, migrations, renderer/files, bot handler with mocks;
- characterization: 16 named tests around current bot/scheduler behavior;
- provider adapters: fully mocked Responses clients, including usage/secrets/incomplete;
- Telegram: mocked, no live Bot API;
- production-like: one 60-scenario synthetic quality test, but it calls `generate_llm()` directly rather than the full Telegram event lifecycle.

Strengths:

- direct provider/Markov/media mutual exclusion;
- provider failure without retry;
- pending hard/soft/TTL/user/reply cases;
- memory cursor success/failure;
- media cleanup success/error;
- chat action cleanup/reuse;
- persona and behavior-mode preservation.

Weaknesses:

- many exact prompt-string, probability and private-member assertions are brittle;
- several one-call tests duplicate service-level behavior while the actual bot+ingest orchestration gap remains;
- production-like test labels each direct `generate_llm` call an event and therefore cannot detect summary + reply;
- no real concurrent workers, SIGTERM, SQLite busy, HTTP timeout or Telegram delivery-commit test.

Critical missing tests:

1. complete ordinary event due for summary and selected AI -> total one LLM permit;
2. full handler matrix asserting exactly one Telegram send/producer;
3. delivery failure does not commit generated/media/pending success;
4. same-user concurrent pending claim;
5. same-chat user event overlapping autonomous/summary/media;
6. global LLM/media concurrency bound and OOM envelope;
7. shutdown during LLM/render/send and temp cleanup;
8. `/forget_chat` enumerates and clears every intended table/physical artifact;
9. configured OpenAI telemetry uses correct provider/route;
10. scheduler claim plus transient send failure is retryable/idempotent.

---

# T. Future module boundaries

Only boundaries tied to observed problems are recommended:

| Boundary | Owns | Prevents / improves |
|---|---|---|
| `EventNormalizer` | Telegram -> immutable core event, identity/direct/special metadata | duplicate parsing and caption/text divergence |
| `ResponseOrchestrator` | priority, one event permit, one response plan | summary+reply, branch divergence, single-producer proof |
| `ContextSnapshotBuilder` | one coherent read of dialogue/summary/stable/media/recent output/settings | 22–31 connections, duplicate prompt state |
| `GenerationCoordinator` | provider/local selection, quality and semantic fallback | branch-specific fallback and provider-name leaks |
| `MemoryFacade` | ingest, summary scheduling, cursor/retention | foreground summary call, raw-window loss |
| `MediaCoordinator` | plan, resource permit, render, abort/finalize | OOM and meme source leaks |
| `DeliveryService` | Telegram send and typed receipt/error | pre-delivery commit inconsistency |
| `EventCommitter` or repository unit-of-work | atomic post-delivery state/telemetry | partial transactions and phantom state |

These can remain modules/classes inside one process and one Docker container. No Kafka, Kubernetes, service mesh or microservices are justified.

---

# U. Refactoring roadmap

## R0 — safety baseline

Goal: close active security/invariant gaps before structural moves.  
Files: `.env.example`, `.dockerignore`, `docker-compose.yml`, tests, small instrumentation around event/provider.  
Dependencies: none.  
Risk: MEDIUM.  
Expected benefit: secret safety; enforce one LLM per event; observe delivery/concurrency.  
Tests required: secret scan, ordinary summary+reply invariant, event correlation, calls/event.  
Recommended Codex model/reasoning: strongest coding model, high reasoning.

## R1 — event normalization and full characterization

Goal: one immutable event representation and frozen priority matrix without behavior change.  
Files: `bot.py`, new core event module, direct/pending preprocessing tests.  
Dependencies: R0 correlation id.  
Risk: HIGH.  
Expected benefit: removes repeated trigger/text/reply parsing and makes handler matrix testable.  
Tests required: every special/direct/pending/photo/document branch; one Telegram producer.  
Recommended Codex model/reasoning: strongest coding model, high reasoning.

## R2 — ResponsePlan and delivery receipt

Goal: separate decision from delivery and commit; represent one final producer.  
Files: `bot.py`, `service.py`, media/pending/telemetry sender paths.  
Dependencies: R1.  
Risk: HIGH.  
Expected benefit: fixes pre-delivery commits, makes fallback/delivery explicit.  
Tests required: send success/failure, no double send, idempotent commit, pending restore/finalize.  
Recommended Codex model/reasoning: strongest coding model, high or extra-high reasoning.

## R3 — ContextSnapshot

Goal: build event context once and pass it through policy/media/persona/generation.  
Files: `service.py`, `memory_service.py`, `chat_state.py`, `media_service.py`, repository query facade.  
Dependencies: R1/R2 plan shape.  
Risk: MEDIUM.  
Expected benefit: fewer connections, coherent concurrency snapshot, smaller prompts.  
Tests required: context equivalence, query-count budgets, current summary/stable/callback behavior.  
Recommended Codex model/reasoning: strong coding model, high reasoning.

## R4 — concurrency hardening

Goal: bounded LLM/media work and per-chat atomic arbitration.  
Files: orchestrator, scheduler, chat action, media coordinator.  
Dependencies: R2 plan/commit; R0 metrics.  
Risk: HIGH.  
Expected benefit: OOM prevention, pending/cooldown correctness, ordered replies.  
Tests required: deterministic thread barriers, same-chat and cross-chat concurrency, shutdown.  
Recommended Codex model/reasoning: strongest coding model, extra-high reasoning.

## R5 — memory lifecycle

Goal: move summary off foreground event, guarantee unsummarized retention, prune candidates.  
Files: `memory_service.py`, `repository.py`, scheduler/background runner.  
Dependencies: R0 event budgets, R4 summary semaphore.  
Risk: HIGH.  
Expected benefit: strict one-call invariant, durable memory, bounded DB.  
Tests required: outage >50 messages, cursor/idempotency, promotion/pruning, restart.  
Recommended Codex model/reasoning: strongest coding model, high reasoning.

## R6 — LearningService decomposition

Goal: convert LearningService into thin compatibility facade over established boundaries.  
Files: `service.py` plus extracted orchestrator/generation/memory/media modules.  
Dependencies: R2–R5.  
Risk: MEDIUM after prior stages, HIGH if attempted earlier.  
Expected benefit: cohesion, smaller tests, independent reasons to change.  
Tests required: existing 318+ characterization and API compatibility.  
Recommended Codex model/reasoning: strong coding model, high reasoning.

## R7 — SQLite migrations, retention and query cleanup

Goal: numbered migrations, complete forget, retention, targeted indexes/connection strategy.  
Files: `repository.py`, migration module, diagnostics.  
Dependencies: stable repository APIs from R3/R6.  
Risk: MEDIUM.  
Expected benefit: safe upgrades, bounded growth, predictable operations.  
Tests required: upgrade matrix from old schemas, backup/restore, retention, EXPLAIN assertions where stable.  
Recommended Codex model/reasoning: strong coding model, high reasoning.

## R8 — legacy deletion and documentation

Goal: remove proven no-caller APIs/settings and align README/env/status with live behavior.  
Files: legacy symbols listed in ARCH-22, README, `.env.example`, tests.  
Dependencies: all compatibility migrations complete.  
Risk: LOW/MEDIUM.  
Expected benefit: smaller surface, less accidental divergence.  
Tests required: caller search, public API smoke, deployment config validation.  
Recommended Codex model/reasoning: balanced coding model, medium/high reasoning.

---

# V. Product improvements after refactor (not implementation work)

1. Context quality: rank one shared snapshot and eliminate duplicated memory/callback facts before prompt assembly.
2. Memory: use older daily summaries as a relevance source or stop retaining them; preserve only product-useful long-term facts.
3. TrollMode: retain 50/50 selection but monitor satisfaction/proxy signals by mode and fallback separately.
4. useful_answer: add behavior-level completeness checks by purpose, not provider retries.
5. troll_user: diversify roast structures using existing lexical tracker and recent-structure history, without weakening personality.
6. Media: preflight image decode under media semaphore and adapt pixel cap to VPS RAM.
7. Memes/GIF: learn contextual tag metadata from explicit operator curation or safe local signals, not another LLM call.
8. Markov: keep it independent, but track source-age/quality outcomes and improve corpus selection rather than expanding model complexity.
9. Cost/latency: expose per-purpose budget and actual token/cost/latency percentiles; stop presenting ineffective `REPLY_MAX_OUTPUT_TOKENS`.
10. Conference personality: preserve callbacks/inside jokes while adding per-person targeting cooldown so repeated autonomous/direct attention is distributed fairly.

---

# W. What should not be changed now

- Do not replace SQLite solely for production optics.
- Do not split into microservices or add distributed infrastructure.
- Do not rewrite the project.
- Do not simplify/remove CyberChair persona, TrollMode 50/50, callbacks, Markov, LocalResponder or meme vocabulary.
- Keep `GenerateRequest`/`SummarizeRequest` and the shared Responses adapter pattern.
- Keep provider failure local and non-retrying under the one-call rule.
- Keep per-chat physical DB isolation unless future multi-chat analytics proves otherwise.
- Keep raw message/generated hard bounds and secret/link filters.
- Keep transactional `finalize_summary()` semantics; fix retention around it rather than replacing it.
- Keep `ConversationPolicy`, `DirectAddressRouter`, `AutonomousPolicy` and `MediaService` transport-free.
- Keep data-driven media catalog and renderer cleanup `finally` paths.
- Keep deterministic media roll and explicit priority characterization tests.
- Do not centralize every number into `.env`; policy/persona constants with tests are valid code.

---

# X. Final verdict

**C. Накопился серьёзный architectural debt — требуется крупный staged refactor.**

Обоснование:

- есть воспроизводимое нарушение ключевого LLM-call invariant;
- секреты находятся в deployable tracked path;
- текущая concurrency envelope допускает OOM и state races;
- plan/delivery/commit не разделены;
- один god-object и adapter layer совместно принимают бизнес-решения;
- DB/context work многократно дублируется;
- retention и shutdown неполны.

При этом partial rewrite не нужен: core algorithms, persona, policies, provider adapter, media catalog, bounded raw memory and test suite are reusable and mostly sound. Правильная стратегия — safety-first migration inside one modular monolith, with compatibility facade and behavior characterization at each stage.

После этого отчёта рефакторинг не начинается автоматически.
