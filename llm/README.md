# llm/ LLM layer (client + prompts)

## Role

The **LLM layer of the whole system** — every LLM call goes through `llm/client.py`:

- **orchestration** uses it for query understanding / review / rewriting (Paper Search's own LLM calls)
- the **downstream task** (reserved folder, TBD) will use it when defined
- **eval** (RAGAS) uses it for metric computation

`llm/jev.py` is the one exception: an optional client for TypeSafe's Jev, a decision-only model
that returns typed answers (choice / score / yes-no probability) instead of text. It is used only
by the optional Jev reviewer (`orchestration.review.backend: jev`); everything that generates text
still goes through `llm/client.py`.

## Implementation requirements

### client.py — the single LLM client
- OpenAI-compatible protocol; configuration via `.env`: `LLM_BASE_URL` / `LLM_MODEL` / `DEEPSEEK_API_KEY`
  (DeepSeek by default; swappable to OpenAI etc., see .env.example)
- `get_client()` returns a process-level singleton `LLMClient`
- Methods: `chat(messages, temperature, max_tokens) -> str`; `chat_json(messages, schema_desc) -> dict|list`
  (retry on JSON parse failure per config `llm.retries`); `stream_chat(...)` (UI streaming)
- Unified: tenacity retries (429/timeout/5xx, exponential backoff); token usage accounting
  (accumulated and written to `data/results/metrics/llm_usage.jsonl` for cost reporting)
- Constraint: no other module imports the OpenAI client directly; evaluation scripts also go through this module

### prompt.py — prompt templates
- `build_understanding_prompt(input_text)` — query understanding (used by orchestration)
- `build_hyde_prompt(input_text)` — optional hypothetical-document retrieval expansion
- `build_review_prompt(candidates, criteria, input_text)` — review (used by orchestration)
- `build_rewrite_prompt(missing, existing)` — feedback-based rewriting (used by orchestration)
- Record `TEMPLATE_VERSION`; the report cites template versions for reproducibility
- (Task-specific prompts — generation/self-check — are defined later with the downstream task)

## Definition of Done (see README: Definition of Done)
- Sanity: `client.py`/`prompt.py` import cleanly (covered by `scripts/sanity_check.py`)
- Smoke: mock-LLM run of `chat`/`chat_json` parsing and retry behavior (offline)
- Integration: real API call verified via a small end-to-end run
