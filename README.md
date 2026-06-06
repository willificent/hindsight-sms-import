# hindsight-sms-import

Extract high-value facts from SMS message archives into persistent long-term memory using a 4-phase pipeline. Built for [Hindsight](https://github.com/nousresearch/hindsight) but adaptable to any knowledge graph or vector store.

## Why?

Most people carry a decade of SMS history — relationships, preferences, health updates, career moves, travel — that's never searchable, never structured, and eventually lost when you switch phones. This pipeline mines that archive and distills it into structured, searchable facts.

**50,000 messages → 2,176 raw extracted facts → 1,362 filtered facts → Hindsight long-term memory**

## Architecture

```
SMS Backup XML  +  Contacts VCF
        |                |
   parse_contacts.py     |
        |                |
   phase1_chunk.py  <----+
        |
   chunks/*.json
        |
   phase2_extract.py  (LLM on Ollama Cloud)
        |
   extracted/*.json
        |
   phase3_filter.py  (pure Python, zero tokens)
        |
   filtered_facts.json
        |
   phase4_ingest.py  (Hindsight async REST API)
        |
   Hindsight long-term memory
```

Each phase writes intermediate files, so you can iterate on Phase 3+4 without re-running the expensive LLM extraction.

## Phase 1: Chunking & Pre-processing

**Scripts:** `parse_contacts.py` + `phase1_chunk.py`

### parse_contacts.py

Parses a VCF (vCard) contacts export and produces `contacts_filter.json` — a phone-number-keyed lookup with names, organizations, emails, and notes. This serves two purposes:
1. **Filtering**: Only messages from recognized contacts are included (no spam, no group chats from unknown numbers, no 2FA codes)
2. **Entity normalization**: Phase 3 uses this file to resolve "Ben", "Ben S", and "Benjamin Smith" into the same canonical entity

```bash
python3 parse_contacts.py contacts-2025-01-01.vcf
# Output: contacts_filter.json
```

### phase1_chunk.py

Reads the [SMS Backup & Restore](https://play.google.com/store/apps/details?id=com.riteshsahu.SMSBackupRestore) XML format, groups messages by contact, filters by contacts list, and chunks conversations into ~200-message groups for LLM extraction.

```bash
python3 phase1_chunk.py \
  --input sms-backup.xml \
  --output chunks/ \
  --contacts contacts_filter.json \
  --min-messages 20
```

**Key behaviors:**
- Short codes (2FA, automated) are filtered out
- Empty-body messages (delivery receipts) are skipped
- Messages not in your contacts are excluded
- Conversations exceeding 200 messages are split into time-ordered sub-chunks

## Phase 2: LLM Fact Extraction

**Script:** `phase2_extract.py`
**Model:** gemma4:31b via Ollama Cloud (configurable)

Each chunk is sent to the LLM with a structured extraction prompt requesting JSON facts with: content, temporal references, entities, salience (1-5), category, and confidence (0-1).

```bash
# Full run (uses OLLAMA_COMPLETIONS_API_KEY env var)
python3 phase2_extract.py

# Test with 3 chunks
python3 phase2_extract.py --limit 3

# Resume after interruption
python3 phase2_extract.py --resume

# Use a different model
python3 phase2_extract.py --model mistral-nemo:12b
```

**Environment variables:**
- `OLLAMA_COMPLETIONS_API_KEY` — Required for Ollama Cloud auth
- `OLLAMA_API_URL` — Override the API endpoint (default: `https://ollama.com/v1/chat/completions`)
- `OLLAMA_MODEL` — Override the default model (default: `gemma4:31b`)

**Extraction prompt discards:**
- Greetings, "lol", "ok", "thanks"
- Scheduling logistics (unless they reveal preferences)
- Vague small talk

**Extraction prompt keeps:**
- Revealed preferences (food, music, design)
- Significant life events (moves, job changes, relationships)
- Health mentions
- Career updates, travel plans, family dynamics
- Specific opinions tied to named entities

**Performance:** ~6 chunks/min on gemma4:31b via Ollama Cloud. A full run of 206 chunks takes ~34 minutes. Each chunk produces 5-15 facts depending on conversation density.

## Phase 3: Post-processing & Filtering

**Script:** `phase3_filter.py`
**Dependencies:** `contacts_filter.json` from Phase 1
**Runtime:** ~8 seconds for 2,176 facts (pure Python, no LLM calls)

Pipeline steps:
1. **Structural validity** — content length >= 10 chars, >= 3 words, >= 5 alpha chars
2. **Salience filter** — keep only facts with salience >= 3 (discard trivial observations)
3. **Confidence filter** — keep only facts with confidence >= 0.7
4. **Entity normalization** — resolve aliases using contacts_filter.json (419 mappings in our test)
5. **Deduplication** — entity-blocked shingled Jaccard similarity with 0.75 threshold

```bash
# Default settings
python3 phase3_filter.py

# Preview stats without writing
python3 phase3_filter.py --dry-run

# Higher salience threshold
python3 phase3_filter.py --salience 4

# Filter to specific categories
python3 phase3_filter.py --categories health career relationship
```

**Customizing entity normalization:**

Edit the top of `phase3_filter.py` to configure self/partner names:

```python
SELF_NAMES = {"yourname", "nickname"}  # Will normalize to Yourname
PARTNER_NAMES = {"partner"}             # Will normalize to "Partner Name"
PARTNER_CANONICAL = "Partner Full Name"
```

**Deduplication approach:**

Original O(n^2) SequenceMatcher comparison timed out on 2,000+ facts. The current implementation uses entity-based blocking — facts are only compared against other facts that share at least one entity or category. This reduces comparisons from ~4.7M to ~15K, running in ~8 seconds instead of 120+.

**Results in our test:**
| Step | Input | Output | Removed |
|------|-------|--------|---------|
| Structural validity | 2,176 | 2,176 | 0 |
| Salience >= 3 | 2,176 | 1,455 | 721 |
| Confidence >= 0.7 | 1,455 | 1,455 | 0 |
| Entity normalization | — | (302 touched) | — |
| Deduplication | 1,455 | 1,362 | 93 |

## Phase 4: Hindsight Ingestion

**Script:** `phase4_ingest.py`
**Target:** Hindsight local REST API (configurable)

Batches filtered facts (default 10 per batch), converts to Hindsight item format with content, context, tags, entities, and metadata, then sends via async bulk API.

```bash
# Full ingestion
python3 phase4_ingest.py

# Preview without sending
python3 phase4_ingest.py --dry-run

# Test with 30 facts
python3 phase4_ingest.py --limit 30

# Resume from last successful batch
python3 phase4_ingest.py --resume

# Smaller batches for slower connections
python3 phase4_ingest.py --batch-size 5 --delay 2.0
```

**Environment variables:**
- `HINDSIGHT_HOST` — Default: `localhost`
- `HINDSIGHT_PORT` — Default: `9077`
- `HINDSIGHT_BANK` — Default: `default`

**Each fact becomes a Hindsight memory item with:**
- `content`: the factual claim
- `context`: "SMS conversation with [contact name]"
- `tags`: `sms:<category>` and `contact:<normalized-name>`
- `entities`: extracted person/org/location references
- `timestamp`: temporal reference (if valid ISO 8601)

**Pitfall:** Bare years like `"1996"` cause 422 errors in Hindsight's API. The script sanitizes these to `None` (omitted from the request).

**Pitfall:** The async flag is `"async_": true` in the JSON body, NOT `retain_async=true` as a query parameter.

**Pitfall (fixed in v1.1):** The original Phase 4 script read `source.get('contact', 'unknown')` from a nested key that didn't exist — the contacts were stored at the top level as `source_contact`. This produced "SMS conversation with unknown" for every fact. The fix reads `fact.get('source_contact', '')` correctly. If you ran the original script, use `cleanup_unknown.py` to delete the affected documents before re-ingesting.

## Installation

```bash
git clone https://github.com/willificent/hindsight-sms-import.git
cd hindsight-sms-import
pip install requests  # Only dependency beyond stdlib
```

No other dependencies required — all four scripts use Python 3.10+ standard library except `phase4_ingest.py` which needs the `requests` package.

## Prerequisites

1. **SMS export**: Use [SMS Backup & Restore](https://play.google.com/store/apps/details?id=com.riteshsahu.SMSBackupRestore) on Android to export your messages as XML

2. **Contacts export**: Export contacts as VCF from Google Contacts, Apple Contacts, or your phone's contact app

3. **LLM access**: An OpenAI-compatible API endpoint (Ollama Cloud, local Ollama, or any compatible provider). Set `OLLAMA_API_URL` and `OLLAMA_COMPLETIONS_API_KEY`.

4. **Hindsight**: Running locally or accessible via network (for Phase 4 only)

## Development Notes

### Why cloud over local for extraction?

A 31B model (gemma4:31b on Ollama Cloud) produces significantly cleaner structured extraction than 8-9B local alternatives. The difference in salience judgment — deciding what's a memorable fact vs. small talk — is substantial. We tested local models and found 3-4x more hallucinated facts and poorer category assignment at smaller sizes.

The cloud run cost roughly 34 minutes of compute for 2,176 facts from 206 chunks. With the go-between file architecture, this only needs to run once.

### Why go-between files?

Each phase writes intermediate files. This means:
- Phase 2 (expensive LLM calls) only runs once
- Phase 3 can be iterated on (adjust threshold, tune dedup) without re-burning tokens
- Phase 4 can be resumed if interrupted, without touching Phase 2 or 3

### Entity normalization edge cases

The current normalization handles:
- First name → full name (Ben → Benjamin Smith)
- Self-reference variants (jon → Jonathan)
- First name + last initial (Ben S. → Benjamin Smith)
- Possessive forms (Jonathan's → Jonathan)

Known gaps:
- Spelling variants (Reagan vs Regan) require manual alias mapping
- Possessive constructions like "Jonathan's grandfather" should be a separate entity, not normalized to "Jonathan"

### Scaling

The pipeline has been tested on ~50,000 messages producing 206 chunks. Performance characteristics:
- Phase 1: ~5 seconds (pure Python, XML parsing)
- Phase 2: ~34 minutes (LLM extraction via Ollama Cloud at ~6 chunks/min)
- Phase 3: ~8 seconds (pure Python, entity-blocked dedup)
- Phase 4: ~15 minutes (async API batching, rate-limited at 1 req/sec)

For larger archives (100K+ messages), Phase 2 will scale linearly. Consider running with a faster model or parallel chunk processing.

## Post-Import Cleanup

If you discover a bug after ingestion (e.g., incorrect context strings), you can delete affected documents and re-ingest:

1. **Identify affected documents:** Use `cleanup_unknown.py --dry-run` to list documents that will be deleted.

2. **Delete them:** `python3 cleanup_unknown.py --delete` removes all SMS documents with incomplete context ("SMS conversation with unknown", generic "SMS conversation", or test contexts). It will NOT touch organically-created documents from normal Hindsight usage.

3. **Re-ingest:** After cleanup, re-run Phase 4 with the fixed script: `python3 phase4_ingest.py`

The cleanup script targets documents by their `retain_params.context` field and `sms:` tags, so it's safe to run alongside other Hindsight data.

## License

MIT

## Acknowledgments

Built with [Hindsight](https://github.com/nousresearch/hindsight) by Nous Research, [Ollama](https://ollama.com) for cloud inference, and Google's Gemma 4 model for fact extraction.