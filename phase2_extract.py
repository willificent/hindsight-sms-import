#!/usr/bin/env python3
"""
Phase 2: LLM Fact Extraction from SMS Chunks

Processes chunk JSON files from Phase 1 through an LLM to extract
salient facts, preferences, relationships, and life events.

Usage:
  python phase2_extract.py --limit 3                    # Test with 3 chunks
  python phase2_extract.py --limit 3 --model gemma3:27b # Specific model
  python phase2_extract.py --resume                      # Skip already processed
  python phase2_extract.py                              # Process all chunks

Environment:
  OLLAMA_COMPLETIONS_API_KEY - Required for Ollama Cloud authentication

Output:
  extracted/<chunk_id>.json - per-chunk extraction results
  extraction_summary.json - overall stats
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

# ── Configuration ──────────────────────────────────────────────────────────────
# Override with --chunks-dir and --output-dir flags
DEFAULT_CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "chunks")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "extracted")
API_URL = os.environ.get("OLLAMA_API_URL", "https://ollama.com/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b")

# Load API key from environment
API_KEY = os.environ.get("OLLAMA_COMPLETIONS_API_KEY", "")

EXTRACTION_PROMPT = """You are a meticulous personal knowledge extractor. You will receive a batch of text messages from a conversation. Your job is to extract ONLY factual, memorable claims — NOT to summarize or paraphrase the conversation.

EXTRACT facts in these categories:
- relationship: how people are connected, relationship dynamics, significant interactions
- preference: likes, dislikes, opinions, tastes, aesthetic choices
- event: significant life events, milestones, travel, gatherings
- health: health conditions, medical info, wellness mentions (treat as CONFIDENTIAL)
- career: work, job, professional updates, academic pursuits
- travel: trips taken, planned travel, places visited
- hobby: hobbies, interests, creative pursuits, entertainment
- family: family relationships, family events, family dynamics

For each fact, output a JSON object with:
- content: the factual claim (concise, specific, not vague)
- temporal: any date/time reference mentioned (ISO format if possible, or natural language)
- entities: list of people, places, or organizations mentioned
- salience: 1-5 (how memorable/noteworthy — 5 = major life event, 1 = trivial)
- category: one of the categories above
- confidence: 0.0-1.0 (how certain you are this is a real fact vs. misunderstanding)

DISCARD:
- Greetings, goodbyes, "lol", "ok", "haha", "thanks"
- Scheduling logistics (unless they reveal a preference or relationship)
- Mundane small talk with no factual content
- Vague statements without specifics ("we should hang out sometime")
- Opinions about media without specifics ("that movie was good")

KEEP:
- Revealed preferences (food, music, design, etc.)
- Significant life events (moves, job changes, relationships)
- Health mentions
- Career updates
- Specific opinions tied to entities
- Travel plans and experiences
- Family dynamics and events

Output a JSON array of fact objects. If no salient facts exist in a message batch, output an empty array: []

RESPOND ONLY WITH THE JSON ARRAY. No preamble, no explanation, no markdown."""


def call_llm(messages, model=DEFAULT_MODEL, max_retries=3):
    """Call Ollama Cloud API with retry logic."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }).encode("utf-8")

    for attempt in range(max_retries):
        try:
            headers = {"Content-Type": "application/json"}
            if API_KEY:
                headers["Authorization"] = f"Bearer {API_KEY}"
            req = urllib.request.Request(
                API_URL,
                data=payload,
                headers=headers,
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())

            content = data["choices"][0]["message"]["content"]
            return content

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")[:500]
            print(f"  HTTP {e.code}: {error_body}", file=sys.stderr)
            if e.code == 429:  # Rate limit
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            if e.code >= 500:  # Server error
                wait = 2 ** (attempt + 1)
                print(f"  Server error, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise

        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise

    return None


def format_chunk_for_llm(chunk):
    """Format a chunk's messages into a readable transcript for the LLM."""
    lines = []
    contact = chunk.get("contact", "Unknown")
    date_range = chunk.get("date_range", "unknown dates")
    lines.append(f"=== Messages with {contact} ({date_range}) ===\n")

    for msg in chunk.get("messages", []):
        direction = "Me" if msg.get("dir") == "sent" else contact
        ts = msg.get("ts", "")[:16]  # Trim to date + time
        text = msg.get("text", "").strip()
        if text:
            lines.append(f"[{ts}] {direction}: {text}")

    return "\n".join(lines)


def parse_llm_response(response_text):
    """Parse LLM response into a list of fact dicts."""
    if not response_text:
        return []

    text = response_text.strip()

    # Strip markdown code block wrapping
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    # Direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "facts" in result:
            return result["facts"]
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Find JSON array in text
    import re
    bracket_match = re.search(r'\[[\s\S]*\]', text)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    print(f"  WARNING: Could not parse LLM response as JSON", file=sys.stderr)
    print(f"  First 200 chars: {text[:200]}", file=sys.stderr)
    return []


def process_chunk(chunk_path, model=DEFAULT_MODEL):
    """Process a single chunk through the LLM."""
    with open(chunk_path) as f:
        chunk = json.load(f)

    chunk_id = chunk.get("chunk_id", os.path.basename(chunk_path).replace(".json", ""))
    transcript = format_chunk_for_llm(chunk)

    if len(transcript) < 50:  # Skip nearly empty chunks
        return {"chunk_id": chunk_id, "facts": [], "msg_count": chunk.get("message_count", 0), "status": "skipped_too_short"}

    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": transcript},
    ]

    start_time = time.time()
    response = call_llm(messages, model=model)
    elapsed = time.time() - start_time

    facts = parse_llm_response(response)

    # Add source metadata to each fact
    for fact in facts:
        fact["source_chunk"] = chunk_id
        fact["source_contact"] = chunk.get("contact", "")
        fact["source_date_range"] = chunk.get("date_range", "")

    return {
        "chunk_id": chunk_id,
        "facts": facts,
        "msg_count": chunk.get("message_count", 0),
        "transcript_length": len(transcript),
        "extraction_time_s": round(elapsed, 1),
        "model": model,
        "status": "extracted" if facts else "no_facts_found",
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract facts from SMS chunks via LLM")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--chunks-dir", default=DEFAULT_CHUNKS_DIR, help="Input chunks directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for extracted facts")
    parser.add_argument("--limit", type=int, help="Only process N chunks (for testing)")
    parser.add_argument("--resume", action="store_true", help="Skip chunks that already have output files")
    parser.add_argument("--start-from", type=int, default=0, help="Start from chunk index N (0-based)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get all chunk files sorted
    chunk_files = sorted(
        [os.path.join(args.chunks_dir, f) for f in os.listdir(args.chunks_dir) if f.endswith(".json")]
    )

    if args.start_from:
        chunk_files = chunk_files[args.start_from:]

    # Skip already-processed chunks when resuming
    if args.resume:
        remaining = []
        for cf in chunk_files:
            chunk_id = os.path.basename(cf).replace(".json", "")
            out_path = os.path.join(args.output_dir, f"{chunk_id}.json")
            if not os.path.exists(out_path):
                remaining.append(cf)
        skipped = len(chunk_files) - len(remaining)
        print(f"Resume mode: skipping {skipped} already-processed chunks")
        chunk_files = remaining

    if args.limit:
        chunk_files = chunk_files[:args.limit]

    total = len(chunk_files)
    print(f"Phase 2: Processing {total} chunks with {args.model}")
    print(f"Output dir: {args.output_dir}")
    print()

    stats = {
        "model": args.model,
        "total_chunks": total,
        "total_facts": 0,
        "total_time_s": 0,
        "results": [],
    }

    for i, chunk_path in enumerate(chunk_files):
        chunk_id = os.path.basename(chunk_path).replace(".json", "")
        print(f"[{i+1}/{total}] {chunk_id}...", end=" ", flush=True)

        try:
            result = process_chunk(chunk_path, model=args.model)
            fact_count = len(result.get("facts", []))
            elapsed = result.get("extraction_time_s", 0)
            print(f"{fact_count} facts in {elapsed}s ({result['status']})")

            # Save per-chunk result
            out_path = os.path.join(args.output_dir, f"{chunk_id}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            stats["total_facts"] += fact_count
            stats["total_time_s"] += elapsed
            stats["results"].append(result)

        except Exception as e:
            print(f"ERROR: {e}")
            stats["results"].append({
                "chunk_id": chunk_id,
                "status": "error",
                "error": str(e),
            })

    # Save summary
    print(f"\nDone! {stats['total_facts']} facts extracted from {total} chunks in {stats['total_time_s']:.0f}s")
    summary_path = os.path.join(os.path.dirname(args.output_dir), "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()