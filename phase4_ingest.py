#!/usr/bin/env python3
"""Phase 4: Ingest filtered facts into Hindsight via async bulk API.

Reads filtered_facts.json, batches facts into groups of 10, and POSTs them
to Hindsight's local REST API with async_=True for background processing.
Returns operation IDs for tracking.

Usage:
    python3 phase4_ingest.py [--input filtered_facts.json] [--batch-size 10] [--dry-run]
    python3 phase4_ingest.py --resume   # Resume from last successfully ingested batch
    python3 phase4_ingest.py --limit 30  # Test with 30 facts

Environment:
    HINDSIGHT_HOST - Hindsight server host (default: localhost)
    HINDSIGHT_PORT - Hindsight server port (default: 9077)
    HINDSIGHT_BANK - Hindsight bank ID (default: hermes)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# ── Configuration ──────────────────────────────────────────────────────────────
HINDSIGHT_HOST = os.environ.get("HINDSIGHT_HOST", "localhost")
HINDSIGHT_PORT = os.environ.get("HINDSIGHT_PORT", "9077")
HINDSIGHT_BANK = os.environ.get("HINDSIGHT_BANK", "default")
HINDSIGHT_URL = f"http://{HINDSIGHT_HOST}:{HINDSIGHT_PORT}"
API_ENDPOINT = f"{HINDSIGHT_URL}/v1/default/banks/{HINDSIGHT_BANK}/memories"
DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "filtered_facts.json")

# Map Phase 3 categories to Hindsight-friendly tags
CATEGORY_TAGS = {
    "relationship": "sms:relationship",
    "preference": "sms:preference",
    "event": "sms:event",
    "health": "sms:health",
    "career": "sms:career",
    "travel": "sms:travel",
    "hobby": "sms:hobby",
    "family": "sms:family",
    "education": "sms:education",
}


def load_filtered_facts(filepath: str) -> list[dict]:
    """Load facts from Phase 3 output."""
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} filtered facts from {filepath}")
    return data


def sanitize_timestamp(ts: str | None) -> str | None:
    """Ensure timestamp is valid ISO 8601 for Hindsight API.

    Accepts: full ISO datetime, date-only, year-month.
    Rejects: bare years, incomplete dates, None, empty strings.
    """
    if not ts or ts in ("null", "None", ""):
        return None

    # Full ISO datetime: 2024-01-15T10:30:00 or 2024-01-15T10:30:00Z
    if re.match(r"^\d{4}-\d{2}-\d{2}T", ts):
        return ts

    # Date only: 2024-01-15
    if re.match(r"^\d{4}-\d{2}-\d{2}$", ts):
        return ts

    # Year-month: 2024-01 -> expand to 2024-01-01
    if re.match(r"^\d{4}-\d{2}$", ts):
        return f"{ts}-01"

    # Bare year like "1996" -> not specific enough, skip
    if re.match(r"^\d{4}$", ts):
        return None

    return None


def build_item(fact: dict) -> dict:
    """Convert a Phase 3 fact into a Hindsight retain item."""
    category = fact.get("category", "event")
    entities = fact.get("entities", [])
    salience = fact.get("salience", 3)
    content = fact.get("content", "")
    # Phase 3 stores contact info at top level: source_contact, source_chunk, source_date_range
    # Not nested under a "source" key
    contact_name = fact.get("source_contact", "")
    chunk_id = fact.get("source_chunk", "")
    date_range = fact.get("source_date_range", "")

    # Build metadata
    metadata = {
        "category": category,
        "salience": str(salience),
        "source_type": "sms",
    }
    if contact_name:
        metadata["contact"] = contact_name
    if chunk_id:
        metadata["chunk_id"] = chunk_id

    # Build tags: category + contact name for scoped retrieval
    tags = [CATEGORY_TAGS.get(category, f"sms:{category}")]
    if contact_name:
        # Normalize to a simple tag format (e.g. "Angela Flynn" -> "angela-flynn")
        contact_tag = contact_name.lower().replace(" ", "-")
        tags.append(f"contact:{contact_tag}")

    # Build entity references
    hindsight_entities = []
    for ent in entities:
        if isinstance(ent, str):
            ent_type = "PERSON"
            if any(kw in ent.lower() for kw in ["university", "college", "school", "hospital", "city", "state"]):
                ent_type = "ORG"
            elif any(kw in ent.lower() for kw in ["road", "street", "ave", "blvd", "drive", "island", "park"]):
                ent_type = "LOCATION"
            hindsight_entities.append({"text": ent, "type": ent_type})
        elif isinstance(ent, dict):
            hindsight_entities.append({
                "text": ent.get("text", ent.get("name", "")),
                "type": ent.get("type", "PERSON"),
            })

    item = {
        "content": content,
        "context": f"SMS conversation with {contact_name}" if contact_name else "SMS conversation",
        "metadata": metadata,
        "tags": tags,
        "entities": hindsight_entities,
    }

    # Add temporal reference if valid
    ts = sanitize_timestamp(fact.get("temporal"))
    if ts:
        item["timestamp"] = ts

    return item


def ingest_batch(items: list[dict], dry_run: bool = False) -> dict:
    """Send a batch of items to Hindsight async ingestion API."""
    if dry_run:
        return {"success": True, "items_count": len(items), "dry_run": True}

    payload = {"items": items}

    try:
        resp = requests.post(
            API_ENDPOINT,
            json={**payload, "async_": True},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return {"success": False, "error": str(e)}


def check_stats() -> dict:
    """Check Hindsight bank stats."""
    try:
        resp = requests.get(
            f"{HINDSIGHT_URL}/v1/default/banks/{HINDSIGHT_BANK}/stats",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Warning: Could not fetch stats: {e}", file=sys.stderr)
        return {}


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Ingest filtered facts into Hindsight")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help="Path to filtered_facts.json from Phase 3",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int, default=10,
        help="Number of facts per API batch (default: 10)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int, default=0,
        help="Limit number of facts to ingest (0 = all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build items but don't send to Hindsight",
    )
    parser.add_argument(
        "--delay",
        type=float, default=1.0,
        help="Seconds between batches (default: 1.0)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last successfully ingested batch (reads progress file)",
    )
    args = parser.parse_args()

    facts = load_filtered_facts(args.input)

    if args.limit > 0:
        facts = facts[:args.limit]
        print(f"Limited to first {args.limit} facts")

    print(f"Preparing {len(facts)} facts for ingestion (batch size: {args.batch_size})")

    # Build all items first
    items = [build_item(f) for f in facts]

    # Resume support
    progress_file = Path(args.input).parent / "phase4_progress.json"
    start_index = 0
    success_count = 0
    fail_count = 0
    operation_ids = []

    if args.resume and progress_file.exists():
        with open(progress_file) as pf:
            progress = json.load(pf)
            start_index = progress.get("last_index", 0)
            success_count = progress.get("success_count", 0)
            fail_count = progress.get("fail_count", 0)
            operation_ids = progress.get("operation_ids", [])
            print(f"Resuming from batch starting at index {start_index} "
                  f"({success_count} already submitted, {fail_count} failed)")

    # Show pre-ingestion stats
    print("\nPre-ingestion stats:")
    stats_before = check_stats()
    if stats_before:
        print(f"  Nodes: {stats_before.get('total_nodes', '?')}, "
              f"Links: {stats_before.get('total_links', '?')}, "
              f"Pending: {stats_before.get('pending_consolidation', '?')}")

    # Ingest in batches
    total_batches = (len(items) - start_index + args.batch_size - 1) // args.batch_size

    print(f"\nIngesting {len(items) - start_index} facts in {total_batches} batches "
          f"(starting from index {start_index})...")

    for i in range(0, len(items), args.batch_size):
        if i < start_index:
            continue

        batch = items[i : i + args.batch_size]
        batch_num = (i - start_index) // args.batch_size + 1

        result = ingest_batch(batch, dry_run=args.dry_run)

        if result.get("success", False) or result.get("dry_run"):
            success_count += len(batch)
            op_id = result.get("operation_id") or result.get("items_count", "?")
            if not args.dry_run and op_id != "?":
                operation_ids.append(op_id)
            print(f"  Batch {batch_num}/{total_batches}: OK ({len(batch)} facts)")
        else:
            fail_count += len(batch)
            error = result.get("error", "unknown")
            print(f"  Batch {batch_num}/{total_batches}: FAILED - {error}")

        # Save progress after each batch
        with open(progress_file, "w") as pf:
            json.dump({
                "last_index": i + args.batch_size,
                "success_count": success_count,
                "fail_count": fail_count,
                "operation_ids": operation_ids,
            }, pf)

        if not args.dry_run and i + args.batch_size < len(items):
            time.sleep(args.delay)

    # Summary
    print(f"\n--- Phase 4 Summary ---")
    print(f"Total facts: {len(facts)}")
    print(f"Successfully submitted: {success_count}")
    print(f"Failed: {fail_count}")
    if operation_ids:
        print(f"Operation IDs: {len(operation_ids)} async operations")
    if args.dry_run:
        print("(DRY RUN - no data sent to Hindsight)")

    # Show post-ingestion stats after a brief wait
    if not args.dry_run and success_count > 0:
        print("\nWaiting 5 seconds for Hindsight to start processing...")
        time.sleep(5)
        stats_after = check_stats()
        if stats_after:
            print(f"Post-ingestion stats:")
            print(f"  Nodes: {stats_after.get('total_nodes', '?')}, "
                  f"Links: {stats_after.get('total_links', '?')}, "
                  f"Pending: {stats_after.get('pending_consolidation', '?')}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()