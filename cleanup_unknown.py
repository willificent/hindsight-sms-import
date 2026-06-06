#!/usr/bin/env python3
"""
Cleanup script: Delete Hindsight documents with 'unknown' context or test context,
then re-ingest facts from filtered_facts.json using the fixed Phase 4 script.

This script finds all SMS-related documents in Hindsight that have incomplete or
placeholder context strings (e.g., "SMS conversation with unknown" or "SMS conversation")
and deletes them, preparing for a clean re-ingestion with correct contact names.

Usage:
    # Step 1: Find and delete unknown/test SMS documents (dry run)
    python3 cleanup_unknown.py

    # Step 2: Actually delete them
    python3 cleanup_unknown.py --delete

    # Step 3: Re-ingest using Phase 4 (after cleanup)
    python3 phase4_ingest.py
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

HINDSIGHT_HOST = os.environ.get("HINDSIGHT_HOST", "localhost")
HINDSIGHT_PORT = os.environ.get("HINDSIGHT_PORT", "9077")
HINDSIGHT_BANK = os.environ.get("HINDSIGHT_BANK", "default")
HINDSIGHT_URL = f"http://{HINDSIGHT_HOST}:{HINDSIGHT_PORT}"


def list_documents():
    """List all documents in the Hindsight bank."""
    url = f"{HINDSIGHT_URL}/v1/default/banks/{HINDSIGHT_BANK}/documents?limit=1000"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get('items', [])


def find_sms_documents(docs):
    """Find all SMS-related documents with 'unknown' context or test context."""
    sms_docs = []
    for doc in docs:
        rp = doc.get('retain_params') or {}
        ctx = rp.get('context', '') if isinstance(rp, dict) else ''
        tags = doc.get('tags', [])
        is_sms = any('sms' in str(t) for t in tags)

        # Match documents that are SMS-related AND have incomplete context
        if is_sms and ('unknown' in ctx.lower() or ctx == 'SMS conversation' or 'test' in ctx.lower()):
            sms_docs.append({
                'id': doc['id'],
                'context': ctx,
                'units': doc.get('memory_unit_count', 0),
                'tags': tags
            })
    return sms_docs


def delete_document(doc_id):
    """Delete a single document from Hindsight."""
    url = f"{HINDSIGHT_URL}/v1/default/banks/{HINDSIGHT_BANK}/documents/{doc_id}"
    req = urllib.request.Request(url, method='DELETE')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status == 200 or resp.status == 204
    except urllib.error.HTTPError as e:
        print(f"  Error deleting {doc_id}: {e.code} {e.reason}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Cleanup Hindsight SMS documents with incomplete context")
    parser.add_argument('--delete', action='store_true', help='Actually delete the documents (default: dry run)')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between deletions in seconds')
    args = parser.parse_args()

    # Step 1: List all documents
    print("Fetching documents from Hindsight...")
    docs = list_documents()
    print(f"Total documents: {len(docs)}")

    # Step 2: Find SMS docs to remove
    sms_docs = find_sms_documents(docs)
    total_units = sum(d['units'] for d in sms_docs)
    print(f"SMS documents to remove: {len(sms_docs)}")
    print(f"Memory units affected: {total_units}")

    if not sms_docs:
        print("\nNo SMS documents with incomplete context found. Nothing to do!")
        return

    # Show breakdown by context type
    unknown_count = sum(1 for d in sms_docs if 'unknown' in d['context'].lower())
    generic_count = sum(1 for d in sms_docs if d['context'] == 'SMS conversation')
    test_count = sum(1 for d in sms_docs if 'test' in d['context'].lower())
    print(f"\nContext breakdown:")
    print(f"  'SMS conversation with unknown': {unknown_count}")
    print(f"  'SMS conversation' (generic): {generic_count}")
    print(f"  Test contexts: {test_count}")

    # Tag distribution
    from collections import Counter
    tag_counts = Counter()
    for d in sms_docs:
        for t in d['tags']:
            tag_counts[t] += 1
    print("\nTag distribution:")
    for t, c in tag_counts.most_common():
        print(f"  {t}: {c}")

    # Show sample
    print("\nSample documents to remove:")
    for d in sms_docs[:5]:
        print(f"  {d['id']}: \"{d['context']}\" ({d['units']} units) {d['tags']}")
    if len(sms_docs) > 5:
        print(f"  ... and {len(sms_docs) - 5} more")

    # Step 3: Delete or dry run
    if not args.delete:
        print(f"\n[DRY RUN] Would delete {len(sms_docs)} documents ({total_units} memory units)")
        print("Run with --delete to actually delete them.")
        return

    print(f"\nDeleting {len(sms_docs)} documents...")
    success = 0
    failures = 0
    for i, doc in enumerate(sms_docs):
        if delete_document(doc['id']):
            success += 1
        else:
            failures += 1
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(sms_docs)} processed")
        time.sleep(args.delay)

    print(f"\nDone! Deleted {success} documents, {failures} failures")
    print(f"Removed {total_units} memory units")
    print("\nNext step: Re-run Phase 4 with the fixed script:")
    print("  python3 phase4_ingest.py")


if __name__ == '__main__':
    main()