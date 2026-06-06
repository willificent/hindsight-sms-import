#!/usr/bin/env python3
"""
Phase 3: Post-processing & Filtering of Extracted SMS Facts

Reads per-chunk JSON files from Phase 2 extraction, then:
  1. Validates structural integrity
  2. Loads and normalizes entity names against contacts_filter.json
  3. Filters by salience >= threshold (default 3)
  4. Filters by confidence >= threshold (default 0.7)
  5. Deduplicates facts (entity-blocked shingled Jaccard similarity)
  6. Normalizes entity names to canonical forms
  7. Writes filtered_facts.json ready for Phase 4 Hindsight ingestion

Usage:
  python phase3_filter.py                          # Default settings
  python phase3_filter.py --salience 3 --confidence 0.7
  python phase3_filter.py --dry-run                 # Show stats without writing
  python phase3_filter.py --input extracted/         # Custom input dir

Output:
  filtered_facts.json — deduplicated, filtered facts
  filter_report.json — statistics and decisions
"""

import argparse
import json
import os
import re
from collections import defaultdict


# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_EXTRACTED_DIR = os.path.join(os.path.dirname(__file__), "extracted")
DEFAULT_CONTACTS_PATH = os.path.join(os.path.dirname(__file__), "contacts_filter.json")
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "filtered_facts.json")
DEFAULT_REPORT_PATH = os.path.join(os.path.dirname(__file__), "filter_report.json")

# Names that refer to the user — customize these for your own setup
# These will all be normalized to the first entry
SELF_NAMES = set()  # e.g., {"jonathan", "jon", "johnny"}
# Names that refer to the user's partner
PARTNER_NAMES = set()  # e.g., {"sam", "samp"}
PARTNER_CANONICAL = ""  # e.g., "Sam Parker"

# Minimum salience and confidence thresholds
DEFAULT_SALIENCE = 3
DEFAULT_CONFIDENCE = 0.7

# Fuzzy match threshold for deduplication (0-1, higher = stricter)
DEDUP_THRESHOLD = 0.75


def load_contacts(path):
    """Load contacts_filter.json and build normalization maps."""
    with open(path) as f:
        raw = json.load(f)

    canonical = {}  # canonical full name -> info
    alias_map = {}  # lowercase variant -> canonical name

    for phone, info in raw.items():
        name = info.get("name", "").strip()
        if not name:
            continue

        if name not in canonical:
            canonical[name] = {"phones": [], "aliases": set()}
        canonical[name]["phones"].append(phone)

        # Add first name as alias
        parts = name.split()
        if len(parts) >= 1:
            canonical[name]["aliases"].add(parts[0].lower())

        # Add first+last as alias (no middle)
        if len(parts) >= 2:
            canonical[name]["aliases"].add(f"{parts[0]} {parts[-1]}".lower())

    # Build alias -> canonical map
    for name, info in canonical.items():
        alias_map[name.lower()] = name
        for alias in info["aliases"]:
            if alias not in alias_map:
                alias_map[alias] = name

    # Add self-names and partner-names
    if SELF_NAMES:
        canonical_name = list(SELF_NAMES)[0].title()
        for alias in SELF_NAMES:
            alias_map[alias] = canonical_name
    if PARTNER_NAMES and PARTNER_CANONICAL:
        for alias in PARTNER_NAMES:
            alias_map[alias] = PARTNER_CANONICAL

    return canonical, alias_map


def normalize_entity(entity, alias_map):
    """Normalize an entity name to its canonical form."""
    e = entity.strip()
    e_lower = e.lower()

    # Direct match in alias map
    if e_lower in alias_map:
        return alias_map[e_lower]

    # Try removing possessives
    if e_lower.endswith("'s"):
        if e_lower[:-2] in alias_map:
            return alias_map[e_lower[:-2]]

    # Try first name + last initial
    parts = e.split()
    if len(parts) == 2 and len(parts[1]) <= 2 and parts[1].endswith("."):
        first = parts[0].lower()
        if first in alias_map:
            return alias_map[first]

    return e


def normalize_entities(fact, alias_map):
    """Normalize all entity names in a fact."""
    if "entities" not in fact:
        return fact

    normalized = []
    for entity in fact["entities"]:
        n = normalize_entity(entity, alias_map)
        if n not in normalized:  # Deduplicate within fact
            normalized.append(n)

    fact["entities"] = normalized
    return fact


def is_valid_fact(fact):
    """Check basic structural validity of a fact."""
    content = fact.get("content", "").strip()
    if len(content) < 10:
        return False
    if content.count(" ") < 2:
        return False
    alpha_count = sum(1 for c in content if c.isalpha())
    if alpha_count < 5:
        return False
    return True


def make_shingles(text, size=3):
    """Create character shingles (n-grams) from text for fast similarity estimation."""
    text = text.lower().strip()
    if len(text) < size:
        return {text}
    return {text[i:i+size] for i in range(len(text) - size + 1)}


def fast_similarity(fact_a, fact_b):
    """Fast similarity using shingle Jaccard + entity overlap + category."""
    content_a = fact_a.get("content", "").lower().strip()
    content_b = fact_b.get("content", "").lower().strip()

    if content_a == content_b:
        return 1.0

    # Shingle-based content similarity
    shingles_a = make_shingles(content_a)
    shingles_b = make_shingles(content_b)
    if shingles_a and shingles_b:
        shingle_sim = len(shingles_a & shingles_b) / len(shingles_a | shingles_b)
    else:
        shingle_sim = 0.0

    # Entity overlap
    entities_a = set(e.lower() for e in fact_a.get("entities", []))
    entities_b = set(e.lower() for e in fact_b.get("entities", []))
    if entities_a and entities_b:
        entity_sim = len(entities_a & entities_b) / len(entities_a | entities_b)
    elif not entities_a and not entities_b:
        entity_sim = 0.5
    else:
        entity_sim = 0.0

    # Category match
    cat_sim = 1.0 if fact_a.get("category") == fact_b.get("category") else 0.0

    return 0.6 * shingle_sim + 0.3 * entity_sim + 0.1 * cat_sim


def deduplicate_facts(facts, threshold=DEDUP_THRESHOLD):
    """Remove duplicate/near-duplicate facts using blocking + fast similarity.
    
    Instead of O(n²) all-pairs comparison, we use entity-based blocking:
    facts are only compared against other facts that share at least one entity.
    """
    if not facts:
        return facts, []

    # Sort by salience descending, then confidence descending
    facts_sorted = sorted(
        facts,
        key=lambda f: (f.get("salience", 0), f.get("confidence", 0)),
        reverse=True,
    )

    entity_index = {}
    kept = []
    duplicates_removed = []

    for fact in facts_sorted:
        is_dup = False
        entities = set(e.lower() for e in fact.get("entities", []))

        # Gather candidate facts that share at least one entity
        candidate_indices = set()
        for entity in entities:
            if entity in entity_index:
                candidate_indices.update(entity_index[entity])

        cat = fact.get("category", "")
        if cat in entity_index:
            candidate_indices.update(entity_index[cat])

        for idx in sorted(candidate_indices):
            existing = kept[idx]
            sim = fast_similarity(fact, existing)
            if sim >= threshold:
                is_dup = True
                duplicates_removed.append({
                    "removed": fact.get("content", "")[:100],
                    "kept": existing.get("content", "")[:100],
                    "similarity": round(sim, 3),
                })
                break

        if not is_dup:
            kept.append(fact)
            kept_idx = len(kept) - 1
            for entity in entities:
                entity_index.setdefault(entity, []).append(kept_idx)
            if cat:
                entity_index.setdefault(cat, []).append(kept_idx)
            if not entities:
                entity_index.setdefault("__no_entities__", []).append(kept_idx)

    return kept, duplicates_removed


def load_all_facts(extracted_dir):
    """Load all fact files from the extraction output directory."""
    all_facts = []
    files_processed = 0
    files_skipped = 0

    for filename in sorted(os.listdir(extracted_dir)):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(extracted_dir, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)

            facts = data.get("facts", [])
            if not facts:
                files_skipped += 1
                continue

            all_facts.extend(facts)
            files_processed += 1

        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Could not load {filename}: {e}")
            files_skipped += 1

    return all_facts, files_processed, files_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Filter and deduplicate extracted SMS facts"
    )
    parser.add_argument(
        "--input", default=DEFAULT_EXTRACTED_DIR,
        help="Input directory (from Phase 2)"
    )
    parser.add_argument(
        "--contacts", default=DEFAULT_CONTACTS_PATH,
        help="Path to contacts_filter.json"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path"
    )
    parser.add_argument(
        "--report", default=DEFAULT_REPORT_PATH, help="Report JSON path"
    )
    parser.add_argument(
        "--salience", type=int, default=DEFAULT_SALIENCE,
        help=f"Minimum salience threshold (default: {DEFAULT_SALIENCE})"
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help=f"Minimum confidence threshold (default: {DEFAULT_CONFIDENCE})"
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=DEDUP_THRESHOLD,
        help=f"Fuzzy dedup similarity threshold (default: {DEDUP_THRESHOLD})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show stats without writing output files"
    )
    parser.add_argument(
        "--categories", nargs="*",
        help="Only include these categories (e.g., --categories relationship health career)"
    )
    args = parser.parse_args()

    # ── Load contacts for entity normalization ──────────────────────────────
    if os.path.exists(args.contacts):
        print(f"Loading contacts from {args.contacts}...")
        canonical, alias_map = load_contacts(args.contacts)
        print(f"  {len(canonical)} canonical names, {len(alias_map)} alias mappings")
    else:
        print(f"Warning: contacts filter not found at {args.contacts}, skipping normalization")
        alias_map = {}
        canonical = {}

    # ── Load all extracted facts ────────────────────────────────────────────
    print(f"\nLoading facts from {args.input}...")
    all_facts, files_processed, files_skipped = load_all_facts(args.input)
    print(f"  Loaded {len(all_facts)} facts from {files_processed} files ({files_skipped} skipped)")

    if not all_facts:
        print("No facts found. Exiting.")
        return

    # ── Step 1: Structural validity ──────────────────────────────────────────
    valid_facts = [f for f in all_facts if is_valid_fact(f)]
    invalid_count = len(all_facts) - len(valid_facts)
    print(f"\nStep 1 — Structural validity: {len(valid_facts)}/{len(all_facts)} passed ({invalid_count} removed)")

    # ── Step 2: Salience filter ─────────────────────────────────────────────
    salience_passed = [f for f in valid_facts if f.get("salience", 0) >= args.salience]
    salience_removed = len(valid_facts) - len(salience_passed)
    print(f"Step 2 — Salience >= {args.salience}: {len(salience_passed)}/{len(valid_facts)} passed ({salience_removed} removed)")

    # ── Step 3: Confidence filter ───────────────────────────────────────────
    confidence_passed = [f for f in salience_passed if f.get("confidence", 0) >= args.confidence]
    confidence_removed = len(salience_passed) - len(confidence_passed)
    print(f"Step 3 — Confidence >= {args.confidence}: {len(confidence_passed)}/{len(salience_passed)} passed ({confidence_removed} removed)")

    # ── Step 4: Category filter (optional) ─────────────────────────────────
    if args.categories:
        category_passed = [f for f in confidence_passed if f.get("category") in args.categories]
        category_removed = len(confidence_passed) - len(category_passed)
        print(f"Step 4 — Categories {args.categories}: {len(category_passed)}/{len(confidence_passed)} passed ({category_removed} removed)")
        filtered = category_passed
    else:
        filtered = confidence_passed

    # ── Step 5: Entity normalization ────────────────────────────────────────
    print(f"\nStep 5 — Entity normalization...")
    entity_changes = 0
    for fact in filtered:
        original = list(fact.get("entities", []))
        normalize_entities(fact, alias_map)
        if fact.get("entities", []) != original:
            entity_changes += 1
    print(f"  Normalized entities in {entity_changes} facts")

    # ── Step 6: Deduplication ───────────────────────────────────────────────
    print(f"\nStep 6 — Deduplication (threshold={args.dedup_threshold})...")
    deduped, duplicates = deduplicate_facts(filtered, args.dedup_threshold)
    print(f"  {len(deduped)}/{len(filtered)} unique facts ({len(duplicates)} duplicates removed)")

    # ── Sort by salience (desc), then by category ──────────────────────────
    deduped.sort(key=lambda f: (-f.get("salience", 0), f.get("category", ""), f.get("content", "")))

    # ── Statistics ───────────────────────────────────────────────────────────
    categories = defaultdict(int)
    for f in deduped:
        categories[f.get("category", "uncategorized")] += 1

    contacts_with_facts = defaultdict(int)
    for f in deduped:
        contact = f.get("source_contact", "unknown")
        contacts_with_facts[contact] += 1

    report = {
        "input_facts": len(all_facts),
        "after_validity": len(valid_facts),
        "after_salience": len(salience_passed),
        "after_confidence": len(confidence_passed),
        "after_dedup": len(deduped),
        "removed": {
            "invalid": invalid_count,
            "low_salience": salience_removed,
            "low_confidence": confidence_removed,
            "duplicates": len(duplicates),
        },
        "categories": dict(sorted(categories.items(), key=lambda x: -x[1])),
        "top_contacts": dict(
            sorted(contacts_with_facts.items(), key=lambda x: -x[1])[:20]
        ),
        "entity_normalizations": entity_changes,
        "salience_threshold": args.salience,
        "confidence_threshold": args.confidence,
        "dedup_threshold": args.dedup_threshold,
        "duplicate_examples": duplicates[:20],
    }

    # ── Output ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL: {len(deduped)} facts from {len(all_facts)} raw extracted facts")
    print(f"  Removed: {invalid_count} invalid, {salience_removed} low-salience, "
          f"{confidence_removed} low-confidence, {len(duplicates)} duplicates")
    print(f"\nCategories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print(f"\nTop contacts by fact count:")
    for contact, count in sorted(contacts_with_facts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {contact}: {count}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(deduped)} facts to {args.output}")
        print(f"[DRY RUN] Would write report to {args.report}")
    else:
        with open(args.output, "w") as f:
            json.dump(deduped, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {len(deduped)} facts to {args.output}")

        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Wrote filter report to {args.report}")


if __name__ == "__main__":
    main()