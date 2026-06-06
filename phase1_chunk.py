#!/usr/bin/env python3
"""
Phase 1: SMS XML Parser & Chunker

Reads SMS Backup & Restore XML format, groups messages by contact,
and outputs structured JSON chunks for LLM extraction.

Input:  sms-*.xml (SMS Backup & Restore format)
Output: chunks/{contact_name}_{date_range}.json files

Usage:
    python3 phase1_chunk.py --input sms-backup.xml --output chunks/ --contacts contacts_filter.json
    python3 phase1_chunk.py --min-messages 20
"""

import xml.etree.ElementTree as ET
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse


def parse_timestamp(ms_str):
    """Convert SMS Backup & Restore millisecond timestamp to datetime."""
    try:
        ts = int(ms_str)
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None


def normalize_phone(raw):
    """Normalize a phone number to E.164 format: +1XXXXXXXXXX
    
    Handles: +13129613928, 3129613928, (312) 961-3928, 13129613928, etc.
    Returns None if not a phone number.
    """
    if not raw:
        return None
    # Strip everything except digits and leading +
    cleaned = raw.strip()
    digits = re.sub(r'[^\d]', '', cleaned)
    
    if not digits:
        return None
    
    # Too short for a phone number
    if len(digits) < 7:
        return None
    
    # Already has country code (US numbers: 11 digits starting with 1)
    if len(digits) == 11 and digits.startswith('1'):
        return f'+{digits}'
    
    # 10-digit US number (no country code)
    if len(digits) == 10:
        return f'+1{digits}'
    
    # International numbers (not US)
    if len(digits) > 11 or (len(digits) == 11 and not digits.startswith('1')):
        return f'+{digits}'
    
    # US number with country code already
    if len(digits) == 12 and digits.startswith('1'):
        return f'+{digits}'
    
    # Fallback
    return f'+{digits}'


def load_contacts(filepath):
    """Load contacts filter from JSON file produced by parse_contacts.py.
    
    The filter file is a flat dict keyed by normalized phone number,
    each value being {name, org, title, notes, emails, phones}.
    """
    if not filepath or not os.path.exists(filepath):
        return None  # None means no filter (include all)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    phone_lookup = {}
    name_lookup = set()
    
    if isinstance(data, dict) and len(data) > 0:
        first_key = next(iter(data))
        first_val = data[first_key]
        
        if isinstance(first_val, dict) and 'name' in first_val:
            # Flat dict format: {phone: {name, org, ...}, ...}
            for phone_key, contact_info in data.items():
                norm = normalize_phone(phone_key)
                if norm:
                    phone_lookup[norm] = contact_info
                for p in contact_info.get('phones', []):
                    norm2 = normalize_phone(p)
                    if norm2 and norm2 not in phone_lookup:
                        phone_lookup[norm2] = contact_info
                
                name = contact_info.get('name', '').strip()
                if name:
                    name_lookup.add(name.lower())
        
        elif 'phone_index' in data:
            for phone, info in data['phone_index'].items():
                norm = normalize_phone(phone)
                if norm:
                    phone_lookup[norm] = info
        
        elif 'contacts' in data:
            for contact in data['contacts']:
                for phone in contact.get('phones', []):
                    norm = normalize_phone(phone)
                    if norm:
                        phone_lookup[norm] = contact
                name = contact.get('name', '').strip()
                if name:
                    name_lookup.add(name.lower())
    
    print(f"Loaded {len(phone_lookup)} phone numbers and {len(name_lookup)} names from filter")
    return {'phones': phone_lookup, 'names': name_lookup}


def normalize_name(name):
    """Normalize a contact name for matching."""
    if not name or name in ('(Unknown)', '(unknown)'):
        return None
    return name.strip().lower()


def determine_message_type(addr):
    """Classify address as shortcode, phone, or other."""
    if not addr:
        return 'unknown'
    digits = re.sub(r'[+\-\s()]', '', addr)
    if len(digits) <= 6 and digits.isdigit():
        return 'shortcode'
    if '@' in addr:
        return 'email'
    return 'phone'


def parse_xml(filepath, progress_interval=10000):
    """Parse SMS Backup & Restore XML, yielding message dicts."""
    print(f"Parsing {filepath}...")
    count = 0
    context = ET.iterparse(filepath, events=('end',))

    for event, elem in context:
        if elem.tag == 'sms':
            addr = elem.get('address', '')
            contact_name = elem.get('contact_name', '')
            date_ms = elem.get('date', '')
            msg_type = elem.get('type', '')  # 1=received, 2=sent
            body = elem.get('body', '') or ''

            ts = parse_timestamp(date_ms)

            yield {
                'address': addr,
                'contact_name': contact_name,
                'timestamp': ts,
                'date_ms': date_ms,
                'type': 'received' if msg_type == '1' else 'sent' if msg_type == '2' else f'unknown_{msg_type}',
                'body': body,
                'msg_type_raw': msg_type,
                'norm_phone': normalize_phone(addr),
            }

            count += 1
            if count % progress_interval == 0:
                print(f"  Parsed {count:,} messages...")

            elem.clear()

    print(f"  Total messages parsed: {count:,}")


def chunk_messages(filepath, output_dir, contacts_filter=None, min_messages=20):
    """
    Parse XML and create per-conversation chunks.
    
    Each chunk covers one contact's messages for a time window,
    with at least min_messages per chunk.
    """
    contact_groups = defaultdict(list)
    skipped_shortcodes = 0
    skipped_empty = 0
    skipped_not_in_filter = 0
    total = 0

    phone_lookup = contacts_filter['phones'] if contacts_filter else {}
    name_lookup = contacts_filter['names'] if contacts_filter else set()

    for msg in parse_xml(filepath):
        total += 1
        body = msg.get('body', '') or ''
        addr = msg.get('address', '')
        norm_phone = msg.get('norm_phone')

        # Skip short codes (automated, 2FA, etc.)
        if determine_message_type(addr) == 'shortcode':
            skipped_shortcodes += 1
            continue

        # Skip empty bodies (delivery receipts, etc.)
        if not body.strip():
            skipped_empty += 1
            continue

        # Determine contact identity
        name_from_sms = msg.get('contact_name', '')
        norm_name = normalize_name(name_from_sms)

        # Apply contact filter
        if contacts_filter is not None:
            matched = False
            if norm_phone and norm_phone in phone_lookup:
                matched = True
            if not matched and norm_name and norm_name in name_lookup:
                matched = True
            if not matched:
                skipped_not_in_filter += 1
                continue
            # Get best display name from contacts data
            if norm_phone and norm_phone in phone_lookup:
                display_name = phone_lookup[norm_phone].get('name', name_from_sms)
            elif norm_name:
                display_name = name_from_sms
            else:
                display_name = name_from_sms or addr
        else:
            display_name = name_from_sms or addr

        # Group key: normalized phone for same-person grouping
        if norm_phone:
            group_key = norm_phone
        else:
            group_key = display_name.lower().strip() if display_name else addr

        msg['_display_name'] = display_name
        msg['_group_key'] = group_key
        contact_groups[group_key].append(msg)

    # Resolve group keys to display names
    group_names = {}
    for group_key, msgs in contact_groups.items():
        if group_key in phone_lookup:
            group_names[group_key] = phone_lookup[group_key].get('name', 'Unknown')
        else:
            name_counts = defaultdict(int)
            for m in msgs:
                dn = m.get('_display_name', '')
                if dn and dn != '(Unknown)':
                    name_counts[dn] += 1
            if name_counts:
                group_names[group_key] = max(name_counts, key=name_counts.get)
            else:
                group_names[group_key] = group_key

    # Filter by minimum messages
    final_groups = {}
    for group_key, msgs in contact_groups.items():
        if len(msgs) >= min_messages:
            final_groups[group_key] = msgs

    print(f"\nGrouping results:")
    print(f"  Total messages in file: {total:,}")
    print(f"  Skipped (shortcodes): {skipped_shortcodes:,}")
    print(f"  Skipped (empty body): {skipped_empty:,}")
    print(f"  Skipped (not in filter): {skipped_not_in_filter:,}")
    print(f"  Contact groups (>= {min_messages} msgs): {len(final_groups)}")
    print(f"  Messages in groups: {sum(len(v) for v in final_groups.values()):,}")

    # Write chunks
    os.makedirs(output_dir, exist_ok=True)
    chunk_count = 0

    for group_key, msgs in sorted(final_groups.items(), key=lambda x: -len(x[1])):
        msgs.sort(key=lambda m: m.get('date_ms', '0').zfill(20))
        display_name = group_names.get(group_key, group_key)

        CHUNK_SIZE = 200
        for i in range(0, len(msgs), CHUNK_SIZE):
            chunk_msgs = msgs[i:i + CHUNK_SIZE]
            dates = [m['timestamp'] for m in chunk_msgs if m['timestamp']]
            if dates:
                start_date = min(dates).strftime('%Y%m%d')
                end_date = max(dates).strftime('%Y%m%d')
            else:
                start_date = 'unknown'
                end_date = 'unknown'

            safe_name = re.sub(r'[^\w\s-]', '', display_name).strip().replace(' ', '_')[:40]
            chunk_id = f"{safe_name}_{start_date}_{end_date}"

            chunk_data = {
                'chunk_id': chunk_id,
                'contact': display_name,
                'phone': group_key if group_key.startswith('+') else '',
                'date_range': f"{start_date} to {end_date}",
                'message_count': len(chunk_msgs),
                'messages': [
                    {
                        'ts': m['timestamp'].isoformat() if m['timestamp'] else m['date_ms'],
                        'dir': m['type'],
                        'text': m['body'],
                    }
                    for m in chunk_msgs
                ]
            }

            filename = f"{chunk_id}.json"
            filepath_out = os.path.join(output_dir, filename)
            with open(filepath_out, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=1)
            chunk_count += 1

    print(f"\nWrote {chunk_count} chunks to {output_dir}")
    return chunk_count


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Parse SMS XML into conversation chunks')
    parser.add_argument('--input', '-i', required=True,
                        help='Input SMS Backup & Restore XML file path')
    parser.add_argument('--output', '-o', default='./chunks',
                        help='Output directory for chunks (default: ./chunks)')
    parser.add_argument('--contacts', '-c', default=None,
                        help='JSON contacts filter file (from parse_contacts.py)')
    parser.add_argument('--min-messages', '-m', type=int, default=20,
                        help='Minimum messages per contact to include (default: 20)')

    args = parser.parse_args()

    contacts_filter = load_contacts(args.contacts)
    chunk_messages(args.input, args.output, contacts_filter, args.min_messages)


if __name__ == '__main__':
    main()