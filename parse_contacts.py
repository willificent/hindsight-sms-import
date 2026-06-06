#!/usr/bin/env python3
"""Parse VCF contacts file and build a contacts filter for the SMS pipeline.

Extracts:
- Full name (FN)
- Phone numbers (TEL), normalized to E.164 format (+1XXXXXXXXXX)
- Email addresses
- Notes (for relationship context)

Outputs contacts_filter.json with phone numbers keyed by normalized form.
"""

import re
import json
import sys
from pathlib import Path


def normalize_phone(raw: str) -> str | None:
    """Normalize a phone number to E.164 format (+1XXXXXXXXXX).
    
    Handles formats like:
    - +13129613928
    - +1 312-316-6550
    - (305) 562-0794
    - 336-601-3321
    - 1 (312) 881-3050
    - 919-609-2293
    """
    # Strip all non-digit characters
    digits = re.sub(r'\D', '', raw)
    
    # Remove leading country code '1' if 11 digits (US number)
    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]
    
    if len(digits) == 10:
        return f"+1{digits}"
    elif len(digits) > 10:
        # International number - keep as is with + prefix
        return f"+{digits}"
    elif len(digits) < 7:
        # Too short to be a real phone number (likely a shortcode or error)
        return None
    else:
        # Ambiguous - return what we can
        return f"+{digits}"


def parse_vcf(vcf_path: str) -> list[dict]:
    """Parse a VCF file and extract contacts with phone numbers."""
    contacts = []
    current = {}
    
    with open(vcf_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n\r')
            
            if line.startswith('BEGIN:VCARD'):
                current = {
                    'name': '',
                    'phones': [],
                    'emails': [],
                    'notes': [],
                    'org': '',
                    'title': '',
                }
            elif line.startswith('END:VCARD'):
                if current.get('name') or current.get('phones'):
                    contacts.append(current)
                current = {}
            elif line.startswith('FN') and ':' in line:
                # Extract full name (handle FN;TYPE=PREF:Name format)
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current['name'] = parts[1].strip()
            elif line.startswith('TEL') and ':' in line:
                # Extract phone number
                parts = line.split(':', 1)
                if len(parts) == 2:
                    raw_phone = parts[1].strip()
                    normalized = normalize_phone(raw_phone)
                    if normalized:
                        current['phones'].append({
                            'raw': raw_phone,
                            'normalized': normalized
                        })
            elif line.startswith('EMAIL') and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    email = parts[1].strip()
                    if email:
                        current['emails'].append(email)
            elif line.startswith('NOTE') and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    note_text = parts[1].strip()
                    if note_text:
                        current['notes'].append(note_text)
            elif line.startswith('ORG') and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current['org'] = parts[1].strip().replace('\\,', ',').replace('\\;', ';')
            elif line.startswith('TITLE') and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current['title'] = parts[1].strip()
            # Skip PHOTO lines and other binary data
    
    return contacts


def build_contacts_filter(contacts: list[dict]) -> dict:
    """Build the contacts_filter.json structure.
    
    Returns a dict mapping normalized phone numbers to contact info.
    Multiple phone numbers for the same contact all point to the same info.
    """
    filter_data = {}
    
    for contact in contacts:
        name = contact['name']
        if not name:
            continue
            
        info = {
            'name': name,
            'org': contact.get('org', ''),
            'title': contact.get('title', ''),
            'notes': contact.get('notes', []),
            'emails': contact.get('emails', []),
            'phones': [p['normalized'] for p in contact.get('phones', [])],
        }
        
        # Add each phone number as a key
        for phone in contact.get('phones', []):
            normalized = phone['normalized']
            if normalized not in filter_data:
                filter_data[normalized] = info
            else:
                # Merge: keep the more detailed entry
                existing = filter_data[normalized]
                if len(info.get('notes', [])) > len(existing.get('notes', [])):
                    filter_data[normalized] = info
    
    return filter_data


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_contacts.py <path_to_contacts.vcf>")
        print("  Output: contacts_filter.json in same directory as the script")
        sys.exit(1)
    
    vcf_path = sys.argv[1]
    
    if not Path(vcf_path).exists():
        print(f"Error: VCF file not found at {vcf_path}")
        sys.exit(1)
    
    print(f"Parsing VCF file: {vcf_path}")
    contacts = parse_vcf(vcf_path)
    
    print(f"Found {len(contacts)} contacts")
    
    # Count contacts with phone numbers
    with_phones = sum(1 for c in contacts if c['phones'])
    print(f"  {with_phones} contacts have phone numbers")
    
    # Build the filter
    filter_data = build_contacts_filter(contacts)
    print(f"  {len(filter_data)} unique phone numbers in filter")
    
    # Save the filter (next to the script by default)
    output_path = Path(args.output) if args.output else Path(__file__).parent / 'contacts_filter.json'
    with open(output_path, 'w') as f:
        json.dump(filter_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved contacts filter to: {output_path}")
    
    # Print summary
    print(f"\n--- Contact Summary ---")
    for phone, info in sorted(filter_data.items()):
        notes_preview = ''
        if info.get('notes'):
            first_note = info['notes'][0][:60]
            notes_preview = f" | {first_note}..."
        org = f" ({info['org']})" if info.get('org') else ''
        print(f"  {phone}: {info['name']}{org}{notes_preview}")

if __name__ == '__main__':
    main()