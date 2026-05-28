import re
import json
import spacy

# Load spaCy model and set up pseudonym maps once
nlp = spacy.load("en_core_web_sm")
person_map = {}
org_map = {}
gpe_map = {}


def redact_pii(text: str) -> str:
    """Strip identifying information using regex + spaCy NER."""
    if not text:
        return text
    
    # Manual overrides for names spaCy keeps missing
    # Add to this list when you spot leaks during review
    manual_names = [
        'Angikaar Singh Chana',
        'Angikaar S. Chana',
        'Angikaar Chana',
        'Singh Chana',
        'Angikaar',
        'Angi',
        'Nicole',
        'Nicole White',
        'Janice',
        'Debora',
        'Debbie',
        'Carroll, Carrie M',
        'Carrie Carroll',
    ]
    for name in manual_names:
        if name not in person_map:
            person_map[name] = f"Person_{chr(65 + len(person_map))}"
        text = re.sub(rf'\b{re.escape(name)}\b', person_map[name], text)
    
    # Regex pass for things spaCy is bad at
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'=\?utf-8\?[BQ]\?[^?]+\?=', '[ENCODED]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[[\w.-]+\.(com|edu|org|gov|net)\]', '[DOMAIN]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{3,5}\s+[A-Z][\w\s]+(?:Road|Street|Avenue|Drive|Lane|Boulevard|Way|Court|Place|Rd|St|Ave|Dr|Ln|Blvd|Ct|Pl)\b', '[ADDRESS]', text)
    
    # spaCy NER pass for names, orgs, locations
    doc = nlp(text)
    replacements = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            mapping, prefix = person_map, "Person"
        elif ent.label_ == "ORG":
            mapping, prefix = org_map, "Org"
        elif ent.label_ == "GPE":
            mapping, prefix = gpe_map, "Location"
        else:
            continue
        
        if ent.text not in mapping:
            mapping[ent.text] = f"{prefix}_{chr(65 + len(mapping))}"
        
        replacements.append((ent.start_char, ent.end_char, mapping[ent.text]))
    
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    
    return text


def get_text(qa_pair: dict) -> str:
    """Clean raw email text for use in BM25 and vector search."""
    question = qa_pair.get('question_text', '')
    subject = qa_pair.get('subject', '')
    
    text = f"{subject} {question}".strip()
    text = re.split(r'\nOn .{10,100}wrote:', text)[0]
    text = re.sub(r'\n?(From|Sent|To|Cc|Subject):.*', '', text)
    text = re.sub(r'CAUTION:.*?(\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'WARNING:.*?(\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[cid:[^\]]+\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


if __name__ == "__main__":
    print("Loading dataset...")
    with open('qa_pairs_dataset.json', 'r') as f:
        data = json.load(f)
    
    total = len(data['qa_pairs'])
    print(f"Loaded {total} Q&A pairs. Starting redaction...")
    
    thread_map = {}  # NEW: track unique threads
    
    for i, pair in enumerate(data['qa_pairs']):
        pair['question_text'] = redact_pii(pair.get('question_text', ''))
        pair['answer_text']   = redact_pii(pair.get('answer_text', ''))
        pair['subject']       = redact_pii(pair.get('subject', ''))
        pair['from_addr']     = redact_pii(pair.get('from_addr', ''))
        pair['to_addr']       = redact_pii(pair.get('to_addr', ''))
        
        pair['qa_id'] = f"qa_{i:04d}"
        pair.pop('question_message_id', None)
        pair.pop('answer_message_id', None)
        
        # NEW: pseudonymize thread_id but preserve grouping
        original_thread = pair.get('thread_id', '')
        if original_thread:
            if original_thread not in thread_map:
                thread_map[original_thread] = f"thread_{len(thread_map):03d}"
            pair['thread_id'] = thread_map[original_thread]
        
        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{total}")
    
    print("Saving redacted file...")
    with open('qa_pairs_dataset_redacted.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Done! Redacted {total} Q&A pairs")
    print(f"Found {len(person_map)} unique persons, {len(org_map)} orgs, {len(gpe_map)} locations, {len(thread_map)} unique threads")