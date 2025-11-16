import json
import re
import time
from pathlib import Path
from deep_translator import GoogleTranslator

def mask_placeholders(text):
    placeholders = []
    def repl(match):
        placeholders.append(match.group(0))
        return f"__PH_{len(placeholders)-1}__"
    masked = placeholder_pattern.sub(repl, text)
    return masked, placeholders

def unmask_placeholders(text, placeholders):
    for idx, placeholder in enumerate(placeholders):
        text = text.replace(f"__PH_{idx}__", placeholder)
    return text

def translate_chunk(texts, retries=3, delay=1.2):
    for attempt in range(1, retries + 1):
        try:
            return translator.translate_batch(texts)
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"Translate error (attempt {attempt}): {exc}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 1.5

placeholder_pattern = re.compile(r"\{[^{}]+\}")
translator = GoogleTranslator(source='en', target='is')
root = Path('locales')
en_path = root / 'en.json'
is_path = root / 'is.json'

en_translations = json.loads(en_path.read_text(encoding='utf-8'))
translated_entries = {}
items = list(en_translations.items())
chunk_size = 40
print(f"Translating {len(items)} entries to Icelandic in {chunk_size}-item chunks...")
for start in range(0, len(items), chunk_size):
    slice_items = items[start:start + chunk_size]
    masked_texts = []
    placeholders_list = []
    for key, text in slice_items:
        masked, placeholders = mask_placeholders(text)
        masked_texts.append(masked)
        placeholders_list.append(placeholders)

    translations = translate_chunk(masked_texts)
    for (key, _), translated, placeholders in zip(slice_items, translations, placeholders_list):
        translated_entries[key] = unmask_placeholders(translated, placeholders)
    done = start // chunk_size + 1
    total = (len(items) + chunk_size - 1) // chunk_size
    print(f"  -> Chunk {done}/{total} done")
    time.sleep(0.5)

is_path.write_text(json.dumps(translated_entries, ensure_ascii=False, indent=4), encoding='utf-8')
print(f"Wrote {is_path}")
