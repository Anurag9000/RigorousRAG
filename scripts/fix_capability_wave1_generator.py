from pathlib import Path

path = Path("scripts/capability_wave1_tests_docs.py")
text = path.read_text(encoding="utf-8")
suffix = '\n\'\'\', encoding="utf-8")\n'
if not text.endswith(suffix):
    raise SystemExit("Expected unmatched generator suffix was not found.")
path.write_text(text[: -len(suffix)] + "\n", encoding="utf-8")
