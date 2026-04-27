import urllib.request
import json

docs_to_test = [
    "interview-prep",
    "project-evolution",
    "research-paper-guide",
    "model-training-guide",
    "data-guide",
    "setup"
]

for slug in docs_to_test:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:8004/api/docs/{slug}', timeout=10) as r:
            data = json.loads(r.read().decode())
            markdown_len = len(data.get('markdown', ''))
            print(f"✓ {slug:25} {markdown_len:6} chars")
    except Exception as e:
        print(f"✗ {slug:25} ERROR: {type(e).__name__}")
