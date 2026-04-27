import urllib.request
import json

try:
    with urllib.request.urlopen('http://127.0.0.1:8004/api/docs/interview-prep', timeout=10) as r:
        data = json.loads(r.read().decode())
        print("Status: 200")
        print(f"Keys: {list(data.keys())}")
        if 'markdown' in data:
            print(f"Markdown length: {len(data['markdown'])}")
            print(f"First 300 chars: {data['markdown'][:300]}")
        else:
            print("ERROR: No 'markdown' key in response!")
            print(f"Response: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
