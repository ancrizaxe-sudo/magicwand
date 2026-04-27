import urllib.request
import json

try:
    with urllib.request.urlopen('http://127.0.0.1:8004/api/blockchain', timeout=10) as r:
        data = json.loads(r.read().decode())
        print("Status: 200")
        print(f"Keys in response: {list(data.keys())}")
        print(f"Status keys: {list(data.get('status', {}).keys())}")
        print(f"Records count: {len(data.get('records', []))}")
        if data.get('records'):
            print(f"\nFirst record keys: {list(data['records'][0].keys())}")
            print(f"First record: {json.dumps(data['records'][0], indent=2)[:500]}")
        else:
            print("\nNo records in ledger yet")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
