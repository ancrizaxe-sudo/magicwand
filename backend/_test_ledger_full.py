import urllib.request
import json

# Run a synthetic analysis first
payload = {
    'bbox': [-54.9, -9.3, -54.7, -9.1],
    'date_t1_start': '2019-06-01',
    'date_t1_end': '2019-08-31',
    'date_t2_start': '2023-06-01',
    'date_t2_end': '2023-08-31',
    'size': 256,
    'allow_blockchain': True,
    'use_synthetic': True,
}

print("Running synthetic analysis to populate ledger...")
try:
    req = urllib.request.Request(
        'http://127.0.0.1:8004/api/analyze',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        result = json.loads(r.read().decode())
        print(f"Analysis completed: id={result.get('id')}")
        if result.get('blockchain'):
            print(f"Blockchain proof: tx_hash={result['blockchain']['tx_hash']}")
except Exception as e:
    print(f"Analysis error: {type(e).__name__}: {e}")

# Now check the ledger
print("\nChecking ledger...")
try:
    with urllib.request.urlopen('http://127.0.0.1:8004/api/blockchain', timeout=10) as r:
        data = json.loads(r.read().decode())
        print(f"Records in ledger: {len(data.get('records', []))}")
        if data.get('records'):
            rec = data['records'][0]
            print(f"Latest record: tx={rec['proof']['tx_hash'][:16]}...")
            print(f"  Carbon: {rec['record'].get('carbon_estimate_tco2')}")
            print(f"  AVLE+: {rec['record'].get('avle_plus_per_km2')}")
except Exception as e:
    print(f"Ledger error: {type(e).__name__}: {e}")
