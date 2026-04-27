import urllib.request
import json

# Check the full ledger response
with urllib.request.urlopen('http://127.0.0.1:8004/api/blockchain?limit=50', timeout=10) as r:
    data = json.loads(r.read().decode())
    
print("=== BLOCKCHAIN RESPONSE ===")
print(f"\nStatus section:")
status = data.get('status', {})
for key, val in status.items():
    print(f"  {key}: {val}")

print(f"\nRecords count: {len(data.get('records', []))}")

if data.get('records'):
    print(f"\n=== FIRST RECORD (should match Ledger.jsx expectations) ===")
    rec = data['records'][0]
    
    print(f"\nRecords structure check:")
    print(f"  Has 'proof'? {('proof' in rec)}")
    print(f"  Has 'record'? {('record' in rec)}")
    
    if 'proof' in rec:
        print(f"\n  proof keys: {list(rec['proof'].keys())}")
        print(f"    block_number: {rec['proof'].get('block_number')} (needed for Ledger table)")
        print(f"    tx_hash: {rec['proof'].get('tx_hash')[:20]}... (needed for Ledger table)")
    
    if 'record' in rec:
        print(f"\n  record keys: {list(rec['record'].keys())}")
        print(f"    region_hash: {rec['record'].get('region_hash')} (needed for Ledger table)")
        print(f"    carbon_estimate_tco2: {rec['record'].get('carbon_estimate_tco2')} (needed for Ledger table)")
        print(f"    avle_plus_per_km2: {rec['record'].get('avle_plus_per_km2')} (needed for Ledger table)")
    
    print(f"\n=== FULL RECORD ===")
    print(json.dumps(rec, indent=2, default=str)[:1000])
