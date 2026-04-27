import urllib.request
import urllib.error
import json

req = urllib.request.Request(
    'http://127.0.0.1:8004/api/analyze',
    data=json.dumps({
        'bbox': [-54.9, -9.3, -54.7, -9.1],
        'date_t1_start': '2019-06-01',
        'date_t1_end': '2019-08-31',
        'date_t2_start': '2023-06-01',
        'date_t2_end': '2023-08-31',
        'size': 384,
        'allow_blockchain': False,
        'use_synthetic': True,
    }).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
)

try:
    with urllib.request.urlopen(req, timeout=300) as r:
        print('status', r.status)
        print(r.read().decode()[:2000])
except urllib.error.HTTPError as e:
    print('HTTP', e.code)
    print(e.read().decode())
except Exception as e:
    print('ERROR', type(e).__name__, e)
