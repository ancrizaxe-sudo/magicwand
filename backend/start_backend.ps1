Set-Location $PSScriptRoot
$env:MONGO_URL = 'mongodb://localhost:27017'
$env:DB_NAME = 'avle'
py -3 -m uvicorn server:app --host 127.0.0.1 --port 8004
