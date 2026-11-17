$env:PROMPTFOO_PYTHON = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
npx promptfoo@latest eval @args
