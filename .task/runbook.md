# Runbook

## 1) Create env with uv

```powershell
uv python install 3.11
uv venv --python 3.11
```

## 2) Install dependencies (GPU)

```powershell
uv sync --extra torch-cuda128
```

Verify CUDA:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3) Prefetch models

```powershell
uv run python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('LongshenOu/m2m_arranger'); AutoModelForCausalLM.from_pretrained('LongshenOu/m2m_pianist_dur'); AutoModelForCausalLM.from_pretrained('LongshenOu/m2m_drummer'); AutoTokenizer.from_pretrained('LongshenOu/m2m_ft'); print('prefetch done')"
```

## 4) Start MCP server

```powershell
uv run python app.py
```

Default server port is `7872`. Override with:

```powershell
$env:M2M_MCP_PORT=7873
uv run python app.py
```

## 5) Quick smoke call

```powershell
uv run python -c "from app import arrange_band_midi; print(arrange_band_midi('tests/smoke_input.mid','outputs','string_trio','',1,'cpu',0))"
```
