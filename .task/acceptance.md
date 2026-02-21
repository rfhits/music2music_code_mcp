# Acceptance checklist

- [ ] `uv sync --extra torch-cuda128` succeeds.
- [ ] Model prefetch command completes without auth errors.
- [ ] `arrange_band_midi` returns valid JSON with at least one output path.
- [ ] `arrange_piano_midi` returns valid JSON with at least one output path.
- [ ] `arrange_drum_midi` returns valid JSON with at least one output path.
- [ ] Output MIDI files are written to requested output directory.
- [ ] `uv run python app.py` starts Gradio MCP server successfully.

## Notes

- Keep generated outputs under `outputs/`.
- If CUDA is unavailable, use `device='cpu'` or `device='auto'`.
- If model downloads are slow, retry once to warm local HF cache.
