# music2music MCP implementation plan

## Goal
Build a lightweight MCP server in this repo that mirrors the simple structure used in `anticipation_mcp/app.py`.

## Scope
- Add MCP entrypoint (`app.py`) with three tools:
  - `arrange_band_midi`
  - `arrange_piano_midi`
  - `arrange_drum_midi`
- Add an inference adapter (`mcp_infer.py`) to wrap `api/arranger.py`.
- Add `pyproject.toml` so dependency management is done with `uv`.
- Keep output protocol simple: JSON text with generated MIDI file paths.

## Milestones
1. Implement MCP tool wrappers and JSON return format.
2. Harden arranger behavior:
   - device auto fallback
   - no mutable default list arguments
   - update history state between bars/segments
3. Configure `uv` project and extras for CPU/CUDA torch.
4. Validate with smoke runs:
   - import checks
   - model prefetch
   - one minimal tool call
   - MCP startup check
