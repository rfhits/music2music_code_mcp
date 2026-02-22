from __future__ import annotations

import os
import sys
from typing import Optional

import gradio as gr

from mcp_infer import arrange_band, arrange_drum, arrange_piano, to_json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def arrange_band_midi(
    midi_path: str,
    output_dir: str,
    use_preset: str = "",
    instrument_and_voice: str = "",
    n_samples: int = 1,
    device: str = "auto",
    seed: Optional[int] = None,
    model_id: str = "",
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    max_length: Optional[int] = None,
) -> str:
    """Arrange input MIDI into a target band instrumentation.

    Returns a JSON array:
    [{"task": "...", "model": "...", "input_midi": "...", "output_midi": "..."}]
    """
    result = arrange_band(
        midi_path=midi_path,
        output_dir=output_dir,
        use_preset=use_preset or None,
        instrument_and_voice=instrument_and_voice,
        n_samples=int(n_samples),
        device=device,
        seed=seed,
        model_id=model_id or None,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_length=max_length,
    )
    return to_json(result)


def arrange_piano_midi(
    midi_path: str,
    output_dir: str,
    n_samples: int = 1,
    device: str = "auto",
    seed: Optional[int] = None,
    model_id: str = "",
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    max_length: Optional[int] = None,
) -> str:
    """Create a piano arrangement from input MIDI.

    Returns a JSON array:
    [{"task": "...", "model": "...", "input_midi": "...", "output_midi": "..."}]
    """
    result = arrange_piano(
        midi_path=midi_path,
        output_dir=output_dir,
        n_samples=int(n_samples),
        device=device,
        seed=seed,
        model_id=model_id or None,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_length=max_length,
    )
    return to_json(result)


def arrange_drum_midi(
    midi_path: str,
    output_dir: str,
    merge_with_input: bool = True,
    n_samples: int = 1,
    device: str = "auto",
    seed: Optional[int] = None,
    model_id: str = "",
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    max_length: Optional[int] = None,
) -> str:
    """Create a drum arrangement from input MIDI.

    Returns a JSON array:
    [{"task": "...", "model": "...", "input_midi": "...", "output_midi": "..."}]
    """
    result = arrange_drum(
        midi_path=midi_path,
        output_dir=output_dir,
        merge_with_input=merge_with_input,
        n_samples=int(n_samples),
        device=device,
        seed=seed,
        model_id=model_id or None,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_length=max_length,
    )
    return to_json(result)


def _build_app() -> gr.TabbedInterface:
    band_iface = gr.Interface(
        fn=arrange_band_midi,
        inputs=[
            gr.Textbox(label="Input MIDI Path", placeholder="C:/path/to/input.mid"),
            gr.Textbox(label="Output Directory", value="outputs"),
            gr.Dropdown(
                label="Preset",
                choices=["", "string_trio", "rock_band", "jazz_band"],
                value="string_trio",
            ),
            gr.Textbox(
                label="Custom Instruments (CSV, optional)",
                placeholder="80,8,33",
                value="",
            ),
            gr.Number(label="Samples", value=1, precision=0),
            gr.Dropdown(label="Device", choices=["auto", "cpu", "cuda"], value="cuda"),
            gr.Number(label="Seed (optional)", value=None, precision=0),
            gr.Textbox(label="Model ID (optional)", value=""),
            gr.Number(label="top_k", value=10, precision=0),
            gr.Number(label="top_p", value=1.0),
            gr.Number(label="temperature", value=1.0),
            gr.Number(label="no_repeat_ngram_size", value=10, precision=0),
            gr.Number(label="max_length (default 800)", value=800, precision=0),
        ],
        outputs=gr.Textbox(label="Output JSON"),
        title="Band Arrangement",
    )

    piano_iface = gr.Interface(
        fn=arrange_piano_midi,
        inputs=[
            gr.Textbox(label="Input MIDI Path", placeholder="C:/path/to/input.mid"),
            gr.Textbox(label="Output Directory", value="outputs"),
            gr.Number(label="Samples", value=1, precision=0),
            gr.Dropdown(label="Device", choices=["auto", "cpu", "cuda"], value="cuda"),
            gr.Number(label="Seed (optional)", value=None, precision=0),
            gr.Textbox(label="Model ID (optional)", value=""),
            gr.Number(label="top_k", value=30, precision=0),
            gr.Number(label="top_p", value=1.0),
            gr.Number(label="temperature", value=1.0),
            gr.Number(label="no_repeat_ngram_size", value=10, precision=0),
            gr.Number(label="max_length (default 800)", value=800, precision=0),
        ],
        outputs=gr.Textbox(label="Output JSON"),
        title="Piano Arrangement",
    )

    drum_iface = gr.Interface(
        fn=arrange_drum_midi,
        inputs=[
            gr.Textbox(label="Input MIDI Path", placeholder="C:/path/to/input.mid"),
            gr.Textbox(label="Output Directory", value="outputs"),
            gr.Checkbox(label="Merge with input tracks", value=True),
            gr.Number(label="Samples", value=1, precision=0),
            gr.Dropdown(label="Device", choices=["auto", "cpu", "cuda"], value="cuda"),
            gr.Number(label="Seed (optional)", value=None, precision=0),
            gr.Textbox(label="Model ID (optional)", value=""),
            gr.Number(label="top_k", value=20, precision=0),
            gr.Number(label="top_p", value=1.0),
            gr.Number(label="temperature", value=1.0),
            gr.Number(label="no_repeat_ngram_size", value=0, precision=0),
            gr.Number(label="max_length (default 800)", value=800, precision=0),
        ],
        outputs=gr.Textbox(label="Output JSON"),
        title="Drum Arrangement",
    )

    return gr.TabbedInterface(
        [band_iface, piano_iface, drum_iface],
        ["Band", "Piano", "Drum"],
        title="Music2Music Arrangement MCP",
    )


if __name__ == "__main__":
    port = int(os.environ.get("M2M_MCP_PORT", "7872"))
    _build_app().launch(mcp_server=True, server_port=port)
