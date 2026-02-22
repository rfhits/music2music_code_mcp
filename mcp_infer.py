from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

DEFAULT_MODEL_IDS = {
    "band": "LongshenOu/m2m_arranger",
    "piano": "LongshenOu/m2m_pianist_dur",
    "drum": "LongshenOu/m2m_drummer",
}

_ARRANGER_CACHE: dict[tuple[str, str, str], Any] = {}


def _ensure_remi_z_assets() -> None:
    """Patch missing YAML asset in remi_z wheel if needed."""
    import remi_z

    pkg_dir = Path(remi_z.__file__).resolve().parent
    ts_target = pkg_dir / "dict_time_signature.yaml"
    if ts_target.exists():
        return

    ts_source = Path(__file__).resolve().parent / "utils_midi" / "ts_dict.yaml"
    if not ts_source.exists():
        raise FileNotFoundError(
            "Missing remi_z dict_time_signature.yaml and fallback ts_dict.yaml"
        )
    shutil.copyfile(ts_source, ts_target)


def _resolve_device(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _validate_midi_path(midi_path: str) -> Path:
    if not midi_path or not isinstance(midi_path, str):
        raise ValueError("midi_path must be a local MIDI file path")
    path = Path(midi_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"MIDI file does not exist: {path}")
    if path.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("midi_path must point to a .mid or .midi file")
    return path


def _parse_instruments(instrument_and_voice: str) -> list[int]:
    if not instrument_and_voice:
        return []
    out: list[int] = []
    for token in instrument_and_voice.split(","):
        part = token.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as exc:
            raise ValueError(
                "instrument_and_voice must be a comma-separated integer list, e.g. '80,8,33'"
            ) from exc
    return out


def _with_generation_overrides(
    arranger: Any,
    *,
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
    no_repeat_ngram_size: Optional[int],
    max_length: Optional[int],
) -> None:
    if not hasattr(arranger, "_default_generate_kwargs"):
        arranger._default_generate_kwargs = dict(arranger.generate_kwargs)
    kwargs = dict(arranger._default_generate_kwargs)

    # Be lenient for UI/MCP numeric inputs: ignore invalid values and keep defaults.
    if top_k is not None:
        try:
            top_k_value = int(top_k)
            if top_k_value > 0:
                kwargs["top_k"] = top_k_value
        except (TypeError, ValueError):
            pass
    if top_p is not None:
        try:
            top_p_value = float(top_p)
            if 0.0 < top_p_value <= 1.0:
                kwargs["top_p"] = top_p_value
        except (TypeError, ValueError):
            pass
    if temperature is not None:
        try:
            temperature_value = float(temperature)
            if temperature_value > 0:
                kwargs["temperature"] = temperature_value
        except (TypeError, ValueError):
            pass
    if no_repeat_ngram_size is not None:
        try:
            ngram_value = int(no_repeat_ngram_size)
            if ngram_value >= 0:
                kwargs["no_repeat_ngram_size"] = ngram_value
        except (TypeError, ValueError):
            pass
    if max_length is not None:
        try:
            max_length_value = int(max_length)
            if max_length_value > 0:
                kwargs["max_length"] = max_length_value
        except (TypeError, ValueError):
            pass
    arranger.generate_kwargs = kwargs


def _get_arranger(task: str, *, device: str, model_id: Optional[str] = None) -> Any:
    task = task.strip().lower()
    if task not in {"band", "piano", "drum"}:
        raise ValueError("task must be one of: band, piano, drum")

    _ensure_remi_z_assets()
    from api.arranger import BandArranger, DrumArranger, PianoArranger

    resolved_device = _resolve_device(device)
    selected_model = model_id.strip() if model_id else DEFAULT_MODEL_IDS[task]
    key = (task, selected_model, resolved_device)
    if key in _ARRANGER_CACHE:
        return _ARRANGER_CACHE[key]

    if task == "band":
        arranger = BandArranger(selected_model, hf_ckpt=True, device=resolved_device)
    elif task == "piano":
        arranger = PianoArranger(selected_model, hf_ckpt=True, device=resolved_device)
    else:
        arranger = DrumArranger(selected_model, hf_ckpt=True, device=resolved_device)
    _ARRANGER_CACHE[key] = arranger
    return arranger


def _prepare_output(output_dir: str, prefix: str, sample_idx: int) -> Path:
    if not output_dir or not isinstance(output_dir, str):
        raise ValueError("output_dir must be a writable directory path")
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{prefix}_{timestamp}_{sample_idx}.mid"


def _seed_if_requested(seed: Optional[int], sample_idx: int) -> None:
    if seed is None:
        return
    value = int(seed) + sample_idx
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def arrange_band(
    *,
    midi_path: str,
    output_dir: str,
    use_preset: Optional[str],
    instrument_and_voice: str,
    n_samples: int,
    device: str,
    seed: Optional[int],
    model_id: Optional[str],
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
    no_repeat_ngram_size: Optional[int],
    max_length: Optional[int],
) -> list[dict[str, Any]]:
    if int(n_samples) < 1:
        raise ValueError("n_samples must be >= 1")
    input_path = _validate_midi_path(midi_path)
    instruments = _parse_instruments(instrument_and_voice)
    arranger = _get_arranger("band", device=device, model_id=model_id)
    _with_generation_overrides(
        arranger,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_length=max_length,
    )

    results: list[dict[str, Any]] = []
    stem = input_path.stem
    for sample_idx in range(int(n_samples)):
        _seed_if_requested(seed, sample_idx)
        arranged = arranger.arrange(
            str(input_path),
            use_preset=use_preset or None,
            instrument_and_voice=instruments,
        )
        output_path = _prepare_output(output_dir, f"{stem}_band", sample_idx)
        arranged.to_midi(str(output_path))
        results.append(
            {
                "task": "band_arrangement",
                "model": model_id or DEFAULT_MODEL_IDS["band"],
                "sample_index": sample_idx,
                "input_midi": str(input_path),
                "output_midi": str(output_path),
                "use_preset": use_preset or None,
                "instrument_and_voice": instruments,
            }
        )
    return results


def arrange_piano(
    *,
    midi_path: str,
    output_dir: str,
    n_samples: int,
    device: str,
    seed: Optional[int],
    model_id: Optional[str],
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
    no_repeat_ngram_size: Optional[int],
    max_length: Optional[int],
) -> list[dict[str, Any]]:
    if int(n_samples) < 1:
        raise ValueError("n_samples must be >= 1")
    input_path = _validate_midi_path(midi_path)
    arranger = _get_arranger("piano", device=device, model_id=model_id)
    _with_generation_overrides(
        arranger,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_length=max_length,
    )

    results: list[dict[str, Any]] = []
    stem = input_path.stem
    for sample_idx in range(int(n_samples)):
        _seed_if_requested(seed, sample_idx)
        arranged = arranger.arrange(str(input_path), use_preset="piano", instrument_and_voice=[])
        output_path = _prepare_output(output_dir, f"{stem}_piano", sample_idx)
        arranged.to_midi(str(output_path))
        results.append(
            {
                "task": "piano_arrangement",
                "model": model_id or DEFAULT_MODEL_IDS["piano"],
                "sample_index": sample_idx,
                "input_midi": str(input_path),
                "output_midi": str(output_path),
                "use_preset": "piano",
            }
        )
    return results


def arrange_drum(
    *,
    midi_path: str,
    output_dir: str,
    merge_with_input: bool,
    n_samples: int,
    device: str,
    seed: Optional[int],
    model_id: Optional[str],
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
    no_repeat_ngram_size: Optional[int],
    max_length: Optional[int],
) -> list[dict[str, Any]]:
    if int(n_samples) < 1:
        raise ValueError("n_samples must be >= 1")
    input_path = _validate_midi_path(midi_path)
    arranger = _get_arranger("drum", device=device, model_id=model_id)
    _with_generation_overrides(
        arranger,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_length=max_length,
    )

    results: list[dict[str, Any]] = []
    stem = input_path.stem
    for sample_idx in range(int(n_samples)):
        _seed_if_requested(seed, sample_idx)
        arranged = arranger.arrange(
            str(input_path),
            use_preset="drum",
            instrument_and_voice=[],
            merge_with_input=bool(merge_with_input),
        )
        output_path = _prepare_output(output_dir, f"{stem}_drum", sample_idx)
        arranged.to_midi(str(output_path))
        results.append(
            {
                "task": "drum_arrangement",
                "model": model_id or DEFAULT_MODEL_IDS["drum"],
                "sample_index": sample_idx,
                "input_midi": str(input_path),
                "output_midi": str(output_path),
                "use_preset": "drum",
                "merge_with_input": bool(merge_with_input),
            }
        )
    return results


def to_json(result: list[dict[str, Any]]) -> str:
    return json.dumps(result, ensure_ascii=False)
