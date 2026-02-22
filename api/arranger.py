"""
API for inference.
"""

from typing import List, Optional

import torch
from remi_z import MultiTrack
from remi_z.core import Bar
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel


def _pick_device(device: str) -> str:
    """Resolve user device preference to an available runtime device."""
    device = (device or "auto").strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _safe_generate_kwargs(model_input, generate_kwargs):
    """
    Ensure generation limits are valid for current input length.
    """
    kwargs = dict(generate_kwargs)
    input_len = int(model_input.shape[-1])
    max_length = kwargs.get("max_length")
    if max_length is not None:
        try:
            max_length = int(max_length)
        except (TypeError, ValueError):
            max_length = 0
        if max_length <= input_len:
            kwargs["max_length"] = input_len + 128
    return kwargs


class BandArranger:
    '''
    Class for arranging music for a band
    '''
    def __init__(self, model_fp, hf_ckpt=True, device="auto"):
        '''
        if hf_ckpt is True, load model from HuggingFace checkpoint
        else, load model from local lightning checkpoint
        '''
        self.device = _pick_device(device)
        if hf_ckpt:
            model = self.from_hf_ckpt(model_fp)
        else:
            raise NotImplementedError("Local checkpoint loading not implemented yet")
        self.model = model.to(self.device)

        # Prepare tokenizer
        tk_fp = 'LongshenOu/m2m_ft'
        self.tk = AutoTokenizer.from_pretrained(tk_fp)

        # Prepare generation kwarg
        self.generate_kwargs = {
            "max_length": 800,
            "use_cache": True,
            "bad_words_ids": [
                [self.tk.pad_token_id],
                [self.tk.convert_tokens_to_ids("[PAD]")],
                [self.tk.convert_tokens_to_ids("[INST]")],
                [self.tk.convert_tokens_to_ids("[SEP]")],
            ],
            "no_repeat_ngram_size": 10,

            # Sampling
            "do_sample": True,  # User searching method
            "top_k": 10,
            "top_p": 1.0,
            "temperature": 1.0,
        }

        # Ensemble preset
        self.preset_instruments = {
            'string_trio': [40, 41, 42],  # violin1, violin2, cello
            'rock_band': [80, 26, 29, 33],  # synth lead, clean e-guitar, overdrive e-guitar, electric bass
            'jazz_band': [64, 40, 61, 26, 0, 44, 33]
        }

    def from_hf_ckpt(self, model_fp) -> GPT2LMHeadModel:
        model = GPT2LMHeadModel.from_pretrained(model_fp)
        model.eval()
        return model

    def arrange(
        self,
        input_midi_fp,
        use_preset: Optional[str],
        instrument_and_voice: Optional[List[int]] = None,
    ) -> MultiTrack:
        '''
        Arrange input music for the specified band instruments
        '''
        # Check preset
        instrument_and_voice = list(instrument_and_voice or [])
        if use_preset is not None:
            assert use_preset in self.preset_instruments, f"Preset {use_preset} not found"
            instrument_and_voice = self.preset_instruments[use_preset]
        assert len(instrument_and_voice) > 0, "No instruments specified for arrangement"

        mt = MultiTrack.from_midi(input_midi_fp)

        arranged = []
        prev_bar = None
        for bar in tqdm(mt):
            arranged_bar = self.arrange_a_bar(bar, instrument_and_voice, prev_bar)
            arranged.append(arranged_bar)
            prev_bar = arranged_bar
        ret = MultiTrack.from_bars(arranged)

        return ret
    
    def arrange_a_bar(self, bar, instrument_and_voice, prev_bar) -> Bar:
        model_input = self.assemble_model_input(bar, prev_bar, instrument_and_voice)
        generate_kwargs = _safe_generate_kwargs(model_input, self.generate_kwargs)
        out = self.model.generate(
            model_input, pad_token_id=self.tk.pad_token_id, **generate_kwargs
        )
        out_str = self.tk.batch_decode(out)[0]  # because we do bs=1 here

        # Select substr between [SEP] and [EOS] as output
        out_str = out_str.split("[SEP]")[1].split("[EOS]")[0].strip()
        
        bar = Bar.from_remiz_str(out_str)

        return bar
    
    def assemble_model_input(self, bar:Bar, prev_bar:Bar, instrument_and_voice:List[int]):
        '''
        input: ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] 
        model output:  original + tgt_seq + ['[EOS]']
        '''
        content_seq = bar.get_content_seq(with_dur=False)
        hist_seq = prev_bar.to_remiz_seq() if prev_bar is not None else []
        inst_seq = [f'i-{inst}' for inst in instrument_and_voice]
        input_seq = ['[BOS]'] + ['[INST]'] + inst_seq + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]']
        input_str = ' '.join(input_seq)

        batch_tokenized = self.tk(
            [input_str],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,  # Don't add eos token
        )["input_ids"].to(self.device)
        return batch_tokenized
    
class PianoArranger:
    '''
    Class for arranging music for piano
    '''
    def __init__(self, model_fp, hf_ckpt=True, device="auto"):
        '''
        if hf_ckpt is True, load model from HuggingFace checkpoint
        else, load model from local lightning checkpoint
        '''
        self.device = _pick_device(device)
        if hf_ckpt:
            model = self.from_hf_ckpt(model_fp)
        else:
            raise NotImplementedError("Local checkpoint loading not implemented yet")
        self.model = model.to(self.device)

        # Prepare tokenizer
        tk_fp = 'LongshenOu/m2m_ft'
        self.tk = AutoTokenizer.from_pretrained(tk_fp)

        # Prepare generation kwarg
        self.generate_kwargs = {
            "max_length": 800,
            "use_cache": True,
            "bad_words_ids": [
                [self.tk.pad_token_id],
                [self.tk.convert_tokens_to_ids("[PAD]")],
                [self.tk.convert_tokens_to_ids("[INST]")],
                [self.tk.convert_tokens_to_ids("[SEP]")],
            ],
            "no_repeat_ngram_size": 10,

            # Sampling
            "do_sample": True,  # User searching method
            "top_k": 30,
            "top_p": 1.0,
            "temperature": 1.0,
        }

        # Ensemble preset
        self.preset_instruments = {
            'piano': [0]
        }

        self.duration_in_input = True

    def from_hf_ckpt(self, model_fp) -> GPT2LMHeadModel:
        model = GPT2LMHeadModel.from_pretrained(model_fp)
        model.eval()
        return model

    def arrange(
        self,
        input_midi_fp,
        use_preset: Optional[str],
        instrument_and_voice: Optional[List[int]] = None,
    ) -> MultiTrack:
        '''
        Arrange input music for the specified band instruments
        '''
        # Check preset
        instrument_and_voice = list(instrument_and_voice or [])
        if use_preset is not None:
            assert use_preset in self.preset_instruments, f"Preset {use_preset} not found"
            instrument_and_voice = self.preset_instruments[use_preset]
        assert len(instrument_and_voice) > 0, "No instruments specified for arrangement"

        mt = MultiTrack.from_midi(input_midi_fp)

        arranged = []
        prev_bar = None
        for bar in tqdm(mt):
            arranged_bar = self.arrange_a_bar(bar, instrument_and_voice, prev_bar)
            arranged.append(arranged_bar)
            prev_bar = arranged_bar
        ret = MultiTrack.from_bars(arranged)

        return ret
    
    def arrange_a_bar(self, bar, instrument_and_voice, prev_bar) -> Bar:
        model_input = self.assemble_model_input(bar, prev_bar, instrument_and_voice)
        generate_kwargs = _safe_generate_kwargs(model_input, self.generate_kwargs)
        out = self.model.generate(
            model_input, pad_token_id=self.tk.pad_token_id, **generate_kwargs
        )
        out_str = self.tk.batch_decode(out)[0]  # because we do bs=1 here

        # Select substr between [SEP] and [EOS] as output
        out_str = out_str.split("[SEP]")[1].split("[EOS]")[0].strip()
        
        bar = Bar.from_remiz_str(out_str)

        return bar
    
    def assemble_model_input(self, bar:Bar, prev_bar:Bar, instrument_and_voice:List[int]):
        '''
        input: ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] 
        model output:  original + tgt_seq + ['[EOS]']
        '''
        content_seq = bar.get_content_seq(with_dur=self.duration_in_input)
        hist_seq = prev_bar.to_remiz_seq() if prev_bar is not None else []
        inst_seq = [f'i-{inst}' for inst in instrument_and_voice]
        input_seq = ['[BOS]'] + ['[INST]'] + inst_seq + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]']
        input_str = ' '.join(input_seq)

        batch_tokenized = self.tk(
            [input_str],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,  # Don't add eos token
        )["input_ids"].to(self.device)
        return batch_tokenized
    
class DrumArranger:
    '''
    Class for arranging music for drums
    '''
    def __init__(self, model_fp, hf_ckpt=True, device="auto"):
        '''
        if hf_ckpt is True, load model from HuggingFace checkpoint
        else, load model from local lightning checkpoint
        '''
        self.device = _pick_device(device)
        if hf_ckpt:
            model = self.from_hf_ckpt(model_fp)
        else:
            raise NotImplementedError("Local checkpoint loading not implemented yet")
        self.model = model.to(self.device)

        # Prepare tokenizer
        tk_fp = 'LongshenOu/m2m_ft'
        self.tk = AutoTokenizer.from_pretrained(tk_fp)

        # Prepare generation kwarg
        self.generate_kwargs = {
            "max_length": 800,
            "use_cache": True,
            "bad_words_ids": [
                [self.tk.pad_token_id],
                [self.tk.convert_tokens_to_ids("[PAD]")],
                [self.tk.convert_tokens_to_ids("[INST]")],
                [self.tk.convert_tokens_to_ids("[SEP]")],
            ],
            "no_repeat_ngram_size": 0,

            # Sampling
            "do_sample": True,  # User searching method
            "top_k": 20,
            "top_p": 1.0,
            "temperature": 1.0,
        }

        # Ensemble preset
        self.preset_instruments = {
            'drum': [128],  # 
        }

    # def arrange(self, input_music):
    #     '''
    #     Arrange input music for drums
    #     Args:
    #         input_music: The input music data
    #     Returns:
    #         arranged_music: The arranged music data
    #     '''
    #     # Placeholder for arrangement logic
    #     arranged_music = self.model.process(input_music)
    #     return arranged_music
    
    def from_hf_ckpt(self, model_fp) -> GPT2LMHeadModel:
        model = GPT2LMHeadModel.from_pretrained(model_fp)
        model.eval()
        return model

    def arrange(
        self,
        input_midi_fp,
        use_preset: Optional[str],
        instrument_and_voice: Optional[List[int]] = None,
        merge_with_input=False,
    ) -> MultiTrack:
        '''
        Arrange a drum track that is compatible with input music

        Use a logic that is same as original drum arranger
        - Original: Arrange for 4-bar segments
        # - Here: Arrange for each bar with  TODO
        '''
        # Check preset
        instrument_and_voice = list(instrument_and_voice or [])
        if use_preset is not None:
            assert use_preset in self.preset_instruments, f"Preset {use_preset} not found"
            instrument_and_voice = self.preset_instruments[use_preset]
        assert len(instrument_and_voice) > 0, "No instruments specified for arrangement"

        mt = MultiTrack.from_midi(input_midi_fp)

        arranged = []
        prev_seg = None
        for i in tqdm(range(0, len(mt), 4)):
            seg_mt = MultiTrack.from_bars(mt.bars[i:i+4])
            arranged_seg = self.arrange_a_segment(seg_mt, prev_seg)

            # Ensure the output is 4-bar
            if len(arranged_seg) > 4:
                arranged_seg = arranged_seg[:4]
            elif len(arranged_seg) < 4:
                last_bar = arranged_seg.bars[-1]
                pad_bar = 4-len(arranged_seg)
                for i in range(pad_bar):
                    arranged_seg.bars.append(last_bar)

            arranged.extend(arranged_seg.bars)
            prev_seg = arranged_seg
        ret = MultiTrack.from_bars(arranged)

        if merge_with_input:
            mt.remove_tracks([128])
            if len(mt) != len(ret):
                min_bars = min(len(mt), len(ret))
                mt = MultiTrack.from_bars(mt.bars[:min_bars])
                ret = MultiTrack.from_bars(ret.bars[:min_bars])
            ret = mt.merge_with(ret, 128)

        return ret
    
    def arrange_a_segment(self, segment:MultiTrack, prev_segment:MultiTrack) -> MultiTrack:
        model_input = self.assemble_model_input(segment, prev_segment, instrument_and_voice=[128])
        generate_kwargs = _safe_generate_kwargs(model_input, self.generate_kwargs)
        out = self.model.generate(
            model_input, pad_token_id=self.tk.pad_token_id, **generate_kwargs
        )
        out_str = self.tk.batch_decode(out)[0]  # because we do bs=1 here

        # Select substr between [SEP] and [EOS] as output
        out_str = out_str.split("[SEP]")[1].split("[EOS]")[0].strip()
        
        arranged_seg = MultiTrack.from_remiz_str(out_str)
        return arranged_seg
    
    def assemble_model_input(self, segment:Bar, prev_segment:Bar, instrument_and_voice:List[int]):
        '''
        input: ['[BOS]'] + ['[INST]'] + insts + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]'] 
        model output:  original + tgt_seq + ['[EOS]']
        '''
        content_seq = segment.get_content_seq(with_dur=False)
        hist_seq = prev_segment.to_remiz_seq() if prev_segment is not None else []
        inst_seq = [f'i-{inst}' for inst in instrument_and_voice]
        input_seq = ['[BOS]'] + ['[INST]'] + inst_seq + ['[PITCH]'] + content_seq + ['[HIST]'] + hist_seq + ['[SEP]']
        input_str = ' '.join(input_seq)

        batch_tokenized = self.tk(
            [input_str],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,  # Don't add eos token
        )["input_ids"].to(self.device)
        return batch_tokenized
    
