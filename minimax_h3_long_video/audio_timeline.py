import torch

from .timeline import FPS


def slice_timeline_audio(audio, segment):
    """Return the source-audio window aligned with one raw H3 AV segment.

    Continuation segments generate a removable AV guide before their delivered
    frames. The reference window therefore starts at output_start-context_frames
    and spans the complete raw H3 frame count, including grid padding.
    """
    if not isinstance(audio, dict):
        raise ValueError("timeline reference audio must be a ComfyUI AUDIO value")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        raise ValueError(
            "timeline reference audio waveform must have shape [batch, channels, samples]")
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        raise ValueError("timeline reference audio has an invalid sample_rate")
    if sample_rate <= 0:
        raise ValueError("timeline reference audio sample_rate must be positive")

    start_frame = segment.output_start - segment.context_frames
    start_sample = round(start_frame / FPS * sample_rate)
    target_samples = round(segment.raw_frames / FPS * sample_rate)
    source_start = max(0, start_sample)
    source_end = min(waveform.shape[-1], start_sample + target_samples)
    if source_end > source_start:
        sliced = waveform[..., source_start:source_end]
    else:
        sliced = waveform[..., :0]
    left_padding = min(target_samples, max(0, -start_sample))
    right_padding = target_samples - left_padding - sliced.shape[-1]
    if right_padding < 0:
        sliced = sliced[..., :target_samples - left_padding]
        right_padding = 0
    if left_padding or right_padding:
        sliced = torch.nn.functional.pad(sliced, (left_padding, right_padding))

    result = dict(audio)
    result["waveform"] = sliced.contiguous()
    result["sample_rate"] = sample_rate
    return result


def slice_timeline_ref_audios(ref_audios, segment):
    return {
        name: None if audio is None else slice_timeline_audio(audio, segment)
        for name, audio in (ref_audios or {}).items()
    }
