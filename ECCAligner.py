import torch
import torchaudio
import pandas as pd
from dataClass import Point, Segment
from utils import timer
from textUtils import preprocess
from audioUtils import load_audio


# Global model and resampler to avoid repeated loading
_model = None
_resampler = None
_labels = None
_device = None
_bundle = None
_target_sample_rate = 16000


def _initialize_model(device):
    """Initialize the global model and resampler on the specified device."""
    global _model, _resampler, _labels, _device, _bundle

    _device = device
    _bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    _model = _bundle.get_model().to(_device)
    _model.eval()  # Set to evaluation mode
    _labels = _bundle.get_labels()
    _resampler = None  # Will be created if needed


def cleanup_model():
    """Clean up global model and resampler to free GPU memory."""
    global _model, _resampler, _labels, _device, _bundle

    _model = None
    _resampler = None
    _labels = None
    _device = None
    _bundle = None

    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def _get_emissions_single(waveform, use_amp):
    """
    Process a single waveform and return emissions.
    """
    global _model, _device

    with torch.inference_mode():
        if use_amp and _device.type == 'cuda':
            with torch.cuda.amp.autocast():
                emissions, _ = _model(waveform)
        else:
            emissions, _ = _model(waveform)

        emissions = torch.log_softmax(emissions, dim=-1)

    # Move to CPU, convert to float32, and detach
    emission = emissions[0].float().cpu().detach()

    # Clean up GPU memory
    del emissions
    torch.cuda.empty_cache() if _device.type == 'cuda' else None

    return emission


def _get_emissions_chunked(waveform, samples_per_chunk, overlap_samples, use_amp):
    """
    Process long audio in chunks with overlap to reduce GPU memory usage.

    Returns combined emissions from all chunks.
    """
    global _model, _device

    num_samples = waveform.shape[1]
    all_emissions = []

    # Calculate chunk boundaries
    start = 0
    while start < num_samples:
        end = min(start + samples_per_chunk, num_samples)
        chunk = waveform[:, start:end]

        # Apply mixed precision if enabled
        with torch.inference_mode():
            if use_amp and _device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    chunk_emissions, _ = _model(chunk)
            else:
                chunk_emissions, _ = _model(chunk)

            chunk_emissions = torch.log_softmax(chunk_emissions, dim=-1)

        # Move to CPU, convert to float32, and detach
        chunk_emissions = chunk_emissions[0].float().cpu().detach()
        all_emissions.append(chunk_emissions)

        # Clean up
        del chunk, chunk_emissions
        torch.cuda.empty_cache() if _device.type == 'cuda' else None

        # Move to next chunk (with overlap)
        start = end - overlap_samples if end < num_samples else num_samples

    # Combine emissions
    # Handle overlap by averaging overlapping regions
    combined = all_emissions[0]
    num_chunks = len(all_emissions)

    for i in range(1, num_chunks):
        prev_len = combined.shape[0]
        curr = all_emissions[i]
        curr_len = curr.shape[0]

        # Calculate overlap in frames (assuming ~320x downsampling)
        downsampling_factor = 320
        overlap_frames = overlap_samples // downsampling_factor

        if overlap_frames > 0 and i < num_chunks - 1:
            # Average overlapping region
            overlap_region_prev = combined[-overlap_frames:]
            overlap_region_curr = curr[:overlap_frames]
            averaged = (overlap_region_prev + overlap_region_curr) / 2

            # Remove non-overlapping part from previous chunk
            combined = combined[:-overlap_frames]

            # Concatenate with averaged overlap and non-overlapping part of current
            combined = torch.cat([combined, averaged, curr[overlap_frames:]], dim=0)
        else:
            # Last chunk - just concatenate
            combined = torch.cat([combined, curr], dim=0)

    # Clean up list after processing
    all_emissions.clear()
    torch.cuda.empty_cache() if _device.type == 'cuda' else None

    return combined


@timer
def gen_segments(dir_mp3, dir_txt, chunk_duration_seconds=30, overlap_seconds=2, use_amp=True):
    """
    Generate word-level segments by aligning transcript with audio.

    Args:
        dir_mp3: Path to audio file
        dir_txt: Path to transcript file
        chunk_duration_seconds: Duration of each processing chunk in seconds
        overlap_seconds: Overlap between chunks in seconds
        use_amp: Whether to use automatic mixed precision

    Returns:
        List of word segments
    """
    global _model, _resampler, _labels, _device, _bundle

    # Initialize if not already done
    if _model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _initialize_model(device)

    samples_per_chunk = int(chunk_duration_seconds * _target_sample_rate)
    overlap_samples = int(overlap_seconds * _target_sample_rate)

    # Load and preprocess audio
    waveform, sample_rate = load_audio(dir_mp3)
    waveform = waveform.to(_device)

    # Create resampler if needed
    global _resampler
    if sample_rate != _target_sample_rate:
        _resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=_target_sample_rate
        ).to(_device)
        waveform = _resampler(waveform)

    # Load transcript (handle both text and parquet files)
    if dir_txt.endswith('.parquet'):
        df = pd.read_parquet(dir_txt)
        # Concatenate all text from the 'text' column
        tmp = ' '.join(df['text'].astype(str).tolist())
    else:
        with open(dir_txt, 'r') as f:
            tmp = f.read().strip()

    transcript = preprocess(tmp)
    dictionary = {c: i for i, c in enumerate(_labels)}
    tokens = [dictionary[c] for c in transcript]
    num_tokens = len(tokens)

    num_samples = waveform.shape[1]

    # Calculate emissions using chunked processing
    if num_samples > samples_per_chunk:
        emission = _get_emissions_chunked(
            waveform,
            samples_per_chunk,
            overlap_samples,
            use_amp
        )
    else:
        emission = _get_emissions_single(waveform, use_amp)

    num_frame = emission.size(0)

    # --- Get trellis ---
    blank_id = 0
    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )

    #  --- Get path ---
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    path = path[::-1]

    # --- Get segments ---
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2

    # --- Get words segments ---
    separator = '|'
    word_segments = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                word_segments.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1

    return word_segments
    
    
    