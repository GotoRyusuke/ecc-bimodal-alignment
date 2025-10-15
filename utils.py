import json
import time
import torchaudio
from functools import wraps
from DataClass import Segment, WordStamp


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Execute the original function
        end_time = time.time()
        runtime = end_time - start_time

        # Presentation of the runtime
        print(f"⏱️ Function '{func.__name__}' ran in: {runtime:.4f} seconds")

        return result
    return wrapper

def remove_punc(word):
    return ''.join(char for char in word if char.isalpha())

def preprocess(text):
    tokens = text.upper().split()
    words = [remove_punc(token) for token in tokens]

    return '|'.join([word for word in words if len(word) > 0])

def load_audio(dir_audio):
    return torchaudio.load(dir_audio)

def gen_audio_segment(torch_wave, start:float, end:float, dir_output:str):
    waveform, sample_rate = torch_wave
    segment = waveform[:, start:end]
    segment_cpu = segment.cpu()

    torchaudio.save(dir_output, segment_cpu, sample_rate)

def gen_timestamp(dir_json:str, sample_rate:float=50):
    with open(dir_json, 'r') as f:
        loaded_data = json.load(f)
    seg = [Segment(**data) for data in loaded_data]

    dict_timestamp = list()
    for idx_segment, segment in enumerate(seg):
        sec_start, sec_end = segment.start / sample_rate, segment.end / sample_rate
        word_timestamp = WordStamp(
            segment.label,
            sec_start,
            sec_end
        )
        dict_timestamp.append(word_timestamp)

    return dict_timestamp
