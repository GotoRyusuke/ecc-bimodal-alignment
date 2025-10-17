import os
import time
from functools import wraps
from audioUtils import gen_audio_segment
from textUtils import preprocess


def gen_speech_timestamp(timestamp, audio, content, save_folder=None):
    speech = content.copy()
    word_times = [(word.start, word.end) for word in timestamp]
    # slices = []

    start_idx = 0
    start_end_times = []
    for i, row in speech.iterrows():
        words = preprocess(row["text"], sep=' ').split()
        end_idx = start_idx + len(words)

        if end_idx > len(word_times):
            raise ValueError(f"Text in row {i} exceeds total words length.")

        start_time = word_times[start_idx][0]
        end_time = word_times[end_idx - 1][1]

        start_end_times.append((start_time, end_time))
        start_idx = end_idx  # Move pointer

        if save_folder:
            session = row['session']
            node = row['node']
            folder = f'{save_folder}/{session}'
            os.makedirs(folder, exist_ok=True)
            dir_save = f'{folder}/{node}.wav'
            gen_audio_segment(audio, start_time, end_time, dir_save)
        # slices.append(audio_slice)
    speech["start_sec"], speech["end_sec"] = zip(*start_end_times)
    # content['slice'] = slices
    return speech

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

