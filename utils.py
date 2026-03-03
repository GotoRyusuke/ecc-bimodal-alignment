import os
import time
from functools import wraps
from audioUtils import gen_audio_segment
from textUtils import preprocess


def gen_speech_timestamp(timestamp, audio, content, save_folder=None):
    speech = content.copy()
    word_times = [(word.start, word.end) for word in timestamp]

    # First pass: Calculate initial start/end times based on word boundaries
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

    # Second pass: Adjust boundaries to make nodes contiguous using midpoint
    for i in range(1, len(start_end_times)):
        prev_end = start_end_times[i - 1][1]
        curr_start = start_end_times[i][0]

        # If there's a gap between nodes, split it at the midpoint
        if curr_start > prev_end:
            midpoint = (prev_end + curr_start) / 2
            start_end_times[i - 1] = (start_end_times[i - 1][0], midpoint)
            start_end_times[i] = (midpoint, start_end_times[i][1])

    # Third pass: Save audio segments with adjusted boundaries
    if save_folder:
        for i, row in speech.iterrows():
            start_time, end_time = start_end_times[i]
            session = row['session']
            node = row['node']
            folder = f'{save_folder}/{session}'
            os.makedirs(folder, exist_ok=True)
            dir_save = f'{folder}/{node}.wav'
            gen_audio_segment(audio, start_time, end_time, dir_save)

    speech["start_sec"], speech["end_sec"] = zip(*start_end_times)
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

