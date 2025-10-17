import json
from dataclasses import asdict
from DataClass import Segment, WordStamp


def load_wordsegments(dir_json:str):
    with open(dir_json, 'r') as f:
        loaded_data = json.load(f)
    return [Segment(**data) for data in loaded_data]

def gen_timestamp(dir_json:str, bundle_sample_rate:float=50):
    seg = load_wordsegments(dir_json)

    dict_timestamp = list()
    for idx_segment, segment in enumerate(seg):
        sec_start, sec_end = segment.start / bundle_sample_rate, segment.end / bundle_sample_rate
        word_timestamp = WordStamp(
            segment.label,
            sec_start,
            sec_end
        )
        dict_timestamp.append(word_timestamp)
    return dict_timestamp

def save_json_segments(segments, dir_output):
    data = [asdict(segment) for segment in segments]
    try:
        with open(dir_output, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\nSuccessfully saved {len(dir_output)} segments to {dir_output}")
    except IOError as e:
        print(f"\nError saving file: {e}")