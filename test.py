import os
import pandas as pd
import torch
import torchaudio
from ECCAligner import gen_segments
from jsonUtils import gen_timestamp
from audioUtils import load_audio
from utils import gen_speech_timestamp
from jsonUtils import save_json_segments


os.chdir('F:/ecc-bimodal-alignment')

## GET SEGMENTS
print(torch.__version__) # CUDA settings
print(torchaudio.__version__) # CUDA settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.random.manual_seed(0)

SPEECH_FILE = 'data/test.wav'
TEXT_FILE = 'data/test.txt'

# Run CTC alignment
word_segments = gen_segments(SPEECH_FILE, TEXT_FILE)

# Save segments
save_json_segments(word_segments, 'data/test_segments.json')

DIR_JSON = 'data/test_segments.json'
dict_timestamp = gen_timestamp(DIR_JSON)

DIR_CONTENT = 'data/test_content.parquet'
content = pd.read_parquet(DIR_CONTENT)
content.to_csv('data/test_content.csv', index=False)

# # Check the length
# from utils import preprocess
# ls_word_labels = [word.label for word in dict_timestamp]
# ls_content = content['text'].to_list()
# ls_text = [preprocess(text, sep=' ').split() for text in ls_content]
# num_labels = len(ls_word_labels)
# num_content = sum([len(text) for text in ls_text])
# -> the number of labels and length of original text (after pre-processing) are the same. One-to-one alignment is viable

# Check the slice
dir_audio = 'data/test_wav.wav'
torch_audio = load_audio(dir_audio)

## GENERATE ALIGNED DF
aligned_content = gen_speech_timestamp(dict_timestamp, torch_audio, content, 'data/test_align')
aligned_content.to_csv('data/test_aligned_content.csv', index=False)