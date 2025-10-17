
import os
import pandas as pd
from utils import load_wordsegments, gen_timestamp
from multiModalCallData import MultiModCall
os.chdir('F:/project_cctranscripts/ecc-bimodal-alignment')

DIR_JSON = 'data/test_segments.json'
dict_timestamp = gen_timestamp(DIR_JSON)
dict_wordsegments = load_wordsegments(DIR_JSON)

DIR_CONTENT = 'data/test_content.parquet'
content = pd.read_parquet(DIR_CONTENT)

panel = pd.read_parquet('data/panel_transcript-recording-merged_2017-2021_R71010.parquet')

idx = 134
test_audio = MultiModCall(
    panel.loc[idx, 'sa_transcript_id'],
    panel.loc[idx, 'factset_transcript_id'],
    panel.loc[idx, 'file'],
    panel.loc[idx, 'title'],
    panel.loc[idx, 'gvkey'],
    panel.loc[idx, 'ticker'],
    panel.loc[idx, 'permno'],
    panel.loc[idx, 'date'],
    panel.loc[idx, 'year'],
    panel.loc[idx, 'fiscal_year'],
    panel.loc[idx, 'fiscal_period'],
    panel.loc[idx, 'duration']
)

DIR_MASTER_MP3 = 'E:/REC/Data_rawMP3'
DIR_MASTER_TXT = 'E:/ECC Transcripts/Data_texts'
test_audio._gen_multimodal(DIR_MASTER_MP3, DIR_MASTER_TXT)

segment = test_audio.gen_slice(0, 18.26, test_audio.sample_rate)

import torchaudio
torchaudio.save(
    "slice.wav",          # output filename
    segment,       # waveform tensor
    test_audio.sample_rate      # sample rate (same as original)
)