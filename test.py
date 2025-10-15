
import os
import pandas as pd
from utils import gen_timestamp
from multiModalCallData import MultiModCall
os.chdir('F:/project_ecc-multimodal')

DIR_JSON = 'test_segments.json'
dict_timestamp = gen_timestamp(DIR_JSON)

DIR_CONTENT = 'test_content.parquet'
content = pd.read_parquet(DIR_CONTENT)

panel = pd.read_parquet('panel_transcript-recording-merged_2017-2021_R71010.parquet')

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

