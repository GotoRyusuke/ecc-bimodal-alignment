import os
import torchaudio
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#Todo: convert all mp3 to wav
DIR_MASTER_MP3 = 'E:/REC/Data_rawMP3'
DIR_SAVE_WAV = 'E:/REC/data_wav'
panel = pd.read_parquet('data/panel_transcript-recording-merged_2017-2021_R71010.parquet')
panel['to_wav'] = 1

for idx in tqdm(panel.index[283:]):
    ticker = panel.loc[idx, 'ticker']
    sa_id = panel.loc[idx, 'sa_transcript_id']

    try:
        dir_mp3 = f'{DIR_MASTER_MP3}/{ticker}/{sa_id}.mp3'
        waveform, sample_rate = torchaudio.load(dir_mp3)

        os.makedirs(f'{DIR_SAVE_WAV}/{ticker}', exist_ok=True)
        dir_wav = f'{DIR_SAVE_WAV}/{ticker}/{sa_id}.wav'
        torchaudio.save(dir_wav, waveform, sample_rate)
    except:
        panel.loc[idx, 'to_wav'] = 0

panel.to_parquet('data/panel_transcript-recording-merged-converted_2017-2021_R71017.parquet')