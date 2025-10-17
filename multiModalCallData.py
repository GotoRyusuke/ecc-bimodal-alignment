import pandas as pd
import torchaudio
from dataclasses import dataclass
from utils import load_audio

@dataclass
class MultiModCall:
    sa_transcript_id: int
    factset_transcript_id: int
    dir_txt: str
    title: str
    gvkey: int
    ticker: str
    permno: int
    date: str
    year: int
    fiscal_year: int
    fiscal_period: int
    duration: float

    def __post_init__(self):
        self.dir_mp3 = f'{self.ticker}/{self.sa_transcript_id}'
        self.dir_txt = f'{self.year}/{self.dir_txt}/content.parquet'

    def __repr__(self):
        return f' TICKER: {self.ticker}\n GVKEY: {self.gvkey}\n DATE: {self.date}\n FISCAL YEAR: {self.fiscal_year}\n FISCAL PERIOD: {self.fiscal_period}\n TITLE: {self.title}\n DURATION: {self.duration:5.2f}mins'

    def gen_full_dir(self, dir_master_mp3, dir_master_txt):
        self.dir_mp3 = f'{dir_master_mp3}/{self.dir_mp3}.mp3'
        self.dir_txt = f'{dir_master_txt}/{self.dir_txt}'

    def gen_multimodal(self):
        self.waveform, self.sample_rate = load_audio(self.dir_mp3)
        self.content = pd.read_parquet(self.dir_txt)

    def gen_slice(self, start: float, end: float, sample_rate:int):
        segment = self.waveform[:, int(start*sample_rate):int(end*sample_rate)]
        return segment.cpu()


