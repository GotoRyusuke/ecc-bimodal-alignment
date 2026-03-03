import os
import pandas as pd
import torch
import torchaudio
from ECCAligner import gen_segments
from utils import gen_speech_timestamp
from jsonUtils import gen_timestamp, load_wordsegments, save_json_segments
from audioUtils import load_audio, gen_audio_segment

class ConfCallAligner:
    # For chunk alignment
    def __init__(self, DIR_PANEL, FD_WAV, FD_TXT, FD_ALIGNED):
        self.DIR_PANE = DIR_PANEL
        self.FD_WAV = FD_WAV
        self.FD_TXT = FD_TXT
        self.FD_ALIGNED = FD_ALIGNED

        self.panel = pd.read_csv(DIR_PANEL)

        print('TORCH VERSION:', torch.__version__)  # CUDA settings
        print('TORCHAUDIO VERSION:', torchaudio.__version__)  # CUDA settings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('DEVICE:', device)
        torch.random.manual_seed(0)

    def _align_single_call(self, call_idx):
        ticker = self.panel.loc[call_idx, 'ticker']
        sa_id = self.panel.loc[call_idx, 'sa_transcript_id']
        dir_wav = f'{self.FD_WAV}/{ticker}/{sa_id}.wav'

        year = self.panel.loc[call_idx, 'year']
        file_name = self.panel.loc[call_idx, 'file']
        dir_txt = f'{self.FD_TXT}/{year}/{file_name}/content.parquet'

        print(f'PROCESSING...')
        print(f'TASK -LOC_ID {call_idx} -TICKER {ticker} -YEAR {year} -TXT_FILE {dir_txt} -WAV_FILE {dir_wav}')

        try:
            segments = gen_segments(dir_wav, dir_txt)
            save_json_segments(segments, f'{self.FD_WAV}/{ticker}/{sa_id}_segments.json')
        except RuntimeError:
            return 0
        dict_timestamp = gen_timestamp(seg=segments)

        print('DONE.')

        # SAVE ALIGNED RECORDS
        content = pd.read_parquet(dir_txt)
        torch_audio = load_audio(dir_wav)
        fd_aligned_wav = f'{self.FD_ALIGNED}/{ticker}/{sa_id}'
        os.makedirs(fd_aligned_wav, exist_ok=True)
        df_aligned = gen_speech_timestamp(dict_timestamp, torch_audio, content, fd_aligned_wav)
        df_aligned.to_csv(f'{fd_aligned_wav}/match.csv', index=False)

        return 1

    def align_calls(self, start_idx=0, end_idx=None, save_panel=True):
        """
        Process calls in the panel and record status codes.

        Args:
            start_idx: Starting index in the panel (default: 0)
            end_idx: Ending index in the panel (default: None, process all)
            save_panel: Whether to save the updated panel with status codes (default: True)

        Returns:
            DataFrame with status codes recorded
        """
        if end_idx is None:
            end_idx = len(self.panel) - 1

        if 'alignment_status' not in self.panel.columns:
            self.panel['alignment_status'] = pd.NA

        print(f'Processing calls from index {start_idx} to {end_idx}...')

        for call_idx in range(start_idx, end_idx + 1):
            print(f'\n=== Processing call {call_idx}/{end_idx} ===')
            try:
                status = self._align_single_call(call_idx)
                self.panel.loc[call_idx, 'alignment_status'] = status
            except Exception as e:
                print(f'ERROR processing call {call_idx}: {e}')
                self.panel.loc[call_idx, 'alignment_status'] = -1

        if save_panel:
            self.panel.to_csv(self.DIR_PANE, index=False)
            print(f'\nPanel with status codes saved to {self.DIR_PANE}')

        return self.panel

if __name__ == '__main__':
    DIR_PANEL = 'panel_transcript-recording-merged_2017-2021_R71010.csv'
    FD_TXT = 'G:/ECC Transcripts/Data_texts'
    FD_WAV = 'G:/REC/data_wav'
    FD_ALIGNED = 'G:/REC/data_txt'

    aligner = ConfCallAligner(DIR_PANEL=DIR_PANEL, FD_WAV=FD_WAV, FD_TXT=FD_TXT, FD_ALIGNED=FD_ALIGNED)
    tmp = aligner._align_single_call(call_idx=0)

