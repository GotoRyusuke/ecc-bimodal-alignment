import torch
import torchaudio
from DataClass import Point, Segment
from utils import timer
from textUtils import preprocess
from audioUtils import load_audio
from jsonUtils import save_json_segments


@timer
def gen_segments(dir_mp3, dir_txt):
    target_sample_rate = 16000

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()

    with torch.inference_mode():
        waveform, sample_rate = load_audio(dir_mp3)
        waveform = waveform.to(device)

        # Resample the waveform if necessary
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            ).to(device)
            waveform = resampler(waveform)

        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()
    
    with open(dir_txt, 'r') as f:
        tmp = f.read().strip()

    transcript = preprocess(tmp)
    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in transcript]

    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # --- Get trellis ---
    blank_id = 0
    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )

    #  --- Get path ---
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    path = path[::-1]

    # --- Get segments ---
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2

    # --- Get words segments ---
    separator = '|'
    word_segments = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                word_segments.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1

    return word_segments

# Ignore this class for it's not done yet
'''
class ConfCallAligner:
    # For chunk alignment 
    def __init__(self, dir_panel, dir_master_wav, dir_master_txt):
        self.dir_panel = dir_panel
        self.panel = pd.read_parquet(dir_panel)
        self.dir_master_wav = dir_master_wav
        self.dir_master_txt = dir_master_txt

    def _align_single_call(self, idx):
        ticker = self.panel.loc[idx, 'ticker']
        sa_id = self.panel.loc[idx, 'sa_transcript_id']
        dir_wav = f'{self.dir_master_wav}/{ticker}/{sa_id}.wav'

        year = self.panel.loc[idx, 'year']
        file_name = self.panel.loc[idx, 'file']
        dir_txt = f'{self.dir_master_txt}/{year}/{file_name}/content.parquet'

        print(dir_wav, dir_txt)

        #segments = gen_segments(dir_mp3, dir_txt)
        #return segments
'''

if __name__ == '__main__':
    # CUDA settings
    print(torch.__version__)
    print(torchaudio.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.random.manual_seed(0)

    SPEECH_FILE = 'data/test.wav'
    TEXT_FILE = 'data/test.txt'

    # Run CTC alignment
    word_segments = gen_segments(SPEECH_FILE, TEXT_FILE)

    # Save segments
    save_json_segments(word_segments, 'test_segments.json')


    # TODO: for chunk alignment
    # DIR_PANEL = 'data/panel_transcript-recording-merged_2017-2021_R71010.parquet'
    # ###
    # DIR_MASTER_WAV = 'E:/REC/Data_rawMP3'
    # ###
    # DIR_MASTER_TXT = 'E:/ECC Transcripts/Data_texts'
    #
    # aligner = ConfCallAligner(DIR_PANEL, DIR_MASTER_WAV, DIR_MASTER_TXT)
    # test = aligner._align_single_call(0)
    
    
    
    