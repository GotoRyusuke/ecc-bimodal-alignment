import torchaudio

def load_audio(dir_audio):
    return torchaudio.load(dir_audio)

def gen_audio_segment(torch_wave, start:float, end:float, dir_output:str=None):
    waveform, sample_rate = torch_wave
    # print(sample_rate)
    multiplier = sample_rate
    segment = waveform[:, int(start*multiplier):int(end*multiplier)]
    segment_cpu = segment.cpu()

    if dir_output is not None:
        torchaudio.save(dir_output, segment_cpu, sample_rate)

    return segment_cpu