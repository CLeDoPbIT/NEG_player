import librosa
import numpy as np

eps = 1e-6

datasets = {
    'speech' : 
    {
        'SR' : 16000,
        'N_FFT' : 512,
        'naive_separation_multiplier' : .5,
    },
    'musdb' :     
    {
        'SR' : 20480,
        'N_FFT' : 512,
        'naive_separation_multiplier' : .1,
    },
    'musdb_drums' :
    {
        'SR' : 20480,
        'N_FFT' : 512,
        'naive_separation_multiplier' : .1,
    },
}
SPEC_TIMESTEPS = 64
WIN_LENGTH_SEC = 0.025
HOP_LENGTH_SEC = 0.01


class Constants:
    def __init__(self, dataset):
        self.dataset = dataset
        self.SR = datasets[self.dataset]['SR'] # audio sample rate
        self.N_FFT = datasets[self.dataset]['N_FFT'] # num FFT coefficients per window
        self.WIN_LENGTH = int(WIN_LENGTH_SEC * self.SR) # in seconds
        self.HOP_LENGTH = int(HOP_LENGTH_SEC * self.SR) # time difference between concecutive frames
        self.N_SPEC = self.N_FFT//2+1 # num of spectrogram bins
        self.naive_separation_multiplier = datasets[dataset]['naive_separation_multiplier']

def spec2aud(specs, consts):
    if specs.ndim == 2:
        return librosa.istft(specs, hop_length=consts.HOP_LENGTH, win_length=consts.WIN_LENGTH)
    auds = []
    for spec in specs:
        auds.append(librosa.istft(spec, hop_length=consts.HOP_LENGTH, win_length=consts.WIN_LENGTH))
    return auds


def aud2spec(audio, consts):
    spec = librosa.stft(audio[0], n_fft=consts.N_FFT, win_length=consts.WIN_LENGTH,
                                          hop_length=consts.HOP_LENGTH)
    spec = spec[:, :int(spec.shape[1]/64)*64]
    spec = spec.reshape((257, int(spec.shape[1]/64), 64))
    spec = spec.transpose((1, 0, 2))
    spec = spec.reshape((spec.shape[0], 1, 257, 64))
    spec_abs = spec.copy()
    spec_abs = spec_abs.astype(np.float32)
    spec_abs = np.abs(spec_abs)
    return spec_abs, spec



def create_merge_speech_noize(speech, noize, consts):
    speech = librosa.resample(speech[0][:,0], speech[1], 16000)
    noize = librosa.resample(noize[0], noize[1], 16000)
    speech = speech[:len(noize)]
    speech = speech + noize
    spec_abs, spec = aud2spec((speech, 1), consts)
    return spec_abs, spec
