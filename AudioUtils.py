import os
import torch
import torchaudio
from scipy.io import wavfile
import scipy.signal as sp
import numpy as np
import math

def save_wav(name, data, fs = 44100):
    wavfile.write(name, fs, data.flatten().astype(np.float32))

def linear_2_db(signal):
    return 20.0 * np.log10(signal)

def db_2_linear(signal_db):
    return 10.0 ** (signal_db / 20.0)

def normalize_max_peak(signal):
    data_max = np.max(np.abs(signal))
    return signal / data_max

def normalize_at_minus_3dB(signal):
    data_max = np.max(np.abs(signal))
    minus_3db_in_linear = 0.707
    return signal * (minus_3db_in_linear / data_max)

def normalize_at_minus_6dB(signal):
    data_max = np.max(np.abs(signal))
    return signal * (0.5 / data_max)

def calculate_rms(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    return rms

def calculate_rms_db(signal):
    rms = calculate_rms(signal)
    rms_db = linear_2_db(rms)
    return rms_db

def calculate_peak_db(signal):
    data_max = np.max(np.abs(signal))
    peak_db = linear_2_db(data_max)
    return peak_db

def convert_to_float(signal):
    data_max = np.max(np.abs(signal))
    treshold_float = 10
    treshold_16bit = 32768
    if (data_max > treshold_float):
        if(data_max <= treshold_16bit):
            scale_factor = 1.0 / treshold_16bit
        else:
            if (data_max > treshold_16bit):
                scale_factor = 1.0 / 2147483648
    else:
        scale_factor = 1.0
    return signal * scale_factor

def load_audio_data(inFile, outFile, offsetSec=1, delay=0):
    # Load and Preprocess Data ###########################################
    in_rate, in_data = wavfile.read(inFile)
    out_rate, out_data = wavfile.read(outFile)

    assert in_rate == out_rate, "Mismatched sample rates"

    offset = int(math.floor(offsetSec * in_rate))

    x_all = in_data.astype(np.float32).flatten()
    x_all = x_all[offset:]  

    y_all = out_data.astype(np.float32).flatten()

    offsetOut = offset-delay
    if offsetOut < 0:
        offsetOut = 0
        
    y_all = y_all[offsetOut:]

    #x_all = normalize_max_peak(x_all).reshape(len(x_all),1)
    #y_all = normalize_at_minus_3dB(y_all).reshape(len(y_all),1)
    x_all = convert_to_float(x_all).reshape(len(x_all),1)
    y_all = convert_to_float(y_all).reshape(len(y_all),1)

    return(x_all, y_all, in_rate)

def check_if_model_exists(name, modelPath='models/'):
    if not os.path.exists(modelPath+name):
        os.makedirs(modelPath+name)
    else:
        print("A model with the same name already exists. Please choose a new name.")
        exit

def torch_2_numpy(tensor):
    if type(tensor) == torch.Tensor:
        return tensor.detach().cpu().numpy()
    if type(tensor) == np.ndarray:
        return tensor

def numpy_2_torch(array):
    if type(array) == np.ndarray:
        return torch.from_numpy(array)
    if type(array) == torch.Tensor:
        return array
    
def load_audio(filename: str) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(filename, channels_first = True)
    return (waveform, sample_rate)

def save_audio(data: torch.Tensor, filename: str, sample_rate: int = 44100):
    torchaudio.save(filename, data, sample_rate)