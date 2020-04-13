import argparse
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--tidfile',    type=str, help="Path to trackid partition file.")
parser.add_argument('--dst',        type=str, help="Path to directory to store intermediate features")
parser.add_argument('--workers',    type=int, help="Number of processes for feature extraction.")
parser.add_argument('--crop', action='store_true', help="Crop longer audio files")
parser.add_argument('--pad', action='store_true', help="Zero-pad shorter audio files")
parser.add_argument('--skip', action='store_true', help="Skip if feature files already exist.")
parser.add_argument('--precision',  type=int, default=32, help="Store features with 16bit or 32bit precision")
parser.add_argument("--log-level",  default=logging.DEBUG, type=lambda x: getattr(logging, x), help="Configure the logging level.")
#parser.add_argument('--test', action='store_true', help="Development parameter. Only process one file")


args = parser.parse_args()
                          
logging.basicConfig(level=args.log_level)

import time
import random
from multiprocessing import Pool

import pandas as pd
import librosa
import audioread
import os
import numpy as np
from tqdm import tqdm
import traceback

import warnings
warnings.filterwarnings('ignore')

DST_PATH = args.dst

DST_SAMPLERATE = 22050
N_FFT          = 1024
HOP_LENGTH     = 256
NUM_MELS       = 128
FMAX           = None
FMIN           = 0.0
MONO           = False

SEG_LENGTH_SEC = 11
SEG_OFFSET_SEC = 3

SEG_DIM        = 880



def extract_melspec(y, sample_rate):

    mel_spec = librosa.feature.melspectrogram(y          = y, 
                                              sr         = sample_rate, 
                                              n_fft      = N_FFT, 
                                              hop_length = HOP_LENGTH, 
                                              n_mels     = NUM_MELS,
                                              fmin       = FMIN,
                                              fmax       = FMAX)

    mel_spec = librosa.core.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec



def extract(track_id, f_name):
    
    dst_fname = DST_PATH + "/" + track_id + ".npz"
    
    if not (args.skip and os.path.exists(dst_fname)):

        try:
                                
            wave_data, sample_rate = librosa.core.load(f_name, 
                                                       sr       = DST_SAMPLERATE, 
                                                       mono     = MONO)
                                                       
                                                       
            if not MONO and (len(wave_data.shape) != 2):
                wave_data = np.asarray([wave_data,wave_data])
                
            
            
            start  = SEG_OFFSET_SEC * sample_rate
            length = sample_rate * SEG_LENGTH_SEC
            end    = start + length
            
            if MONO:
            
                if (wave_data.shape[0] > end):
                    wave_data = wave_data[start:end]
                else:
                    wave_data = wave_data[:length]
                    
            else:
            
                if (wave_data.shape[1] > end):
                    wave_data = wave_data[:,start:end]
                else:
                    wave_data = wave_data[:,:length]
                

            if MONO:
                
                mel_spec = extract_melspec(np.asfortranarray(wave_data), sample_rate) 
                
                if args.crop:
                    mel_spec = mel_spec[:,:SEG_DIM]
                mel_spec = np.expand_dims(mel_spec, 2)
                                                
            else:
                
                mel_spec_ch1 = extract_melspec(np.asfortranarray(wave_data[0,:]), sample_rate)
                mel_spec_ch2 = extract_melspec(np.asfortranarray(wave_data[1,:]), sample_rate)
                
                if args.crop:
                    mel_spec_ch1 = mel_spec_ch1[:,:SEG_DIM]
                    mel_spec_ch2 = mel_spec_ch2[:,:SEG_DIM]
                
                mel_spec_ch1 = np.expand_dims(mel_spec_ch1, 2)
                mel_spec_ch2 = np.expand_dims(mel_spec_ch2, 2)
                
                mel_spec = np.concatenate([mel_spec_ch1, mel_spec_ch2], axis=2)

            
            if (args.pad) and (mel_spec.shape[1] < SEG_DIM):
                                
                zeros = np.zeros((mel_spec.shape[0],SEG_DIM,mel_spec.shape[2]), dtype=np.float32)
                zeros[:mel_spec.shape[0], :mel_spec.shape[1], :mel_spec.shape[2]] = mel_spec
                    
                mel_spec = zeros            
            
            
            if args.precision == 16:
                mel_spec = mel_spec.astype(np.float32)
            elif args.precision == 16:
                mel_spec = mel_spec.astype(np.float16)
            
            np.savez(dst_fname, data=mel_spec)
            
        except Exception as e:
            print(e)
            traceback.print_exc()
            pass

    return track_id


#extract("TRFFYHQ128F147CC95")
#exit(0)

tids = pd.read_csv(args.tidfile, header=None)
tids.columns = ["track_id", "filename"]

pool = Pool(args.workers)

pbar = tqdm(total=tids.shape[0])

def update(*a):
    pbar.update()
    pbar.set_description(str(a))
    
for i in range(pbar.total):
    pool.apply_async(extract, args=(tids.iloc[i].track_id, tids.iloc[i].filename,), callback=update)

pool.close()
pool.join()