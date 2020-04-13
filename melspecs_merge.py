import argparse
import logging
import dask.array as da

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--tidfile',    type=str)
parser.add_argument('--src',        type=str)
parser.add_argument('--dst',        type=str)
parser.add_argument('--crop', action='store_true')
parser.add_argument('--pad', action='store_true')
parser.add_argument("--log-level",  default=logging.INFO, type=lambda x: getattr(logging, x), help="Configure the logging level.")

args = parser.parse_args()
                          
logging.basicConfig(level=args.log_level)


import pandas as pd
import os
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

SEG_DIM        = 880

tids = pd.read_csv(args.tidfile, header=None, index_col=0)

mel_specs = []

pbar = tqdm(total=tids.shape[0])

i = 0

for tid in tids.index:
    
    with np.load(args.src + "/" + tid + ".npz", allow_pickle=True) as npz:
    
        mel_spec = npz["data"].astype(np.float32)
        
        if (args.pad) and (mel_spec.shape[1] < SEG_DIM):
                            
            zeros = np.zeros((mel_spec.shape[0],SEG_DIM,mel_spec.shape[2]), dtype=np.float32)
            zeros[:mel_spec.shape[0], :mel_spec.shape[1], :mel_spec.shape[2]] = mel_spec
                
            mel_spec = zeros
            
        if args.crop:
            mel_spec = mel_spec[:,:SEG_DIM,:]
        
        mel_specs.append(mel_spec)
            
        pbar.update()
        
        #i += 1
        
        #if i > 1000:
        #    break
    
#mel_specs = da.asarray(mel_specs)
#da.to_hdf5(args.dst + '.h5', 'data', mel_specs) 

#np.savez(args.dst + ".npy", mel_specs)
np.savez(args.dst, data=mel_specs, track_ids=tids)