import glob
from pathlib import Path
import numpy as np

from tools.InputFile import InputFile

'''
Script for recording useful information (redshifts, steps and comoving distances) from light cone headers.

First need to unzip headers file in header_dir with:
$ mkdir all_headers; cd all_headers
$ cp /global/project/projectdirs/desi/cosmosim/Abacus/AbacusSummit_base_c000_ph006/lightcones/pid/headers.zip .
$ unzip headers.zip
$ rm headers.zip
'''

def get_lc_info(header_dir):
    # sort headers by step number
    header_fns = sorted(list(header_dir.glob("header_Step*")))
    
    # initialize arrays
    redshifts = np.zeros(len(header_fns))
    steps = np.zeros(len(header_fns),dtype=int)
    coord_dist = np.zeros(len(header_fns))

    # record needed information from each header
    for i in range(len(header_fns)):
        header_fn = str(header_fns[i])
        #print(header_fn, infile.CoordinateDistanceHMpc)
        infile = InputFile(header_fn)

        redshifts[i] = infile.Redshift
        steps[i] = np.int(header_fn.split('Step')[-1])
        coord_dist[i] = infile.CoordinateDistanceHMpc
    return redshifts, steps, coord_dist



def main():
    # location where all headers for sim are saved
    sim_name = "AbacusSummit_highbase_c021_ph000"
    header_dir = Path("/global/homes/b/boryanah/repos/abacus_lc_cat/all_headers") / sim_name
    redshifts, steps, coord_dist = get_lc_info(header_dir)

    print(redshifts)
    print(steps)
    print(coord_dist)
    
    np.save("../data_headers/redshifts.npy",redshifts)
    np.save("../data_headers/steps.npy",steps)
    np.save("../data_headers/coord_dist.npy",coord_dist)

