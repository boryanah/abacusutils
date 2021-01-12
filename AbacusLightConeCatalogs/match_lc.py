#!/usr/bin/env python3

import glob
from pathlib import Path
import asdf
import numpy as np
import sys
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import gc
import os
import argparse

from tools.aid_asdf import save_asdf, load_mt_pid, load_mt_npout, load_lc_pid_rv
from bitpacked import unpack_rvint, unpack_pids
from tools.merger import get_zs_from_headers, get_one_header
from tools.read_headers import get_lc_info
from tools.match_searchsorted import match, match_halo_pids_to_lc_rvint

# these are probably just for testing; should be removed for production
DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_highbase_c021_ph000"
#DEFAULTS['sim_name'] = "AbacusSummit_highbase_c000_ph100"
#DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
#DEFAULTS['light_cone_parent'] = "/mnt/gosling2/bigsims/"
DEFAULTS['light_cone_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus"
#DEFAULTS['catalog_parent'] = "/mnt/gosling1/boryanah/light_cone_catalog/"
DEFAULTS['catalog_parent'] = "/global/cscratch1/sd/boryanah/light_cone_catalog/"
#DEFAULTS['merger_parent'] = "/mnt/gosling2/bigsims/merger/"
DEFAULTS['merger_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus/merger"
DEFAULTS['z_lowest'] = 0.5
DEFAULTS['z_highest'] = 0.8

# return the mt catalog names straddling the given redshift
def get_mt_fns(z_th, zs_mt, chis_mt, cat_lc_dir):
    for k in range(len(zs_mt-1)):
        squish = (zs_mt[k] <= z_th) & (z_th <= zs_mt[k+1])
        if squish == True: break 
    z_low = zs_mt[k]
    z_high = zs_mt[k+1]
    chi_low = chis_mt[k]
    chi_high = chis_mt[k+1]
    fn_low = cat_lc_dir / ("z%.3f/pid_lc.asdf"%(z_low))
    fn_high = cat_lc_dir / ("z%.3f/pid_lc.asdf"%(z_high))
    halo_fn_low = cat_lc_dir / ("z%.3f/halo_info_lc.asdf"%(z_low))
    halo_fn_high = cat_lc_dir / ("z%.3f/halo_info_lc.asdf"%(z_high))

    mt_fns = [fn_high, fn_low]
    mt_zs = [z_high, z_low]
    mt_chis = [chi_high, chi_low]
    halo_mt_fns = [halo_fn_high, halo_fn_low]

    return mt_fns, mt_zs, mt_chis, halo_mt_fns

def extract_steps(fn):
    split_fn = fn.split('Step')[1]
    step = np.int(split_fn.split('.asdf')[0])
    return step

def main(sim_name, z_lowest, z_highest, light_cone_parent, catalog_parent, merger_parent):
    light_cone_parent = Path(light_cone_parent)
    catalog_parent = Path(catalog_parent)
    merger_parent = Path(merger_parent)

    # directory where the merger tree files are kept
    merger_dir = merger_parent / sim_name
    header = get_one_header(merger_dir)
    
    # simulation parameters
    Lbox = header['BoxSize']
    PPD = header['ppd']

    # directory where we have saved the final outputs from merger trees and halo catalogs
    cat_lc_dir = catalog_parent / sim_name / "halos_light_cones"

    # directory where light cones are saved
    lc_dir = light_cone_parent / sim_name / "lightcones"

    # all redshifts, steps and comoving distances of light cones files; high z to low z
    # remove presaving after testing done (or make sure presaved can be matched with simulation)
    if not os.path.exists(Path("data_headers") / sim_name / "coord_dist.npy") or not os.path.exists(Path("data_headers") / sim_name / "redshifts.npy") or not os.path.exists(Path("data_headers") / sim_name / "steps.npy"):
        zs_all, steps, chis_all = get_lc_info("all_headers")
        os.makedirs(Path("data_headers") / sim_name, exist_ok=True)
        np.save(Path("data_headers") / sim_name / "redshifts.npy", zs_all)
        np.save(Path("data_headers") / sim_name / "steps.npy", steps_all)
        np.save(Path("data_headers") / sim_name / "coord_dist.npy", chis_all)
    zs_all = np.load(Path("data_headers") / sim_name / "redshifts.npy")
    steps_all = np.load(Path("data_headers") / sim_name / "steps.npy")
    chis_all = np.load(Path("data_headers") / sim_name / "coord_dist.npy")
    zs_all[-1] = float("%.1f" % zs_all[-1])

    # if merger tree redshift information has been saved, load it (if not, save it)
    if not os.path.exists(Path("data_mt") / sim_name / "zs_mt.npy"):
        # all merger tree snapshots and corresponding redshifts
        snaps_mt = sorted(merger_dir.glob("associations_z*.0.asdf"))
        zs_mt = get_zs_from_headers(snaps_mt)
        os.makedirs(Path("data_mt") / sim_name, exist_ok=True)
        np.save(Path("data_mt") / sim_name / "zs_mt.npy", zs_mt)
    zs_mt = np.load(Path("data_mt") / sim_name / "zs_mt.npy")
    # correct for interpolation out of bounds error
    zs_mt = zs_mt[(zs_mt <= zs_all.max()) & (zs_mt >= zs_all.min())]

    # time step of furthest and closest shell in the light cone files
    step_min = np.min(steps_all)
    step_max = np.max(steps_all)
    
    # get functions relating chi and z
    chi_of_z = interp1d(zs_all,chis_all)
    z_of_chi = interp1d(chis_all,zs_all)
    
    # conformal distance of the mtree catalogs
    chis_mt = chi_of_z(zs_mt)

    # Read light cone file names
    lc_rv_fns = sorted(glob.glob(os.path.join(lc_dir, 'rv/LightCone*')))
    lc_pid_fns = sorted(glob.glob(os.path.join(lc_dir, 'pid/LightCone*')))
    
    # select the final and initial step for computing the convergence map
    step_start = steps_all[np.argmin(np.abs(zs_all-z_highest))]
    step_stop = steps_all[np.argmin(np.abs(zs_all-z_lowest))]
    print("step_start = ",step_start)
    print("step_stop = ",step_stop)

    # these are the time steps associated with each of the light cone files
    step_fns = np.zeros(len(lc_pid_fns),dtype=int)
    for i in range(len(lc_pid_fns)):
        step_fns[i] = extract_steps(lc_pid_fns[i])

    # initialize previously loaded mt file name
    currently_loaded_zs = []
    currently_loaded_headers = []
    currently_loaded_npouts = []
    currently_loaded_pids = []
    currently_loaded_tables = []
    for step in range(step_start,step_stop+1):
        # this is because our arrays start correspond to step numbers: step_start, step_start+1, step_start+2 ... step_stop
        j = step-step_min
        step_this = steps_all[j]
        z_this = zs_all[j]
        chi_this = chis_all[j]

        assert step_this == step, "You've messed up the counts"
        print("light cones step, redshift = ",step_this, z_this)

        # get the two redshifts it's straddling and the mean chi
        mt_fns, mt_zs, mt_chis, halo_mt_fns = get_mt_fns(z_this, zs_mt, chis_mt, cat_lc_dir)

        # get the mean chi
        mt_chi_mean = np.mean(mt_chis)

        # how many shells are we including on both sides, including mid point (total of 2j+1)
        buffer_no = 2

        # is this the redshift that's closest to the bridge between two redshifts 
        mid_bool = (np.argmin(np.abs(mt_chi_mean-chis_all)) <= j+buffer_no) & (np.argmin(np.abs(mt_chi_mean-chis_all)) >= j-buffer_no)
        
        # if not in between two redshifts, we just need one catalog -- the one it is closest to
        if not mid_bool:
            mt_fns = [mt_fns[np.argmin(np.abs(mt_chis-chi_this))]]
            halo_mt_fns = [halo_mt_fns[np.argmin(np.abs(mt_chis-chi_this))]] 
            mt_zs = [mt_zs[np.argmin(np.abs(mt_chis-chi_this))]] 

        # load this and prev
        for i in range(len(mt_fns)):
            # check if catalog already loaded
            if mt_zs[i] in currently_loaded_zs: print("skipped loading catalog ",mt_zs[i]); continue

            # discard the old redshift catalog and record its data
            if len(currently_loaded_zs) >= 2:

                # save the information about that redshift
                save_asdf(currently_loaded_tables[0],"pid_rv_lc",currently_loaded_headers[0],cat_lc_dir / ("z%4.3f"%currently_loaded_zs[0]))
                print("saved catalog = ",currently_loaded_zs[0])

                # discard it from currently loaded
                currently_loaded_zs = currently_loaded_zs[1:]
                currently_loaded_headers = currently_loaded_headers[1:]
                currently_loaded_pids = currently_loaded_pids[1:]
                currently_loaded_tables = currently_loaded_tables[1:]

            # load new merger tree catalog
            mt_pid, header = load_mt_pid(mt_fns[i],Lbox,PPD)
            halo_mt_npout = load_mt_npout(halo_mt_fns[i])
            
            # start the light cones table for this redshift
            lc_table_final = np.empty(len(mt_pid),dtype=[('pid',mt_pid.dtype),('pos',(np.float32,3)),\
                                                         ('vel',(np.float32,3)),('redshift',np.float32)])

            # append the newly loaded catalog
            currently_loaded_zs.append(mt_zs[i])
            currently_loaded_headers.append(header)
            currently_loaded_pids.append(mt_pid)
            currently_loaded_npouts.append(halo_mt_npout)
            currently_loaded_tables.append(lc_table_final)

        print("currently loaded redshifts = ",currently_loaded_zs)    
        print("using redshifts = ",mt_zs)
        
        # find all light cone file names that correspond to this time step
        choice_fns = np.where(step_fns == step_this)[0]
        # number of light cones at this step
        num_lc = len(choice_fns)
        
        assert (num_lc <= 3) & (num_lc > 0), "There can be at most three files in the light cones corresponding to a given step"
        # loop through those one to three light cone files
        for i_choice, choice_fn in enumerate(choice_fns):
            print("light cones file = ",lc_pid_fns[choice_fn])

            # load particles in light cone
            lc_pid, lc_rv = load_lc_pid_rv(lc_pid_fns[choice_fn],lc_rv_fns[choice_fn],Lbox,PPD)
            
            if 'LightCone1' in lc_pid_fns[choice_fn]:
                offset_lc = np.array([0.,0.,Lbox])
            elif 'LightCone2' in lc_pid_fns[choice_fn]:
                offset_lc = np.array([0.,Lbox,0.])
            else: offset_lc = np.array([0.,0.,0.])

            # loop over the one or two closest catalogs 
            for i in range(len(mt_fns)):
                which_mt = np.where(mt_zs[i] == currently_loaded_zs)[0]
                mt_pid = currently_loaded_pids[which_mt[0]]
                halo_mt_npout = currently_loaded_npouts[which_mt[0]]
                header = currently_loaded_headers[which_mt[0]]
                lc_table_final = currently_loaded_tables[which_mt[0]]
                mt_z = currently_loaded_zs[which_mt[0]]

                
                # match merger tree and light cone pids
                print("starting")
                
                # original version start
                t1 = time.time()
                i_sort_lc_pid = np.argsort(lc_pid)
                mt_in_lc = match(mt_pid, lc_pid, arr2_index=i_sort_lc_pid)
                comm2 = mt_in_lc[mt_in_lc > -1]
                comm1 = np.arange(len(mt_pid),dtype=int)[mt_in_lc > -1]
                pid_mt_lc = mt_pid[mt_in_lc > -1]
                print("time = ", time.time()-t1)
                
                # select the intersected positions
                pos_mt_lc, vel_mt_lc = unpack_rvint(lc_rv[comm2],Lbox)
                
                # print percentage of matched pids
                print("at z = %.3f, matched = "%mt_z,len(comm1)*100./(len(mt_pid)))
                # original version end
                
                '''
                # alternative Lehman implementation start
                t1 = time.time()
                comm1, nmatch, hrvint = match_halo_pids_to_lc_rvint(halo_mt_npout, mt_pid, lc_rv, lc_pid)
                print("at z = %.3f, matched = "%mt_z,len(hrvint)*100./(len(mt_pid)))
                print("time = ", time.time()-t1)
                
                pos_mt_lc, vel_mt_lc = unpack_rvint(hrvint,Lbox)
                pid_mt_lc = mt_pid[comm1]                
                # alternative Lehman implementation end
                '''

                # offset depending on which light cone we are at
                pos_mt_lc += offset_lc

                # save the pid, position, velocity and redshift
                lc_table_final['pid'][comm1] = pid_mt_lc
                lc_table_final['pos'][comm1] = pos_mt_lc
                lc_table_final['vel'][comm1] = vel_mt_lc
                lc_table_final['redshift'][comm1] = np.ones(len(pid_mt_lc))*z_this
            print("-------------------")


    # close the two that are currently open
    for i in range(len(currently_loaded_zs)):
        # save the information about that redshift
        save_asdf(currently_loaded_tables[0],"pid_rv_lc",currently_loaded_headers[0],cat_lc_dir / ("z%4.3f"%currently_loaded_zs[0]))
        print("saved catalog = ",currently_loaded_zs[0])

        # discard it from currently loaded
        currently_loaded_zs = currently_loaded_zs[1:]
        currently_loaded_headers = currently_loaded_headers[1:]
        currently_loaded_pids = currently_loaded_pids[1:]
        currently_loaded_tables = currently_loaded_tables[1:]

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_lowest', help='Initial redshift where we start building the trees', type=float, default=DEFAULTS['z_lowest'])
    parser.add_argument('--z_highest', help='Final redshift (inclusive)', type=float, default=DEFAULTS['z_highest'])
    parser.add_argument('--light_cone_parent', help='Light cone output directory', default=(DEFAULTS['light_cone_parent']))
    parser.add_argument('--catalog_parent', help='Light cone catalog directory', default=(DEFAULTS['catalog_parent']))
    parser.add_argument('--merger_parent', help='Merger tree directory', default=(DEFAULTS['merger_parent']))
    
    args = vars(parser.parse_args())
    main(**args)
