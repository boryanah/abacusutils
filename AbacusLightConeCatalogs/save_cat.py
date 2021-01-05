#!/usr/bin/env python3
'''
This is the second script in the "lightcone halo" pipeline.  The goal of this script is to use the output
from build_mt.py (i.e. information about what halos intersect the lightcone and when) and save the relevant
information from the CompaSO halo info catalogs.

Usage
-----
$ ./save_cat.py --help
'''

import sys
import glob
from pathlib import Path
import time
import gc
import os

import asdf
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import argparse
from astropy.table import Table

from compaso_halo_catalog import CompaSOHaloCatalog
from tools.aid_asdf import save_asdf, reindex_pid, reindex_pid_pos
from tools.merger import simple_load, get_halos_per_slab, get_zs_from_headers, get_halos_per_slab_origin, extract_superslab, extract_superslab_minified, unpack_inds

# TODO: copy all halo fields (just get rid of fields=fields_cat)

# these are probably just for testing; should be removed for production
DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_highbase_c021_ph000"
#DEFAULTS['sim_name'] = "AbacusSummit_highbase_c000_ph100"
#DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
#DEFAULTS['compaso_parent'] = "/mnt/gosling2/bigsims")
DEFAULTS['compaso_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus"
#DEFAULTS['catalog_parent'] = "/mnt/gosling1/boryanah/light_cone_catalog/"
DEFAULTS['catalog_parent'] = "/global/cscratch1/sd/boryanah/light_cone_catalog/"
#DEFAULTS['merger_parent'] = "/mnt/gosling2/bigsims/merger"
DEFAULTS['merger_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus/merger"
DEFAULTS['z_start'] = 0.5
DEFAULTS['z_stop'] = 0.8
CONSTANTS = {'c': 299792.458}  # km/s, speed of light

def extract_redshift(fn):
    fn = str(fn)
    redshift = float(fn.split('z')[-1])
    return redshift

def correct_all_inds(halo_ids, N_halo_slabs, slabs, n_chunks):
    # unpack indices into slabs and inds
    slab_ids, ids = unpack_inds(halo_ids)
        
    # total number of halos in the slabs that we have loaded
    offsets = np.zeros(n_chunks, dtype=int)
    offsets[1:] = np.cumsum(N_halo_slabs)[:-1]

    # select the halos belonging to given slab
    for counter in range(n_chunks):
        select = np.where(slab_ids == slabs[counter])[0]
        ids[select] += offsets[counter]
    return ids

def main(sim_name, z_start, z_stop, compaso_parent, catalog_parent, merger_parent, save_pos=False):

    compaso_parent = Path(compaso_parent)
    catalog_parent = Path(catalog_parent)
    merger_parent = Path(merger_parent)

    # directory where the CompaSO halo catalogs are saved
    cat_dir = compaso_parent / sim_name / "halos"

    # fields to extract from the CompaSO catalogs
    fields_cat = ['id','npstartA','npoutA','N','x_L2com','v_L2com','sigmav3d_L2com']

    # obtain the redshifts of the CompaSO catalogs
    redshifts = glob.glob(os.path.join(cat_dir,"z*"))
    zs_cat = [extract_redshift(redshifts[i]) for i in range(len(redshifts))]
    
    # directory where we save the final outputs
    cat_lc_dir = catalog_parent / sim_name / "halos_light_cones"

    # directory where the merger tree files are kept
    merger_dir = merger_parent / sim_name
    
    # more accurate, slightly slower
    if not os.path.exists("data/zs_mt.npy"):
        # all merger tree snapshots and corresponding redshifts
        snaps_mt = sorted(merger_dir.glob("associations_z*.0.asdf"))
        zs_mt = get_zs_from_headers(snaps_mt)
        np.save("data/zs_mt.npy", zs_mt)
    zs_mt = np.load("data/zs_mt.npy")

    # names of the merger tree file for a given redshift
    merger_fns = list(merger_dir.glob("associations_z%4.3f.*.asdf"%zs_mt[0]))
    
    # number of chunks
    n_chunks = len(merger_fns)
    print("number of chunks = ",n_chunks)
    
    # all redshifts, steps and comoving distances of light cones files; high z to low z
    # remove presaving after testing done (or make sure presaved can be matched with simulation)
    if not os.path.exists("data_headers/coord_dist.npy") or not os.path.exists("data_headers/redshifts.npy"):
        zs_all, steps, chis_all = get_lc_info("all_headers")
        np.save("data_headers/redshifts.npy", zs_all)
        np.save("data_headers/coord_dist.npy", chis_all)
    zs_all = np.load("data_headers/redshifts.npy")
    chis_all = np.load("data_headers/coord_dist.npy")
    zs_all[-1] = float("%.1f" % zs_all[-1])  # LHG: I guess this is trying to match up to some filename or something?

    # get functions relating chi and z
    chi_of_z = interp1d(zs_all,chis_all)
    z_of_chi = interp1d(chis_all,zs_all)
    
    # initial redshift where we start building the trees
    ind_start = np.argmin(np.abs(zs_mt-z_start))
    ind_stop = np.argmin(np.abs(zs_mt-z_stop))

    # loop over each merger tree redshift
    for i in range(ind_start,ind_stop+1):
        
        # starting snapshot
        z_mt = zs_mt[i]
        z_cat = zs_cat[np.argmin(np.abs(z_mt-zs_cat))]
        print("Redshift = %.3f %.3f"%(z_mt,z_cat))

        # names of the merger tree file for this redshift
        merger_fns = list(merger_dir.glob("associations_z%4.3f.*.asdf"%z_mt))
        for counter in range(len(merger_fns)):
            merger_fns[counter] = str(merger_fns[counter])

        # slab indices and number of halos per slab
        N_halo_slabs, slabs = get_halos_per_slab(merger_fns, minified=False)
        
        # names of the light cone merger tree file for this redshift
        merger_lc_fns = list((cat_lc_dir / ("z%.3f"%z_mt)).glob("Merger_lc*.asdf"))
        for counter in range(len(merger_lc_fns)):
            merger_lc_fns[counter] = str(merger_lc_fns[counter])

        # slab indices, origins and number of halos per slab
        N_halo_slabs_lc, slabs_lc, origins_lc = get_halos_per_slab_origin(merger_lc_fns, minified=False)

        # total number of halos in this light cone redshift
        N_lc = np.sum(N_halo_slabs_lc)
        print("total number of lc halos = ", N_lc)

        Merger_lc = Table(
            {'HaloIndex':np.zeros(N_lc, dtype=np.int64),
             'InterpolatedVelocity': np.zeros(N_lc, dtype=(np.float32,3)),
             'InterpolatedPosition': np.zeros(N_lc, dtype=(np.float32,3)),
             'InterpolatedComoving': np.zeros(N_lc, dtype=np.float32)
            }
        )
        
        # initialize index for filling halo information
        start = 0; file_no = 0

        # offset for correcting halo indices
        offset = 0

        # loop over each chunk
        for k in range(n_chunks):
            # assert chunk number is correct
            assert slabs[k] == k, "the chunks are not matching"
            
            # origins for which information is available
            origins_k = origins_lc[slabs_lc == k]

            # loop over each observer origin
            for o in origins_k:
                # load the light cone arrays
                with asdf.open(cat_lc_dir / ("z%.3f"%z_mt) / ("Merger_lc%d.%02d.asdf"%(o,k)), lazy_load=True, copy_arrays=True) as f:
                    merger_lc = f['data']

                # number of halos in this file
                num = N_halo_slabs_lc[file_no]
                file_no += 1

                # the files should be congruent
                N_halo_lc = len(merger_lc['HaloIndex'])
                assert N_halo_lc == num, "file order is messed up"
                
                # translate information from this file to the complete array
                for key in Merger_lc.keys():
                    Merger_lc[key][start:start+num] = merger_lc[key][:]
                
                # add halos in this file
                start += num

            # offset all halos in given chunk
            offset += N_halo_slabs[k]


        
            
        # unpack the fields of the merger tree catalogs
        halo_ind_lc = Merger_lc['HaloIndex'][:]
        pos_interp_lc = Merger_lc['InterpolatedPosition'][:]
        vel_interp_lc = Merger_lc['InterpolatedVelocity'][:]
        chi_interp_lc = Merger_lc['InterpolatedComoving'][:]
        del Merger_lc

        # unpack halo indices
        halo_ind_lc = correct_all_inds(halo_ind_lc, N_halo_slabs, slabs, n_chunks)
            
        # catalog directory
        catdir = str(cat_dir / ("z%.3f"%z_cat))

        # load halo catalog, setting unpack to False for speed
        if save_pos:
            try:
                cat = CompaSOHaloCatalog(catdir, load_subsamples='A_halo_all', fields=fields_cat, unpack_bits = False)
                loaded_pos = True
            except:
                cat = CompaSOHaloCatalog(catdir, load_subsamples='A_halo_pid', fields=fields_cat, unpack_bits = False)
                print("Particle positions are not available for this redshift. Saving only PIDs")
                loaded_pos = False
        else:
            cat = CompaSOHaloCatalog(catdir, load_subsamples='A_halo_pid', fields=fields_cat, unpack_bits = False)
            loaded_pos = False
            
        # halo catalog
        halo_table = cat.halos[halo_ind_lc]
        header = cat.header
        N_halos = len(cat.halos)
        print("N_halos = ",N_halos)
        
        # load the particle ids
        pid = cat.subsamples['pid']
        if save_pos and loaded_pos:
            pos = cat.subsamples['pos']
        del cat

        # reindex npstart and npout for the new catalogs
        npstart = halo_table['npstartA']
        npout = halo_table['npoutA']
        if save_pos and loaded_pos:
            pid_new, pos_new, npstart_new, npout_new = reindex_pid_pos(pid, pos, npstart, npout)
            del pid, pos
        else:
            pid_new, npstart_new, npout_new = reindex_pid(pid, npstart, npout)
            del pid
        halo_table['npstartA'] = npstart_new
        halo_table['npoutA'] = npout_new
        del npstart, npout
        del npstart_new, npout_new
        
        # create particle array
        if save_pos and loaded_pos:
            pid_table = Table({'pid': np.zeros(len(pid_new), pid_new.dtype), 'pos': np.zeros(pos_new.shape, pos_new.dtype)})
            pid_table['pos'] = pos_new
        else:
            pid_table = Table({'pid': np.zeros(len(pid_new), pid_new.dtype)})
        pid_table['pid'] = pid_new
        del pid_new

        # isolate halos that did not have interpolation and get the velocity from the halo info files
        not_interp = (np.sum(np.abs(vel_interp_lc),axis=1) - 0.) < 1.e-6
        vel_interp_lc[not_interp] = halo_table['v_L2com'][not_interp]
        print("percentage not interpolated = ", 100.*np.sum(not_interp)/len(not_interp))
        
        # append new fields
        halo_table['index_halo'] = halo_ind_lc
        halo_table['pos_interp'] = pos_interp_lc
        halo_table['vel_interp'] = vel_interp_lc
        halo_table['redshift_interp'] = z_of_chi(chi_interp_lc)
        
        del halo_ind_lc, pos_interp_lc, vel_interp_lc, not_interp, chi_interp_lc
        
        # save to files
        save_asdf(halo_table, "halo_info_lc", header, cat_lc_dir / ("z%4.3f"%z_mt))
        save_asdf(pid_table, "pid_lc", header, cat_lc_dir / ("z%4.3f"%z_mt))
        
        # delete things at the end
        del pid_table
        del halo_table

        gc.collect()
            
#dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_start', help='Initial redshift where we start building the trees', type=float, default=DEFAULTS['z_start'])
    parser.add_argument('--z_stop', help='Final redshift (inclusive)', type=float, default=DEFAULTS['z_stop'])
    parser.add_argument('--compaso_parent', help='CompaSO directory', default=(DEFAULTS['compaso_parent']))
    parser.add_argument('--catalog_parent', help='Light cone catalog directory', default=(DEFAULTS['catalog_parent']))
    parser.add_argument('--merger_parent', help='Merger tree directory', default=(DEFAULTS['merger_parent']))
    parser.add_argument('--save_pos', help='Want to save positions', action='store_true')
    
    args = vars(parser.parse_args())
    main(**args)
