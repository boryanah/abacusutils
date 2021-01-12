#!/usr/bin/env python3
'''
This is the first script in the "lightcone halo" pipeline.  The goal of this script is to use merger
tree information to flag halos that intersect the lightcone and make a unique determination of which
halo catalog epoch from which to draw the halo.

Usage
-----
$ ./build_mt.py --help
'''

import sys
import glob
import time
import gc
import os
from pathlib import Path

import asdf
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from astropy.table import Table

from tools.InputFile import InputFile
from tools.merger import simple_load, get_zs_from_headers, get_halos_per_slab, get_one_header, unpack_inds, pack_inds, reorder_by_slab, mark_ineligible
from tools.aid_asdf import save_asdf
from tools.read_headers import get_lc_info
from tools.compute_dist import dist, wrapping

# these are probably just for testing; should be removed for production
DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_highbase_c021_ph000"
#DEFAULTS['sim_name'] = "AbacusSummit_highbase_c000_ph100"
#DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
#DEFAULTS['merger_parent'] = "/mnt/gosling2/bigsims/merger"
DEFAULTS['merger_parent'] = "/global/project/projectdirs/desi/cosmosim/Abacus/merger"
#DEFAULTS['catalog_parent'] = "/mnt/gosling1/boryanah/light_cone_catalog/"
DEFAULTS['catalog_parent'] = "/global/cscratch1/sd/boryanah/light_cone_catalog/"
DEFAULTS['z_start'] = 0.5
DEFAULTS['z_stop'] = 0.8
CONSTANTS = {'c': 299792.458}  # km/s, speed of light

def correct_inds(halo_ids, N_halos_slabs, slabs, inds_fn):
    '''
    Reorder indices for given halo index array with 
    corresponding n halos and slabs for its time epoch
    '''

    # number of halos in the loaded chunks
    N_halos_load = np.array([N_halos_slabs[i] for i in inds_fn])
    
    # unpack slab and index for each halo
    slab_ids, ids = unpack_inds(halo_ids)

    # total number of halos in the slabs that we have loaded
    N_halos = np.sum(N_halos_load)
    offsets = np.zeros(len(inds_fn), dtype=int)
    offsets[1:] = np.cumsum(N_halos_load)[:-1]
    
    # determine if unpacking halos for only one file (Merger_this['HaloIndex']) -- no need to offset 
    if len(inds_fn) == 1: return ids

    # select the halos belonging to given slab
    for i, ind_fn in enumerate(inds_fn):
        select = np.where(slab_ids == slabs[ind_fn])[0]
        ids[select] += offsets[i]

    return ids

def get_mt_info(fn_load, fields, minified):
    '''
    Load merger tree and progenitors information
    '''

    # loading merger tree info
    mt_data = simple_load(fn_load, fields=fields)
        
    # turn data into astropy table
    Merger = mt_data['merger']
    Merger.add_column(np.empty(len(Merger['HaloIndex']), dtype=np.float32), copy=False, name='ComovingDistance')

    # if loading all progenitors
    if "Progenitors" in fields:
        num_progs = Merger["NumProgenitors"]
        # get an array with the starting indices of the progenitors array
        start_progs = np.zeros(len(num_progs), dtype=int)
        start_progs[1:] = num_progs.cumsum()[:-1]
        Merger.add_column(start_progs, name='StartProgenitors', copy=False)
        
    return mt_data

def solve_crossing(r1, r2, pos1, pos2, chi1, chi2, Lbox, origin, extra=4.):
    '''
    Solve when the crossing of the light cones occurs and the
    interpolated position and velocity
    '''

    # periodic wrapping of the positions of the particles
    r1, r2, pos1, pos2 = wrapping(r1, r2, pos1, pos2, chi1, chi2, Lbox, origin)
    
    # assert wrapping worked
    assert np.all(((r2 <= chi1) & (r2 > chi2)) | ((r1 <= chi1) & (r1 > chi2))), "Wrapping didn't work"
    
    # solve for chi_star, where chi(z) = eta(z=0)-eta(z)
    # equation is r1+(chi1-chi)/(chi1-chi2)*(r2-r1) = chi, with solution:
    chi_star = (r1 * (chi1 - chi2) + chi1 * (r2 - r1)) / ((chi1 - chi2) + (r2 - r1))

    # get interpolated positions of the halos
    v_avg = (pos2 - pos1) / (chi1 - chi2)
    pos_star = pos1 + v_avg * (chi1 - chi_star[:, None])

    # enforce boundary conditions by periodic wrapping
    pos_star[pos_star >= Lbox/2.] = pos_star[pos_star >= Lbox/2.] - Lbox
    pos_star[pos_star < -Lbox/2.] = pos_star[pos_star < -Lbox/2.] + Lbox
    
    # interpolated velocity [km/s]
    vel_star = v_avg * CONSTANTS['c']  # vel1+a_avg*(chi1-chi_star)

    # x is comoving position; r = x a; dr = a dx; r = a x; dr = da x + dx a; a/H
    #vel_star = dx/deta = dr/dt − H(t)r -> r is real space coord dr/dt = vel_star + a H(t) x 
    # 'Htime', 'HubbleNow', 'HubbleTimeGyr', 'HubbleTimeHGyr'
    
    # mark True if closer to chi2 (this snapshot)
    bool_star = np.abs(chi1 - chi_star) > np.abs(chi2 - chi_star)

    # condition to check whether halo in this light cone band
    assert np.all(((chi_star <= chi1+extra) & (chi_star > chi2-extra))), "Solution is out of bounds"

    return chi_star, pos_star, vel_star, bool_star
    
def offset_pos(pos,ind_origin,all_origins):
    '''
    Offset the interpolated positions to create continuous light cones
    '''

    # location of initial observer
    first_observer = all_origins[0]
    current_observer = all_origins[ind_origin]
    offset = (first_observer-current_observer)
    pos += offset
    return pos

def main(sim_name, z_start, z_stop, merger_parent, catalog_parent, resume=False, plot=False):
    '''
    Main function.
    The algorithm: for each merger tree epoch, for 
    each superslab, for each light cone origin,
    compute the intersection of the light cone with
    each halo, using the interpolated position
    to the previous merger epoch (and possibly a 
    velocity correction).  If the intersection is
    between the current and previous merger epochs, 
    then record the closer one as that halo's
    epoch and mark its progenitors as ineligible.
    Will need one padding superslab in the previous
    merger epoch.  Can process in a rolling fashion.
    '''
    
    # turn directories into Paths
    merger_parent = Path(merger_parent)
    catalog_parent = Path(catalog_parent)
    merger_dir = merger_parent / sim_name
    header = get_one_header(merger_dir)
    
    # simulation parameters
    Lbox = header['BoxSize']
    # location of the LC origins in Mpc/h
    origins = np.array(header['LightConeOrigins']).reshape(-1,3)

    
    # just for testing with highbase. remove!
    if 'highbase' in sim_name:
        origins /= 2.
    
    
    # directory where we save the final outputs
    cat_lc_dir = catalog_parent / sim_name / "halos_light_cones/"
    os.makedirs(cat_lc_dir, exist_ok=True)

    # directory where we save the current state if we want to resume
    os.makedirs(cat_lc_dir / "tmp", exist_ok=True)
    with open(cat_lc_dir / "tmp" / "tmp.log", "a") as f:
        f.writelines(["# Starting light cone catalog construction in simulation %s \n"%sim_name])
    
    # all redshifts, steps and comoving distances of light cones files; high z to low z
    # remove presaving after testing done (or make sure presaved can be matched with simulation)
    if not os.path.exists(Path("data_headers") / sim_name / "coord_dist.npy") or not os.path.exists(Path("data_headers") / sim_name / "redshifts.npy"):
        zs_all, steps, chis_all = get_lc_info("all_headers")
        os.makedirs(Path("data_headers") / sim_name, exist_ok=True)
        np.save(Path("data_headers") / sim_name / "redshifts.npy", zs_all)
        np.save(Path("data_headers") / sim_name / "steps.npy", steps_all)
        np.save(Path("data_headers") / sim_name / "coord_dist.npy", chis_all)
    zs_all = np.load(Path("data_headers") / sim_name / "redshifts.npy")
    chis_all = np.load(Path("data_headers") / sim_name / "coord_dist.npy")
    zs_all[-1] = float("%.1f" % zs_all[-1])  # LHG: I guess this is trying to match up to some filename or something?

    # get functions relating chi and z
    chi_of_z = interp1d(zs_all, chis_all)
    z_of_chi = interp1d(chis_all, zs_all)
    
    # if merger tree redshift information has been saved, load it (if not, save it)
    if not os.path.exists(Path("data_mt") / sim_name / "zs_mt.npy"):
        # all merger tree snapshots and corresponding redshifts
        snaps_mt = sorted(merger_dir.glob("associations_z*.0.asdf"))
        zs_mt = get_zs_from_headers(snaps_mt)
        os.makedirs(Path("data_mt") / sim_name, exist_ok=True)
        np.save(Path("data_mt") / sim_name / "zs_mt.npy", zs_mt)
    zs_mt = np.load(Path("data_mt") / sim_name / "zs_mt.npy")

    # number of chunks
    n_chunks = len(list(merger_dir.glob("associations_z%4.3f.*.asdf"%zs_mt[0])))
    print("number of chunks = ",n_chunks)

    # starting and finishing redshift indices indices
    ind_start = np.argmin(np.abs(zs_mt - z_start))
    ind_stop = np.argmin(np.abs(zs_mt - z_stop))

    if resume:
        # if user wants to resume from previous state, create padded array for marking whether chunk has been loaded
        resume_flags = np.ones((n_chunks, origins.shape[1]), dtype=bool)
        
        # previous redshift, distance between shells
        infile = InputFile(cat_lc_dir / "tmp" / "tmp.log")
        z_this_tmp = infile.z_prev
        delta_chi_old = infile.delta_chi
        chunk = infile.super_slab
        assert (np.abs(n_chunks-1 - chunk) < 1.0e-6), "Your recorded state did not complete all chunks, can't resume from old"
        assert (np.abs(zs_mt[ind_start] - z_this_tmp) < 1.0e-6), "Your recorded state is not for the correct redshift, can't resume from old"
        with open(cat_lc_dir / "tmp" / "tmp.log", "a") as f:
            f.writelines(["# Resuming from redshift z = %4.3f \n"%z_this_tmp])
    else:
        # delete the exisiting temporary files
        tmp_files = list((cat_lc_dir / "tmp").glob("*"))
        for i in range(len(tmp_files)):
            os.unlink(str(tmp_files[i]))
        resume_flags = np.zeros((n_chunks, origins.shape[0]), dtype=bool)

    # fields to extract from the merger trees
    fields_mt = ['HaloIndex','HaloMass','Position','MainProgenitor','Progenitors','NumProgenitors']
    # lighter version
    #fields_mt = ['HaloIndex', 'Position', 'MainProgenitor']

    # redshift of closest point on wall between original and copied box
    z1 = z_of_chi(0.5 * Lbox - origins[0][0])
    # redshift of closest point where all three boxes touch
    z2 = z_of_chi((0.5*Lbox-origins[0][0])*np.sqrt(2))
    # furthest point where all three boxes touch;
    z3 = z_of_chi((0.5 * Lbox - origins[0][0]) * np.sqrt(3))

    # initialize difference between the conformal time of last two shells
    delta_chi_old = 0.0
    
    for i in range(ind_start, ind_stop + 1):

        # this snapshot redshift and the previous
        z_this = zs_mt[i]
        z_prev = zs_mt[i + 1]
        print("redshift of this and the previous snapshot = ", z_this, z_prev)

        # coordinate distance of the light cone at this redshift and the previous
        assert z_this >= np.min(zs_all), "You need to set starting redshift to the smallest value of the merger tree"
        chi_this = chi_of_z(z_this)
        chi_prev = chi_of_z(z_prev)
        delta_chi = chi_prev - chi_this
        print("comoving distance between this and previous snapshot = ", delta_chi)

        # read merger trees file names at this and previous snapshot from minified version 
        fns_this = merger_dir.glob(f'associations_z{z_this:4.3f}.*.asdf.minified')
        fns_prev = merger_dir.glob(f'associations_z{z_prev:4.3f}.*.asdf.minified')
        fns_this = list(fns_this)
        fns_prev = list(fns_prev)
        minified = True

        # if minified files not available,  load the regular files
        if len(list(fns_this)) == 0 or len(list(fns_prev)) == 0:
            fns_this = merger_dir.glob(f'associations_z{z_this:4.3f}.*.asdf')
            fns_prev = merger_dir.glob(f'associations_z{z_prev:4.3f}.*.asdf')
            fns_this = list(fns_this)
            fns_prev = list(fns_prev)
            minified = False

        # turn file names into strings
        for counter in range(len(fns_this)):
            fns_this[counter] = str(fns_this[counter])
            fns_prev[counter] = str(fns_prev[counter])

        # number of merger tree files
        print("number of files = ", len(fns_this), len(fns_prev))
        assert n_chunks == len(fns_this) and n_chunks == len(fns_prev), "Incomplete merger tree files"
        # reorder file names by super slab number
        fns_this = reorder_by_slab(fns_this, minified)
        fns_prev = reorder_by_slab(fns_prev, minified)

        # get number of halos in each slab and number of slabs
        N_halos_slabs_this, slabs_this = get_halos_per_slab(fns_this, minified)
        N_halos_slabs_prev, slabs_prev = get_halos_per_slab(fns_prev, minified)
        
        # maybe we want to support resuming from arbitrary superslab
        first_ss = 0 

        # We're going to be loading slabs in a rolling fashion:
        # reading the "high" slab at the leading edge, discarding the trailing "low" slab
        # and moving the mid to low. But first we need to read all three to prime the queue
        mt_prev = {}  # indexed by slab num
        mt_prev[(first_ss-1)%n_chunks] = get_mt_info(fns_prev[(first_ss-1)%n_chunks], fields=fields_mt, minified=minified)
        mt_prev[first_ss] = get_mt_info(fns_prev[first_ss], fields=fields_mt, minified=minified)

        # loop over each chunk
        for k in range(first_ss,n_chunks):
            # starting and finishing superslab chunks
            klow = (k-1)%n_chunks
            khigh = (k+1)%n_chunks
            
            # slide down by one
            if (klow-1)%n_chunks in mt_prev:
                del mt_prev[(klow-1)%n_chunks]
            mt_prev[khigh] = get_mt_info(fns_prev[khigh], fields_mt, minified)
            
            # starting and finishing superslab chunks
            inds_fn_this = [k]
            inds_fn_prev = np.array([klow,k,khigh],dtype=int)
            print("chunks loaded in this and previous redshifts = ",inds_fn_this, inds_fn_prev)
            
            # get merger tree data for this snapshot and for the previous one
            mt_data_this = get_mt_info(fns_this[k], fields_mt, minified)
            
            # number of halos in this step and previous step; this depends on the number of files requested
            N_halos_this = np.sum(N_halos_slabs_this[inds_fn_this])
            N_halos_prev = np.sum(N_halos_slabs_prev[inds_fn_prev])
            print("N_halos_this = ", N_halos_this)
            print("N_halos_prev = ", N_halos_prev)

            # organize data into astropy tables
            Merger_this = mt_data_this['merger']
            cols = {col:np.empty(N_halos_prev, dtype=(Merger_this[col].dtype, Merger_this[col].shape[1] if 'Position' in col else 1)) for col in Merger_this.keys()}
            Merger_prev = Table(cols, copy=False)
            offset = 0
            for key in mt_prev.keys():
                size_chunk = len(mt_prev[key]['merger']['HaloIndex'])
                Merger_prev[offset:offset+size_chunk] = mt_prev[key]['merger'][:]
                offset += size_chunk
                
            # mask where no merger tree info is available (because we don'to need to solve for eta star for those)
            noinfo_this = Merger_this['MainProgenitor'] <= 0
            info_this = Merger_this['MainProgenitor'] > 0
            
            # print percentage where no information is available or halo not eligible
            print("percentage no info = ", np.sum(noinfo_this) / len(noinfo_this) * 100.0)

            # no info is denoted by 0 or -999 (or regular if ineligible), but -999 messes with unpacking, so we set it to 0
            Merger_this['MainProgenitor'][noinfo_this] = 0

            # rework the main progenitor and halo indices to return in proper order
            Merger_this['HaloIndex'] = correct_inds(
                Merger_this['HaloIndex'],
                N_halos_slabs_this,
                slabs_this,
                inds_fn_this,
            )
            Merger_this['MainProgenitor'] = correct_inds(
                Merger_this['MainProgenitor'],
                N_halos_slabs_prev,
                slabs_prev,
                inds_fn_prev,
            )
            Merger_prev['HaloIndex'] = correct_inds(
                Merger_prev['HaloIndex'],
                N_halos_slabs_prev,
                slabs_prev,
                inds_fn_prev,
            )
            
            # loop over all origins
            for o in range(len(origins)):
                
                # location of the observer
                origin = origins[o]
                
                # comoving distance to observer
                Merger_this['ComovingDistance'][:] = dist(Merger_this['Position'], origin)
                Merger_prev['ComovingDistance'][:] = dist(Merger_prev['Position'], origin)
                
                # merger tree data of main progenitor halos corresponding to the halos in current snapshot
                Merger_prev_main_this = Merger_prev[Merger_this['MainProgenitor']].copy()
                
                # if eligible, can be selected for light cone redshift catalog;
                if i != ind_start or resume_flags[k, o]:
                    # dealing with the fact that these files may not exist for all origins and all chunks
                    if os.path.exists(cat_lc_dir / "tmp" / ("eligibility_prev_z%4.3f_lc%d.%02d.npy"%(z_this, o, k))):
                        eligibility_this = np.load(cat_lc_dir / "tmp" / ("eligibility_prev_z%4.3f_lc%d.%02d.npy"%(z_this, o, k)))
                    else:
                        eligibility_this = np.ones(N_halos_this, dtype=bool)
                else:
                    eligibility_this = np.ones(N_halos_this, dtype=bool)
                
                # for a newly opened redshift, everyone is eligible to be part of the light cone catalog
                eligibility_prev = np.ones(N_halos_prev, dtype=bool)

                # mask for eligible halos for light cone origin with and without information
                mask_noinfo_this = noinfo_this & eligibility_this
                mask_info_this = info_this & eligibility_this

                # halos that have merger tree information
                Merger_this_info = Merger_this[mask_info_this].copy()
                Merger_prev_main_this_info = Merger_prev_main_this[mask_info_this]
                
                # halos that don't have merger tree information
                Merger_this_noinfo = Merger_this[mask_noinfo_this].copy()
                
                # select objects that are crossing the light cones
                cond_1 = ((Merger_this_info['ComovingDistance'] > chi_this) & (Merger_this_info['ComovingDistance'] <= chi_prev))
                cond_2 = ((Merger_prev_main_this_info['ComovingDistance'] > chi_this) & (Merger_prev_main_this_info['ComovingDistance'] <= chi_prev))
                mask_lc_this_info = cond_1 | cond_2
                del cond_1, cond_2

                # for hals that have no merger tree information, we simply take their current position
                mask_lc_this_noinfo = (
                    (Merger_this_noinfo['ComovingDistance'] > chi_this - delta_chi_old / 2.0)
                    & (Merger_this_noinfo['ComovingDistance'] <= chi_this + delta_chi / 2.0)
                )

                # spare the computer the effort and avert empty array errors
                # TODO: perhaps revise, as sometimes we might have no halos in
                # noinfo but some in info and vice versa
                if np.sum(mask_lc_this_info) == 0 or np.sum(mask_lc_this_noinfo) == 0: continue

                # percentage of objects that are part of this or previous snapshot
                print(
                    "percentage of halos in light cone %d with and without progenitor info = "%o,
                    np.sum(mask_lc_this_info) / len(mask_lc_this_info) * 100.0,
                    np.sum(mask_lc_this_noinfo) / len(mask_lc_this_noinfo) * 100.0,
                )

                # select halos with mt info that have had a light cone crossing
                Merger_this_info_lc = Merger_this_info[mask_lc_this_info]
                Merger_prev_main_this_info_lc = Merger_prev_main_this_info[mask_lc_this_info]

                if plot:
                    x_min = -Lbox/2.+k*(Lbox/n_chunks)
                    x_max = x_min+(Lbox/n_chunks)

                    x = Merger_this_info_lc['Position'][:,0]
                    choice = (x > x_min) & (x < x_max)
                    
                    y = Merger_this_info_lc['Position'][choice,1]
                    z = Merger_this_info_lc['Position'][choice,2]
                    
                    plt.figure(1)
                    plt.scatter(y, z, color='dodgerblue', s=0.1, label='current objects')

                    plt.legend()
                    plt.axis('equal')
                    plt.savefig('this_%d_%d_%d.png'%(i,k,o))
                    
                    x = Merger_prev_main_this_info_lc['Position'][:,0]
                    
                    choice = (x > x_min) & (x < x_max)

                    y = Merger_prev_main_this_info_lc['Position'][choice,1]
                    z = Merger_prev_main_this_info_lc['Position'][choice,2]
                    
                    plt.figure(2)
                    plt.scatter(y, z, color='orangered', s=0.1, label='main progenitor')

                    plt.legend()
                    plt.axis('equal')
                    plt.savefig('prev_%d_%d_%d.png'%(i,k,o))
                    plt.close()
                    
                # select halos without mt info that have had a light cone crossing
                Merger_this_noinfo_lc = Merger_this_noinfo[mask_lc_this_noinfo]

                # add columns for new interpolated position, velocity and comoving distance
                Merger_this_info_lc.add_column('InterpolatedPosition',copy=False)
                Merger_this_info_lc.add_column('InterpolatedVelocity',copy=False)
                Merger_this_info_lc.add_column('InterpolatedComoving',copy=False)

                # get chi star where lc crosses halo trajectory; bool is False where closer to previous
                (
                    Merger_this_info_lc['InterpolatedComoving'],
                    Merger_this_info_lc['InterpolatedPosition'],
                    Merger_this_info_lc['InterpolatedVelocity'],
                    bool_star_this_info_lc,
                ) = solve_crossing(
                    Merger_prev_main_this_info_lc['ComovingDistance'],
                    Merger_this_info_lc['ComovingDistance'],
                    Merger_prev_main_this_info_lc['Position'],
                    Merger_this_info_lc['Position'],
                    chi_prev,
                    chi_this,
                    Lbox,
                    origin
                )

                # number of objects in this light cone
                N_this_star_lc = np.sum(bool_star_this_info_lc)
                N_this_noinfo_lc = np.sum(mask_lc_this_noinfo)

                if i != ind_start or resume_flags[k, o]:
                    # check if we have information about this light cone origin, chunk and epoch
                    if os.path.exists(cat_lc_dir / "tmp" / ("Merger_next_z%4.3f_lc%d.%02d.asdf"%(z_this,o,k))):
                        
                        # load leftover halos from previously loaded redshift
                        with asdf.open(cat_lc_dir / "tmp" / ("Merger_next_z%4.3f_lc%d.%02d.asdf"%(z_this,o,k)), lazy_load=True, copy_arrays=True) as f:
                            Merger_next = f['data']

                        # adding contributions from the previously loaded redshift
                        N_next_lc = len(Merger_next['HaloIndex'])
                    else:
                        N_next_lc = 0
                else:
                    N_next_lc = 0

                # total number of halos belonging to this light cone superslab and origin
                N_lc = N_this_star_lc + N_this_noinfo_lc + N_next_lc
                print("in this snapshot: interpolated, no info, next, total = ", N_this_star_lc * 100.0 / N_lc, N_this_noinfo_lc * 100.0 / N_lc, N_next_lc * 100.0 / N_lc, N_lc)
                
                # save those arrays
                Merger_lc = Table(
                    {'HaloIndex':np.zeros(N_lc, dtype=Merger_this_info_lc['HaloIndex'].dtype),
                     'InterpolatedVelocity': np.zeros(N_lc, dtype=(np.float32,3)),
                     'InterpolatedPosition': np.zeros(N_lc, dtype=(np.float32,3)),
                     'InterpolatedComoving': np.zeros(N_lc, dtype=np.float32)
                    }
                )

                # record interpolated position and velocity for those with info belonging to current redshift
                Merger_lc['InterpolatedPosition'][:N_this_star_lc] = Merger_this_info_lc['InterpolatedPosition'][bool_star_this_info_lc]
                Merger_lc['InterpolatedVelocity'][:N_this_star_lc] = Merger_this_info_lc['InterpolatedVelocity'][bool_star_this_info_lc]
                Merger_lc['InterpolatedComoving'][:N_this_star_lc] = Merger_this_info_lc['InterpolatedComoving'][bool_star_this_info_lc]
                Merger_lc['HaloIndex'][:N_this_star_lc] = Merger_this_info_lc['HaloIndex'][bool_star_this_info_lc]

                # record interpolated position and velocity of the halos in the light cone without progenitor information
                Merger_lc['InterpolatedPosition'][N_this_star_lc:N_this_star_lc+N_this_noinfo_lc] = Merger_this_noinfo_lc['Position']
                Merger_lc['InterpolatedVelocity'][N_this_star_lc:N_this_star_lc+N_this_noinfo_lc] = np.zeros_like(Merger_this_noinfo_lc['Position'])
                Merger_lc['InterpolatedComoving'][N_this_star_lc:N_this_star_lc+N_this_noinfo_lc] = np.ones(Merger_this_noinfo_lc['Position'].shape[0])*chi_this
                Merger_lc['HaloIndex'][N_this_star_lc:N_this_star_lc+N_this_noinfo_lc] = Merger_this_noinfo_lc['HaloIndex']
                del Merger_this_noinfo_lc

                # pack halo indices for all halos but those in Merger_next
                Merger_lc['HaloIndex'][:(N_this_star_lc + N_this_noinfo_lc)] = pack_inds(Merger_lc['HaloIndex'][:(N_this_star_lc + N_this_noinfo_lc)], k)
                
                # record information from previously loaded redshift that was postponed
                if i != ind_start or resume_flags[k, o]:
                    if N_next_lc != 0:
                        Merger_lc['InterpolatedPosition'][-N_next_lc:] = Merger_next['InterpolatedPosition'][:]
                        Merger_lc['InterpolatedVelocity'][-N_next_lc:] = Merger_next['InterpolatedVelocity'][:]
                        Merger_lc['InterpolatedComoving'][-N_next_lc:] = Merger_next['InterpolatedComoving'][:]
                        Merger_lc['HaloIndex'][-N_next_lc:] = Merger_next['HaloIndex'][:]
                        del Merger_next
                    resume_flags[k, o] = False
                
                # offset position to make light cone continuous
                Merger_lc['InterpolatedPosition'] = offset_pos(Merger_lc['InterpolatedPosition'], ind_origin = o, all_origins=origins)

                # create directory for this redshift
                os.makedirs(cat_lc_dir / ("z%.3f"%z_this), exist_ok=True)

                # write table with interpolated information
                save_asdf(Merger_lc, ("Merger_lc%d.%02d"%(o,k)), header, cat_lc_dir / ("z%.3f"%z_this))
                
                # TODO: Need to make sure no bugs with eligibility
                # version 1: only the main progenitor is marked ineligible
                # if halo belongs to this redshift catalog or the previous redshift catalog;
                eligibility_prev[Merger_prev_main_this_info_lc['HaloIndex']] = False

                
                # version 2: all progenitors of halos belonging to this redshift catalog are marked ineligible 
                # run version 1 AND 2 to mark ineligible Merger_next objects to avoid multiple entries
                # Note that some progenitor indices are zeros;
                # For best result perhaps combine Progs with MainProgs 
                if "Progenitors" in fields_mt:
                    nums = Merger_this_info_lc['NumProgenitors'][bool_star_this_info_lc]
                    starts = Merger_this_info_lc['StartProgenitors'][bool_star_this_info_lc]
                    # for testing purposes (remove in final version)
                    main_progs = Merger_this_info_lc['MainProgenitor'][bool_star_this_info_lc]
                    progs = mt_data_this['progenitors']['Progenitors']
                    halo_ind_prev = Merger_prev['HaloIndex']

                    N_halos_load = np.array([N_halos_slabs_prev[i] for i in inds_fn_prev])
                    slabs_prev_load = np.array([slabs_prev[i] for i in slabs_prev[inds_fn_prev]],dtype=np.int64)
                    offsets = np.zeros(len(inds_fn_prev), dtype=np.int64)
                    offsets[1:] = np.cumsum(N_halos_load)[:-1]
                    
                    mark_ineligible(nums, starts, main_progs, progs, halo_ind_prev, eligibility_prev, offsets, slabs_prev_load)
                                    
                # information to keep for next redshift considered
                N_next = np.sum(~bool_star_this_info_lc)
                Merger_next = Table(
                    {'HaloIndex': np.zeros(N_next, dtype=Merger_lc['HaloIndex'].dtype),
                     'InterpolatedVelocity': np.zeros(N_next, dtype=(np.float32,3)),
                     'InterpolatedPosition': np.zeros(N_next, dtype=(np.float32,3)),
                     'InterpolatedComoving': np.zeros(N_next, dtype=np.float32)
                    }
                )
                Merger_next['HaloIndex'][:] = Merger_prev_main_this_info_lc['HaloIndex'][~bool_star_this_info_lc]
                Merger_next['InterpolatedVelocity'][:] = Merger_this_info_lc['InterpolatedVelocity'][~bool_star_this_info_lc]
                Merger_next['InterpolatedPosition'][:] = Merger_this_info_lc['InterpolatedPosition'][~bool_star_this_info_lc]
                Merger_next['InterpolatedComoving'][:] = Merger_this_info_lc['InterpolatedComoving'][~bool_star_this_info_lc]
                del Merger_this_info_lc, Merger_prev_main_this_info_lc
                
                if plot:
                    # select the halos in the light cones
                    pos_choice = Merger_lc['InterpolatedPosition']

                    # selecting thin slab
                    pos_x_min = -Lbox/2.+k*(Lbox/n_chunks)
                    pos_x_max = x_min+(Lbox/n_chunks)

                    ijk = 0
                    choice = (pos_choice[:, ijk] >= pos_x_min) & (pos_choice[:, ijk] < pos_x_max)

                    circle_this = plt.Circle(
                        (origins[0][1], origins[0][2]), radius=chi_this, color="g", fill=False
                    )
                    circle_prev = plt.Circle(
                        (origins[0][1], origins[0][2]), radius=chi_prev, color="r", fill=False
                    )

                    # clear things for fresh plot
                    ax = plt.gca()
                    ax.cla()

                    # plot particles
                    ax.scatter(pos_choice[choice, 1], pos_choice[choice, 2], s=0.1, alpha=1., color="dodgerblue")

                    # circles for in and prev
                    ax.add_artist(circle_this)
                    ax.add_artist(circle_prev)
                    plt.xlabel([-Lbox/2., Lbox*1.5])
                    plt.ylabel([-Lbox/2., Lbox*1.5])
                    plt.axis("equal")
                    plt.savefig('interp_%d_%d_%d.png'%(i,k,o))
                    #plt.show()
                    plt.close()
                    
                gc.collect()

                # pack halo indices for the halos in Merger_next
                offset = 0
                for idx in inds_fn_prev:
                    print("k, idx = ",k,idx)
                    choice_idx = (offset <= Merger_next['HaloIndex'][:]) & (Merger_next['HaloIndex'][:] < offset+N_halos_slabs_prev[idx])
                    Merger_next['HaloIndex'][choice_idx] = pack_inds(Merger_next['HaloIndex'][choice_idx]-offset, idx)
                    offset += N_halos_slabs_prev[idx]
                
                
                # split the eligibility array over three files for the three chunks it's made up of
                offset = 0
                for idx in inds_fn_prev:
                    eligibility_prev_idx = eligibility_prev[offset:offset+N_halos_slabs_prev[idx]]
                    # combine current information with previously existing
                    if os.path.exists(cat_lc_dir / "tmp" / ("eligibility_prev_z%4.3f_lc%d.%02d.npy"%(z_prev, o, idx))):
                        eligibility_prev_old = np.load(cat_lc_dir / "tmp" / ("eligibility_prev_z%4.3f_lc%d.%02d.npy"%(z_prev, o, idx)))
                        eligibility_prev_idx = eligibility_prev_old & eligibility_prev_idx
                        print("Exists!")
                    else:
                        print("Doesn't exist")
                    np.save(cat_lc_dir / "tmp" / ("eligibility_prev_z%4.3f_lc%d.%02d.npy"%(z_prev, o, idx)), eligibility_prev_idx)
                    offset += N_halos_slabs_prev[idx]

                # write as table the information about halos that are part of next loaded redshift
                save_asdf(Merger_next, ("Merger_next_z%4.3f_lc%d.%02d"%(z_prev, o, k)), header, cat_lc_dir / "tmp")

                # save redshift of catalog that is next to load and difference in comoving between this and prev
                with open(cat_lc_dir / "tmp" / "tmp.log", "a") as f:
                    f.writelines(["# Next iteration: \n", "z_prev = %.8f \n"%z_prev, "delta_chi = %.8f \n"%delta_chi, "light_cone = %d \n"%o, "super_slab = %d \n"%k])
                
            del Merger_this, Merger_prev

        # update values for difference in comoving distance
        delta_chi_old = delta_chi

# dict_keys(['HaloIndex', 'HaloMass', 'HaloVmax', 'IsAssociated', 'IsPotentialSplit', 'MainProgenitor', 'MainProgenitorFrac', 'MainProgenitorPrec', 'MainProgenitorPrecFrac', 'NumProgenitors', 'Position', 'Progenitors'])
    

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_start', help='Initial redshift where we start building the trees', type=float, default=DEFAULTS['z_start'])
    parser.add_argument('--z_stop', help='Final redshift (inclusive)', type=float, default=DEFAULTS['z_stop'])
    parser.add_argument('--merger_parent', help='Merger tree directory', default=(DEFAULTS['merger_parent']))
    parser.add_argument('--catalog_parent', help='Light cone catalog directory', default=(DEFAULTS['catalog_parent']))
    parser.add_argument('--resume', help='Resume the calculation from the checkpoint on disk', action='store_true')
    parser.add_argument('--plot', help='Want to show plots', action='store_true')
    
    args = vars(parser.parse_args())
    main(**args)
