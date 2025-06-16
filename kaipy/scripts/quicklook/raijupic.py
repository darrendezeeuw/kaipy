#!/usr/bin/env python

# Import standard modules.
import os
import argparse
import sys
import datetime

# Import supplemental modules.
import h5py as h5
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmasher as cmr
import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

# Import project-specific modules.
import kaipy.kaiTools as kt
import kaipy.kaiViz as kv
import kaipy.kaiH5 as kh5
import kaipy.raiju.raijuUtils as ru
import kaipy.raiju.raijuViz as rv


# Program constants and defaults
# Program description.
description = """Creates simple multi-panel RAIJU figure MAGE run.
"""

# Default identifier for results to read.
default_runid = "msphere"
default_vidOut = "qkRaijuVid"
default_step = -1
default_ut = -1
default_nStride = 10
domain_opts = ["ACTIVE", "BUFFER"]
default_domain="ACTIVE"
default_ncpus = 1
default_start = 1

def get_bVol_dipole(f5):
    colats = ru.getVar(f5,'X')  # [rad]
    R = f5['Grid']['ShellGrid'].attrs['radius']  # [Rp]
    Leq = R/( np.sin(colats[:,0])*np.sin(colats[:,0]) )
    Ni, Nj = colats.shape
    bVol_dipole_1D = np.zeros(Ni)
    for i in range(Ni):
        bVol_dipole_1D[i] = kt.L_to_bVol(Leq[i])
    bVol_dipole_2D = np.broadcast_to(bVol_dipole_1D[:,np.newaxis], (Ni,Nj))
    return bVol_dipole_2D

def parse_command_line():
    """

    """

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-d", type=str, metavar="directory", default=os.getcwd(),
        help="Directory containing data to read (default: %(default)s)"
    )
    parser.add_argument(
        "-id", type=str, metavar="runid", default=default_runid,
        help="Run ID of data (default: %(default)s)"
    )
    parser.add_argument(
        "-n", "-ne", type=int, metavar="step", default=default_step,
        help="Time slice to plot (last slice if video; default: %(default)s)"
    )
    parser.add_argument(
        "-ns", type=int, metavar="startstep", default=default_start,
        help="First slice of video; default: %(default)s)"
    )
    parser.add_argument(
        "-dn", type=int, metavar="stride", default=default_nStride,
        help="Step stride in case of vid (default: %(default)s)"
    )
    parser.add_argument(
        "-uts", type=str, metavar=kt.isotfmt, default=default_ut,
        help="First UT of video (default: %(default)s)"
    )
    parser.add_argument(
        "-ut", "-ute", type=str, metavar=kt.isotfmt, default=default_ut,
        help="UT to plot (overrides -n if provided; last UT if video; default: %(default)s)"
    )
    parser.add_argument(
        '-domain', choices=domain_opts, default=default_domain,
        help="Domain to include (always includes active) (default: %(default)s)"
    )
    parser.add_argument(
        '-diff',action='store_true', default=False,
        help="Do difference of some quantities from dipole (default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        '-vid', action='store_true', default=False,
        help="Make a video and store in mixVid directory (default: %(default)s)"
    )
    parser.add_argument(
        "-vidOut", type=str, metavar="dir", default=default_vidOut,
        help="Output directory if doing video (default: %(default)s)"
    )
    parser.add_argument(
        '-overwrite', action='store_true', default=False,
        help="Overwrite existing vid files (default: %(default)s)"
    )
    parser.add_argument(
        '-nohash', action='store_true', default=False,
        help="Don't display branch/hash info (default: %(default)s)"
    )
    parser.add_argument(
        '--ncpus', type=int, default=default_ncpus,
        help="Number of threads for parallel vid (default: %(default)s)"
    )

    args = parser.parse_args()
    config = {
        "indir"      : args.d,
        "id"         : args.id,
        "nPlt"       : args.n,
        "nStride"    : args.dn,
        "utPlt"      : args.ut,
        "domain"     : args.domain,
        "doDiff"     : args.diff,
        "doVerbose"  : args.verbose,
        "doVid"      : args.vid,
        "vidOut"     : args.vidOut,
        "doOverwrite": args.overwrite,
        "noHash"     : args.nohash,
        "nCPU"       : args.ncpus,
        "nStart"     : args.ns,
    }
    return config

def plot_frame(pairs,raiI: ru.RAIJUInfo, config):

    iPlt, nPlt = pairs
    f5 = h5.File(raiI.fname,'r')

    fgSize = (13,7)
    eqBnds = [-15,10,-10,10]
    norm_press = kv.genNorm(05e-2,50,doLog=False)
    norm_den   = kv.genNorm(0,10, doLog=False)
    norm_bvol = kv.genNorm(1e-4,1e0,doLog=True)
    norm_diff_bvol = kv.genNorm(-1e2,1e2,doSymLog=True)
    norm_ent  = kv.genNorm(1e-4,0.5,doLog=True)
    norm_wimag = kv.genNorm(0,1,doLog=False)
    norm_bMin = kv.genNorm(0.1,1000,doLog=True)
    norm_Tb = kv.genNorm(0,180,doLog=False)

    cmap_diff = 'RdBu_r'
    cmap_press = cmr.lilac
    cmap_bvol = cmr.dusk
    cmap_ent = 'gist_earth'

    tickIn_pad = -15

    nRows = 4
    nCols = 4
    fig = plt.figure(figsize=fgSize)
    gs = gridspec.GridSpec(nRows,nCols,height_ratios=[0.1,1,1,0.1],hspace=0.2,wspace=0.18)
    axs = []
    for iRow in range(nRows):
        axRow = []
        for iCol in range(nCols):
            axRow.append(fig.add_subplot(gs[iCol,iRow]))
        axs.append(axRow)

    axCol = axs[0]
    kv.genCB(axCol[0], norm_press, "Proton Pressure [nPa]",cmap_press)
    axCol[0].xaxis.set_ticks_position("top")
    axCol[0].xaxis.set_label_position("top")
    kv.genCB(axCol[3], norm_press, "Electron Pressure [nPa]",cmap_press)
    axCol = axs[1]
    kv.genCB(axCol[0], norm_den, "Proton Density [#/cc]")
    axCol[0].xaxis.set_ticks_position("top")
    axCol[0].xaxis.set_label_position("top")
    kv.genCB(axCol[3], norm_den, "Electron Density [#/cc]")
    axCol = axs[2]
    if config['doDiff']:
        bVol_dipole = get_bVol_dipole(f5)
        bVol_dip_cc = kt.to_center2D(bVol_dipole)
        kv.genCB(axCol[0], norm_diff_bvol, "(V-V$_d$)/V$_d$",cM=cmap_diff)
    else:
        kv.genCB(axCol[0], norm_bvol, "bVol",cmap_bvol)
    axCol[0].xaxis.set_ticks_position("top")
    axCol[0].xaxis.set_label_position("top")
    kv.genCB(axCol[3], norm_ent , "Flux tube entropy",cmap_ent)
    axCol = axs[3]
    kv.genCB(axCol[0], norm_wimag, "wIMAG")
    #kv.genCB(axCol[0], norm_bMin, "bMin [nT]")
    axCol[0].xaxis.set_ticks_position("top")
    axCol[0].xaxis.set_label_position("top")
    kv.genCB(axCol[3], norm_Tb, "Tbounce [s]")


    def fillPlots(nPlt):
        for iRow in range(1,nRows-1):
            for iCol in range(nCols):
                axs[iCol][iRow].clear()
        if raiI.stepStrs[nPlt] == "Step#0":
            print("WE HAVE ZERO????", nPlt, raiI.stepStrs[nPlt])
            return
        s5 = f5[raiI.stepStrs[nPlt]]
        # H+  press | H+ den  | bVol    | wIMAG
        # Ele press | Ele den | entropy | Tbounce

        #spc_p = raiI.getSpcFromFlav(ru.flavs_s['HOTP'])
        #spc_e = raiI.getSpcFromFlav(ru.flavs_s['HOTE'])
        #spc_psph = raiI.getSpcFromFlav(ru.flavs_s['PSPH'])
        spcIdx_p = ru.spcIdx(raiI.species, ru.flavs_s['HOTP'])
        spcIdx_e = ru.spcIdx(raiI.species, ru.flavs_s['HOTE'])
        spcIdx_psph = ru.spcIdx(raiI.species, ru.flavs_s['PSPH'])

        fig.suptitle(raiI.UTs[nPlt])

        xmin = ru.getVar(s5,'xmin')
        ymin = ru.getVar(s5,'ymin')
        xmincc = kt.to_center2D(xmin)
        ymincc = kt.to_center2D(ymin)
        topo = ru.getVar(s5,'topo')
        active = ru.getVar(s5,'active')
        if config['domain'] == "ACTIVE":
            mask_cc = active != ru.domain['ACTIVE']
        elif config['domain'] == "BUFFER":
            mask_cc = active != ru.domain['INACTIVE']
        mask_corner = topo==ru.topo['OPEN']
        press_all = ru.getVar(s5,'Pressure',mask=mask_cc,broadcast_dims=(2,))
        den_all   = ru.getVar(s5,'Density'  ,mask=mask_cc,broadcast_dims=(2,))
        tBounce = ru.getVar(s5,'Tbounce',mask=mask_cc)
        vaFrac  = ru.getVar(s5,'vaFrac',mask=mask_corner)
        pstd_rc  = ru.getVar(s5,'Pstd_in',mask=mask_cc,broadcast_dims=(2,))[:,:,0]
        bMin  = ru.getVar(s5,'bminZ',mask=mask_corner)
        bVol    = ru.getVar(s5,'bVol'   ,mask=mask_corner)
        bVol_cc = kt.to_center2D(bVol)

        pot_corot = ru.getVar(s5, 'pot_corot', mask=mask_corner)
        pot_iono  = ru.getVar(s5, 'pot_iono' , mask=mask_corner)
        pot_total = pot_corot + pot_iono

        press_p = press_all[:,:,spcIdx_p+1]  # First slot is bulk
        press_e = press_all[:,:,spcIdx_e+1]
        den_p = den_all[:,:,spcIdx_p+1]
        den_e = den_all[:,:,spcIdx_e+1]
        entropy = press_all[:,:,0]*bVol_cc**(5./3.)  # Wolf units [nPa * (Rx/nT)^(5/3)]
        den_psph =  den_all[:,:,spcIdx_psph+1]
        levels_psphDen = [1,10,100,1000]
        levels_pot = np.arange(-250, 255, 5)

        axCol = axs[0]
        rv.plotXYMin(axCol[1], xmin, ymin, press_p,norm=norm_press,bnds=eqBnds,cmap=cmap_press)
        axCol[1].contour(xmin, ymin, pot_total, levels=levels_pot, colors='white',linewidths=0.5, alpha=0.3)
        rv.plotXYMin(axCol[2], xmin, ymin, press_e,norm=norm_press,bnds=eqBnds,cmap=cmap_press)
        axCol[2].contour(xmin, ymin, pot_total, levels=levels_pot, colors='white',linewidths=0.5, alpha=0.3)

        axCol = axs[1]
        rv.plotXYMin(axCol[1], xmin, ymin, den_p,norm=norm_den,bnds=eqBnds)
        axCol[1].contour(xmincc, ymincc, den_psph,levels=levels_psphDen,colors='white',linewidths=0.5,alpha=0.4)
        rv.plotXYMin(axCol[2], xmin, ymin, den_e,norm=norm_den,bnds=eqBnds)

        axCol = axs[2]
        if config['doDiff']:
            d_bVol_cc = (bVol_cc - bVol_dip_cc)/bVol_dip_cc
            rv.plotXYMin(axCol[1], xmin, ymin, d_bVol_cc,norm=norm_diff_bvol,cmap=cmap_diff,bnds=eqBnds)
        else:
            rv.plotXYMin(axCol[1], xmin, ymin, bVol_cc,norm=norm_bvol,bnds=eqBnds,cmap=cmap_bvol)
        axCol[1].contour(xmincc,ymincc,active,levels=[0.5],colors='orange')
        rv.plotXYMin(axCol[2], xmin, ymin, entropy,norm=norm_ent ,bnds=eqBnds,cmap=cmap_ent)

        axCol = axs[3]
        rv.plotXYMin(axCol[1], xmin, ymin, vaFrac,norm=norm_wimag,bnds=eqBnds)
        #axCol[1].contour(xmin, ymin, vaFrac, levels=np.arange(0,1.1,0.1),colors='black',alpha=0.4)
        #axCol[1].contour(xmin, ymin, bMin, levels=[5,10], colors='black', alpha=0.4)
        #axCol[1].contour(xmincc, ymincc, pstd_rc, levels=[0.5], colors='black', alpha=0.4)
        axCol[1].contour(xmincc,ymincc,active,levels=[0.5],colors='orange')
        rv.plotXYMin(axCol[2], xmin, ymin, tBounce,norm=norm_Tb,bnds=eqBnds)
        #axCol[1].tick_params(axis="x",direction="in", pad=tickIn_pad)
        #axCol[2].tick_params(axis="x",direction="in", pad=tickIn_pad)
        #axCol[1].yaxis.set_major_formatter(mtk.NullFormatter())
        #axCol[2].yaxis.set_major_formatter(mtk.NullFormatter())

    # Add Branch and Hash info
    if not config['noHash']:
        fig.text(0.1,0.95,f"branch/commit: {config['branch']}/{config['hash']}", fontsize=6)

    if config['doVid'] == False:
        fillPlots(nPlt)
        kv.savePic('qkraijupic.png')
    else:
        if nPlt == 0:
            print("WE HAVE ZERO??")
        outdir = config['vidOut']
        kh5.CheckDirOrMake(outdir)
        fillPlots(nPlt)
        fname = "vid.{:>04d}.png".format(iPlt)
        kv.savePic(os.path.join(outdir, fname))

    # Clean up
    plt.close()

def makeSuperPlot(raiI, config):

    if not config['doVid']:
        plot_frame((0,config['nPlt']),raiI, config)  # keep your original single-frame mode
        return

    # Create the plots in a memory buffer.
    mpl.use('Agg')

    # Set global plot font options.
    mpl.rc('mathtext', fontset='stixsans', default='regular')
    mpl.rc('font', size=10)

    os.makedirs(config['vidOut'], exist_ok=True)
    frame_indices = list(range(config['nStart'], config['nPlt'], config['nStride']))
    frame_pairs = list(enumerate(frame_indices))

    # Pass only necessary lightweight args
    worker = partial(
        plot_frame,
        raiI=raiI,
        config=config
    )
    nCPU = config.get("nCPU", os.cpu_count() or 8)

    with ProcessPoolExecutor(max_workers=nCPU) as executor:
        list(tqdm.tqdm(executor.map(worker, frame_pairs), total=len(frame_pairs)))

if __name__ == "__main__":
    config = parse_command_line()
    print(config)

    fname = os.path.join(config['indir'],
                        config['id']+".raiju.h5")
    raiI = ru.RAIJUInfo(fname,useTAC=True)
    if (config['doVerbose']): raiI.printStepInfo()

    # Determine plot time
    if config['utPlt'] != default_ut:
        utPlot = datetime.datetime.strptime(config['utPlt'],kt.isotfmt)
        config['nPlt'] = np.abs(raiI.UTs-utPlot).argmin()
    else:
        if config['nPlt'] == -1 or config['nPlt'] > raiI.Nt:
            config['nPlt'] = raiI.Nt-1
        #nPlot = raiI.steps[config['nPlt']]
    # Get branch/hash info
    if not config['noHash']:
        config['branch'] = kh5.GetBranch(fname)
        config['hash'] = kh5.GetHash(fname)
    print("Plotting RAIJU step {} (t={}, UT={})".format(config['nPlt'],raiI.times[config['nPlt']], raiI.UTs[config['nPlt']]))

    makeSuperPlot(raiI, config)
