"""
Diagnostic tools to check mapping MHD moments to etas and back
"""

import os
import argparse
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
import datetime

import kaipy.kdefs as kd
import kaipy.kaiViz as kv
import kaipy.kaiTools as kt
import kaipy.raiju.raijuUtils as ru
import kaipy.raiju.raijuViz as rv

@dataclass
class m2mData_pnt:  # Maybe overkill idk
    p_mhd: float
    d_mhd: float
    kt_mhd: float
    p_rai: float
    d_rai: float
    kt_rai: float

@dataclass
class m2mData_step:
    frac_P: np.ndarray
    frac_D: np.ndarray

isotfmt = '%Y-%m-%dT%H:%M:%S'

def parse_command_line():
    """
    
    """

    description = """Do some diagnostics for mapping MHD moments to etas and back.
    """

    default_runid = "msphere"
    default_nStride = 10
    default_uts = -1
    default_ute = -1
    mode_choices = ["summary", "step", "point"]
    
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
        "-uts", type=str, metavar=isotfmt, default=default_uts,
        help="First UT (default: %(default)s)"
    )
    parser.add_argument(
        "-ute", type=str, metavar=isotfmt, default=default_ute,
        help="Last UT (default: %(default)s)"
    )
    parser.add_argument(
        "-dt", type=int, metavar="stride", default=default_nStride,
        help="Step stride (in minutes)  (default: %(default)s)"
    )
    parser.add_argument(
        "-phi0", type=int, metavar="int", default=180,
        help=" Phi value [deg] to probe in 'point' mode (default: %(default)s)"
    )
    parser.add_argument(
        "-mode", choices=mode_choices, default=mode_choices[0],
        help="What do you wanna do? (default: %(default)s)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print verbose output (default: %(default)s)."
    )

    args = parser.parse_args()
    config = {
        "indir"     : args.d,
        "id"        : args.id,
        "ut_start"  : args.uts,
        "ut_end"    : args.ute,
        "del_min"   : args.dt,
        "phi0"      : args.phi0,
        "mode"      : args.mode,
        "doVerbose" : args.verbose,
    }
    return config


def checkMom2Mom_pnt(raiI: ru.RAIJUInfo, s5: h5.Group, 
                     i0: int, j0: int, spcID="BLK",
                     doIntenPlot=False, doConsoleIO=False) -> m2mData_pnt:

    tiote = 4.0

    iStep = np.argmin(np.abs(raiI.times - s5.attrs['time']))

    if spcID=="BLK":
        idx_mhd = 0
        idx_rai = 0
        doIntenPlot = False
    if spcID=="HOTP":
        idx_mhd = 0
        idx_rai = ru.spcIdx(raiI.species, ru.flavs_s["HOTP"])+1
        spc = raiI.species[idx_rai-1]
        mass = kd.Mp_cgs*1e-3
    if spcID=="HOTE":
        idx_mhd = 0
        idx_rai = ru.spcIdx(raiI.species, ru.flavs_s["HOTE"])+1
        spc = raiI.species[idx_rai-1]
        mass = kd.Me_cgs*1e-3

    p_mhd = ru.getVar(s5, "Pavg_in")[i0,j0,idx_mhd]
    d_mhd = ru.getVar(s5, "Davg_in")[i0,j0,idx_mhd]
    p_rai = ru.getVar(s5, "Pressure")[i0,j0,idx_rai]
    d_rai = ru.getVar(s5, "Density" )[i0,j0,idx_rai]
    if spcID=="HOTP": p_mhd = p_mhd / (1 + 1/tiote)  # Get just ions
    if spcID=="HOTE": p_mhd = p_mhd / (1 +   tiote)  # Get just electrons
    t_mhd = 6.25*p_mhd/d_mhd
    t_rai = 6.25*p_rai/d_rai

    if doConsoleIO:
        print("MHD:")
        print("  P={:0.6f} nPa, D={:0.6f} #/cc, T={:0.6f} keV".format(p_mhd,d_mhd,t_mhd))
        print("RAIJU:")
        print("  P={:0.6f} nPa, D={:0.6f} #/cc, T={:0.6f} keV".format(p_rai,d_rai,t_rai))

    if doIntenPlot:
        inten = ru.getVar(s5, "intensity")[i0,j0,spc.kStart:spc.kEnd]
        bVol_cc = ru.getVar(s5,'bVol_cc')[i0,j0]
        xmin_cc = kt.to_center2D(ru.getVar(s5, 'xmin'))[i0,j0]
        ymin_cc = kt.to_center2D(ru.getVar(s5, 'ymin'))[i0,j0]
        energies = np.abs(spc.alami)*bVol_cc**(-2./3.)*1e-3  # [keV]
        energies_cc = np.abs(spc.alamc)*bVol_cc**(-2./3.)*1e-3  # [keV]

        Nk = energies_cc.shape[0]
        inten_maxwell = np.zeros(Nk)
        for i in range(Nk):
            inten_maxwell[i] = ru.intensity_maxwell(d_mhd, mass,energies_cc[i], t_mhd)

        plt.figure(figsize=(8,8))
        plt.plot(energies_cc, inten_maxwell, '--',linewidth=3,alpha=0.8,label='MHD (Maxwellian)')
        plt.scatter(energies_cc, inten_maxwell, s=20,alpha=0.8)
        plt.plot(energies_cc, inten, label='RAIJU')
        plt.scatter(energies_cc, inten,s=10)
        plt.xlabel("Energy [keV]")
        plt.ylabel("Intensity [...]")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim([1e4,1e8])
        plt.title("{}\ni0={}, j0={}\n(X,Y)=({:0.2f},{:0.2f}) R$_E$".format(raiI.UTs[iStep],i0,j0,xmin_cc,ymin_cc))
        fname = f"mom2mom_{i0}_{j0}.png"
        kv.savePic(fname)
        print(f"Saved plot {fname}")


    return m2mData_pnt(p_mhd,d_mhd,t_mhd,p_rai,d_rai,t_rai)


def checkMom2Mom_step(raiI: ru.RAIJUInfo, s5: h5.Group, 
                      mask_cc:np.ndarray = None, spcID="BLK",
                      doPlot=False,doConsoleIO=False) -> m2mData_step:
    
    active = ru.getVar(s5, 'active')
    if mask_cc is None:
        mask_cc = active != ru.domain['BUFFER']

    frac_P = np.ma.ones((raiI.Ni, raiI.Nj))
    frac_D = np.ma.ones((raiI.Ni, raiI.Nj))
    kt_mhd = np.ma.ones((raiI.Ni, raiI.Nj))
    frac_P.mask = mask_cc
    frac_D.mask = mask_cc
    
    for j in range(raiI.Nj):
        for i in range(raiI.Ni):
            if mask_cc[i,j]: continue
            m2m = checkMom2Mom_pnt(raiI, s5, i, j, spcID=spcID,doIntenPlot=False)
            frac_P[i,j] = m2m.p_rai/m2m.p_mhd
            frac_D[i,j] = m2m.d_rai/m2m.d_mhd
            kt_mhd[i,j] = m2m.kt_mhd

    if doPlot:
        iStep = np.argmin(np.abs(raiI.times - s5.attrs['time']))

        plt.figure(figsize=(8,8))
        norm = kv.genNorm(0.8,1.2)
        cmap = 'RdBu_r'
        colat2D = ru.getVar(s5.file,'X')
        lon2D = ru.getVar(s5.file,'Y')
        mp = rv.plotLonColat(plt.gca(),lon2D,colat2D,frac_P,norm=norm,cmap=cmap)
        plt.colorbar(mappable=mp,label='frac_P')
        plt.title(raiI.UTs[iStep])
        fname = "frac_P.png"
        kv.savePic(fname)
        print(f"Saved plot {fname}")
        
        plt.clf()
        mp = rv.plotLonColat(plt.gca(),lon2D,colat2D,frac_D,norm=norm,cmap=cmap)
        plt.colorbar(mappable=mp,label='frac_D')
        plt.title(raiI.UTs[iStep])
        fname = "frac_D.png"
        kv.savePic(fname)
        print(f"Saved plot {fname}")
        
        plt.clf()
        plt.scatter(kt_mhd,frac_P,facecolors='none',edgecolors='r',alpha=0.2)
        xlim = plt.xlim()
        plt.plot(xlim,[1,1],'k--',alpha=0.5)
        plt.xlabel('Temp [keV]')
        plt.ylabel('P$_{rai}$/P$_{mhd}$')
        plt.xscale('log')
        plt.title(raiI.UTs[iStep])
        fname = "kt_vs_fracP.png"
        kv.savePic(fname)
        print(f"Saved plot {fname}")

        plt.clf()
        plt.scatter(kt_mhd,frac_D,facecolors='none',edgecolors='r',alpha=0.2)
        xlim = plt.xlim()
        plt.plot(xlim,[1,1],'k--',alpha=0.5)
        plt.xlabel('Temp [keV]')
        plt.ylabel('D$_{rai}$/D$_{mhd}$')
        plt.xscale('log')
        plt.title(raiI.UTs[iStep])
        fname = "kt_vs_fracD.png"
        kv.savePic(fname)
        print(f"Saved plot {fname}")
    if doConsoleIO:
        print("frac_P:")
        print(" min={:06f}, max={:06f}, avg={:06f}".format(np.min(frac_P), np.max(frac_P), np.average(frac_P)))
        print("frac_D:")
        print(" min={:06f}, max={:06f}, avg={:06f}".format(np.min(frac_D), np.max(frac_D), np.average(frac_D)))

    return m2mData_step(frac_P,frac_D)


def checkMom2Mom_summary(raiI:ru.RAIJUInfo, ut_start:datetime.datetime=None, ut_end:datetime.datetime=None, stride_minutes:float=5):

    f5 = h5.File(raiI.fname,'r')

    if ut_start is None:
        iStart = 0
    else:
        iStart = np.argmin(np.abs(raiI.UTs-ut_start))
    if ut_end is None:
        iEnd = raiI.Nt - 1
    else:
        iEnd = np.argmin(np.abs(raiI.UTs-ut_end))
    if stride_minutes is None:
        iStep = 10
    else:
        iStep = int( (stride_minutes*60) / (raiI.UTs[-1]-raiI.UTs[-2]).seconds )

    Nt = int( (iEnd - iStart)/iStep )+1
    
    timeArr = np.empty(Nt,dtype=datetime.datetime)
    min_P = np.zeros(Nt)
    max_P = np.zeros(Nt)
    avg_P = np.zeros(Nt)
    min_D = np.zeros(Nt)
    max_D = np.zeros(Nt)
    avg_D = np.zeros(Nt)

    for i, iRai in enumerate(tqdm(range(iStart,iEnd,iStep))):
        s5 = f5[raiI.stepStrs[iRai]]
        m2m = checkMom2Mom_step(raiI,s5,spcID="HOTP")
        timeArr[i] = raiI.UTs[iRai]
        min_P[i] = np.min(m2m.frac_P)
        max_P[i] = np.max(m2m.frac_P)
        avg_P[i] = np.average(m2m.frac_P)
        min_D[i] = np.min(m2m.frac_D)
        max_D[i] = np.max(m2m.frac_D)
        avg_D[i] = np.average(m2m.frac_D)

    clr_P = "red"
    clr_D = "blue"
    alpha = 0.4
    plt.figure(figsize=(12,8))
    plt.plot(timeArr, avg_P     ,c=clr_P,label="Pressure")
    plt.plot(timeArr, min_P,'--',c=clr_P,alpha=alpha)
    plt.plot(timeArr, max_P,'--',c=clr_P,alpha=alpha)
    plt.plot(timeArr, avg_D     ,c=clr_D,label="Density")
    plt.plot(timeArr, min_D,'--',c=clr_D,alpha=alpha)
    plt.plot(timeArr, max_D,'--',c=clr_D,alpha=alpha)
    plt.legend()
    plt.grid(alpha=0.3)
    fname = "m2m_summary.png"
    kv.savePic(fname)
    print(f"Saved plot {fname}")


def main(config=None):

    if config is None:
        config = parse_command_line()
    indir = os.getcwd()
    fname = "msphere.raiju.h5"

    raiI = ru.RAIJUInfo(os.path.join(indir,fname),useTAC=True)

    if config['ut_start'] == -1:
        config['ut_start'] = raiI.UTs[0]
    else:
        config['ut_start'] = datetime.datetime.strptime(config['ut_start'],isotfmt)
    if config['ut_end'] == -1:
        config['ut_end'] = raiI.UTs[-1]
    else:
        config['ut_end'] = datetime.datetime.strptime(config['ut_end'],isotfmt)
    
    if config['mode']=="summary":
        checkMom2Mom_summary(raiI,config['ut_start'],config['ut_end'],config['del_min'])
    if config['mode']=="step":
        iUT = np.argmin(np.abs(raiI.UTs - config['ut_end']))
        f5 = h5.File(raiI.fname, 'r')
        s5 = f5[raiI.stepStrs[iUT]]
        checkMom2Mom_step(raiI, s5, doConsoleIO=True,doPlot=True)
    if config['mode']=="point":
        iUT = np.argmin(np.abs(raiI.UTs - config['ut_end']))
        f5 = h5.File(raiI.fname, 'r')
        s5 = f5[raiI.stepStrs[iUT]]
        phicc = kt.to_center1D(ru.getVar(f5,'Y')[0,:])*180/np.pi
        iPhi = np.argmin(np.abs(phicc - config['phi0']))
        active = ru.getVar(s5,'active')[:,iPhi]
        iColat = np.argmin(np.abs(active - ru.domain['BUFFER']))
        checkMom2Mom_pnt(raiI, s5, iColat, iPhi,spcID="HOTP",doIntenPlot=True,doConsoleIO=True)
    

if __name__=="__main__":
    main()