#%%
# Standard modules
import os
import argparse

# Third-party modules
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Kaipy modules
import kaipy.kdefs as kd
import kaipy.kaiTools as kt
import kaipy.kaiViz as kv
import kaipy.kaiH5 as kh5
import kaipy.raiju.raijuUtils as ru


def calcMRIonoVars(s5):
    f5 = s5.file
    thetaIono = kt.to_center2D((90 - f5['Y'][:])*np.pi/180)
    phiIono   = kt.to_center2D((f5['X'][:])*np.pi/180)
    Ri_m = kd.RionE*1e6
    br = np.zeros(thetaIono.shape)
    for i in range(thetaIono.shape[0]):
        br[i,:] = kd.EarthM0g*kd.G2nT \
                    / (Ri_m/kd.REarth)**3 \
                    * 2*np.cos(thetaIono[i,:])
    
    areaCC = np.zeros(thetaIono.shape)
    dTheta = thetaIono[0,1] - thetaIono[0,0]
    dPhi = phiIono[1,0] - phiIono[0,0]
    for i in range(areaCC.shape[1]):
        areaCC[:,i] = (Ri_m/kd.REarth)**2*np.sin(thetaIono[:,i])*dTheta*dPhi
    
    return br, areaCC


def DPSDst_raiju(raiI: ru.RAIJUInfo, s5: h5.Group, rmax=6):
    """
    Calculate DPSDst [nT] for a raiju step
    If return2D=True, gives back full 2D map, where each value is that cell's contribution to DPSDst.
        This will ignore isGood, if provided, but will manually ignore RAIJUINACTIVE cells
    If return2D=False (default), returns single value that sums over all isGood=True
    """

    xmin = ru.getVar(s5,'xmin')
    ymin = ru.getVar(s5,'ymin')
    xmcc = kt.to_center2D(xmin)
    ymcc = kt.to_center2D(ymin)
    rmin = np.sqrt(xmcc**2+ymcc**2)

    active = ru.getVar(s5, 'active')
    isGood = np.logical_and(rmin<rmax, active == ru.domain['ACTIVE'])

    idx_ele = ru.spcIdx(raiI.species, ru.flavs_s['HOTE'])

    f5 = s5.file
    Ri_m = f5['Planet'].attrs['Rad_ionosphere']
    Rp_m = f5['Planet'].attrs['Rad_surface']
    areaCC = ru.getVar(f5, 'Grid/areaCC')
    brcc   = ru.getVar(f5, 'Grid/BrCC')
    press_tot  = ru.getVar(s5, 'Pressure', mask = isGood==False, broadcast_dims=(2,))[:,:,0]  # Total pressure
    press_ele  = ru.getVar(s5, 'Pressure', mask = isGood==False, broadcast_dims=(2,))[:,:,idx_ele+1]  # Electron pressure
    #bvolcc = np.ma.masked_where(isGood==False, kt.to_center2D(ru.getVar(s5, 'bVol')))
    bvolcc = ru.getVar(s5, 'bVol_cc', mask=isGood==False)

    energyDen_tot = (press_tot*1e-9)*(bvolcc*Rp_m*1e9)*(brcc*1e-9)/ kd.kev2J
    energyDen_ele = (press_ele*1e-9)*(bvolcc*Rp_m*1e9)*(brcc*1e-9)/ kd.kev2J

    energy_tot = (press_tot*1e-9)*(bvolcc*Rp_m*1e9)*(brcc*1e-9)*(areaCC*Ri_m**2) / kd.kev2J  # p[J/m^3] * bVol[m/T] * B[T] * Re^2[m^2] = [J] * keV/J = [keV]
    energy_ele = (press_ele*1e-9)*(bvolcc*Rp_m*1e9)*(brcc*1e-9)*(areaCC*Ri_m**2) / kd.kev2J  # p[J/m^3] * bVol[m/T] * B[T] * Re^2[m^2] = [J] * keV/J = [keV]
    dpsdst_2D_tot = -4.2*(1.0e-30)*energy_tot  # [nT]
    dpsdst_2D_ele = -4.2*(1.0e-30)*energy_ele  # [nT]
    #dpsdst_sum = np.ma.sum(dpsdst_2D)

    return energyDen_tot, energyDen_ele, dpsdst_2D_tot, dpsdst_2D_ele


def DPSDst_mhdrcm(rmI: kh5.H5Info, s5: h5.Group, rmax=6, doMHD=False, br=None, areaCC=None):
    """
    Same as DPSDst_raiju but for RCM
    """

    f5 = s5.file
        
    xmin = s5['xMin'][:]
    ymin = s5['yMin'][:]
    rmin = np.sqrt(xmin**2+ymin**2)
    iopen = s5['IOpen'][:]
    active = np.full(iopen.shape, ru.domain['BUFFER'])
    active[iopen > 0.5] = ru.domain['INACTIVE']
    active[iopen < -0.5] = ru.domain['ACTIVE']

    isGood = np.logical_and(rmin < rmax, active == ru.domain['ACTIVE'])

    press_tot = s5['P'][:]
    press_ele = s5['Pe'][:]

    press_tot = np.ma.masked_where(isGood==False, press_tot)
    press_ele = np.ma.masked_where(isGood==False, press_ele)
    bvol = np.ma.masked_where(isGood==False, s5['bVol'][:])
    
    if br is None or areaCC is None:
        br, areaCC = calcMRIonoVars(s5)

    Ri_m = kd.RionE*1e6
    energyDen_tot = (press_tot*1e-9)*(bvol*Ri_m*1e9)*(br*1e-9)/ kd.kev2J
    energyDen_ele = (press_ele*1e-9)*(bvol*Ri_m*1e9)*(br*1e-9)/ kd.kev2J

    energy_tot = (press_tot*1e-9)*(bvol*Ri_m*1e9)*(br*1e-9)*(areaCC*Ri_m**2) / kd.kev2J  # p[J/m^3] * bVol[m/T] * B[T] * Re^2[m^2] = [J] * keV/J = [keV]
    energy_ele = (press_ele*1e-9)*(bvol*Ri_m*1e9)*(br*1e-9)*(areaCC*Ri_m**2) / kd.kev2J  # p[J/m^3] * bVol[m/T] * B[T] * Re^2[m^2] = [J] * keV/J = [keV]
    dpsdst_2D_tot = -4.2*(1.0e-30)*energy_tot  # [nT]
    dpsdst_2D_ele = -4.2*(1.0e-30)*energy_ele  # [nT]
    #dpsdst_sum = np.ma.sum(dpsdst_2D)

    return energyDen_tot, energyDen_ele, dpsdst_2D_tot, dpsdst_2D_ele


def plotDstTS(raiI: ru.RAIJUInfo, rmax=6, mrI: kh5.H5Info = None):

    doMR = isinstance(mrI, kh5.H5Info)

    rai5 = h5.File(raiI.fname, 'r')
    dps_rai_ts     = np.zeros(raiI.Nt)
    dps_rai_ts_ele = np.zeros(raiI.Nt)

    if doMR: 
        mr5 = h5.File(mrI.fname)
        t0 = raiI.times[0]
        t1 = raiI.times[-1]
        i0 = np.abs(mrI.times - t0).argmin()
        i1 = np.abs(mrI.times - t1).argmin()
        Nt_mr = i1-i0+1
        dps_mr_ts     = np.zeros(Nt_mr)
        dps_mr_ts_ele = np.zeros(Nt_mr)

    for t in tqdm(range(raiI.Nt), desc="Calculating RAIJU DPS-Dst"):
        _, _, dst_tot, dst_ele = DPSDst_raiju(raiI, rai5[raiI.stepStrs[t]], rmax=rmax)
        dps_rai_ts[t]     = np.sum(dst_tot)
        dps_rai_ts_ele[t] = np.sum(dst_ele)
    dps_rai_noEle = dps_rai_ts - dps_rai_ts_ele


    if doMR:
        br, areaCC = calcMRIonoVars(mr5[mrI.stepStrs[0]])
        for t in tqdm(range(Nt_mr), desc="Calculating MHDRCM DPS-Dst"):
            _, _, dst_tot, dst_ele = DPSDst_mhdrcm(mrI, mr5[mrI.stepStrs[i0+t]], rmax=rmax, br=br, areaCC=areaCC)
            dps_mr_ts[t]     = np.sum(dst_tot)
            dps_mr_ts_ele[t] = np.sum(dst_ele)
        dps_mr_noEle = dps_mr_ts - dps_mr_ts_ele

    print("Plotting")
    plt.figure(figsize=(8,4))
    plt.plot(raiI.UTs[1:], dps_rai_noEle[1:] , 'b-' , label='RAIJU (Tot-ele)')
    plt.plot(raiI.UTs[1:], dps_rai_ts_ele[1:], 'b--', label='RAIJU (Ele)')

    if doMR:
        plt.plot(mrI.UTs[i0:i1+1], dps_mr_noEle ,'r-' , label='RCM (Tot-ele)')
        plt.plot(mrI.UTs[i0:i1+1], dps_mr_ts_ele,'r--', label='RCM (Ele)')
    plt.legend()
    plt.grid(alpha=0.3)

    kv.savePic("DPSDst.png")
        
#%%
def create_command_line_parser():
    fdir = '.'
    ftag = 'raijuOWD'
    ftag_mr = 'msphere'
    rmax = 6
    doMR = True

    MainS = """Creates series of XMF files from MPI-decomposed Gamera run
    """
    parser = argparse.ArgumentParser(description=MainS, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d',type=str,metavar="directory",default=fdir,help="Directory to read from (default: %(default)s)")
    parser.add_argument('-id',type=str,metavar="runid",default=ftag,help="RunID of raiju h5 (default: %(default)s)")
    parser.add_argument('-mr',type=str,metavar="runid",default=ftag_mr,help="RunID of mhdrcm h5 (default: %(default)s)")
    parser.add_argument('-rmax',type=float,metavar="Re",default=rmax,help="Maximum Req value to eval DPSDst out to (default: %(default)s)")
    parser.add_argument('--nomr',action='store_true',default=False,help="Don't do mhdrcm stuff (default: %(default)s)")
    
    return parser
#%%
def main():
    fdir = '.'
    ftag = 'raijuOWD'
    ftag_mr = 'msphere'
    rmax = 6
    doMR = True

    parser = create_command_line_parser()
    args = parser.parse_args()
    
    #args = parser.parse_args(args=['-d', '/glade/derecho/scratch/sciola/raijudev/owdTests/dinoKO', '-id', 'raijuOWD_bvolFix'])


    fdir = args.d
    ftag = args.id
    ftag_mr = args.mr
    rmax = args.rmax
    doMR = not args.nomr

    print("Reading RAIJU info")
    rai_fname = ftag+".raiju.h5"
    raiI = ru.RAIJUInfo(os.path.join(fdir, rai_fname))
    
    if doMR:
        print("Reading mhdrcm info")
        mrfname = ftag_mr+'.mhdrcm.h5'
        mrI = kh5.H5Info(os.path.join(fdir, mrfname))
    else:
        mrI = None
    
    #%%
    plotDstTS(raiI, rmax, mrI)
    # %%


if __name__ == "__main__":
    main()