import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import kaipy.kaiTools as kt
import kaipy.kaiViz as kv

import kaipy.raiju.raijuUtils as ru

inset_bnds = [-15,10,-12.5,12.5]
inset_border_color="dodgerblue"

def plotLonColat(Ax: plt.Axes, lon: np.ndarray, colat: np.ndarray, var: np.ndarray, unit="deg", norm=None, cmap='viridis') -> mpl.collections.QuadMesh:
    """ Simple plot function in longitude - colatitude projection
        lon and colat inputs should be in radians
        lon and colat should be size Ni+1,Nj+1
    """

    if unit=="deg":
        uStr = "[deg]"
        colat = colat*180/np.pi
        lon   = lon  *180/np.pi
    else:
        uStr = "[rad]"

    cMin = np.min(colat)
    cMax = np.max(colat)

    mp = Ax.pcolormesh(lon, colat, var, norm=norm, cmap=cmap)
    Ax.set_xlabel("Longitude " + uStr)
    Ax.set_ylabel("Colatitude " + uStr)
    Ax.set_ylim([cMax, cMin])
    return mp

def plotIono(Ax: plt.Axes, lon: np.ndarray, colat: np.ndarray, var: np.ndarray, norm=None, cmap='viridis', Riono=1) -> mpl.collections.QuadMesh:
    """ Simple plot function to show ionosphere projection
        lon and colat inputs should be in radians
        lon and colat should be size Ni+1,Nj+1
    """

    # First get coordinates in polar projection
    r, theta = kt.rtp2rt(Riono, colat, lon)
    mp = Ax.pcolormesh(theta, r, var, norm=norm, cmap=cmap)
    Ax.axis([0,2*np.pi,0,r.max()])
    Ax.grid(True)

    # Re-do Colat labels
    colatStride = 10*np.pi/180  # rad
    colatMax = np.max(colat) # rad
    colatTickLocs_rad = np.arange(10*np.pi/180, colatMax, colatStride)
    rTickLocs = Riono*np.sin(colatTickLocs_rad)
    rTickLabels = ["{:2n}".format(c*180/np.pi) for c in colatTickLocs_rad]
    Ax.set_yticks(rTickLocs)
    Ax.set_yticklabels(rTickLabels)
    return mp


def plotXYMin(Ax: plt.Axes, xmin: np.ndarray, ymin: np.ndarray, var: np.ndarray, shading='auto',
              norm=None, cmap='viridis', bnds=None, 
              alpha_grid = 0.25, lblsize=None)-> mpl.collections.QuadMesh:
    """ Simple plot function to show xy bmin projection
        If xBnds or yBnds is not none, it will use x or y bnds and rescale the other so that aspect ratio is 1
    """
    if (type(xmin)==np.ma.core.MaskedArray):
        mp = Ax.pcolor(xmin, ymin, var, norm=norm, cmap=cmap, shading=shading, rasterized=True)
    else:
        mp = Ax.pcolormesh(xmin, ymin, var, norm=norm, cmap=cmap, shading=shading, rasterized=True)
    if alpha_grid > 0:
        Ax.grid(alpha=alpha_grid)

    if lblsize is not None:
        Ax.set_xlabel('X [R$_p$]', fontsize=lblsize)
        Ax.set_ylabel('Y [R$_p$]', fontsize=lblsize)

    if bnds is not None:
        #kv.SetAx(bnds,ax=Ax,Adj='datalim')
        kv.SetAx(bnds,ax=Ax,Adj='datalim')
        #Ax.set_xlim([bnds[0],bnds[1]])
        #Ax.set_ylim([bnds[2],bnds[3]])

    kv.addEarth2D(ax=Ax)
    return mp

def drawGrid(ax, X, Y, color='slategrey', linewidth='0.15', alpha=0.8, mask=None):
    if mask is not None:
        Xplt = np.ma.masked_where(mask, X)
        Yplt = np.ma.masked_where(mask, Y)
    else:
        Xplt = X
        Yplt = Y
    ax.plot(Xplt  , Yplt  , color=color, alpha=alpha, linewidth=linewidth)
    ax.plot(Xplt.T, Yplt.T, color=color, alpha=alpha, linewidth=linewidth)


def plotInset(ax: plt.Axes, raiI: ru.RAIJUInfo, stepStr: str, norm_press: mpl.colors.Normalize, ax_gam = None) -> list:
    import h5py as h5
    f5 = h5.File(raiI.fname)
    rai_idx = raiI.stepStrs.index(stepStr)
    s5 = f5[stepStr]

    idx_psph = ru.spcIdx(raiI.species, ru.flavs_s["PSPH"])

    xmin = ru.getVar(s5, 'xmin')
    ymin = ru.getVar(s5, 'ymin')
    xmcc = kt.to_center2D(xmin)
    ymcc = kt.to_center2D(ymin)
    active = ru.getVar(s5, 'active')
    mask_cc = active != ru.domain['ACTIVE']
    mask_corner = ru.getMask_cornerByActive(s5)
    press_tot = ru.getVar(s5,'Pressure',mask=mask_cc,broadcast_dims=(2,))[:,:,0]  # Bulk
    density_psph = ru.getVar(s5,'Density',mask=mask_cc,broadcast_dims=(2,))[:,:,idx_psph+1]  # +1 to account for bulk at idx 0

    plotXYMin(ax, xmin, ymin, press_tot, norm=norm_press, cmap='viridis',bnds=inset_bnds)
    drawGrid(ax, xmin, ymin, mask=mask_corner,alpha=0.4)

    psph_lvls = [1,10,100,1000]
    ax.contour(xmcc,ymcc,density_psph,levels=psph_lvls,colors='white',alpha=0.5,linewidths=1)

    

    # Decoration
    kv.SetAxLabs(ax,xLab=None,yLab=None)
    kv.addEarth2D(ax=ax)
    ax.spines['bottom'].set_color(inset_border_color)
    ax.spines['top'].set_color(inset_border_color) 
    ax.spines['right'].set_color(inset_border_color)
    ax.spines['left'].set_color(inset_border_color)
    ax.grid(alpha=0.4)
    ax.set_title("RAIJU Pressure",fontsize="small",color=inset_border_color)
    # Axes grid
    gCol = "cyan"
    gLW = 0.15
    xS = [-10,-5,0,5]
    yS = [-10,-5,0,5,10]
    for x in xS:
        ax.axvline(x,linewidth=gLW,color=gCol)
    for y in yS:
        ax.axhline(y,linewidth=gLW,color=gCol)

    # Extra stuff on gam eq panel
    if ax_gam is not None:
        ax_gam.contour(xmcc, ymcc, active, levels=[ru.domain['ACTIVE']-0.9],colors='orange',linewidths=0.5,alpha=0.7)
        ax.contour(    xmcc, ymcc, active, levels=[ru.domain['ACTIVE']-0.5],colors='orange',linewidths=0.5,alpha=1.0)
        #ax.contour(xmin,ymin,mask_corner.astype(int),levels=[0], colors='orange',linewidths=0.5,alpha=1.0)

    return inset_bnds