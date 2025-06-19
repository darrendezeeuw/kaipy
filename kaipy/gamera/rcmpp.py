#Various routines for plotting RCM output data

# Standard modules
import os

# Third-party modules
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import h5py
import numpy.ma as ma
import matplotlib.patches as patches
import matplotlib.ticker as plticker

# Kaipy modules
import kaipy.kaiViz as kv
import kaipy.gamera.gampp as gampp

rMin = 1.25
rMax = 35.0
rcBds = [-15,10.0,-12.5,12.5]
pCMap = "viridis"
eCol = "slategrey"
rcmCol = "dodgerblue"
gCol = "cyan"
gLW = 0.15

doXYZ = True #How to calculate "equator"
doCut = False
pCut = 1.0e-8
MHDCol = "red"
eLW = 0.05
MHDLW = 0.5

doEll = True

Psi0 = 92.4 #kV
RioRe = 1.018 #Ionosphere height
doCorot = True #Add corotation potential
DenPP = 50.0
ppCol = "orange"

#Get equatorial coordinates, masked if asked to
def RCMEq(rcmdata, nStp, doMask=False, doXYZ=doXYZ, doVerb=True):
	"""
	Calculate the RCMEq values for the given rcmdata and time step.

	Args:
		rcmdata: The rcmdata object containing the data.
		nStp: The time step index.
		doMask: A boolean indicating whether to apply a mask to the calculated values. Default is False.
		doXYZ: A boolean indicating whether to calculate the values in XYZ coordinates. Default is doXYZ.
		doVerb: A boolean indicating whether to print verbose output. Default is True.

	Returns:
		bmX: The calculated X values.
		bmY: The calculated Y values.
	"""
	bmX = rcmdata.GetVar("xMin", nStp, doVerb=doVerb)
	bmY = rcmdata.GetVar("yMin", nStp, doVerb=doVerb)
	if (doXYZ):
		bmZ = rcmdata.GetVar("zMin", nStp, doVerb=doVerb)
		bmP = np.arctan2(bmY, bmX)
		bmR = np.sqrt(bmX * bmX + bmY * bmY + bmZ * bmZ)
		bmX = bmR * np.cos(bmP)
		bmY = bmR * np.sin(bmP)
	if (doMask):
		I = GetMask(rcmdata, nStp)
		bmX = ma.masked_array(bmX, mask=I)
		bmY = ma.masked_array(bmY, mask=I)
	return bmX, bmY


def GetVarMask(rcmdata, nStp, Qid="P", I=None, doVerb=True):
	"""
	Get the variable mask for a given RCM data.

	Args:
		rcmdata: The RCM data object.
		nStp: The time step index.
		Qid: The variable identifier (default is "P").
		I: The mask array (default is None).
		doVerb: Whether to print verbose information (default is True).

	Returns:
		Q: The variable array with the mask applied.
	"""
	if I is None:
		I = GetMask(rcmdata, nStp)
	Q = rcmdata.GetVar(Qid, nStp, doVerb=doVerb)
	Q = ma.masked_array(Q, mask=I)
	return Q


def GetPotential(rcmdata, nStp, I=None, NumCP=25, doVerb=True):
	"""
	Calculate the potential and potential values for a given rcmdata.

	Args:
		rcmdata: The rcmdata object containing the data.
		nStp: The time step index.
		I: The mask array. If None, it will be calculated using GetMask.
		NumCP: The number of potential values to generate.
		doVerb: Whether to print verbose output.

	Returns:
		pot: The calculated potential.
		pVals: The potential values.

	"""
	if (I is None):
		I = GetMask(rcmdata, nStp)
	pot = (1.0e-3) * rcmdata.GetVar("pot", nStp, doVerb=doVerb)

	if (doCorot):
		# Add corotation potential
		colat = GetVarMask(rcmdata, nStp, "colat", I, doVerb=doVerb)
		pcorot = -Psi0 * (RioRe) * (np.sin(colat) ** 2.0)
		pot = pot + pcorot
	pMag = np.abs(pot).max()
	pVals = np.linspace(-pMag, pMag, NumCP)

	return pot, pVals


#Calculate mask
#doRCM: Do RCM domain or full closed region
def GetMask(rcmdata, nStp, doVerb=True):
	"""
	Generate a mask based on the given rcmdata, nStp, and doVerb parameters.

	Args:
		rcmdata: The rcmdata object containing the data.
		nStp: The value of nStp parameter.
		doVerb: A boolean indicating whether to enable verbose mode (default is True).

	Returns:
		I: The generated mask.

	"""
	IOpen = rcmdata.GetVar("IOpen", nStp, doVerb=doVerb)
	
	if (doEll):
		ioCut = -0.5
	else:
		ioCut = 0.5
	bmX = rcmdata.GetVar("xMin", nStp, doVerb=doVerb)
	bmY = rcmdata.GetVar("yMin", nStp, doVerb=doVerb)
	bmR = np.sqrt(bmX * bmX + bmY * bmY)

	Ir = (bmR < rMin) | (bmR > rMax)
	if (doCut):
		Prcm = rcmdata.GetVar("P", nStp, doVerb=doVerb)
		I = Ir | (IOpen > ioCut) | (Prcm < pCut)
	else:
		I = Ir | (IOpen > ioCut)
	return I


#Take axis and rcmdata object and add pressure plot
def RCMInset(AxRCM, rcmdata, nStp, vP, pCol="k", doPP=True):
	"""
	Plot the RCM pressure inset on a given axis.

	Args:
		AxRCM (matplotlib.axes.Axes): The axis on which to plot the RCM pressure inset.
		rcmdata (numpy.ndarray): The RCM data.
		nStp (int): The number of steps.
		vP (matplotlib.colors.Normalize): The normalization object for the color map.
		pCol (str, optional): The color of the contour lines. Defaults to "k".
		doPP (bool, optional): Whether to plot the Npp contour. Defaults to True.
	"""
	if (AxRCM is None):
		AxRCM = plt.gca()

	bmX,bmY = RCMEq(rcmdata,nStp,doMask=True)
	I = GetMask(rcmdata,nStp)
	Ni = (~I).sum()

	if (Ni == 0):
		return

	Prcm  = GetVarMask(rcmdata,nStp,"P"    ,I)
	toMHD = GetVarMask(rcmdata,nStp,"toMHD",I)
	pot,pVals = GetPotential(rcmdata,nStp,I,NumCP=11)
	if (doPP):
		Npp  = GetVarMask(rcmdata,nStp,"Npsph"    ,I)
		
	#Start plotting
	AxRCM.pcolor(bmX,bmY,Prcm,norm=vP,cmap=pCMap,shading='auto')
	AxRCM.plot(bmX,bmY,color=eCol,linewidth=eLW)
	AxRCM.plot(bmX.T,bmY.T,color=eCol,linewidth=eLW)
	if (pCol is not None):
		AxRCM.contour(bmX,bmY,pot,pVals,colors=pCol,linewidths=MHDLW,alpha=0.5)
	if (doPP):
		AxRCM.contour(bmX,bmY,Npp,[DenPP],colors=ppCol,linewidths=2*MHDLW,alpha=0.8)
	doCon = (nStp>0) and (toMHD.min()<0.5) and (toMHD.max()>0.5)
	#Add MHD ingestion contour
	if (doCon):
		CS1 = AxRCM.contour(bmX,bmY,toMHD,[0.5],colors=MHDCol,linewidths=MHDLW)
		manloc = [(0.0,8.0)]

		fmt = {}
		fmt[0.5] = 'MHD'
		AxRCM.clabel(CS1,CS1.levels[::2],inline=True,fmt=fmt,fontsize=5,inline_spacing=25,manual=manloc)

	#Add grid
	xS = [-10,-5,0,5]
	yS = [-10,-5,0,5,10]
	for x in xS:
		AxRCM.axvline(x,linewidth=gLW,color=gCol)
	for y in yS:
		AxRCM.axhline(y,linewidth=gLW,color=gCol)

	kv.addEarth2D(ax=AxRCM)
	kv.SetAx(rcBds,AxRCM)
	kv.SetAxLabs(AxRCM,xLab=None,yLab=None)
	AxRCM.spines['bottom'].set_color(rcmCol)
	AxRCM.spines['top'].set_color(rcmCol) 
	AxRCM.spines['right'].set_color(rcmCol)
	AxRCM.spines['left'].set_color(rcmCol)

	AxRCM.set_title("RCM Pressure",fontsize="small",color=rcmCol)

	return rcBds

#Add RCM box to other plot
def AddRCMBox(Ax):
	"""
	Add a rectangular box to the given Axes object.

	Args:
		Ax: The Axes object to which the rectangle will be added.

	Returns:
		None
	"""
	if (Ax is None):
		Ax = plt.gca()
	xy0 = (rcBds[0],rcBds[2])
	H = rcBds[3]-rcBds[2]
	W = rcBds[1]-rcBds[0]
	rcmRec = patches.Rectangle( (xy0),W,H, fill=False,edgecolor=rcmCol)
	Ax.add_patch(rcmRec)
