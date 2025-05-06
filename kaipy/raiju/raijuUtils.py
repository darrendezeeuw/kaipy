import h5py as h5
import numpy as np
from typing import List
import scipy.special as sp

#from bidict import bidict

import kaipy.kdefs as kd
import kaipy.kaiTools as kt
import kaipy.kaiH5 as kh5

import kaipy.raiju.lambdautils.AlamParams as aP

dim = {"THETA": 0,
       "PHI": 1}

topo = {"OPEN" : 0,
        "CLOSED" : 1}

domain = {"INACTIVE" : -1,
          "BUFFER" : 0,
          "ACTIVE" : 1}

flavs_s = {"PSPH" : 0,  # Flav dict, lookup by string name
           "HOTE" : 1,
           "HOTP" : 2}
#flavs_n = bidict(flavs_s).inv  # Flav dict, lookup by index
flavs_n = {0 : "PSPH",
           1 : "HOTE",
           2 : "HOTP"}

spcs_s = {"IDK": 0,
          "ELE": 1,
          "H+" : 2,
          "O+" : 3}
spcs_n = {0: "IDK",
          1: "ELE",
          2: "H+" ,
          3: "O+" }

#------
# Containers
#------


class SpeciesInfo(object):

    def __init__(self, grp):
        """
        grp: h5 Species group to read from
        """
        att = grp.attrs
        self.N        = att['N']
        self.flav     = att['flav']
        self.spcType  = att['spcType']
        self.kStart   = att['kStart']
        self.kEnd     = att['kEnd']+1
        self.numNuc_p = att['numNuc_p']
        self.numNuc_n = att['numNuc_p']
        self.amu      = att['amu']
        self.q        = att['q']
        self.alami    = grp['alami'][:]
        self.alamc    = 0.5*(self.alami[1:] + self.alami[:-1])

class RAIJUInfo(kh5.H5Info):
    """
    Extends H5Info to grab RAIJU-specific info
    """

    species: SpeciesInfo

    def __init__(self, h5fname, noSubsec=True, useTAC=False, useBars=True):
        super().__init__(h5fname, noSubsec, useTAC=useTAC, useBars=useBars)
        
        self.species = []
        with h5.File(h5fname) as f5:
            # Spatial info
            Ni_corner, Nj_corner = getVar(f5, 'X').shape
            self.Ni = Ni_corner - 1
            self.Nj = Nj_corner - 1

            # Species things
            for spc_key in f5['Species'].keys():
                self.species.append(SpeciesInfo(f5['Species'][spc_key]))
            self.nSpc = len(self.species)
            self.Nk = 0
            for s in self.species: self.Nk += s.N
            
            # Planet info
            self.planetInfo = {}
            for k in f5['Planet'].attrs.keys():
                v = f5['Planet'].attrs[k]
                if isinstance(v, bytes):
                    v = v.decode('utf-8')
                self.planetInfo[k] = v

    def getSpcFromFlav(self, flav: int) -> SpeciesInfo:
        idx = spcIdx(self.species, flav)
        if idx == -1:
            return None
        return self.species[idx]


            
       

#------
# Data handlers
#------

def getVar(grp: h5.Group, varName: str, mask:bool=None, broadcast_dims=None) -> np.ndarray:
    """ Get vars from h5 file this way so that we're sure everyone agrees on type and shape
    """
    try:
        var = grp[varName][:].T  # .T to go from Fortran to python indexing order
        if mask is not None:
            if broadcast_dims is not None:
                for d in broadcast_dims:
                    mask = np.expand_dims(mask, axis=d)
                mask = np.broadcast_to(mask, var.shape)
            var = np.ma.masked_where(mask, var)
        return var
    except KeyError:
        print("Error: {} not in keys".format(varName))
        return None

#------
# Species helpers
#------

def spcIdx(spcList: List[SpeciesInfo], flav: int) -> int:
    # Get index of a certain species based on its flavor
    # spcList: list of SpeciesInfo
    for idx, s in enumerate(spcList):
        if s.flav == flav:
            return idx
    # If here, we didn't find index. Complain
    print("Warning: spcIdx didn't find flav '{}' in spcList".format(flav))
    return -1

def getSpcFromNkArr():
    pass


def getMask(s5, dom="ACTIVE"):
    # s5 = step#X group object

    mask = getVar(s5, 'active') != domain[dom]
        # Only include domain specified by caller
    return mask


def getMask_cornerByActive(s5, dom="ACTIVE"):
    Ni, Nj = getVar(s5.file,'X').shape
    is_active = getVar(s5, 'active') == domain[dom]
    # True = good point while we do our operations, convert to true np.mask meaning on return
    mask_corner = np.full((Ni,Nj),False)

    # Any corner bordering an active cell is a good point
    mask_corner[:-1,:-1] = is_active
    mask_corner[1:,:-1] = np.logical_or(mask_corner[1:,:-1], is_active)
    mask_corner[:-1,1:] = np.logical_or(mask_corner[:-1,1:], is_active)
    mask_corner[1:,1:]  = np.logical_or(mask_corner[1:,1:] , is_active)

    return ~mask_corner


#------
# Some analytic stuff
#------

def intensity_maxwell(n, mass, E, kT):
    """ Intensity for energy E in an analytic Maxwellian profile
        n: density in #/cc
        mass: mass in kg
        E: energy in keV
        kT: temp in keV
        j: Intensity [1/(s*sr*keV*cm^2)]
    """
    f = n * (mass/(2*np.pi*kT))**(3/2) * np.exp(-E/kT)
    j = 2*E/mass**2 * f  * 1e2 * np.sqrt(kd.kev2J)
    return j

def intensity_kappa(n, mass, E, kT, kappa=6):
    """ Intensity for energy E in an analytic Maxwellian profile
        n: density in #/cc
        mass: mass in kg
        E: energy in keV
        kT: temp in keV
        j: Intensity [1/(s*sr*keV*cm^2)]
    """
    gamfac = sp.gamma(kappa+1)/sp.gamma(kappa-0.5)

    #f = n * (mass/(2*np.pi*kappa*kT))**(3/2) * gamfac * (1+(E/kappa/kT))**(-kappa-1)  # From Baumjohann & Treumann
    kap15 = kappa-1.5
    E0 = kT*kap15/kappa
    kArg = 1 + (E/E0)/kap15
    f = n * (mass/(2*np.pi*E0*kap15))**(3/2) * gamfac * kArg**(-kappa-1)

    j = 2*E/mass**2 * f  * 1e2 * np.sqrt(kd.kev2J)
    return j

#------
# Conversions
#------

def lambda2Energy(lambdas, bvol):
    """
    lambdas = [eV*(Rx/nT)^(2/3)]
    bVol = [Rx/nT]
    return: energy [keV]
    """

    return lambdas*bvol**(-2/3)*1e-3

def etak2Press(etak, alamc, bVol):
    """
    etak [#/cc * Rx/T]
    alamc: [eV*(Rx/nT)^(2/3)]
    bVol [Rx/nT]

    Returns: pressure [nPa]
    """
    return 2./3.*etak*alamc*bVol**(-5./3.) * kd.ev2J * 1.e6

def etak2Den(etak, bVol):
    """
    etak [#/cc * Rx/T]
    bVol [Rx/nT]

    Returns: density [#/cc]
    """

    return (etak*1.0E-9)/bVol  # [#/cc]

# TODO:
#   Calc vcorot, vgc, veffective