# Third party modules
import h5py as h5
import numpy as np

class wmParams:
    """
    Class representing the dimension of the parameters for the chorus wave model
    
    Attributes:
        dim: number of parameters in the electron lifetime
        nKp, nMLT,nL,nEk: Dimension of Kp, MLT, L and Ek as the parameter of the model.   
    """
    #All energies in eV
    def __init__(self, dim=4, nKp=6, nMLT=97, nL=41, nEk=155):
        self.dim = dim
        self.nKp = nKp
        self.nMLT = nMLT
        self.nL = nL
        self.nEk = nEk

    def getAttrs(self):
        return {
            'tauDim': self.dim,
            'nKp': self.nKp,
            'nMLT': self.nMLT,
            'nL': self.nL,
            'nEk': self.nEk,
        }

