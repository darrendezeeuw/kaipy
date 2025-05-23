# Standard modules
import os

# Third-party modules
import numpy as np
import h5py as h5
from scipy.interpolate import RectBivariateSpline

# Kaipy modules
from kaipy.rcm.wmutils.wmData import wmParams


def genWM(params):
        """
        Generate WM (Wave Model: Chorus wave) using the given parameters.

        Args:
                params (dict): A dictionary containing the parameters for generating the WM.

        Returns:
                object: The returns from genChorus function.

        Raises:
                None

        Example:
                params = {
                        'param1': value1,
                        'param2': value2,
                        ...
                }
                genWM(params)
        """



        fInChorus = 'chorus_polynomial.txt'
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        fInChorus = os.path.join(__location__,fInChorus)

        print("Reading %s"%fInChorus)

        return genChorus(params,fInChorus)

# Add wpi-induced electron lifetime model to input file and create an output file
# Writes arrays to file in rcmconfig.h5 format
def genh5(fIn, fOut, inputParams):
        """
        Write the wave model to an HDF5 file (rcmconfig.h5) with data and attributes based on the input parameters.

        Args:
                fIn (str): The input HDF5 file path.
                fOut (str): The output HDF5 file path.
                inputParams (dict): A dictionary containing input parameters.

        Returns:
                None
        """
        if fIn != fOut:
                oH5 = h5.File(fOut, 'w')
                iH5 = h5.File(fIn, 'r')
                for Q in iH5.keys():
                        sQ = str(Q)
                        oH5.create_dataset(sQ, data=iH5[sQ])
                oH5.attrs.update(iH5.attrs)
        else:
                oH5 = h5.File(fOut, 'r+')

        if not ('Taui' in oH5.keys()):
                kpi, mlti, li, eki, taui = genWM(inputParams)
                attrs = inputParams.getAttrs()

                oH5.create_dataset('Kpi', data=kpi)
                oH5.create_dataset('MLTi', data=mlti)
                oH5.create_dataset('Li', data=li)
                oH5.create_dataset('Eki', data=eki)
                oH5.create_dataset('Taui', data=taui)
                for key in attrs.keys():
                        oH5.attrs[key] = attrs[key]
        oH5.close()


#read parameters of the polynomial fit, Wang+,2023
def readPoly(fIn):
        """
        Read a polynomial fit parameters of chorus wave from an input file.

        Args:
                fIn (str): The path to the input file.

        Returns:
                tuple: A tuple containing the length of each row and a numpy array of the parameter table.

        """
        table = []
        with open(fIn, 'r') as file:
                # Skip the first row
                next(file)
                for line in file:
                        row = line.strip().split('\t')[2:-1]  # Discard the first two elements of each row
                        row = [float(x) for x in row]  # Convert the strings to float
                        rowLen = len(row)
                        table.append(np.array(row))
        return (rowLen, np.array(table))

#Chorus polynomial fit for the electron lifetime
def ChorusPoly(Li, Eki, polyArray):
    """
    Calculate the electron lifetime caused by interaction with Chorus Waves using polynomial fit coefficients.

    Parameters:
    Li (array-like): Array of L values.
    Eki (array-like): Array of Ek values.
    polyArray (ndarray): Array of polynomial fit coefficients.

    Returns:
    ndarray: Array of electron lifetimes.

    Notes:
    - The polynomial fit coefficients are obtained from a research paper by Dedong Wang et al. (in preparation).
    - The function calculates the electron lifetime for each Kp and MLT value based on the given Li and Eki values.

    References:
    - [Research Paper](https://doi.org/will be provided)
    """

    lenKp, lenMLT, lenParam = polyArray.shape
    # Extend polyArray
    polyArrayX = polyArray[:, :, :, np.newaxis, np.newaxis]
    # Extend Li and Ki
    lenL = len(Li)
    lenEki = len(Eki)
    Lx = np.tile(Li, (lenEki, 1)).T
    Lx = Lx[np.newaxis, np.newaxis, :, :]
    Ex = np.tile(Eki, (lenL, 1))
    Ex = Ex[np.newaxis, np.newaxis, :, :]

    tau = np.ones((lenKp, lenMLT, lenL, lenEki))

    c0 = polyArrayX[:, :, 0, :, :]  # Intercept
    c1 = polyArrayX[:, :, 1, :, :]  # L
    c2 = polyArrayX[:, :, 2, :, :]  # log10(E)
    c3 = polyArrayX[:, :, 3, :, :]  # L^2
    c4 = polyArrayX[:, :, 4, :, :]  # log10(E)^2
    c5 = polyArrayX[:, :, 5, :, :]  # L^3
    c6 = polyArrayX[:, :, 6, :, :]  # log10(E)^3
    c7 = polyArrayX[:, :, 7, :, :]  # log10(E)*L
    c8 = polyArrayX[:, :, 8, :, :]  # log10(E)*L^2
    c9 = polyArrayX[:, :, 9, :, :]  # log10(E)^2*L

    tau = c0 * tau + \
            c1 * Lx + c2 * Ex + \
            c3 * np.power(Lx, 2) + \
            c4 * np.power(Ex, 2) + \
            c5 * np.power(Lx, 3) + \
            c6 * np.power(Ex, 3) + \
            c7 * Lx * Ex + \
            c8 * np.power(Lx, 2) * Ex + \
            c9 * Lx * np.power(Ex, 2)  # in log10(days)

    tau = 10.0 ** tau * (60. * 60. * 24.)  # in seconds

    return tau

def ReSample(L, MLT, Qp, xMLT):
        """
        Resamples the input data based on the given parameters.

        Args:
                L (array-like): Array of L-shell values.
                MLT (array-like): Array of magnetic local time values.
                Qp (array-like): Array of input data.
                xMLT (float): Magnetic local time value to resample at.

        Returns:
                array-like: 2D (L,MLT) smoothed inupt data resampled in the MLT dimension.

        Raises:
                None

        Notes:
                - This function adds ghosts in MLT to handle periodic boundary.
                - The resampling is performed by setting the center and then left/right strips.
                - Equality at the overlap point is enforced.

        """
        Nr, Np = Qp.shape
        #Add ghosts in MLT to handle periodic boundary
        Ng = 2
        Npg = Np+Ng*2
        gMLT = np.arange(0-Ng,24+Ng+1)
        Qpg = np.zeros((Nr,Npg))
        #Set center and then left/right strips
        Qpg[:,2:-2] = Qp
        Qpg[:,1] = Qp[:,-1]
        Qpg[:,0] = Qp[:,-2]
        Qpg[:,-1] = Qp[:,0]
        Qpg[:,-2] = Qp[:,1]

        Q = np.log10(Qpg)
        upQ = RectBivariateSpline(L,gMLT,Q,s=10)

        Qu = upQ(L,xMLT)
        xQp = 10.0**(Qu)
        #Enforce equality at overlap point
        tauP = 0.5*(xQp[:,0]+xQp[:,-1])
        xQp[:, 0] = tauP
        xQp[:,-1] = tauP

        return xQp

def genChorus(params, fInChorus):
        """
        Generate chorus wave model parameters.

        Args:
                params (object): Object containing the parameters for the chorus wave model.
                fInChorus (str): File path to the input file.

        Returns:
                tuple: A tuple containing the following arrays:
                        - Kpi (numpy.ndarray): Array of Kp indices.
                        - xMLTi (numpy.ndarray): Array of MLT indices.
                        - Li (numpy.ndarray): Array of L indices.
                        - Eki (numpy.ndarray): Array of E indices.
                        - tauX (numpy.ndarray): Array of tau values.

        """
        print("Dimension of parameters in Chorus wave model, Kp:",params.nKp,"MLT:",params.nMLT,"L:",params.nL,"Ek:",params.nEk)
        dimKp = params.nKp #maximum Kp allowed
        rowLen,paramArray = readPoly(fInChorus)
        polyArray = paramArray.reshape(24,7,rowLen) #dim MLT: 24, Dim Kp: 7
        polyArray = polyArray[:,:dimKp,:] #use Kp = 1,2,...,maxKp
        polyArray = polyArray.transpose(1, 0, 2) #shape (,24,rowLen)
        lenMLT = 24
        #Kpi
        startValue = 1.0 #Kp index in real number
        endValue = float(dimKp)
        lenKp = dimKp
        Kpi = np.linspace(startValue, endValue, num=lenKp) 
        #Eki
        startValue = 1.0e-3 #in MeV
        endValue = 2.0  
        lenEk = 155  
        Eki = np.linspace(np.log10(startValue), np.log10(endValue), lenEk) #in log10(MeV)
        #Li
        startValue = 3.0 #in Re 
        endValue = 7.0
        lenL = 41  
        Li = np.linspace(startValue, endValue, num=lenL) 
        #Tau from polynomial fit
        tauP = ChorusPoly(Li,Eki,polyArray)
        #expand MLT from 0-23 to 0-24
        extraMLT0 = tauP[:, 0, :, :][:,np.newaxis,:,:]
        tauE = np.concatenate((tauP, extraMLT0), axis=1)
        tauE = tauE.T
        #Interpolation in the MLT dimesion
        xFac = 4
        lenMLTx = lenMLT*xFac+1 # 97
        MLTi = np.linspace(0,24,lenMLT+1)
        xMLTi = np.linspace(0,24,lenMLTx) 
        tauX = np.zeros((lenEk,lenL,lenMLTx,lenKp))
        # Smoothing in MLT
        for i, j in np.ndindex(tauX.shape[0], tauX.shape[3]):
            Q = tauE[i, :, :, j]
            tauX[i, :, :, j] = ReSample(Li, MLTi, Q, xMLTi)
        Eki = 10.0**Eki #in MeV

        return Kpi,xMLTi,Li,Eki,tauX


