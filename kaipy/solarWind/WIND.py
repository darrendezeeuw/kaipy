# Custom
import kaipy.transform
from kaipy.solarWind.SolarWind import SolarWind
from kaipy.solarWind.OMNI import OMNI

# 3rd party
import numpy
import cdasws as cdas

# Standard
import datetime
import re

class WIND(OMNI):
    """
    WIND Solar Wind from file.
    Data stored in GSE coordinates.

    Parameters:
        fSWE (str): Filepath for the SWE data.
        fMFI (str): Filepath for the MFI data.
        fOMNI (str): Filepath for the OMNI data.
        xloc (str): Location of the data.
        tOffset (int): Time offset.
        t0 (datetime.datetime): Start time.
        t1 (datetime.datetime): End time.

    Attributes:
        filter (bool): Flag indicating whether filtering is enabled.
        sigma (float): The number of standard deviations used for filtering.
        bad_data (list): List of values considered as bad data.
        data (TimeSeries): TimeSeries object to store the solar wind data.

    Methods:
        __init__(self, filename=None, doFilter=False, sigmaVal=3.0): Initializes the OMNI object.
        __read(self, filename): Reads the solar wind file and stores the results in self.data TimeSeries object.
        __readData(self, fh): Reads the variables from the file and returns a 2D array containing the data.
        __appendMetaData(self, date, filename): Adds standard metadata to the data dictionary.
        _removeBadData(self, data, datanames=['Time','Bx','By','Bz','Vx','Vy','Vz','n','Temp','AE','AL','AU','SYMH','BowShockX','BowShockY','BowShockZ'], hasBeenInterpolated=None): Linearly interpolates over bad data in the data array.
        _coarseFilter(self, dataArray, hasBeenInterpolated): Uses coarse noise filtering to remove outliers from the data array.
    """

    def __init__(self, fSWE,fMFI,t0,t1,doFilter = False, sigmaVal = 3.0):
        """
        Initialize the WIND class.

        Args:
            fSWE (str): Filepath for the SWE data.
            fMFI (str): Filepath for the MFI data.
            t0 (datetime.datetime): Start time.
            t1 (datetime.datetime): End time.
            doFilter (bool, optional): Flag indicating whether to apply filtering. Defaults to False.
            sigmaVal (float, optional): The sigma value for filtering. Defaults to 3.0.
        """
        SolarWind.__init__(self)

        self.filter = doFilter
        self.bad_data = [-999.900, 
                         99999.9, # V
                         9999.99, # B
                         999.990, # density
                         1.00000E+07, # Temperature
                         99999, # Activity indices 
                         9999000, # SWE del_time
                         -1e31 # SWE & MFI                         
                         ]
        self.good_quality = [4098,14338]#

        self.bad_datetime = [   datetime.datetime(2015,3,17,hour=17,minute=27,second=7, microsecond=488*1000),
                                datetime.datetime(2015,3,18,hour=21,minute=53,second=49,microsecond=140*1000),
                                datetime.datetime(2024,5,9, hour=20,minute=41,second=20,microsecond=906*1000),
                                datetime.datetime(2024,5,9, hour=21,minute=26,second=39,microsecond=515*1000),
                                datetime.datetime(2024,5,9, hour=21,minute=50,second=2, microsecond=515*1000),
                                datetime.datetime(2024,5,9, hour=23,minute=0, second=17,microsecond=273*1000),                                datetime.datetime(2024,5,10,hour=4, minute=22,second=55,microsecond=589*1000),
                                datetime.datetime(2024,5,10,hour=5, minute=26,second=23,microsecond=335*1000),
                                datetime.datetime(2024,5,10,hour=5, minute=27,second=58,microsecond=912*1000),
                                datetime.datetime(2024,5,10,hour=8, minute=40,second=29,microsecond=865*1000),
                                datetime.datetime(2024,5,10,hour=8, minute=43,second=47,microsecond=765*1000),
                                datetime.datetime(2024,5,10,hour=23,minute=42,second=25,microsecond=781*1000),
                                datetime.datetime(2024,5,11,hour=7, minute=32,second=59,microsecond= 31*1000)
                            ]

        print('Retrieving solar wind data from CDAWeb')
        self.__read(fSWE,fMFI,t0,t1)

    def __read(self, fSWE,fMFI,t0,t1):
        """
        Read the solar wind file & store results in self.data TimeSeries object.

        Args:
            fSWE (str): Filepath for the SWE data.
            fMFI (str): Filepath for the MFI data.
            t0 (datetime.datetime): Start time.
            t1 (datetime.datetime): End time.

        Returns:
            None
        """
        (SWEstartDate, MFIstartDate, DSTstartDate, SWEdata, MFIdata, DSTdata, SWEqf) = self.__readData(fSWE,fMFI,t0,t1)


        (SWEdata) = self.__checkGoodData(SWEdata,SWEqf)
        (SWEdataArray, SWEhasBeenInterpolated)   = self._removeBadData(SWEdata )
        (MFIdataArray, MFIhasBeenInterpolated)   = self._removeBadData(MFIdata )
        (DSTdataArray, DSThasBeenInterpolated)   = self._removeBadData(DSTdata)

        if self.filter:
            (SWEdataArray, SWEhasBeenInterpolated)   = self._coarseFilter(SWEdataArray , SWEhasBeenInterpolated )
            (MFIdataArray, MFIhasBeenInterpolated)   = self._coarseFilter(MFIdataArray , MFIhasBeenInterpolated )
            (DSTdataArray, DSThasBeenInterpolated)   = self._coarseFilter(DSTdataArray , DSThasBeenInterpolated)

        (dates, dataArray, hasBeenInterpolated)  = self.__joinData(SWEdataArray, SWEhasBeenInterpolated, 
                                                                    MFIdataArray, MFIhasBeenInterpolated, 
                                                                    DSTdataArray, DSThasBeenInterpolated,
                                                                    t0,t1)
        self.__storeDataDict(dates, dataArray, hasBeenInterpolated)
        self.__appendMetaData(t0, SWEstartDate, fSWE)
        self._appendDerivedQuantities()

        
    def __readData(self, fhSWE,fhMFI,t0,t1):
        """
        Read the solar wind file and store the results in the self.data TimeSeries object.

        Args:
            fhSWE (str): Filepath for the SWE data.
            fhMFI (str): Filepath for the MFI data.
            t0 (datetime.datetime): Start time.
            t1 (datetime.datetime): End time.

        Returns:
            tuple: A tuple containing the start times and data arrays for SWE, MFI, and OMNI.
        """
        tSWE = fhSWE['Epoch']               # datetime
        v    = fhSWE['V_GSE']               # km/s
        qfv  = fhSWE['QF_V']                #
        qfn  = fhSWE['QF_Np']               #
        n    = fhSWE['Np']                  # #/cc
        cs   = fhSWE['THERMAL_SPD']         # km/s
        pos  = fhSWE['SC_pos_gse']          # km
        xloc = numpy.mean(pos[:,0])            # km

        vx   = v[:,0]            #km/s
        vy   = v[:,1]            #km/s
        vz   = v[:,2]            #km/s

        tMFI = fhMFI['Epoch']               #datetime
        b    = fhMFI['BGSEc']                #nT
        bx   = b[:,0]            #nT
        by   = b[:,1]            #nT
        bz   = b[:,2]            #nT

        tshift = ((0 - xloc) / numpy.mean(vx[abs(vx) < 3000]))/60. # t = (x - x_0)/Vx where X = 0, x_0 = xloc, and Vx is mean of Vx in data block in km/s.
        print('tshift:',tshift,"min from ",xloc,"km with mean vx of",numpy.mean(vx[abs(vx) < 3000]),"km/s")

        SWEdates = []
        SWErows = []
        SWEqf = []
        SWEstartTime = tSWE[0]

        dsttime,dst = self._getDst(t0,t1)

        for i in range(len(tSWE)):
            for itime in range(len(self.bad_datetime)):
                if ( abs(self.__deltaMinutes(tSWE[i],self.bad_datetime[itime])) <= 3./60. ):
                    qfv[i] = 0
                    qfn[i] = 0
                
            #calculating minutes from the start time
            nMin = self.__deltaMinutes(tSWE[i],t0)+tshift

            data = [nMin,n[i],vx[i],vy[i],vz[i],cs[i]]

            qf = [qfv[i],qfn[i]]

            SWEdates.append( tSWE[i] )
            SWErows.append ( data    )
            SWEqf.append   ( qf      )

        MFIdates = []
        MFIrows = []
        MFIstartTime = tMFI[0]
        for i in range(len(tMFI)):
          
            #calculating minutes from the start time
            nMin = self.__deltaMinutes(tMFI[i],t0)+tshift
            if abs(bz[i]) > 150:
                bx[i] = 9999.99
                by[i] = 9999.99
                bz[i] = 9999.99

            data = [nMin,bx[i],by[i],bz[i]]

            MFIdates.append( tMFI[i] )
            MFIrows.append( data )

        DSTdates = []
        DSTrows = []
        DSTstartTime = dsttime[0]
        for i in range(len(dsttime)):

            nMin = self.__deltaMinutes(dsttime[i],t0)

            data = [nMin,0,0,0,dst[i],0,0,0]

            DSTdates.append( dsttime[i] )
            DSTrows.append( data )

        return ( SWEstartTime, MFIstartTime, DSTstartTime, SWErows, MFIrows, DSTrows, SWEqf )

    def __checkGoodData(self, data, qf):
        """
        Check the quality flag and set the data to bad if it is flagged as bad.

        Args:
            data (list): The data to be checked.
            qf (list): The quality flag for each data point.

        Returns:
            list: The data with bad values set to the bad data value.
        """
        nvar = len(data[0])
        nqf = len(qf[0])
        ntime = len(data)
        #print(numpy.shape(data),nvar,nqf,ntime)
        for itime in range(ntime):
            for iq in range(nqf):
                if qf[itime][iq] not in self.good_quality:
                    for ivar in range(1,nvar):
                        data[itime][ivar] = -1e31 # SWE & MFI
        return ( data )

    def __joinData(self, SWEdataArray, SWEhasBeenInterpolated, MFIdataArray, MFIhasBeenInterpolated, DSTdataArray, DSThasBeenInterpolated,t0,t1):
        """
        Joins and interpolates data from different arrays based on time.

        Args:
            SWEdataArray (numpy.ndarray): Array containing SWE data.
            SWEhasBeenInterpolated (numpy.ndarray): Array indicating whether SWE data has been interpolated.
            MFIdataArray (numpy.ndarray): Array containing MFI data.
            MFIhasBeenInterpolated (numpy.ndarray): Array indicating whether MFI data has been interpolated.
            DSTdataArray (numpy.ndarray): Array containing DST data.
            DSThasBeenInterpolated (numpy.ndarray): Array indicating whether DST data has been interpolated.
            t0 (datetime.datetime): Start time.
            t1 (datetime.datetime): End time.

        Returns:
            tuple: A tuple containing the following:
                - dates (list): List of datetime objects representing the interpolated time points.
                - dataArray (numpy.ndarray): Array containing the joined and interpolated data.
                - hasBeenInterpolated (numpy.ndarray): Array indicating whether the joined data has been interpolated.
        """
        ntime = self.__deltaMinutes(t1,t0)
        nMin = range(int(ntime))
        n  = numpy.interp(nMin,SWEdataArray[:,0],SWEdataArray[:,1])
        vx = numpy.interp(nMin,SWEdataArray[:,0],SWEdataArray[:,2])
        vy = numpy.interp(nMin,SWEdataArray[:,0],SWEdataArray[:,3])
        vz = numpy.interp(nMin,SWEdataArray[:,0],SWEdataArray[:,4])
        cs = numpy.interp(nMin,SWEdataArray[:,0],SWEdataArray[:,5])
        bx = numpy.interp(nMin,MFIdataArray[:,0],MFIdataArray[:,1])
        by = numpy.interp(nMin,MFIdataArray[:,0],MFIdataArray[:,2])
        bz = numpy.interp(nMin,MFIdataArray[:,0],MFIdataArray[:,3])
        ae = numpy.interp(nMin,DSTdataArray[:,0],DSTdataArray[:,1])
        al = numpy.interp(nMin,DSTdataArray[:,0],DSTdataArray[:,2])
        au = numpy.interp(nMin,DSTdataArray[:,0],DSTdataArray[:,3])
        symh = numpy.interp(nMin,DSTdataArray[:,0],DSTdataArray[:,4])
        xBow = numpy.interp(nMin,DSTdataArray[:,0],DSTdataArray[:,5])
        yBow = numpy.interp(nMin,DSTdataArray[:,0],DSTdataArray[:,6])
        zBow = numpy.interp(nMin,DSTdataArray[:,0],DSTdataArray[:,7])
        nI  = numpy.interp(nMin,SWEdataArray[:,0],SWEhasBeenInterpolated[:,0])
        vxI = numpy.interp(nMin,SWEdataArray[:,0],SWEhasBeenInterpolated[:,1])
        vyI = numpy.interp(nMin,SWEdataArray[:,0],SWEhasBeenInterpolated[:,2])
        vzI = numpy.interp(nMin,SWEdataArray[:,0],SWEhasBeenInterpolated[:,3])
        csI = numpy.interp(nMin,SWEdataArray[:,0],SWEhasBeenInterpolated[:,4])
        bxI = numpy.interp(nMin,MFIdataArray[:,0],MFIhasBeenInterpolated[:,0])
        byI = numpy.interp(nMin,MFIdataArray[:,0],MFIhasBeenInterpolated[:,1])
        bzI = numpy.interp(nMin,MFIdataArray[:,0],MFIhasBeenInterpolated[:,2])
        aeI = numpy.interp(nMin,DSTdataArray[:,0],DSThasBeenInterpolated[:,0])
        alI = numpy.interp(nMin,DSTdataArray[:,0],DSThasBeenInterpolated[:,1])
        auI = numpy.interp(nMin,DSTdataArray[:,0],DSThasBeenInterpolated[:,2])
        symhI = numpy.interp(nMin,DSTdataArray[:,0],DSThasBeenInterpolated[:,3])
        xBowI = numpy.interp(nMin,DSTdataArray[:,0],DSThasBeenInterpolated[:,4])
        yBowI = numpy.interp(nMin,DSTdataArray[:,0],DSThasBeenInterpolated[:,5])
        zBowI = numpy.interp(nMin,DSTdataArray[:,0],DSThasBeenInterpolated[:,6])
        
        dates = []
        dataArray = []
        interped = []
        hasBeenInterpolated = []
        for i in nMin:
            dates.append(t0+datetime.timedelta(minutes=i))

            arr = [nMin[i],bx[i],by[i],bz[i],vx[i],vy[i],vz[i],n[i],cs[i],ae[i],al[i],au[i],symh[i],xBow[i],yBow[i],zBow[i]]
            dataArray.append(arr)
            arr = [bxI[i],byI[i],bzI[i],vxI[i],vyI[i],vzI[i],nI[i],csI[i],aeI[i],alI[i],auI[i],symhI[i],xBowI[i],yBowI[i],zBowI[i]]
            hasBeenInterpolated.append(arr)
    
        return (dates, numpy.array(dataArray,float), numpy.array(hasBeenInterpolated))

    def __storeDataDict(self, dates, dataArray, hasBeenInterpolated):
        """
        Populates the self.data TimeSeries object using the 2D dataArray read from the file.

        Args:
            dates (numpy.ndarray): Array of dates.
            dataArray (numpy.ndarray): 2D array containing the data.
            hasBeenInterpolated (numpy.ndarray): 2D array indicating whether each data point has been interpolated from bad data.

        Returns:
            None
        """
        self._gse2gsm(dates, dataArray)

        self.data.append('time_min', 'Time (Minutes since start)', 'min', dataArray[:,0])

        # Magnetic field
        self.data.append('bx', 'Bx (gsm)', r'$\mathrm{nT}$', dataArray[:,1])
        self.data.append('isBxInterped', 'Is index i of By interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,0])
        
        self.data.append('by', 'By (gsm)', r'$\mathrm{nT}$', dataArray[:,2])
        self.data.append('isByInterped', 'Is index i of By interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,1])

        self.data.append('bz', 'Bz (gsm)', r'$\mathrm{nT}$', dataArray[:,3])
        self.data.append('isBzInterped', 'Is index i of Bz interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,2])

        # Velocity
        self.data.append('vx', 'Vx (gsm)', r'$\mathrm{km/s}$', dataArray[:,4])
        self.data.append('isVxInterped', 'Is index i of Vx interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,3])

        self.data.append('vy', 'Vy (gsm)', r'$\mathrm{km/s}$', dataArray[:,5])
        self.data.append('isVyInterped', 'Is index i of Vy interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,4])

        self.data.append('vz', 'Vz (gsm)', r'$\mathrm{km/s}$', dataArray[:,6])
        self.data.append('isVzInterped', 'Is index i of Vz interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,5])

        # Density
        self.data.append('n', 'Density', r'$\mathrm{1/cm^3}$', dataArray[:,7])
        self.data.append('isNInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,6])

        ## Temperature
        #self.data.append('t', 'Temperature', r'$\mathrm{kK}$', dataArray[:,8]*1e-3)
        #self.data.append('isTInterped', 'Is index i of T interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,7])

        # Sound Speed (Thermal Speed)
        self.data.append('cs', 'Sound speed', r'$\mathrm{km/s}$', dataArray[:,8])
        self.data.append('isCsInterped', 'Is index i of Cs interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,7])

        # Activity Indices
        self.data.append('ae', 'AE-Index', r'$\mathrm{nT}$', dataArray[:,9])
        self.data.append('isAeInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,8])

        self.data.append('al', 'AL-Index', r'$\mathrm{nT}$', dataArray[:,10])
        self.data.append('isAlInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,9])

        self.data.append('au', 'AU-Index', r'$\mathrm{nT}$', dataArray[:,11])
        self.data.append('isAuInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,10])

        self.data.append('symh', 'SYM/H', r'$\mathrm{nT}$', dataArray[:,12])
        self.data.append('isSymHInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,11])

        # Bowshock Location
        self.data.append('xBS', 'BowShockX (gsm)', r'$\mathrm{RE}$', dataArray[:,13])
        self.data.append('isxBSInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,12])

        self.data.append('yBS', 'BowShockY (gsm)', r'$\mathrm{RE}$', dataArray[:,14])
        self.data.append('isyBSInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,13])

        self.data.append('zBS', 'BowShockZ (gsm)', r'$\mathrm{RE}$', dataArray[:,15])
        self.data.append('iszBSInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,14])
        
    def __appendMetaData(self, date, dateshift, filename):
        """
        Append metadata to the TimeSeries object.

        Args:
            date (datetime.datetime): The start date for the data retrieval.
            dateshift (datetime.timedelta): The time difference between the start date and the SWE data start date.
            filename (str): The filepath of the solar wind data.

        Returns:
            None
        """
        metadata = {'Model': 'WIND',
                    'Source': filename,
                    'Date processed': datetime.datetime.now(),
                    'Start date': date,
                    }
        
        self.data.append(key='meta',
                         name='Metadata for WIND CDAWeb',
                         units='n/a',
                         data=metadata)

    
    def __deltaMinutes(self, t1, startDate):
        """
        Returns the number of minutes elapsed between t1 and startDate.

        Args:
            t1 (datetime.datetime): End time.
            startDate (datetime.datetime): Start time.

        Returns:
            float: Number of minutes elapsed.
        """
        diff = t1 - startDate

        return (diff.days*24.0*60.0 + diff.seconds/60.0)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
