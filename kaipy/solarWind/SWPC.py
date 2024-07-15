# Custom
import kaipy.transform
from kaipy.solarWind.SolarWind import SolarWind
from kaipy.solarWind.OMNI import OMNI
from kaipy.kdefs import *

# 3rd party
import numpy
import netCDF4 as nc
from cdasws import CdasWs

# Standard
import datetime
import re
import os

class DSCOVRNC(OMNI):
    """
    Args:
        t0 (datetime.datetime): The start time for the data.
        t1 (datetime.datetime): The end time for the data.
        doFilter (bool): Whether to apply a filter to the data.
        sigmaVal (float): The sigma value for the filter.

    Attributes:
        filter (bool): Whether to apply a filter to the data.
        sigma (float): The sigma value for the filter.
        bad_data (list): A list of values considered as bad data.
        data (TimeSeries): The TimeSeries object to store the data.

    Methods:
        __init__(self, t0, filename=None, doFilter=False, sigmaVal=3.0): Initializes the DSCOVR object.
        __read(self, filename, t0): Reads the solar wind file and stores the results in self.data TimeSeries object.
        __readData(self, filename, t0): Reads the data from the file and returns a 2D array containing the data.
        _storeDataDict(self, dates, dataArray, hasBeenInterpolated): Populates self.data TimeSeries object with the data.
        __appendMetaData(self, date, filename): Adds standard metadata to the data dictionary.
    """

    def __init__(self,t0,t1,doFilter = False, sigmaVal = 3.0):
        """
        Initializes an instance of the DSCOVRNC class.

        Args:
            t0 (datetime.datetime): Start time.
            t1 (datetime.datetime): End time.
            filename (str, optional): The name of the file to read data from. Defaults to None.
            doFilter (bool, optional): Flag indicating whether to apply filtering. Defaults to False.
            sigmaVal (float, optional): The sigma value for filtering. Defaults to 3.0.
        """
        SolarWind.__init__(self)

        self.filter = doFilter
        self.sigma = sigmaVal

        self.bad_data = [-999.900, 
                         99999.9, # V
                         9999.99, # B
                         999.990, # density
                         1.00000E+07, # Temperature
                         9999999.0, # Temperature
                         99999, # Activity indices 
                         -99999,
                         1e+20
                         ]
        self.__read(t0,t1)

    def __read(self, t0,t1):
        """
        Read the solar wind file and store the results in the self.data TimeSeries object.

        Args:
            t0 (datetime): The start time of the data.
            t1 (datetime): The end time of the data.

        Returns:
            tuple: A tuple containing the start date, dates, and rows of the data.

        Raises:
            Exception: If the file list is not the same or if there are missing files for the given date range.
        """
        (startDate, dates, data) = self.__readData(t0,t1)
        (dataArray, hasBeenInterpolated) = self._removeBadData(data)
        if self.filter:
            (dataArray, hasBeenInterpolated) = self._coarseFilter(dataArray, hasBeenInterpolated)
        self._storeDataDict(dates, dataArray, hasBeenInterpolated)
        self.__appendMetaData(startDate)
        self._appendDerivedQuantities()

    def __readData(self, t0,t1):
        """
        Read the solar wind file and store the results in the self.data TimeSeries object.

        Args:
            t0 (datetime): The start time of the data.
            t1 (datetime): The end time of the data.

        Returns:
            tuple: A tuple containing the start date, dates, and rows of the data.

        Note:
             read the fmt and figure out which column is which.  This would make things easier
        and more user friendly.  However, for now, we'll just assume the file is exactly
        these quantities in this order
        """

        filelist = os.listdir()
        pop = []
        f1m = []
        m1m = []
        fmt1 = '%Y%m%d%H%M%S'
        fmt2 = '%Y%m%d%H%M%S'
        jud0 = datetime.datetime(1970,1,1,0,0,0,0)
        
        for f in filelist:
            if f[0:2] == 'oe':
                ctime = datetime.datetime.strptime(f[15:29],fmt1)
                etime = datetime.datetime.strptime(f[31:45],fmt2)
                if (ctime >= t0 and ctime <=t1) or (t0 <= ctime and ctime <= t1) or (t0 <= etime and etime <= t1):
                    if 'pop' in f:
                        pop.append(f)
                    if 'f1m' in f:
                        f1m.append(f)
                    if 'm1m' in f:
                        m1m.append(f)
                        
        pop = np.sort(pop)
        f1m = np.sort(f1m)
        m1m = np.sort(m1m)
        
        if len(pop) != len(f1m) or len(f1m) != len(m1m) or len(pop) != len(m1m):
            raise Exception('file list not the same')
        if len(pop) == 0 or len(f1m) == 0 or len(m1m) == 0:
            raise Exception('missing files for this daterange')
        
        mtime = []
        ftime = []
        ptime = []
        n = []
        vx = []
        vy = []
        vz = []
        temp = []
        bx = []
        by = []
        bz = []
        satx = []
        for i in range(len(pop)):
            pfn = pop[i]
            ffn = f1m[i]
            mfn = m1m[i]
            pds = nc.Dataset(pfn) #time, sat_x_gse
            fds = nc.Dataset(ffn) #time,proton_density, proton_vx_gse, proton_vy_gse, proton_vz_gse, proton_temperature
            mds = nc.Dataset(mfn) #time, bx_gse, by_gse, bz_gse
            for k in range(len(mds['time'])):
                mtime.append(jud0 + datetime.timedelta(milliseconds=mds['time'][:][k]))
                bx.append(mds['bx_gse'][:][k])
                by.append(mds['by_gse'][:][k])
                bz.append(mds['bz_gse'][:][k])
            for k in range(len(fds['time'])):
                ftime.append(jud0 + datetime.timedelta(milliseconds=fds['time'][:][k]))
                n.append(fds['proton_density'][:][k])
                vx.append(fds['proton_vx_gse'][:][k])
                vy.append(fds['proton_vy_gse'][:][k])
                vz.append(fds['proton_vz_gse'][:][k])
                temp.append(fds['proton_temperature'][:][k])
            for k in range(len(pds['time'])):
                ptime.append(jud0 + datetime.timedelta(milliseconds=pds['time'][:][k]))
                satx.append(pds['sat_x_gse'][:][k])
        
        dates = []
        rows  = []

        timeshift = int(np.round((np.mean(satx)*-1)/(np.nanmean(vx))/60.0))
        startTime = t0 + datetime.timedelta(minutes=timeshift)

        dsttime,dst = self._getDst(t0,t1)
        ntimes = t1 - t0
        ntimes = int(ntimes.total_seconds()/60.0)

        print("Starting Time: ",startTime.isoformat())
        print("We are using a constant timeshift of: ", timeshift ," minutes")
        #itp = 0 #ptime
        itf = 0 #ftime
        itm = 0 #mtime
        itd = 0 #dsttime

        for i in range(ntimes):
            #currentTime = datetime.datetime(int(yrs[i]),1,1,hour=int(hrs[i]),minute=int(mns[i])) + datetime.timedelta(int(doy[i])-1)
            currentTime = t0 + datetime.timedelta(minutes=i)
            #calculating minutes from the start time
            #nMin = self.__deltaMinutes(currentTime,startTime)
            while(mtime[itm] + datetime.timedelta(minutes=timeshift) < currentTime):
                itm = itm+1
            while(ftime[itf] + datetime.timedelta(minutes=timeshift) < currentTime):
                itf = itf+1
            while(dsttime[itd] < currentTime):
                itd = itd+1
            nMin = i
            
            data = [nMin,bx[itm],by[itm],bz[itm],vx[itf],vy[itf],vz[itf],n[itf],temp[itf],0,0,0,dst[itd],0,0,0]

            dates.append( currentTime )
            rows.append( data )

        return (t0, dates, rows)

    def __appendMetaData(self, date):
        """
        Add standard metadata to the data dictionary.

        Args:
            date (datetime.datetime): The start date of the data.

        Returns:
            None

        """
        metadata = {'Model': 'DSCOVRNC',
                    'Source': 'NOAA DSCOVR NetCDF File',
                    'Date processed': datetime.datetime.now(),
                    'Start date': date
                    }
        
        self.data.append(key='meta',
                         name='Metadata for OMNI Solar Wind file',
                         units='n/a',
                         data=metadata)


class ACESWPC(DSCOVRNC):
    """
    Args:
        t0 (datetime.datetime): The start time for the data.
        t1 (datetime.datetime): The end time for the data.
        doFilter (bool): Whether to apply a filter to the data.
        sigmaVal (float): The sigma value for the filter.

    Attributes:
        filter (bool): Whether to apply a filter to the data.
        sigma (float): The sigma value for the filter.
        bad_data (list): A list of values considered as bad data.
        data (TimeSeries): The TimeSeries object to store the data.

    Methods:
        __init__(self, t0, filename=None, doFilter=False, sigmaVal=3.0): Initializes the DSCOVR object.
        __read(self, filename, t0): Reads the solar wind file and stores the results in self.data TimeSeries object.
        __readData(self, filename, t0): Reads the data from the file and returns a 2D array containing the data.
        _storeDataDict(self, dates, dataArray, hasBeenInterpolated): Populates self.data TimeSeries object with the data.
        __appendMetaData(self, date, filename): Adds standard metadata to the data dictionary.
    """

    def __init__(self,t0,t1,doFilter = False, sigmaVal = 3.0):
        """
        Initializes an instance of the ACESWPC class.

        Args:
            t0 (datetime.datetime): Start time.
            t1 (datetime.datetime): End time.
            doFilter (bool, optional): Flag indicating whether to apply filtering. Defaults to False.
            sigmaVal (float, optional): The sigma value for filtering. Defaults to 3.0.
        """
        SolarWind.__init__(self)

        self.filter = doFilter
        self.sigma = sigmaVal

        self.bad_data = [-999.900,
                         99999.9, # V
                         9999.99, # B
                         999.990, # density
                         1.00000E+07, # Temperature
                         9999999.0, # Temperature
                         99999, # Activity indices
                         -99999,
                         1e+20
                         ]
        self.__read(t0,t1)

    def __read(self, t0,t1):
        """
        Read the solar wind file & store results in self.data TimeSeries object.

        Args:
            t0 (datetime.datetime): Start time.
            t1 (datetime.datetime): End time.

        Returns:
            None
        """
        (startDate, dates, data) = self.__readData(t0,t1,dodcx=dodcx,dcxfile=dcxfile)
        (dataArray, hasBeenInterpolated) = self._removeBadData(data)
        if self.filter:
            (dataArray, hasBeenInterpolated) = self._coarseFilter(dataArray, hasBeenInterpolated)
        self._storeDataDict(dates, dataArray, hasBeenInterpolated)
        self.__appendMetaData(startDate)
        self._appendDerivedQuantities()

    def __readData(self, t0,t1):
        """
        Read the solar wind file and store the results in the self.data TimeSeries object.

        Args:
            t0 (datetime.datetime): Start time.
            t1 (datetime.datetime): End time.

        Returns:
            tuple: A tuple containing the start times and data arrays for SWE, MFI, and OMNI.
        """

        self.__downloadACE(t0,t1)

        filelist = os.listdir()
        pop = []
        f1m = []
        m1m = []
        fmt1 = '%Y%m%d%H%M%S'
        fmt2 = '%Y%m%d%H%M%S'
        jud0 = datetime.datetime(1970,1,1,0,0,0,0)

        for f in filelist:
            if f[0:2] == 'oe':
                ctime = datetime.datetime.strptime(f[15:29],fmt1)
                etime = datetime.datetime.strptime(f[31:45],fmt2)
                if (ctime >= t0 and ctime <=t1) or (t0 <= ctime and ctime <= t1) or (t0 <= etime and etime <= t1):
                    if 'pop' in f:
                        pop.append(f)
                    if 'f1m' in f:
                        f1m.append(f)
                    if 'm1m' in f:
                        m1m.append(f)

        pop = np.sort(pop)
        f1m = np.sort(f1m)
        m1m = np.sort(m1m)

        if len(pop) != len(f1m) or len(f1m) != len(m1m) or len(pop) != len(m1m):
            raise Exception('file list not the same')
        if len(pop) == 0 or len(f1m) == 0 or len(m1m) == 0:
            raise Exception('missing files for this daterange')

        mtime = []
        ftime = []
        ptime = []
        n = []
        vx = []
        vy = []
        vz = []
        temp = []
        bx = []
        by = []
        bz = []
        satx = []
        for i in range(len(pop)):
            pfn = pop[i]
            ffn = f1m[i]
            mfn = m1m[i]
            pds = nc.Dataset(pfn) #time, sat_x_gse
            fds = nc.Dataset(ffn) #time,proton_density, proton_vx_gse, proton_vy_gse, proton_vz_gse, proton_temperature
            mds = nc.Dataset(mfn) #time, bx_gse, by_gse, bz_gse
            for k in range(len(mds['time'])):
                mtime.append(jud0 + datetime.timedelta(milliseconds=mds['time'][:][k]))
                bx.append(mds['bx_gse'][:][k])
                by.append(mds['by_gse'][:][k])
                bz.append(mds['bz_gse'][:][k])
            for k in range(len(fds['time'])):
                ftime.append(jud0 + datetime.timedelta(milliseconds=fds['time'][:][k]))
                '''
                if fds['overall_quality'][:][k] == 0:
                    n.append(fds['proton_density'][:][k])
                    vx.append(fds['proton_vx_gse'][:][k])
                    vy.append(fds['proton_vy_gse'][:][k])
                    vz.append(fds['proton_vz_gse'][:][k])
                    temp.append(fds['proton_temperature'][:][k])
                else:
                    n.append(numpy.nan)
                    vx.append(numpy.nan)
                    vy.append(numpy.nan)
                    vz.append(numpy.nan)
                    temp.append(numpy.nan)
                '''
                n.append(fds['proton_density'][:][k])
                vx.append(fds['proton_vx_gse'][:][k])
                vy.append(fds['proton_vy_gse'][:][k])
                vz.append(fds['proton_vz_gse'][:][k])
                temp.append(fds['proton_temperature'][:][k])
            for k in range(len(pds['time'])):
                ptime.append(jud0 + datetime.timedelta(milliseconds=pds['time'][:][k]))
                satx.append(pds['sat_x_gse'][:][k])

        dates = []
        rows  = []

        timeshift = int(np.round((np.mean(satx)*-1)/(np.nanmean(vx))/60.0))
        startTime = t0 + datetime.timedelta(minutes=timeshift)
        dsttime,dst = self._getDst(t0,t1)
        ntimes = t1 - t0
        ntimes = int(ntimes.total_seconds()/60.0)

        print("Starting Time: ",startTime.isoformat())
        print("We are using a constant timeshift of: ", timeshift ," minutes")
        #itp = 0 #ptime
        itf = 0 #ftime
        itm = 0 #mtime
        itd = 0 #dsttime

        for i in range(ntimes):
            currentTime = t0 + datetime.timedelta(minutes=i)
            #calculating minutes from the start time
            while(mtime[itm] + datetime.timedelta(minutes=timeshift) < currentTime):
                itm = itm+1
            while(ftime[itf] + datetime.timedelta(minutes=timeshift) < currentTime):
                itf = itf+1
            while(dsttime[itd] < currentTime):
                itd = itd+1
            nMin = i

            data = [nMin,bx[itm],by[itm],bz[itm],vx[itf],vy[itf],vz[itf],n[itf],temp[itf],0,0,0,dst[itd],0,0,0]

            dates.append( currentTime )
            rows.append( data )

        return (t0, dates, rows)

    def __downloadACE(self, t0, t1):
        """
        Downloads the ACE data within the specified time range.

        Args:
        t0 (datetime): The start time of the desired time range.
        t1 (datetime): The end time of the desired time range.

        """

        swpcdir = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"
        magpost = "_ace_mag_1m.txt"
        swepost = "_ace_swepam_1m.txt"
        dday = (t1-t0).days
        current_time = t0
        for i in range(dday):
            current_time = t0 + datetime.timedelta(days=i)
            datestr = current_time.strftime('%Y%m%d')
            magstr = swpcdir + datestr + magpost
            swestr = swpcdir + datestr + swepost
            print("Downloading ACE data: ",datestr + magpost)
            print("Downloading ACE data: ",datestr + swepost)
            urllib.request.urlretrieve(magstr,datestr + magpost)
            urllib.request.urlretrieve(swestr,datestr + swepost)
            self.__readACE(datestr,magpost,swepost)

    def __readACE(self,datestr,magpost,swepost):
        """
        Reads the downloaded ACE data within the specified time range.

        Args:
        datestr (datetime): The current time
        magpost (str): Ending string for mag data.
        swepost (str): Ending string for particle data.

        Returns:
        tuple: A tuple containing two lists - 'dsttime' and 'dst'. 'dsttime' contains the datetime objects within the specified time range, and 'dst' contains the corresponding DST values.

        NOTE: THIS ALGORITHM DOES NOT YET KNOW HOW TO SPLIT ACE VELOCITY INTO COMPONENTS. THIS IS INCOMPLETE.

        """
        rdfile = open(datestr+magpost,'r')
        text = rdfile.readlines()
        for i,j in enumerate(text):
            if j[0] == '2' or j[0] == '1':
                magskip = i
                break
        rdfile.close()
        rdfile = open(datestr+swepost,'r')
        text = rdfile.readlines()
        for i,j in enumerate(text):
            if j[0] == '2' or j[0] == '1':
                sweskip = i
                break
        rdfile.close()

        dat = np.genfromtxt(datestr + magpost,skip_header=magskip, autostrip=True,dtype=None)
        magtime = []
        bx = []
        by = []
        bz = []
        for i in dat:
            currenttime = datetime.datetime(i[0],i[1],i[2],i[3],i[5])
            print(currenttime)
            if currenttime >= t0 and currenttime <= t1:
                magtime.append(currenttime)
                bx.append(i[7])
                bx.append(i[8])
                bx.append(i[9])
        dat = np.genfromtxt(datestr + swepost,skip_header=sweskip, autostrip=True,dtype=None)
        swetime = []
        n = []
        v = []
        t = []
        for i in dat:
            currenttime = datetime.datetime(i[0],i[1],i[2],i[3],i[5])
            print(currenttime)
            if currenttime >= t0 and currenttime <= t1:
                magtime.append(currenttime)
                bx.append(i[7])
                bx.append(i[8])
                bx.append(i[9])

        return (dsttime, dst)

    def __appendMetaData(self, date):
        """
        Add standard metadata to the data dictionary.

        Args:
            date (datetime.datetime): The start date of the data.

        Returns:
            None

        """
        metadata = {'Model': 'ACESPWC',
                    'Source': 'NOAA ACE SWPC',
                    'Date processed': datetime.datetime.now(),
                    'Start date': date
                    }

        self.data.append(key='meta',
                         name='Metadata for ACE Solar Wind',
                         units='n/a',
                         data=metadata)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
