import datetime

import numpy

import kaipy.transform
from kaipy.solarWind.TimeSeries import TimeSeries 

class SolarWind(object):
    """
    This class serves as an abstract base class for Solar Wind processing. 

    Derived classes should implement the necessary methods to read in Solar Wind data and store the results in a standard way using the `kaipy.solarWind.TimeSeries` object. The derived classes should also make a call to the `_appendDerivedQuantities()` method to compute additional solar wind variables.

    Attributes:
        data (TimeSeries): The TimeSeries object that stores all the Solar Wind data.

    Methods:
        __init__(): Initializes the SolarWind object. See file for variables that must be set.
        _getTiltAngle(dateTime): Get the tilt angle with respect to season for the current Date & Time.
        _gsm2sm(dateTime, x, y, z): Convert from GSM to SM coordinates for the current Date & Time.
        bxFit(): Compute and return coefficients for a multiple linear regression fit of Bx to By & Bz.
        _appendDerivedQuantities(): Calculate and append standard derived quantities to the data dictionary.
    """

    def __init__(self):
        """
        Derived classes must read in Solar Wind data and store the
        results in a standard way via the a kaipy.solarWind.TimeSeries object.
        Variables that must be set are:

        key       |  name                         |  units
        ----------------------------------------------
        n         |  Density                      |  #/cc
        vx        |  Velocity vector (gsm)        |  km/s
        vy        |  Velocity vector (gsm)        |  km/s
        vz        |  Velocity vector (gsm)        |  km/s
        t         |  Temperature                  |  k Kelvin
        cs        |  Sound speed                  |  km/s
        bx        |  Magnetic field vector (gsm)  |  nT
        by        |  Magnetic field vector (gsm)  |  nT
        bz        |  Magnetic field vector (gsm)  |  nT
        b         |  Magnetic field vector        |  nT
        time_min  |  Elapsed time since start     |  minutes
        -----------------------------------------------
        meta     | Metadata about the run.  Must  |  n/a
                 | include ['data']['Start Date'] |
                 | for coordinate transforms.     |

        Derived classes should make a call to
        SolarWind._appendDerivedQuantities() to compute some
        additional solar wind variables.
        """
        # The TimeSeries object stores all the Solar Wind data.
        self.data = TimeSeries()
        
            
    def _getTiltAngle(self, dateTime):
        """
        Get the tilt angle for the current Date & Time.

        Args:
            dateTime (datetime): The date and time for which the tilt angle is calculated.

        Returns:
            float: The tilt angle in radians.
        """
        (x,y,z) = kaipy.transform.SMtoGSM(0,0,1, dateTime)

        return numpy.arctan2(x,z)

    def _gsm2sm(self, dateTime, x,y,z):
        """
        Convert from GSM to SM coordinates for the current Date & Time.

        Args:
            dateTime (datetime): The date and time for which the coordinates are converted.

        Returns:
            tuple: The converted coordinates (x, y, z) in SM coordinates.
        """        
        return kaipy.transform.GSMtoSM(x,y,z, dateTime)


    def bxFit(self):
        """
        Compute and return coefficients for a multiple linear regression fit of Bx to By & Bz.

        Args:
            None

        Returns:
            numpy.ndarray: The coefficients of the linear regression fit.

        Notes:
            - The linear regression fit is applied to the Bx, By, and Bz data stored in the SolarWind object.
            - Before performing the fit, the Bx, By, and Bz data are converted to SM coordinates.
            - The fit is performed using the OLS method from umpy.linalg.lstsq
        """
        # Before doing anything, convert to SM coordinates.
        bx_sm = []
        by_sm = []
        bz_sm = []

        for i,time in enumerate(self.data.getData('time_min')):
            b_sm = self._gsm2sm(self.data.getData('meta')['Start date']+datetime.timedelta(minutes=time),
                                self.data.getData('bx')[i],
                                self.data.getData('by')[i],
                                self.data.getData('bz')[i])
            bx_sm.append(b_sm[0])
            by_sm.append(b_sm[1])
            bz_sm.append(b_sm[2])

        bx_sm = numpy.squeeze(numpy.array(bx_sm))
        by_sm = numpy.squeeze(numpy.array(by_sm))
        bz_sm = numpy.squeeze(numpy.array(bz_sm))

        # Now that we're in SM, do the fit!

        A = numpy.vstack(( by_sm, bz_sm, numpy.ones_like(by_sm))).T
        npcoeffs = numpy.linalg.lstsq(A, bx_sm, rcond=None)[0]
        reoderedcoeffs = numpy.array([npcoeffs[2], npcoeffs[0], npcoeffs[1]])

        return reoderedcoeffs

    def _appendDerivedQuantities(self):
        """
        Calculate and append standard derived quantities to the data dictionary.

        Note: single '_' underscore so this function can be called by derived classes.
        """

        # --- Magnetic Field magnitude
        if 'b' not in self.data:
            b = numpy.sqrt(self.data.getData('bx')**2 +
                           self.data.getData('by')**2 +
                           self.data.getData('bz')**2)
            self.data.append('b', 'Magnitude of Magnetic Field', r'$\mathrm{nT}$', b)
        else:
            b = self.data.getData('b')

        # --- Velocity Field magnitude
        if 'v' not in self.data:
            v = numpy.sqrt(self.data.getData('vx')**2 +
                           self.data.getData('vy')**2 +
                           self.data.getData('vz')**2)
            self.data.append('v', 'Magnitude of Velocity', r'$\mathrm{km/s}$', v)
        else:
            v = self.data.getData('v')

        # -- Sound Speed
        if 'cs' not in self.data:
            try:
                cs = numpy.sqrt(5.0*1e3*self.data.getData('t')*(1.38e-23)/(3.0*1.67e-27)/(1.0e6))
                self.data.append('cs', 'Sound Speed', r'$\mathrm{km/s}$', cs)
            except KeyError:
                raise KeyError('Could not find temperature \'t\'.  Cannot compute sound speed (cs) without Temperature (t)!')

        # --- Alfven speed
        if 'va' not in self.data:
            va = (self.data.getData('b') * 1.0e-10 /
                  numpy.sqrt(1.97e-24*4*numpy.pi*
                             self.data.getData('n')) )
            self.data.append('va', 'Alfven Speed', r'$\mathrm{km/s}$', va)
        
        # --- Magnetosonic mach number (dimensionless)
        if 'ms' not in self.data:
            ms = v / self.data.getData('cs')
            self.data.append('ms', 'Magnetosonic Mach Number', '', ms)

        # --- Alfvenic Mach Number (dimensionless)
        if 'ma' not in self.data:
            ma = v/va
            self.data.append('ma', 'Alfvenic Mach Number', '', ma)

        # --- Temperature (Kelvin)
        if 't' not in self.data:
            t = 1e-3*(self.data.getData('cs')**2)*1.0e6*1.67e-27/1.38e-23
            self.data.append('t', 'Temperature', r'$\mathrm{kK}$', t)

        # --- Hours since start
        if 'time_hr' not in self.data:
            hr = self.data.getData('time_min')/60.0
            self.data.append('time_hr', 'Time (hours since start)', r'$\mathrm{hour}$', hr)

        # --- datetime
        if 'time' not in self.data:
            time = []
            for minute in self.data.getData('time_min'):
                time.append( self.data.getData('meta')['Start date'] + datetime.timedelta(minutes=minute) )
            self.data.append('time', 'Date and Time', r'$\mathrm{Date/Time}$', time)
            
        # --- Compute & store day of year
        if 'day' not in self.data:
            doy = []
            for dt in self.data.getData('time'):
                tt = dt.timetuple()
                dayFraction = (tt.tm_hour+tt.tm_min/60.+tt.tm_sec/(60.*60.))/24.
                doy.append( float(tt.tm_yday) + dayFraction )
            self.data.append('time_doy', 'Day of Year', r'$\mathrm{Day}$', doy)
