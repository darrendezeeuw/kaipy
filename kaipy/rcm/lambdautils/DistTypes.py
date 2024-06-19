from dataclasses import dataclass
#from dataclasses_json import dataclass_json
from dataclasses import asdict as dc_asdict
from typing import Optional, List

import numpy as np

# dataclasses_json isn't a default package. Since its only used for reading, don't want to make it a requirement for everyone
try:
    from dataclasses_json import dataclass_json
    dataclasses_json_module_imported = True
except ModuleNotFoundError:
    dataclass_json = None
    dataclasses_json_module_imported = False

def conditional_decorator(dec, dataclasses_json_module_imported):
    """
    A decorator that conditionally applies another decorator based on the availability of the dataclasses_json module.

    Args:
        dec: The decorator to be applied if the dataclasses_json module is imported.
        dataclasses_json_module_imported: A boolean indicating whether the dataclasses_json module is imported.

    Returns:
        The decorated function if the dataclasses_json module is imported, otherwise returns the function unchanged.
    """
    def decorator(func):
        if not dataclasses_json_module_imported:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator


def getDistTypeFromKwargs(**kwargs):
    """This function takes a set of keyword arguments and determines which DistType implementation they belong to based on the 'name' key.
    
    Args:
        **kwargs: A set of keyword arguments.
        
    Returns:
        An object of the corresponding DistType implementation based on the 'name' key.
    """
    if 'name' in kwargs.keys():    
        if kwargs['name'] == 'Wolf':
            return DT_Wolf.from_dict(kwargs)
        elif kwargs['name'] == 'ValueSpec':
            return DT_ValueSpec.from_dict(kwargs)
    else:
        return DistType.from_dict(kwargs)


#------
# Parameters needed to determine lambda distribution
#------

@dataclass
class DistType:
    """
    Represents a distribution type.

    Attributes:
        name (str): The name of the distribution type.
    """
    name: str = "Empty"

    
#------
# Specific implementations of DistType
#------

#@dataclass_json
@conditional_decorator(dataclass_json, dataclasses_json_module_imported)
@dataclass
class DT_Manual(DistType):
    """
    Represents a manual distribution type.

    This class can be used to allow users to add any additional information they want to save for a manual distribution type.

    Attributes:
        name (str): The name of the distribution type.
    """

    def __post_init__(self):
        if self.name == "Empty":
            self.name = "Manual"

#@dataclass_json
@conditional_decorator(dataclass_json, dataclasses_json_module_imported)
@dataclass
class DT_Wolf(DistType):
    """Lambda channel spacing based on Wolf's notes. 
    Ask Anthony Sciola or Frank Toffoletto for a copy
    With the addition that there can be 2 p values for the start and end, and pStar transitions between them

    Attributes:
        p1 (float): The p value for the start.
        p2 (float): The p value for the end.
    """

    p1: float = None
    p2: float = None

    def __post_init__(self):
        """Initialize the object after its creation."""
        self.name = "Wolf"

    def genAlamsFromSpecies(self, sP):
        """
        Generate Alams from Species.

        This method generates Alams based on the given Species object.

        Parameters:
        - sP: The Species object containing the necessary information.

        Returns:
        - The generated Alams.

        """
        return self.genAlams(sP.n, sP.amin, sP.amax)


    def genAlams(self, n, amin, amax, kmin=0, kmax=-1):
        """
        Generate a list of 'n' lambda values based on the given parameters.

        Args:
            n (int): The number of lambda values to generate.
            amin (float): The minimum lambda value.
            amax (float): The maximum lambda value.
            kmin (int, optional): The minimum channel range. Defaults to 0.
            kmax (int, optional): The maximum channel range. Defaults to -1.

        Returns:
            list: A list of 'n' lambda values.

        """
        if kmax == -1: kmax = n

        alams = []
        for k in range(n):
            kfrac = (k-kmin)/(kmax-kmin)  # How far through the channel range are we
            pstar = (1-kfrac)*self.p1 + kfrac*self.p2
            lammax = amax-amin
            lam = lammax*((k - kmin + 0.5)/(kmax-kmin + 0.5))**pstar + amin
            alams.append(lam)
        return alams


#@dataclass_json
@conditional_decorator(dataclass_json, dataclasses_json_module_imported)
@dataclass
class ValueSpec:
    """
    Represents a value specification with start, end, scale type, and optional parameters.

    Attributes:
        start (float): The starting value.
        end (float): The ending value.
        scaleType (str): The scale type. Must be one of ['lin', 'log', 'spacing_lin'].
        n (Optional[int]): The number of channels. Default is None.
        c (Optional[float]): The spacing parameter. Default is None.

    Methods:
        __post_init__(): Initializes the ValueSpec object and performs validation.
        eval(doEnd): Performs the appropriate operation based on the scale type and returns a list of values.
    """
    start: float
    end: float
    scaleType: str  # See 'goodScaleTypes' below for valid strings
    n: Optional[int] = None  # Number of channels
    c: Optional[float] = None
    """ c has different meanings depending on scaleType
                lin: If given, will use set spacing c
                log: If given, will use log of base c
                spacing_lin: If given, will use as Sum_{k=1}^{N} c*k, where N is calculated based on start, end, and c
    """

    def __post_init__(self):
        """
        Initializes the ValueSpec object and performs validation.

        If the scaleType is not one of ['lin', 'log', 'spacing_lin'], it defaults to 'lin'.
        If neither n nor c is provided, and the scaleType is 'lin', c defaults to 1.
        """
        goodScaleTypes = ['lin', 'log', 'spacing_lin']
        if self.scaleType not in goodScaleTypes:
            print("Error in ValueSpec, scaleType must be in {}, not {}. Defaulting to {}".format(goodScaleTypes, self.scaleType, goodScaleTypes[0]))
            self.scaleType = goodScaleTypes[0]
        if self.n is None and self.c is None:
            if self.scaleType == 'lin': self.c = 1
            print("Error in ValueSpec, must provide either (n) or (c). See source code to see what (c) does for each scaleType. Defaulting to " + str(self.c))
            

    def eval(self, doEnd):
        """
        Performs the appropriate operation given self's attributes and returns a list of values.

        Args:
            doEnd (bool): Indicates whether to include the end value in the output.

        Returns:
            list: A list of values based on the scale type and other attributes of the ValueSpec object.
        """
        if self.scaleType == 'lin':
            line = np.linspace(sL.start, sL.end, sL.n, endpoint=doEnd)
        elif self.scaleType == 'log':
            lbase = self.c
            sign = 1 if self.start > 0 else -1
            start = np.log(np.abs(self.start))/np.log(lbase)
            end = np.log(np.abs(self.end))/np.log(lbase)
            line = np.logspace(start, end, self.n, base=lbase, endpoint=doEnd)
        elif self.scaleType == 'spacing_lin':
            diff = self.end-self.start
            if self.c is not None:
                #Set n based on c if needed
                #But also force n to be an integer
                self.n = int(0.5*(np.sqrt(8*diff/self.c + 1) + 1))
            #(Re)calculate c based on integer n
            self.c = 2*diff/(self.n**2 + self.n)
            print("Spacing_lin: n={}, c={}".format(self.n, self.c))

            spacings = np.array([self.c*k for k in range(self.n)])
            line = np.array([self.start + np.sum(spacings[:k]) for k in range(self.n)])

        return line

#@dataclass_json
@conditional_decorator(dataclass_json, dataclasses_json_module_imported)
@dataclass
class DT_ValueSpec(DistType):
    """Lambda channel spacing based on a series of slope specifications.

    Attributes:
        specList (List[ValueSpec]): List of ValueSpec objects representing the slope specifications.

    Methods:
        __post_init__: Post-initialization method to check if all slopes are contiguous.
        genAlamsFromSpecies: Generates alams from species parameters, adjusting the endpoints if necessary.
        genAlams: Generates alams based on the slope specifications.

    """

    specList: List[ValueSpec] = None

    def __post_init__(self):
        self.name = "ValueSpec"
        # Check to see if all slopes are contiguous
        if len(self.specList) > 1:
            tol = 1E-4
            for i in range(len(self.specList)-1):
                if np.abs(self.specList[i].end - self.specList[i+1].start) > tol:
                    print("Error creating a DistType_SlopeSpec: SlopeSpec[{}].end ({}) != SlopeSpec[{}].start ({}). Undefined behavior"\
                        .format(i, self.specList[i].end, i+1, self.specList[i+1].start))

    def genAlamsFromSpecies(self, sP):
        """Generates alams from species parameters, adjusting the endpoints if necessary.

        Args:
            sP: Species parameters object.

        Returns:
            List[float]: List of alams.

        """
        # See if end points match up
        tol = 1E-4
        if np.abs(self.specList[0].start - sP.amin) > tol:
            print("SpecList[0].start={}, SpecParams.amin={}. Overwriting SpecParams.amin to SpecList[0].start".format(self.specList[0].start, sP.amin))
            sP.amin = self.specList[0].start
        if np.abs(self.specList[-1].end - sP.amax) > tol:
            print("SpecList[-1].end={}, SpecParams.amax={}. Overwriting SpecParams.amax to SpecList[-1].end".format(self.specList[0].start, sP.amin))
            sP.amax = self.specList[-1].end
        return self.genAlams(sP.n, sP.amin,sP.amax)

    def genAlams(self, n, amin, amax):
        """Generates alams based on the slope specifications.

        Args:
            n (int): Number of alams to generate.
            amin (float): Minimum value for alams.
            amax (float): Maximum value for alams.

        Returns:
            List[float]: List of alams.

        """
        nSL = len(self.specList)
        alams = np.array([])
        for i in range(nSL):
            doEnd = False if i < nSL-1 else True
            alams = np.append(alams, self.specList[i].eval(doEnd))
        return alams.tolist()







