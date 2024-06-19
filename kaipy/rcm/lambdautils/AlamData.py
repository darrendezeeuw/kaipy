from dataclasses import dataclass
#from dataclasses_json import dataclass_json
from dataclasses import asdict as dc_asdict
from typing import Optional, List

import kaipy.rcm.lambdautils.AlamParams as aP

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
		dec: The decorator to be applied.
		dataclasses_json_module_imported: A boolean value indicating whether the dataclasses_json module is imported.

	Returns:
		The decorated function if the dataclasses_json module is imported, otherwise returns the function unchanged.
	"""
	def decorator(func):
		if not dataclasses_json_module_imported:
			# Return the function unchanged, not decorated.
			return func
		return dec(func)
	return decorator

@dataclass
class Species:
	"""
	Represents a species in the RCM model.

	Attributes:
		n (int): Number of channels.
		alams (List[float]): Lambda channel values.
		amins (List[float]): Lower lambda bounds for species.
		amaxs (List[float]): Upper lambda bound for species.
		flav (int): "Flavor", used to distinguish species types in RCM.
					1 = electrons, 2 = protons.
		fudge (Optional[float], optional): "Fudge factor" loss ratio.
		params (Optional[aP.SpecParams], optional): Parameters used to generate this instance of Species.
		name (Optional[str], optional): Name of the species.
	"""
	n: int
	alams: List[float]
	amins: List[float]
	amaxs: List[float]
	flav: int
	fudge: Optional[float] = 0
	params: Optional[aP.SpecParams] = None
	name: Optional[str] = None

#@dataclass_json
@conditional_decorator(dataclass_json, dataclasses_json_module_imported)
@dataclass
class AlamData:
	"""Main class that most things will interact with.

	Attributes:
		doUsePsphere (bool): Whether or not this dataset includes a zero-energy plasmasphere channel.
		specs (List[Species]): List of Species objects.
		params (Optional[aP.AlamParams], optional): Parameters used to generate this instance of AlamData.
	"""
	doUsePsphere: bool
	specs: List[Species]
	params: Optional[aP.AlamParams] = None

