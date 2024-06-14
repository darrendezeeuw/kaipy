import os
import json

#Packages needed to handle variable types
import numpy as np
import datetime

#todo move to kaidefs
dtformat = "%Y-%m-%dT%H:%M:%SZ"

#======
#Custom handlers for non-standard types
#======

#TODO: Handle saving/loading attributes for hdf5 data
class CustomEncoder(json.JSONEncoder):
	'''
	Custom JSON encoder that extends the functionality of the base JSONEncoder class.
	'''

	def default(self, obj):
		'''
		Override the default method to handle custom encoding for specific object types.
		Extends the base JSONEncoder class to handle encoding of datetime objects,
		numpy arrays, and numpy base variable types.

		Args:
			obj: The object to be encoded.

		Returns:
			The encoded representation of the object.

		Raises:
			TypeError: If the object cannot be encoded.
		'''
		if isinstance(obj, (datetime.time, datetime.datetime)):
			return obj.strftime(dtformat)

		if isinstance(obj, np.ndarray):
			return {'shape': obj.shape, 'data': obj.tolist()}

		# Handling annoying numpy types
		if isinstance(obj, np.float32):
			return "_f32_{:.16f}".format(obj)
		if isinstance(obj, np.int64):
			return "_i64_{}".format(obj)

		return json.JSONEncoder.default(self, obj)

def customhook(dct):
	'''
	Custom hook function for JSON decoding.

	This function is used as a hook for the `json.loads` function to customize the decoding process.
	It handles specific cases for datetime objects, numpy arrays, and numpy base variable types.

	Parameters:
		dct (dict): The dictionary representing the JSON object being decoded.

	Returns:
		dict: The modified dictionary after applying the custom decoding logic.
	'''
	for key in dct.keys():
		# Handle datetime
		try:
			datetime.datetime.strptime(dct[key][0], dtformat)
			# If we're still here, it worked
			# So go ahead and replace this whole list with the proper datetime objects
			newlist = [datetime.datetime.strptime(dtStr, dtformat) for dtStr in dct[key]]
			dct[key] = newlist
		except:
			pass

		# Handle numpy arrays
		try:
			if 'shape' in dct[key].keys():
				shape = tuple(dct[key]['shape'])
				newdata = np.array(dct[key]['data']).reshape(shape)
				dct[key] = newdata
		except:
			pass

		# Handle numpy base variable types
		try:
			if type(dct[key]) == str and '_f32_' in dct[key]:
				dct[key] = np.float32(dct[key].split('_f32_')[1])
			if type(dct[key]) == str and '_i64_' in dct[key]:
				dct[key] = np.int64(dct[key].split('_i64_')[1])
		except:
			pass

	return dct

#======
#Main functions
#======
def dump(fname, data, action='w'):
	'''
	Store data [dict] in file fname.

	Args:
		fname (str): The file name or path where the data will be stored.
		data (dict): The data to be stored in the file.
		action (str, optional): The action to perform on the file. Defaults to 'w' (write).

	Raises:
		FileNotFoundError: If the specified file path does not exist.

	'''
	with open(fname, action) as jfile:
		json.dump(data, jfile, indent=4, cls=CustomEncoder)

def load(fname):
	'''
	Load JSON data from a file.

	Args:
		fname (str): The path to the JSON file.

	Returns:
		dict: The loaded JSON data as a dictionary.

	Raises:
		FileNotFoundError: If the specified file doesn't exist.
	'''
	if not os.path.exists(fname):
		print("File " + fname + " doesn't exist, can't load json")
		return

	with open(fname, 'r') as jfile:
		data = json.load(jfile, object_hook=customhook)

	return data

def dumps(data, noIndent=False):
	'''
	Returns a string with the given dictionary in JSON format.

	Parameters:
		data: The dictionary to be converted to JSON.
		noIndent: Optional parameter to specify whether to include indentation in the JSON string. Default is False.

	Returns:
		A string representing the dictionary in JSON format.
	'''
	if noIndent:
		return json.dumps(data, cls=CustomEncoder)
	else:
		return json.dumps(data, indent=4, cls=CustomEncoder)

def loads(dataString):
	'''
	Parses a string containing JSON data and returns a dictionary.

	Args:
		dataString (str): The string containing JSON data.

	Returns:
		dict: A dictionary representing the parsed JSON data.

	'''
	return json.loads(dataString, object_hook=customhook)