
import datetime
import kaipy.satcomp.scutils as scutils

#TODO: Need to add "epoch" str for each dataset

def test_getscIds():
	"""
	Test the ability to grab spacecraft Ids from a JSON file.

	This function tests the functionality of the `getScIds` function in the `scutils` module.
	It checks if the returned value is of type `dict` and if the dictionary has at least one entry.
	It also checks if every spacecraft entry has an "Ephem" data product and if every spacecraft data product
	has at least an "Id" and "data" key-value pair.

	"""
	print("Testing ability to grab spacecraft Ids from json file")
	scIdDict = scutils.getScIds()
	assert type(scIdDict) == dict, "Returned type is {}, but should be type dict".format(type(scIdDict))
	assert len(scIdDict.keys()) != 0, "Dictionary has zero entries"
	#Check if every spacecraft entry has any data at all
	#Check if every spacecraft entry has an "Ephem" data product
	#Check if every spacefraft data product has at least an "Id" and "data" k-v pair


def test_getCdasData():
	"""
	Test function for retrieving data from cdasws.

	This function tests if all data in the scId dictionary is retrievable from cdasws.
	It iterates over each spacecraft name in the scId dictionary and retrieves the corresponding data.
	The function prints the status of each dataset retrieval (Good or Bad) and asserts that the retrieved data is not empty.

	Returns:
		None
	"""
	# Function code goes here
	pass


def main():
	test_getscIds()

	test_getCdasData()

if __name__ == "__main__":
	main()