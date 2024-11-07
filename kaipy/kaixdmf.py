#Various routines to help scripts that create XMF files from H5 data
import numpy as np
import h5py
import xml.etree.ElementTree as et
import kaipy.kdefs as kdefs
import kaipy.kaiH5 as kh5

#Add grid info to step
#Geom is topology subelement, iDims is grid size string
def AddGrid(fname, Geom, iDims, coordStrs):
	"""
	Add a grid to the given Geometry xdmf element.

	Args:
		fname (str): The file name.
		Geom (Element): The geometry xdmf element to add the grid to.
		iDims (str): The dimensions of the grid.
		coordStrs (list): A list of coordinate strings.

	Returns:
		None
	"""
	for coordStr in coordStrs:
		xC = et.SubElement(Geom, "DataItem")
		xC.set("Dimensions", iDims)
		xC.set("NumberType", "Float")
		xC.set("Precision", "4")
		xC.set("Format", "HDF")

		text = fname + ":/" + coordStr
		xC.text = text

#Add data to slice
def AddData(Grid, fname, vID, vLoc, xDims, s0=None):
	"""
	AddData function adds data to the given Grid xdmf element.

	Parameters:
		Grid: The parent element to which the data will be added.
		fname: The file name or path.
		vID: The ID of the attribute.
		vLoc: The location of the attribute.
		xDims: The dimensions of the data item.
		s0: The step number (integer, optional). If not provided, will point to data in root of h5 file.

	Returns:
		None
	"""
	if vLoc != 'Other':
		# Add attribute
		vAtt = et.SubElement(Grid, "Attribute")
		vAtt.set("Name", vID)
		vAtt.set("AttributeType", "Scalar")
		vAtt.set("Center", vLoc)
		
		# Add data item
		aDI = et.SubElement(vAtt, "DataItem")
		aDI.set("Dimensions", xDims)
		aDI.set("NumberType", "Float")
		aDI.set("Precision", "4")
		aDI.set("Format", "HDF")
		
		if s0 is None:
			aDI.text = "%s:/%s" % (fname, vID)
		else:
			aDI.text = "%s:/Step#%d/%s" % (fname, s0, vID)

#Add data item to passed element
def AddDI(elt, h5F, nStp, cDims, vId):
	"""
	Adds a DataItem element to the given element.

	Parameters:
		elt: The parent element to which the DataItem element will be added.
		h5F: The HDF file path.
		nStp: The step number.
		cDims: The dimensions of the DataItem.
		vId: The variable ID.

	Returns:
		None
	"""
	aDI = et.SubElement(elt, "DataItem")
	aDI.set("Dimensions", cDims)
	aDI.set("NumberType", "Float")
	aDI.set("Precision", "4")
	aDI.set("Format", "HDF")
	if nStp >= 0:
		aDI.text = "%s:/Step#%d/%s" % (h5F, nStp, vId)
	else:
		aDI.text = "%s:/%s" % (h5F, vId)


#Get root variables
def getRootVars(fname, gDims):
	"""
	Retrieve the root variables from an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.
		gDims (list[int]): The global dimensions of the data.

	Returns:
		tuple: A tuple containing two lists - `vIds` and `vLocs`.
		`vIds` (list): A list of variable IDs.
		`vLocs` (list): A list of variable locations.

	"""
	with h5py.File(fname, 'r') as hf:
		vIds = []
		vLocs = []
		for k in hf.keys():
			vID = str(k)
			doV = True  # Whether or not to include variable with name k
			if ("Step" in vID or kdefs.grpTimeCache in vID):  # Ignore Steps and timeAttributeCache
				doV = False
			if ((vID == "X") or (vID == "Y") or (vID == "Z")):  # Ignore root geom
				doV = False
			if (isinstance(hf[k], h5py._hl.group.Group)):
				# Ignore any root groups (technically makes Step check unncessary,
				#  but keeping it just in case this gets removed or changed for some reason)
				doV = False

			if (doV):
				Nv = hf[k].shape
				vLoc = getLoc(gDims, Nv)
				# If location is Other (isn't well-described by Cell or Node), don't want to try to make vtk read it
				if (vLoc != "Other"):
					vIds.append(vID)
					vLocs.append(vLoc)
				else:
					print("Excluding %s" % (vID))

	return vIds, vLocs

#Get variables in initial Step
def getVars(fname, gId, gDims, recursive=False):
	"""
	Retrieves the variable names and locations from the specified group ID in an HDF5 file.

	Parameters:
		fname (str): The path to the HDF5 file.
		gId (str): The group ID.
		gDims (list[int]): The dimensions of the group.
		recursive (bool, optional): Flag indicating whether to recursively retrieve variables from subgroups. Defaults to False.

	Returns:
		tuple: A tuple containing two lists - the variable names and their corresponding locations.
	"""
	def getVarName(gId, k, recursive):
		"""
		Returns the variable name based on the given group ID, index, and recursive flag.

		Parameters:
			gId (str): The group ID.
			k (int): The index.
			recursive (bool): Flag indicating whether to include the group name in the variable name.

		Returns:
			str: The variable name.
		"""
		if recursive:
			# Want to keep the group name, since all vars need to be at the same level in xdmf
			spl = gId.split('/')
			return '/'.join(spl[1:]) + '/' + str(k)
		else:
			return str(k)

	with h5py.File(fname, 'r') as hf:
		stp0 = hf[gId]
		vIds = []
		vLocs = []
		for k in stp0.keys():
			vID = getVarName(gId, k, recursive)
			
			# If its a group, recursively call this function to get to the vars and add them to the list
			if isinstance(stp0[k], h5py._hl.group.Group):
				vIdR, vLR = getVars(fname, gId+'/'+k, gDims, recursive=True)
				for vi, vl in zip(vIdR, vLR):
					vIds.append(vi)
					vLocs.append(vl)
			else:
				Nv = stp0[k].shape
				vLoc = getLoc(gDims, Nv)
				if (vLoc != "Other"):
					vIds.append(vID)
					vLocs.append(vLoc)
				else:
					print("Excluding %s" % (vID))
	return vIds, vLocs


def printVidAndLocs(vIds: list, vLocs: list):
	"""
	Prints variable ids and their locations (cell vs node)

	Parameters:
		vIds: List of variable IDs/names
		vLocs: List of variable locations

	Returns:
		None

	"""
	if len(vIds) == 0:
		print("None")
		return
	
	print("Variable ID/name and grid location:")
	for vId, vLoc in zip(vIds, vLocs):
		print("{}\t: {}".format(vId, vLoc))

def AddVectors(Grid, fname, vIds, cDims, vDims, Nd, nStp):
	"""
	Adds vector attributes to the given Grid element based on the provided parameters.

	Parameters:
		Grid: The Grid element to which the vector attributes will be added.
		fname: The file name.
		vIds: A list of vector identifiers.
		cDims: The cell dimensions.
		vDims: The vector dimensions.
		Nd: The number of dimensions.
		nStp: The step number.

	Returns:
		None
	"""
	# Velocity (2D)
	if (Nd == 2) and ("Vx" in vIds) and ("Vy" in vIds):
		vAtt = et.SubElement(Grid, "Attribute")
		vAtt.set("Name", "VecV")
		vAtt.set("AttributeType", "Vector")
		vAtt.set("Center", "Cell")
		fDI = et.SubElement(vAtt, "DataItem")
		fDI.set("ItemType", "Function")
		fDI.set("Dimensions", vDims)
		fDI.set("Function", "JOIN($0 , $1)")
		AddDI(fDI, fname, nStp, cDims, "Vx")
		AddDI(fDI, fname, nStp, cDims, "Vy")

	# Velocity (3D)
	if (Nd == 3) and ("Vx" in vIds) and ("Vy" in vIds) and ("Vz" in vIds):
		vAtt = et.SubElement(Grid, "Attribute")
		vAtt.set("Name", "VecV")
		vAtt.set("AttributeType", "Vector")
		vAtt.set("Center", "Cell")
		fDI = et.SubElement(vAtt, "DataItem")
		fDI.set("ItemType", "Function")
		fDI.set("Dimensions", vDims)
		fDI.set("Function", "JOIN($0 , $1 , $2)")
		AddDI(fDI, fname, nStp, cDims, "Vx")
		AddDI(fDI, fname, nStp, cDims, "Vy")
		AddDI(fDI, fname, nStp, cDims, "Vz")

	# Magnetic field (2D)
	if (Nd == 2) and ("Bx" in vIds) and ("By" in vIds):
		vAtt = et.SubElement(Grid, "Attribute")
		vAtt.set("Name", "VecB")
		vAtt.set("AttributeType", "Vector")
		vAtt.set("Center", "Cell")
		fDI = et.SubElement(vAtt, "DataItem")
		fDI.set("ItemType", "Function")
		fDI.set("Dimensions", vDims)
		fDI.set("Function", "JOIN($0 , $1)")
		AddDI(fDI, fname, nStp, cDims, "Bx")
		AddDI(fDI, fname, nStp, cDims, "By")

	if (Nd == 3) and ("Bx" in vIds) and ("By" in vIds) and ("Bz" in vIds):
		vAtt = et.SubElement(Grid, "Attribute")
		vAtt.set("Name", "VecB")
		vAtt.set("AttributeType", "Vector")
		vAtt.set("Center", "Cell")
		fDI = et.SubElement(vAtt, "DataItem")
		fDI.set("ItemType", "Function")
		fDI.set("Dimensions", vDims)
		fDI.set("Function", "JOIN($0 , $1 , $2)")
		AddDI(fDI, fname, nStp, cDims, "Bx")
		AddDI(fDI, fname, nStp, cDims, "By")
		AddDI(fDI, fname, nStp, cDims, "Bz")

	#Velocity (2D)
	if ( (Nd == 2) and ("Vx" in vIds) and ("Vy" in vIds) ):
		vAtt = et.SubElement(Grid,"Attribute")
		vAtt.set("Name","VecV")
		vAtt.set("AttributeType","Vector")
		vAtt.set("Center","Cell")
		fDI = et.SubElement(vAtt,"DataItem")
		fDI.set("ItemType","Function")
		fDI.set("Dimensions",vDims)
		fDI.set("Function","JOIN($0 , $1)")
		AddDI(fDI,fname,nStp,cDims,"Vx")
		AddDI(fDI,fname,nStp,cDims,"Vy")

	#Velocity (3D)
	if ( (Nd == 3) and ("Vx" in vIds) and ("Vy" in vIds) and ("Vz" in vIds)):
		vAtt = et.SubElement(Grid,"Attribute")
		vAtt.set("Name","VecV")
		vAtt.set("AttributeType","Vector")
		vAtt.set("Center","Cell")
		fDI = et.SubElement(vAtt,"DataItem")
		fDI.set("ItemType","Function")
		fDI.set("Dimensions",vDims)
		fDI.set("Function","JOIN($0 , $1 , $2)")
		AddDI(fDI,fname,nStp,cDims,"Vx")
		AddDI(fDI,fname,nStp,cDims,"Vy")
		AddDI(fDI,fname,nStp,cDims,"Vz")

	#Magnetic field (2D)
	if ( (Nd == 2) and ("Bx" in vIds) and ("By" in vIds) ):
		vAtt = et.SubElement(Grid,"Attribute")
		vAtt.set("Name","VecB")
		vAtt.set("AttributeType","Vector")
		vAtt.set("Center","Cell")
		fDI = et.SubElement(vAtt,"DataItem")
		fDI.set("ItemType","Function")
		fDI.set("Dimensions",vDims)
		fDI.set("Function","JOIN($0 , $1)")
		AddDI(fDI,fname,nStp,cDims,"Bx")
		AddDI(fDI,fname,nStp,cDims,"By")

	if ( (Nd == 3) and ("Bx" in vIds) and ("By" in vIds) and ("Bz" in vIds)):
		vAtt = et.SubElement(Grid,"Attribute")
		vAtt.set("Name","VecB")
		vAtt.set("AttributeType","Vector")
		vAtt.set("Center","Cell")
		fDI = et.SubElement(vAtt,"DataItem")
		fDI.set("ItemType","Function")
		fDI.set("Dimensions",vDims)
		fDI.set("Function","JOIN($0 , $1 , $2)")
		AddDI(fDI,fname,nStp,cDims,"Bx")
		AddDI(fDI,fname,nStp,cDims,"By")
		AddDI(fDI,fname,nStp,cDims,"Bz")	

#Decide on centering
def getLoc(gDims, vDims):
	"""
	Determines the location type based on the given global dimensions and variable dimensions.

	Parameters:
		gDims (list): A list of integers representing the global dimensions.
		vDims (list): A list of integers representing the variable dimensions.

	Returns:
		str: The location type, which can be "Cell", "Node", or "Other".

	Note:
		If the lengths of gDims and vDims are not equal, the function returns "Other".
		The function checks each dimension and determines the location type based on the comparison
		between the global dimension and the variable dimension. If all dimensions have the same
		location type, the function returns that type. Otherwise, it returns "Other".
	"""
	vDims = np.array(vDims, dtype=int)
	dimLocs = []
	if len(gDims) != len(vDims):
		return "Other"
	for d in range(len(gDims)):
		Ngd = gDims[d] - 1
		Nvd = vDims[d]
		if Ngd == Nvd:
			dimLocs.append("Cell")
		elif Ngd == Nvd - 1:
			dimLocs.append("Node")
		else:
			dimLocs.append("Other")

	if "Other" in dimLocs:
		return "Other"
	# If all the same, we have consensus
	if all(x == dimLocs[0] for x in dimLocs):
		return dimLocs[0]
	else:
		return "Other"

def addHyperslab(Grid, vName, dSetDimStr, vdimStr, startStr, strideStr, numStr, origDSetDimStr, fileText):
	"""
	Add a hyperslab to the given Grid element.

	Parameters:
		Grid: The parent element to which the hyperslab will be added.
		vName: The name of the attribute.
		dSetDimStr: The dimensions of the hyperslab.
		vdimStr: The dimensions of the cut.
		startStr: The start position of the hyperslab.
		strideStr: The stride of the hyperslab.
		numStr: The number of elements in the hyperslab.
		origDSetDimStr: The dimensions of the original dataset.
		fileText: The text of the file.

	Returns:
		None
	"""
	vAtt = et.SubElement(Grid, "Attribute")
	vAtt.set("Name", vName)
	vAtt.set("AttributeType", "Scalar")
	vAtt.set("Center", "Node")
	slabDI = et.SubElement(vAtt, "DataItem")
	slabDI.set("ItemType", "HyperSlab")
	slabDI.set("Dimensions", dSetDimStr)
	slabDI.set("Type", "HyperSlab")  # Not sure if redundant, but it works
	cutDI = et.SubElement(slabDI, "DataItem")
	cutDI.set("Dimensions", vdimStr)
	cutDI.set("Format", "XML")
	cutDI.text = "\n{}\n{}\n{}\n".format(startStr, strideStr, numStr)
	datDI = et.SubElement(slabDI, "DataItem")
	datDI.set("Dimensions", origDSetDimStr)
	datDI.set("DataType", "Float")
	datDI.set("Precision", "4")
	datDI.set("Format", "HDF")
	datDI.text = fileText

