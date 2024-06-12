#Various routines to help with restart upscaling
#Updated to work w/ new-form restarts

import h5py
import numpy as np
import scipy
from scipy.spatial import ConvexHull
import kaipy.kaiH5 as kh5
import os
import kaipy.gamera.gamGrids as gg

TINY = 1.0e-8
NumG = 4
IDIR = 0
JDIR = 1
KDIR = 2

#Compare two restart sets to check if they're identical
def CompRestarts(iStr,oStr,nRes,Ri,Rj,Rk):
	"""
	Compare restart files and calculate the differences between corresponding variables.

	Args:
		iStr (str): Input file name pattern for the original restart files.
		oStr (str): Input file name pattern for the modified restart files.
		nRes (int): Number of restart files.
		Ri (int): Number of iterations in the i-direction.
		Rj (int): Number of iterations in the j-direction.
		Rk (int): Number of iterations in the k-direction.

	Returns:
		None
	"""
	vIDs = ["X","Y","Z","Gas","oGas","Bxyz","oBxyz","magFlux","omagFlux"]

	for i in range(Ri):
		for j in range(Rj):
			for k in range(Rk):
				fIn1 = kh5.genName(iStr,i,j,k,Ri,Rj,Rk,nRes)
				fIn2 = kh5.genName(oStr,i,j,k,Ri,Rj,Rk,nRes)

				kh5.CheckOrDie(fIn1)
				i1H5 = h5py.File(fIn1,'r')
				kh5.CheckOrDie(fIn2)
				i2H5 = h5py.File(fIn2,'r')

				# Perform comparison for each variable
				for v in vIDs:
					Q1 = i1H5[v][:]
					Q2 = i2H5[v][:]
					# Calculate the differences and print the results
					# ...

				# Close the files
				i1H5.close()
				i2H5.close()

				
#Get full data from a tiled restart file
def PullRestartMPI(bStr, nRes, Ri, Rj, Rk):
	"""
	Pulls and restarts MPI data from input files.

	Args:
		bStr (str): Base string for file names.
		nRes (int): Number of resolution levels.
		Ri (int): Number of blocks in the i-direction.
		Rj (int): Number of blocks in the j-direction.
		Rk (int): Number of blocks in the k-direction.

	Returns:
		tuple: A tuple containing the following arrays and variables:
			- X (ndarray): X-coordinate array.
			- Y (ndarray): Y-coordinate array.
			- Z (ndarray): Z-coordinate array.
			- nG (ndarray): Global gas array.
			- nM (ndarray): Global magnetic flux array.
			- nB (ndarray): Global magnetic field array.
			- oG (ndarray): Ghost gas array.
			- oM (ndarray): Ghost magnetic flux array.
			- oB (ndarray): Ghost magnetic field array.
			- G0 (ndarray): Initial gas array.
			- fIn0 (str): Initial input file name.
	"""
	doInit = True
	for i in range(Ri):
		for j in range(Rj):
			for k in range(Rk):
				fIn = kh5.genName(bStr, i, j, k, Ri, Rj, Rk, nRes)
				kh5.CheckOrDie(fIn)
				# Open input file
				iH5 = h5py.File(fIn, 'r')

				if doInit:
					fIn0 = fIn
					# Get size (w/ halos)
					Ns, Nv, Nk, Nj, Ni = iH5['Gas'].shape
					doGas0 = 'Gas0' in iH5.keys()
					Nkp = Nk - 2 * NumG
					Njp = Nj - 2 * NumG
					Nip = Ni - 2 * NumG

					# Get combined size (w/ ghosts)
					NkT = Rk * Nkp + 2 * NumG
					NjT = Rj * Njp + 2 * NumG
					NiT = Ri * Nip + 2 * NumG

					# Allocate arrays (global)
					nG = np.zeros((Ns, Nv, NkT, NjT, NiT))
					oG = np.zeros((Ns, Nv, NkT, NjT, NiT))

					if doGas0:
						G0 = np.zeros((Ns, Nv, NkT, NjT, NiT))
					else:
						G0 = None
					nM = np.zeros((3, NkT + 1, NjT + 1, NiT + 1))
					oM = np.zeros((3, NkT + 1, NjT + 1, NiT + 1))
					nB = np.zeros((3, NkT, NjT, NiT))
					oB = np.zeros((3, NkT, NjT, NiT))

					X = np.zeros((NkT + 1, NjT + 1, NiT + 1))
					Y = np.zeros((NkT + 1, NjT + 1, NiT + 1))
					Z = np.zeros((NkT + 1, NjT + 1, NiT + 1))

					# Fill with nans for testing
					nM[:] = np.nan
					nG[:] = np.nan
					nB[:] = np.nan
					oM[:] = np.nan
					oG[:] = np.nan
					oB[:] = np.nan

					X[:] = np.nan
					Y[:] = np.nan
					Z[:] = np.nan

					doInit = False

				# Get grids
				Corner2Global(X, "X", iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp)
				Corner2Global(Y, "Y", iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp)
				Corner2Global(Z, "Z", iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp)

				# Get cell-centered quantities
				Gas2Global(nG, "Gas", iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp)
				Gas2Global(oG, "oGas", iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp)
				if doGas0:
					Gas2Global(G0, "Gas0", iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp)

				Bvec2Global(nB, "Bxyz", iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp)
				Bvec2Global(oB, "oBxyz", iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp)

				# Get face-centered quantities
				Flux2Global(nM, "magFlux", iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp)
				Flux2Global(oM, "omagFlux", iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp)

				# Close the file
				iH5.close()

	return X, Y, Z, nG, nM, nB, oG, oM, oB, G0, fIn0

	doInit = True
	for i in range(Ri):
		for j in range(Rj):
			for k in range(Rk):
				fIn = kh5.genName(bStr,i,j,k,Ri,Rj,Rk,nRes)
				kh5.CheckOrDie(fIn)
				#print("Reading from %s"%(fIn))
				#Open input file
				iH5 = h5py.File(fIn,'r')

				if (doInit):
					fIn0 = fIn
					#Get size (w/ halos)
					Ns,Nv,Nk,Nj,Ni = iH5['Gas'].shape
					doGas0 = ('Gas0' in iH5.keys())
					Nkp = Nk-2*NumG
					Njp = Nj-2*NumG
					Nip = Ni-2*NumG

					#Get combined size (w/ ghosts)
					NkT = Rk*Nkp + 2*NumG
					NjT = Rj*Njp + 2*NumG
					NiT = Ri*Nip + 2*NumG

					#Allocate arrays (global)
					nG = np.zeros((Ns,Nv,NkT,NjT,NiT))
					oG = np.zeros((Ns,Nv,NkT,NjT,NiT))

					if (doGas0):
						G0 = np.zeros((Ns,Nv,NkT,NjT,NiT))
					else:
						G0 = None
					nM = np.zeros((3,NkT+1,NjT+1,NiT+1))
					oM = np.zeros((3,NkT+1,NjT+1,NiT+1))
					nB = np.zeros((3,NkT  ,NjT  ,NiT  ))
					oB = np.zeros((3,NkT  ,NjT  ,NiT  ))

					X = np.zeros((NkT+1,NjT+1,NiT+1))
					Y = np.zeros((NkT+1,NjT+1,NiT+1))
					Z = np.zeros((NkT+1,NjT+1,NiT+1))

					#Fill w/ nans for testing
					nM[:] = np.nan
					nG[:] = np.nan
					nB[:] = np.nan
					oM[:] = np.nan
					oG[:] = np.nan
					oB[:] = np.nan
					
					X[:] = np.nan
					Y[:] = np.nan
					Z[:] = np.nan

					doInit = False

			#Get grids
				Corner2Global(X,"X",iH5,i,j,k,Ri,Rj,Rk,Nip,Njp,Nkp)
				Corner2Global(Y,"Y",iH5,i,j,k,Ri,Rj,Rk,Nip,Njp,Nkp)
				Corner2Global(Z,"Z",iH5,i,j,k,Ri,Rj,Rk,Nip,Njp,Nkp)

			#Get cell-centered quantities
				Gas2Global(nG, "Gas",iH5,i,j,k,Ri,Rj,Rk,Nip,Njp,Nkp)
				Gas2Global(oG,"oGas",iH5,i,j,k,Ri,Rj,Rk,Nip,Njp,Nkp)
				if (doGas0):
					Gas2Global(G0,"Gas0",iH5,i,j,k,Ri,Rj,Rk,Nip,Njp,Nkp)

				Bvec2Global(nB, "Bxyz",iH5,i,j,k,Ri,Rj,Rk,Nip,Njp,Nkp)
				Bvec2Global(oB,"oBxyz",iH5,i,j,k,Ri,Rj,Rk,Nip,Njp,Nkp)
			#Get face-centered quantities
				Flux2Global(nM, "magFlux",iH5,i,j,k,Ri,Rj,Rk,Nip,Njp,Nkp)
				Flux2Global(oM,"omagFlux",iH5,i,j,k,Ri,Rj,Rk,Nip,Njp,Nkp)

				#print("\tMPI (%d,%d,%d) = [%d,%d]x[%d,%d]x[%d,%d]"%(i,j,k,iS,iE,jS,jE,kS,kE))
				#print("\tGrid indices = (%d,%d)x(%d,%d)x(%d,%d)"%(iSg,iEg,jSg,jEg,kSg,kEg))

				#Close up
				iH5.close()

# print(np.isnan(G).sum(),G.size)
# print(np.isnan(X).sum(),X.size)
# print(np.isnan(M).sum(),M.size)

	return X,Y,Z,nG,nM,nB,oG,oM,oB,G0,fIn0

def Gas2Global(G, vID, iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp):
	"""
	Copies a local gas field to the corresponding global indices in a larger gas field.

	Parameters:
		G (ndarray): The larger gas field with shape (Ns, Nv, NkT, NjT, NiT).
		vID (int): The index of the local gas field to be copied.
		iH5 (ndarray): The local gas field with shape (Ns, Nv, Nk, Nj, Ni).
		i (int): The current i-index of the local gas field.
		j (int): The current j-index of the local gas field.
		k (int): The current k-index of the local gas field.
		Ri (int): The total number of i-indices in the larger gas field.
		Rj (int): The total number of j-indices in the larger gas field.
		Rk (int): The total number of k-indices in the larger gas field.
		Nip (int): The number of i-indices in the local gas field.
		Njp (int): The number of j-indices in the local gas field.
		Nkp (int): The number of k-indices in the local gas field.

	Returns:
		None
	"""
	Ns, Nv, NkT, NjT, NiT = G.shape
	Nk = Nkp + 2 * NumG
	Nj = Njp + 2 * NumG
	Ni = Nip + 2 * NumG

	locG = np.zeros((Ns, Nv, Nk, Nj, Ni))
	locG = iH5[vID][:]

	# Set general global/local indices
	iS, iE, iSg, iEg = ccL2G(i, Ri, Nip)
	jS, jE, jSg, jEg = ccL2G(j, Rj, Njp)
	kS, kE, kSg, kEg = ccL2G(k, Rk, Nkp)

	G[:, :, kSg:kEg, jSg:jEg, iSg:iEg] = locG[:, :, kS:kE, jS:jE, iS:iE]

def Bvec2Global(B, vID, iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp):
	"""
	Convert a local B vector to the global B vector.

	Args:
		B (ndarray): The global B vector.
		vID (int): The ID of the vector.
		iH5 (ndarray): The local B vector.
		i (int): The i index.
		j (int): The j index.
		k (int): The k index.
		Ri (float): The Ri value.
		Rj (float): The Rj value.
		Rk (float): The Rk value.
		Nip (int): The Nip value.
		Njp (int): The Njp value.
		Nkp (int): The Nkp value.

	Returns:
		None
	"""
	Nd, NkT, NjT, NiT = B.shape
	Nk = Nkp + 2 * NumG
	Nj = Njp + 2 * NumG
	Ni = Nip + 2 * NumG

	locB = np.zeros((Nd, Nk, Nj, Ni))
	locB = iH5[vID][:]

	# Set general global/local indices
	iS, iE, iSg, iEg = ccL2G(i, Ri, Nip)
	jS, jE, jSg, jEg = ccL2G(j, Rj, Njp)
	kS, kE, kSg, kEg = ccL2G(k, Rk, Nkp)

	B[:, kSg:kEg, jSg:jEg, iSg:iEg] = locB[:, kS:kE, jS:jE, iS:iE]

def Flux2Global(M, vID, iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp):
	"""
	Converts local flux data to global flux data and updates the global flux array.

	Parameters:
		M (ndarray): The global flux array with shape (Nd, NkT1, NjT1, NiT1).
		vID (int): The index of the local flux data in the iH5 array.
		iH5 (ndarray): The array containing the local flux data.
		i, j, k (int): The indices of the local flux data in the global flux array.
		Ri, Rj, Rk (int): The number of grid points in the i, j, k directions.
		Nip, Njp, Nkp (int): The number of grid points in the i, j, k directions excluding ghost cells.

	Returns:
		None
	"""
	Nd, NkT1, NjT1, NiT1 = M.shape

	Nk = Nkp + 2 * NumG
	Nj = Njp + 2 * NumG
	Ni = Nip + 2 * NumG
	locM = np.zeros((Nd, Nk + 1, Nj + 1, Ni + 1))
	locM = iH5[vID][:]

	# Set global/local indices
	iS, iE, iSg, iEg = ccL2G(i, Ri, Nip)
	jS, jE, jSg, jEg = ccL2G(j, Rj, Njp)
	kS, kE, kSg, kEg = ccL2G(k, Rk, Nkp)
	M[:, kSg:kEg + 1, jSg:jEg + 1, iSg:iEg + 1] = locM[:, kS:kE + 1, jS:jE + 1, iS:iE + 1]

def ccL2G(i, Ri, Nip):
	"""
	Convert local indices to global indices.

	Args:
		i (int): The local index.
		Ri (int): The total number of local indices.
		Nip (int): The number of indices per process.

	Returns:
		tuple: A tuple containing the start and end indices for the local and global indices.
	"""
	# Set general case of only active
	iS = NumG
	iE = iS + Nip + 1
	iSg = NumG + i * Nip
	iEg = iSg + Nip + 1

	if i == 0:
		# Need low-i ghosts
		iS = 0
		iSg = iSg - NumG
	if i == (Ri - 1):
		iE = iE + NumG
		iEg = iEg + NumG

	return iS, iE, iSg, iEg

#Pull corner-centered variables w/ halos into global sized grid
def Corner2Global(Q, vID, iH5, i, j, k, Ri, Rj, Rk, Nip, Njp, Nkp):
	"""
	Transfers a corner of data from a local grid to the global grid.

	Args:
		Q (ndarray): Global grid.
		vID (int): Index of the corner data in iH5.
		iH5 (ndarray): Local grid.
		i (int): Index of the corner in the i-direction.
		j (int): Index of the corner in the j-direction.
		k (int): Index of the corner in the k-direction.
		Ri (float): Grid resolution in the i-direction.
		Rj (float): Grid resolution in the j-direction.
		Rk (float): Grid resolution in the k-direction.
		Nip (int): Number of points in the i-direction.
		Njp (int): Number of points in the j-direction.
		Nkp (int): Number of points in the k-direction.

	Returns:
		None
	"""
	iS = i * Nip
	iE = iS + Nip
	jS = j * Njp
	jE = jS + Njp
	kS = k * Nkp
	kE = kS + Nkp
	iSg = iS - NumG + NumG
	iEg = iS + Nip + NumG + 1 + NumG
	jSg = jS - NumG + NumG
	jEg = jS + Njp + NumG + 1 + NumG
	kSg = kS - NumG + NumG
	kEg = kS + Nkp + NumG + 1 + NumG

	Q[kSg:kEg, jSg:jEg, iSg:iEg] = iH5[vID][:]

#Push restart data w/ ghosts to an output tiling
def PushRestartMPI(outid, nRes, Ri, Rj, Rk, X, Y, Z, nG, nM, nB, oG, oM, oB, G0, f0, dtScl=1.0):
	"""
	Pushes restart data to multiple output files in a parallel MPI simulation.

	Args:
		outid (int): Output identifier.
		nRes (int): Number of restarts.
		Ri (int): Number of MPI processes in the i-direction.
		Rj (int): Number of MPI processes in the j-direction.
		Rk (int): Number of MPI processes in the k-direction.
		X (ndarray): Array of X-coordinates.
		Y (ndarray): Array of Y-coordinates.
		Z (ndarray): Array of Z-coordinates.
		nG (ndarray): Array of cell-centered values.
		nM (ndarray): Array of face fluxes.
		nB (ndarray): Array of cell-centered magnetic field values.
		oG (ndarray): Array of old cell-centered values.
		oM (ndarray): Array of old face fluxes.
		oB (ndarray): Array of old cell-centered magnetic field values.
		G0 (ndarray): Array of initial cell-centered values.
		f0 (str): File path of the input file.
		dtScl (float, optional): Scaling factor for the time step. Defaults to 1.0.
	"""
	if G0 is not None:
		doGas0 = True

	# Read attributes from the input file
	print("Reading attributes from %s" % f0)
	iH5 = h5py.File(f0, 'r')

	# Loop over output slices and create restarts
	for i in range(Ri):
		for j in range(Rj):
			for k in range(Rk):
				fOut = kh5.genName(outid, i, j, k, Ri, Rj, Rk, nRes)

				if os.path.exists(fOut):
					os.remove(fOut)

				# Open output file
				oH5 = h5py.File(fOut, 'w')

				# Transfer attributes to output
				for ak in iH5.attrs.keys():
					aStr = str(ak)
					if aStr == "dt0":
						oH5.attrs.create(ak, dtScl * iH5.attrs[aStr])
						print(dtScl * iH5.attrs[aStr])
					else:
						oH5.attrs.create(ak, iH5.attrs[aStr])

				# Base indices
				iS = i * Nip
				iE = iS + Nip
				jS = j * Njp
				jE = jS + Njp
				kS = k * Nkp
				kE = kS + Nkp

				# Do subgrids
				iSg = iS - NumG + NumG
				iEg = iS + Nip + NumG + 1 + NumG
				jSg = jS - NumG + NumG
				jEg = jS + Njp + NumG + 1 + NumG
				kSg = kS - NumG + NumG
				kEg = kS + Nkp + NumG + 1 + NumG
				ijkX = X[kSg:kEg, jSg:jEg, iSg:iEg]
				ijkY = Y[kSg:kEg, jSg:jEg, iSg:iEg]
				ijkZ = Z[kSg:kEg, jSg:jEg, iSg:iEg]

				# Do face fluxes (same indices as subgrids)
				ijknM = nM[:, kSg:kEg, jSg:jEg, iSg:iEg]
				ijkoM = oM[:, kSg:kEg, jSg:jEg, iSg:iEg]

				# Do cell-centered values
				iSg = iS - NumG + NumG
				iEg = iS + Nip + NumG + NumG
				jSg = jS - NumG + NumG
				jEg = jS + Njp + NumG + NumG
				kSg = kS - NumG + NumG
				kEg = kS + Nkp + NumG + NumG

	if (G0 is not None):
		doGas0 = True

	#print(X.shape,nG.shape,nM.shape,nB.shape)
	Ns,Nv,NkT,NjT,NiT = nG.shape
	Nkp = (NkT-2*NumG)//Rk
	Njp = (NjT-2*NumG)//Rj
	Nip = (NiT-2*NumG)//Ri

	print("Reading attributes from %s"%(f0))
	iH5 = h5py.File(f0,'r')

	#print("Splitting (%d,%d,%d) cells into (%d,%d,%d) x (%d,%d,%d) [Cells,MPI]"%(Ni,Nj,Nk,Nip,Njp,Nkp,Ri,Rj,Rk))
	#Loop over output slices and create restarts
	for i in range(Ri):
		for j in range(Rj):
			for k in range(Rk):
				fOut = kh5.genName(outid,i,j,k,Ri,Rj,Rk,nRes)
				#print("Writing to %s"%(fOut))

				if (os.path.exists(fOut)):
					os.remove(fOut)
				#Open output file
				oH5 = h5py.File(fOut,'w')

				#Transfer attributes to output
				for ak in iH5.attrs.keys():
					aStr = str(ak)
					#print(aStr)
					if (aStr == "dt0"):
						oH5.attrs.create(ak,dtScl*iH5.attrs[aStr])
						print(dtScl*iH5.attrs[aStr])
					else:
						
						oH5.attrs.create(ak,iH5.attrs[aStr])
			#Base indices
				iS =  i*Nip
				iE = iS+Nip
				jS =  j*Njp
				jE = jS+Njp
				kS =  k*Nkp
				kE = kS+Nkp
			#Do subgrids
				iSg =  iS    -NumG   + NumG 
				iEg =  iS+Nip+NumG+1 + NumG 
				jSg =  jS    -NumG   + NumG 
				jEg =  jS+Njp+NumG+1 + NumG 
				kSg =  kS    -NumG   + NumG 
				kEg =  kS+Nkp+NumG+1 + NumG 
				ijkX = X[kSg:kEg,jSg:jEg,iSg:iEg]
				ijkY = Y[kSg:kEg,jSg:jEg,iSg:iEg]
				ijkZ = Z[kSg:kEg,jSg:jEg,iSg:iEg]
			#Do face fluxes (same indices as subgrids)
				ijknM = nM [:,kSg:kEg,jSg:jEg,iSg:iEg]
				ijkoM = oM [:,kSg:kEg,jSg:jEg,iSg:iEg]

			#Do cell-centered values
				iSg =  iS    -NumG + NumG 
				iEg =  iS+Nip+NumG + NumG 
				jSg =  jS    -NumG + NumG 
				jEg =  jS+Njp+NumG + NumG 
				kSg =  kS    -NumG + NumG 
				kEg =  kS+Nkp+NumG + NumG

				ijknG = nG[:,:,kSg:kEg,jSg:jEg,iSg:iEg]
				ijknB = nB[:,kSg:kEg,jSg:jEg,iSg:iEg]
				ijkoG = oG[:,:,kSg:kEg,jSg:jEg,iSg:iEg]
				ijkoB = oB[:,kSg:kEg,jSg:jEg,iSg:iEg]

				if (doGas0):
					ijkG0 = G0[:,:,kSg:kEg,jSg:jEg,iSg:iEg]

			#Write vars
				oH5.create_dataset( "Gas"    ,data=ijknG)
				oH5.create_dataset( "magFlux",data=ijknM)
				oH5.create_dataset( "Bxyz"   ,data=ijknB)
				oH5.create_dataset("oGas"    ,data=ijkoG)
				oH5.create_dataset("omagFlux",data=ijkoM)
				oH5.create_dataset("oBxyz"   ,data=ijkoB)

				oH5.create_dataset("X",data=ijkX)
				oH5.create_dataset("Y",data=ijkY)
				oH5.create_dataset("Z",data=ijkZ)

				if (doGas0):
					oH5.create_dataset("Gas0",data=ijkG0)

				#Close this output file
				oH5.close()

	#Close input file
	iH5.close()

#Upscale a grid (with ghosts, k-j-i order)
def upGrid(X, Y, Z):
	"""
	Upscales a grid by embedding old points into a new grid.

	Parameters:
		X (ndarray): 3D array representing the X-coordinates of the original grid.
		Y (ndarray): 3D array representing the Y-coordinates of the original grid.
		Z (ndarray): 3D array representing the Z-coordinates of the original grid.

	Returns:
		ndarray: Upscaled X-coordinates of the new grid.
		ndarray: Upscaled Y-coordinates of the new grid.
		ndarray: Upscaled Z-coordinates of the new grid.
	"""
	Ngk, Ngj, Ngi = X.shape

	Nk = Ngk - 2 * NumG - 1
	Nj = Ngj - 2 * NumG - 1
	Ni = Ngi - 2 * NumG - 1

	# Assuming LFM-style grid, get upper half plane

	xx = X[NumG, NumG:-NumG, NumG:-NumG].T
	yy = Y[NumG, NumG:-NumG, NumG:-NumG].T

	# Now half cells in r-phi polar
	rr = np.sqrt(xx ** 2.0 + yy ** 2.0)
	pp = np.arctan2(yy, xx)

	NiH = 2 * Ni
	NjH = 2 * Nj
	NkH = 2 * Nk

	rrH = np.zeros((NiH + 1, NjH + 1))
	ppH = np.zeros((NiH + 1, NjH + 1))

	# Embed old points into new grid
	for i in range(Ni + 1):
		for j in range(Nj + 1):
			rrH[2 * i, 2 * j] = rr[i, j]
			ppH[2 * i, 2 * j] = pp[i, j]

	# Create I midpoints
	for i in range(Ni):
		rrH[2 * i + 1, :] = 0.5 * (rrH[2 * i, :] + rrH[2 * i + 2, :])
		ppH[2 * i + 1, :] = 0.5 * (ppH[2 * i, :] + ppH[2 * i + 2, :])

	# Create J midpoints
	for j in range(Nj):
		rrH[:, 2 * j + 1] = 0.5 * (rrH[:, 2 * j] + rrH[:, 2 * j + 2])
		ppH[:, 2 * j + 1] = 0.5 * (ppH[:, 2 * j] + ppH[:, 2 * j + 2])

	# Create I-J midpoints
	for i in range(Ni):
		for j in range(Nj):
			rrH[2 * i + 1, 2 * j + 1] = 0.25 * (
				rrH[2 * i, 2 * j]
				+ rrH[2 * i, 2 * j + 2]
				+ rrH[2 * i + 2, 2 * j]
				+ rrH[2 * i + 2, 2 * j + 2]
			)
			ppH[2 * i + 1, 2 * j + 1] = 0.25 * (
				ppH[2 * i, 2 * j]
				+ ppH[2 * i, 2 * j + 2]
				+ ppH[2 * i + 2, 2 * j]
				+ ppH[2 * i + 2, 2 * j + 2]
			)

	# Convert back to 2D Cartesian
	xxH = rrH * np.cos(ppH)
	yyH = rrH * np.sin(ppH)

	# Augment w/ 2D ghosts
	xxG, yyG = gg.Aug2D(xxH, yyH, doEps=True, TINY=TINY)

	# Augment w/ 3D ghosts
	X, Y, Z = gg.Aug3D(xxG, yyG, Nk=NkH, TINY=TINY)

	return X.T, Y.T, Z.T

	Ngk,Ngj,Ngi = X.shape
	
	Nk = Ngk-2*NumG-1
	Nj = Ngj-2*NumG-1
	Ni = Ngi-2*NumG-1
	

	#Assuming LFM-style grid, get upper half plane
	
	xx = X[NumG,NumG:-NumG,NumG:-NumG].T
	yy = Y[NumG,NumG:-NumG,NumG:-NumG].T

	#Now half cells in r-phi polar
	rr = np.sqrt(xx**2.0+yy**2.0)
	pp = np.arctan2(yy,xx)

	NiH = 2*Ni
	NjH = 2*Nj
	NkH = 2*Nk

	rrH = np.zeros((NiH+1,NjH+1))
	ppH = np.zeros((NiH+1,NjH+1))

	#Embed old points into new grid
	for i in range(Ni+1):
		for j in range(Nj+1):
			rrH[2*i,2*j] = rr[i,j]
			ppH[2*i,2*j] = pp[i,j]

	#Create I midpoints
	for i in range(Ni):
		rrH[2*i+1,:] = 0.5*( rrH[2*i,:] + rrH[2*i+2,:] )
		ppH[2*i+1,:] = 0.5*( ppH[2*i,:] + ppH[2*i+2,:] )

	#Create J midpoints
	for j in range(Nj):
		rrH[:,2*j+1] = 0.5*( rrH[:,2*j] + rrH[:,2*j+2])
		ppH[:,2*j+1] = 0.5*( ppH[:,2*j] + ppH[:,2*j+2])

	#Create I-J midpoints
	for i in range(Ni):
		for j in range(Nj):
			rrH[2*i+1,2*j+1] = 0.25*( rrH[2*i,2*j] + rrH[2*i,2*j+2] + rrH[2*i+2,2*j] + rrH[2*i+2,2*j+2] )
			ppH[2*i+1,2*j+1] = 0.25*( ppH[2*i,2*j] + ppH[2*i,2*j+2] + ppH[2*i+2,2*j] + ppH[2*i+2,2*j+2] )


	#Convert back to 2D Cartesian
	xxH = rrH*np.cos(ppH)
	yyH = rrH*np.sin(ppH)

	#Augment w/ 2D ghosts
	xxG,yyG = gg.Aug2D(xxH,yyH,doEps=True,TINY=TINY)

	#Augment w/ 3D ghosts
	X,Y,Z = gg.Aug3D(xxG,yyG,Nk=NkH,TINY=TINY)


	return X.T,Y.T,Z.T

#Return cell centered volume from grid X,Y,Z (w/ ghosts)
def Volume(Xg, Yg, Zg):
	"""
	Calculate the volume of a grid.

	Args:
		Xg (ndarray): X coordinates of the grid.
		Yg (ndarray): Y coordinates of the grid.
		Zg (ndarray): Z coordinates of the grid.

	Returns:
		dV (ndarray): Volume of the grid.

	Note:
		The function calculates the volume of a grid by performing a convex hull calculation on each grid cell.
		The grid is assumed to have LFM-like symmetry.
	"""

	Ngk1, Ngj1, Ngi1 = Xg.shape
	Nk = Ngk1 - 1
	Nj = Ngj1 - 1
	Ni = Ngi1 - 1

	print("Calculating volume of grid of size (%d,%d,%d)" % (Ni, Nj, Nk))
	dV = np.zeros((Nk, Nj, Ni))
	ijkPts = np.zeros((8, 3))

	# Assuming LFM-like symmetry
	k = 0
	for j in range(Nj):
		for i in range(Ni):
			ijkPts[:, 0] = Xg[k:k + 2, j:j + 2, i:i + 2].flatten()
			ijkPts[:, 1] = Yg[k:k + 2, j:j + 2, i:i + 2].flatten()
			ijkPts[:, 2] = Zg[k:k + 2, j:j + 2, i:i + 2].flatten()

			dV[:, j, i] = ConvexHull(ijkPts, incremental=False).volume
	return dV

	Ngk1,Ngj1,Ngi1 = Xg.shape
	Nk = Ngk1-1
	Nj = Ngj1-1
	Ni = Ngi1-1

	print("Calculating volume of grid of size (%d,%d,%d)"%(Ni,Nj,Nk))
	dV = np.zeros((Nk,Nj,Ni))
	ijkPts = np.zeros((8,3))

	#Assuming LFM-like symmetry
	k = 0
	for j in range(Nj):
		for i in range(Ni):
			ijkPts[:,0] = Xg[k:k+2,j:j+2,i:i+2].flatten()
			ijkPts[:,1] = Yg[k:k+2,j:j+2,i:i+2].flatten()
			ijkPts[:,2] = Zg[k:k+2,j:j+2,i:i+2].flatten()

			dV[:,j,i] = ConvexHull(ijkPts,incremental=False).volume
	return dV

#Upscale magnetic fluxes (M) on grid X,Y,Z (w/ ghosts) to doubled grid
def upFlux(M):
	"""
	Upscales the face fluxes of a given input array.

	Args:
		M (numpy.ndarray): The input array representing the face fluxes.

	Returns:
		Mu (numpy.ndarray): The upscaled face fluxes.

	"""
	#Chop out outer two cells, upscale inside
	cM = M[:,2:-2,2:-2,2:-2]

	Nd,Nkc,Njc,Nic = cM.shape
	Nk = Nkc-1
	Nj = Njc-1
	Ni = Nic-1

	Mu = np.zeros((Nd,2*Nk+1,2*Nj+1,2*Ni+1))

	#Loop over coarse grid
	print("Upscaling face fluxes ...")
	for k in range(Nk):
		for j in range(Nj):
			for i in range(Ni):
				ip = 2*i
				jp = 2*j
				kp = 2*k

				#West i face (4)
				Mu[IDIR,kp:kp+2,jp:jp+2,ip] = 0.25*cM[IDIR,k,j,i]
				#East i face (4)
				Mu[IDIR,kp:kp+2,jp:jp+2,ip+2] = 0.25*cM[IDIR,k,j,i+1]

				#South j face (4)
				Mu[JDIR,kp:kp+2,jp,ip:ip+2] = 0.25*cM[JDIR,k,j,i]
				#North j face (4)
				Mu[JDIR,kp:kp+2,jp+2,ip:ip+2] = 0.25*cM[JDIR,k,j+1,i]

				#Bottom k face (4)
				Mu[KDIR,kp,jp:jp+2,ip:ip+2] = 0.25*cM[KDIR,k,j,i]
				#Top k face (4)
				Mu[KDIR,kp+2,jp:jp+2,ip:ip+2] = 0.25*cM[KDIR,k+1,j,i]

				#Now all exterior faces are done
				#12 remaining interior faces

				#Interior i faces (4)
				Mu[IDIR,kp:kp+2,jp:jp+2,ip+1] = 0.5*( Mu[IDIR,kp:kp+2,jp:jp+2,ip] + Mu[IDIR,kp:kp+2,jp:jp+2,ip+2] )

				#Interior j faces (4)
				Mu[JDIR,kp:kp+2,jp+1,ip:ip+2] = 0.5*( Mu[JDIR,kp:kp+2,jp,ip:ip+2] + Mu[JDIR,kp:kp+2,jp+2,ip:ip+2] )

				#Interior k faces (4)
				Mu[KDIR,kp+1,jp:jp+2,ip:ip+2] = 0.5*( Mu[KDIR,kp,jp:jp+2,ip:ip+2] + Mu[KDIR,kp+2,jp:jp+2,ip:ip+2] )
	#MaxDiv(M)
	#MaxDiv(Mu)

	return Mu

#Upscale gas variable (G) on grid X,Y,Z (w/ ghosts) to doubled grid
def upGas(G,dV0,dVu,vID="Gas"):
	def upGas(G, dV0, dVu, vID="Gas"):
		"""
		Upscales the given gas variables.

		Args:
			G (numpy.ndarray): The gas variables to be upscaled.
			dV0 (numpy.ndarray): The volume of each cell in the original grid.
			dVu (numpy.ndarray): The volume of each cell in the upscaled grid.
			vID (str, optional): Identifier for the gas variables. Default is "Gas".

		Returns:
			Gu (numpy.ndarray): The upscaled gas variables.

		"""
		#Chop out outer two cells, upscale inside
		cG = G[:,:,2:-2,2:-2,2:-2]
		cdV0 = dV0[2:-2,2:-2,2:-2]
		Ns, Nv, Nk, Nj, Ni = cG.shape
		
		Gu = np.zeros((Ns, Nv, 2*Nk, 2*Nj, 2*Ni))
		print("Upscaling %s variables ..." % (vID))
		#Loop over coarse grid
		for s in range(Ns):
			for v in range(Nv):
				print("\tUpscaling Species %d, Variable %d" % (s, v))
				Gu[s, v, :, :, :] = upVarCC(cG[s, v, :, :, :], cdV0, dVu)
		return Gu
	
#Upscale Bxyz on grid X,Y,Z (w/ ghosts) to doubled grid
def upCCMag(B,dV0,dVu,vID="Bxyz"):
	"""
	Upscales the given magnetic field data by a factor of 2 in each dimension.

	Args:
		B (numpy.ndarray): The input magnetic field data with shape (Nv, Nk, Nj, Ni).
		dV0 (numpy.ndarray): The cell volume data with shape (Nk, Nj, Ni).
		dVu (float): The upscaled cell volume.
		vID (str, optional): The identifier for the variable being upscaled. Default is "Bxyz".

	Returns:
		Bu (numpy.ndarray): The upscaled magnetic field data with shape (Nv, 2*Nk, 2*Nj, 2*Ni).
	"""
	#Chop out outer two cells, upscale inside
	cB = B[:,2:-2,2:-2,2:-2]
	cdV0 = dV0[2:-2,2:-2,2:-2]

	Nv,Nk,Nj,Ni = cB.shape

	Bu = np.zeros((Nv,2*Nk,2*Nj,2*Ni))
	print("Upscaling %s variables ..."%(vID))

	#Loop over coarse grid
	for v in range(Nv):
		print("\tUpscaling Variable %d"%(v))
		Bu[v,:,:,:] = upVarCC(cB[v,:,:,:],cdV0,dVu)

	return Bu

#Upscale single cell-centered variable Q, dV0=dV on coarse, dVu=dV on fine grid
def upVarCC(Q, dV0, dVu):
	"""
	Upscale a variable defined on a coarse grid to a finer grid using conservative interpolation.

	Parameters:
		Q (ndarray): Variable defined on the coarse grid.
		dV0 (ndarray): Volumes of the coarse grid cells.
		dVu (ndarray): Volumes of the finer subgrid cells.

	Returns:
		ndarray: Upscaled variable on the finer grid.

	Note:
		The function loops over the coarse grid and calculates the weighted contribution of each subcell on the finer grid.
		It then scales back the density to obtain the upscaled variable.
		The conservation of the variable is tested by comparing the total values before and after the upscaling process.
	"""

	Nk, Nj, Ni = Q.shape
	Qu = np.zeros((2*Nk, 2*Nj, 2*Ni))

	# Loop over coarse grid
	for k in range(Nk):
		for j in range(Nj):
			for i in range(Ni):
				QdV = Q[k, j, i] * dV0[k, j, i]
				dVijk = dVu[2*k:2*k+2, 2*j:2*j+2, 2*i:2*i+2]  # Volumes of the finer subgrid
				vScl = dVijk.sum()  # Total volume of the finer subchunks
				QdVu = QdV * (dVijk / vScl)  # Give weighted contribution to each subcell
				Qu[2*k:2*k+2, 2*j:2*j+2, 2*i:2*i+2] = QdVu / dVijk  # Scale back to density

	# Test conservation
	print("\t\tCoarse (Total) = %e" % (Q[:,:,:] * dV0).sum())
	print("\t\tFine   (Total) = %e" % (Qu[:,:,:] * dVu).sum())
	return Qu


	Nk,Nj,Ni = Q.shape
	Qu = np.zeros((2*Nk,2*Nj,2*Ni))

	#Loop over coarse grid
	for k in range(Nk):
		for j in range(Nj):
			for i in range(Ni):
				QdV = Q[k,j,i]*dV0[k,j,i]
				dVijk = dVu[2*k:2*k+2,2*j:2*j+2,2*i:2*i+2] #Volumes of the finer subgrid
				vScl = dVijk.sum() #Total volume of the finer subchunks
				QdVu = QdV*(dVijk/vScl) #Give weighted contribution to each subcell
				Qu[2*k:2*k+2,2*j:2*j+2,2*i:2*i+2] = QdVu/dVijk #Scale back to density

	#Test conservation
	print("\t\tCoarse (Total) = %e"%(Q [:,:,:]*dV0).sum())
	print("\t\tFine   (Total) = %e"%(Qu[:,:,:]*dVu).sum())
	return Qu

#Calculates maximum divergence of mag flux data
def MaxDiv(M):
	"""
	Calculate the maximum divergence and mean divergence of a given 3D array.

	Args:
		M (numpy.ndarray): The input 3D array with shape (Nd, Nkc, Njc, Nic).

	Returns:
		numpy.ndarray: The divergence array with shape (Nk, Nj, Ni).

	"""
	Nd, Nkc, Njc, Nic = M.shape
	Nk = Nkc - 1
	Nj = Njc - 1
	Ni = Nic - 1

	Div = np.zeros((Nk, Nj, Ni))
	for k in range(Nk):
		for j in range(Nj):
			for i in range(Ni):
				Div[k, j, i] = M[IDIR, k, j, i + 1] - M[IDIR, k, j, i] + M[JDIR, k, j + 1, i] - M[JDIR, k, j, i] + M[KDIR, k + 1, j, i] - M[KDIR, k, j, i]

	mDiv = np.abs(Div).max()
	bDiv = np.abs(Div).mean()
	print("Max/Mean divergence = %e,%e" % (mDiv, bDiv))
	print("Sum divergence = %e" % (Div.sum()))

	return Div
