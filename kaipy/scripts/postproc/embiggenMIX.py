#!/usr/bin/env python
#Takes MIX restart and up/down-scales it

# Standard modules
import argparse
import os
from argparse import RawTextHelpFormatter

# Third-party modules
import numpy as np
import h5py

# Kaipy modules
import kaipy.kaiH5 as kh5
import kaipy.gamera.magsphereRescale as upscl

def create_command_line_parser():
	"""Create the command-line argument parser.
	Create the parser for command-line arguments.
	Returns:
		argparse.ArgumentParser: Command-line argument parser for this script.
	"""
	dIn = os.getcwd()


	inid  = "msphere"
	outid = "msphereX"

	nRes = "0"

	MainS = """Up/down-scales a ReMIX restart

	inid/nres : Run ID string and restart number, i.e. input file = inid.MPISTUFF.Res.#nres.h5
	outid : Output Run ID

	"""	
	parser = argparse.ArgumentParser(description=MainS, formatter_class=RawTextHelpFormatter)
	parser.add_argument('-i',metavar='inid',default=inid,help="Input Run ID string (default: %(default)s)")
	parser.add_argument('-n',type=int,metavar="nres",default=0,help="Restart number (default: %(default)s)")
	parser.add_argument('-o',type=str,metavar="outid",default=outid,help="Output run ID (default: %(default)s)")	
	parser.add_argument('--down',action='store_true',default=False,help='Downscale instead of upscale (default: %(default)s)')
	return parser

def main():
	MainS = """Upscales and retiles a Gamera MPI resart
	"""
	parser = create_command_line_parser()
	#Finalize parsing
	args = parser.parse_args()
	bStr = args.i
	nRes = args.n
	outid = args.o
	doUp = not args.down

	fIn  = bStr  + ".mix.Res.%05d.h5"%(nRes)
	fOut = outid + ".mix.Res.%05d.h5"%(nRes)

	print("Reading from %s and writing to %s"%(fIn,fOut))

	vIDs = kh5.getRootVars(fIn)

	#Open input and output
	iH5 = h5py.File(fIn ,'r')
	oH5 = h5py.File(fOut,'w')

	#Start by scraping attributes
	for k in iH5.attrs.keys():
		aStr = str(k)
		print(aStr)
		oH5.attrs.create(k,iH5.attrs[aStr])

	for vID in vIDs:
		print(vID)
		Q = iH5[vID][:]
		
		if (doUp):
			Qr = upscl.upMIX(Q)
		else:
			Qr = upscl.downMIX(Q)
		oH5.create_dataset(vID,data=Qr)
		print(Q.shape)
		print(Qr.shape)
	#Now get 
	iH5.close()
	oH5.close()

if __name__ == "__main__":
	main()