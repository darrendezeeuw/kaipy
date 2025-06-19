#!/usr/bin/env python
#Creates a time vs distance plot from a 2D slice created by slice.x

# Standard modules
import argparse
from argparse import RawTextHelpFormatter
import os
import datetime
import sys

# Kaipy modules
import kaipy.kaiH5 as kh5
import kaipy.solarWind.swBCplots as swBCplots

def create_command_line_parser():
	"""Create the command-line argument parser.

	Create the parser for command-line arguments.

	Returns:
		argparse.ArgumentParser: Command-line argument parser for this script.
	"""
	#Defaults
	fdir = os.getcwd()
	swtag = "bcwind.h5"
	imgtype = 'png'

	MainS = """Creates simple multi-panel figure for the 
	solar wind conditions with the  bcwind file and saves it as swBCplot.imgtype
	"""

	parser = argparse.ArgumentParser(description=MainS, formatter_class=RawTextHelpFormatter)
	parser.add_argument('-d',type=str,metavar="directory",default=fdir,help="Directory to read from (default: %(default)s)")
	parser.add_argument('-id',type=str,metavar="swid",default=swtag,help="Solar wind file used (default: %(default)s)")
	parser.add_argument('-type',type=str,metavar="type",default=imgtype,help="Image type (default: %(default)s)")

	return parser

def main():
	#Defaults

	# Set up the command-line parser.
	parser = create_command_line_parser()
	#Finalize parsing
	args = parser.parse_args()
	fdir = args.d
	swtag = args.id
	imgtype = args.type

	allowTypes = ['pdf','png','jpeg','jpg']
	if not (imgtype in allowTypes):
		print('Image type not supported please try',*allowTypes)
		sys.exit()

	if (imgtype == 'pdf'):
		#For some reason trimming PDF files doesn't work
		doTrim = False
	else:
		doTrim = True

	swIn = fdir+'/'+swtag
	kh5.CheckOrDie(swIn)

	# Name the output file the same as the solarwind file with the image extension
	fOut = swtag.split('.')[0]+'.'+imgtype

	# pulling UT variable for plotting
	t0Fmts = ["%Y-%m-%d %H:%M:%S","%Y-%m-%dT%H:%M:%S.%f"]
	utfmt='%H:%M \n%Y-%m-%d'

	UTall  = kh5.PullVar(swIn,"UT")

	#Identify the correct time format
	t0Fmt = None
	for tfmt in t0Fmts:
		try:
			datetime.datetime.strptime(UTall[1].decode('utf-8'),tfmt)
			t0Fmt = tfmt
			break # datetime parse succeeded
		except ValueError:
			pass # datetime parse failed
	if t0Fmt is None:
		print("Time format in bcwind.h5 did not match any expected format.")
		sys.exit()

	utall = []
	for n in range(len(UTall)):
		utall.append(datetime.datetime.strptime(UTall[n].decode('utf-8'),t0Fmt))
	
	# pulling the solar wind values from the table
	varlist = kh5.getRootVars(swIn)
	D = kh5.PullVar(swIn,"D")
	Vx = kh5.PullVar(swIn,"Vx")
	Vy = kh5.PullVar(swIn,"Vy")
	Vz = kh5.PullVar(swIn,"Vz")
	Bx = kh5.PullVar(swIn,"Bx")
	By = kh5.PullVar(swIn,"By")
	Bz = kh5.PullVar(swIn,"Bz")
	Temp = kh5.PullVar(swIn,"Temp")
	Tsec = kh5.PullVar(swIn,"T")
	SYMH = kh5.PullVar(swIn,"symh")
	if ('Interped' in varlist):
		pltInterp = kh5.PullVar(swIn,"Interped")
	else:
		pltInterp = 0*D
	doEps = False
	swBCplots.swQuickPlot(UTall,D,Temp,Vx,Vy,Vz,Bx,By,Bz,SYMH,pltInterp,fOut,doEps=doEps,doTrim=doTrim,t0fmt=t0Fmt)


if __name__ == "__main__":
	main()