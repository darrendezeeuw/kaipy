#!/usr/bin/env python3
"""
Overengineered script to print the time of each restart file by parsing h5dump output
Uses only packages available on Cheyenne
"""
# Standard modules
import subprocess
import glob
import argparse
from argparse import RawTextHelpFormatter
import h5py as h5
import kaipy.kaiTools as kt

def sortFn(elem): #Used to sort final list in order of nRes
    	return int(elem['nRes'])
		
def getAttrKeyValue(lineList):
	for line in lineList:
		if 'ATTRIBUTE' in line:
			key = line.split('"')[1]
		if '(0)' in line:
			value = line.split('(0):')[1]
	return key, value

#Return dictionary of attrs for a single restart file
def getKVsFromFile(fName):
	
	f5 = h5.File(fName)
	attrs = {}
	attrs['nRes'] = f5.attrs['nRes']
	attrs['time'] = f5.attrs['time']
	attrs['UT'] = kt.MJD2UT(f5.attrs['MJD'])
	return attrs

def create_command_line_parser():
	"""Create the command-line argument parser.
	Create the parser for command-line arguments.
	Returns:
		argparse.ArgumentParser: Command-line argument parser for this script.
	"""
	idStr_noMPI = ".volt.Res.*.h5"

	ftag = "msphere"
	timeFmt = "m"
	MainS = """Overengineered script to print the time of each restart file by parsing h5dump output
	"""

	parser = argparse.ArgumentParser(description=MainS, formatter_class=RawTextHelpFormatter)
	parser.add_argument('-id',type=str,metavar="runid",default=ftag,help="RunID of data (default: %(default)s)")
	parser.add_argument('-f',type=str,metavar="timeFmt",default=timeFmt,help="Time format [s,m,h] (default: %(default)s)")
	parser.add_argument('-ut',action='store_true',default=False,help="Print UT instead of simulation time (default: %(default)s)")
	return parser

def main():
	idStr_noMPI = ".volt.Res.*.h5"
	ftag = "msphere"
	timeFmt = "m"
	MainS = """Overengineered script to print the time of each restart file by parsing h5dump output
	"""

	parser = create_command_line_parser()
	args = parser.parse_args()
	ftag = args.id
	timeFormat = args.f
	doUT = args.ut

	if timeFormat not in ['s','m','h']:
		print('Unrecognized value "%s" for time format. Using "%s"'%(timeFormat, timeFmt))
		timeFormat = timeFmt
	if timeFormat == 's':
		timeFormat = 'sec'
		timeMult = 1
	elif timeFormat == 'h':
		timeFormat = 'hr'
		timeMult = 1./3600
	else:
		timeFormat = 'min'
		timeMult = 1./60
	
	#Get list of files
	globStr = ftag + idStr_noMPI
	print("Looking for voltron restarts...",end='')
	fileList = glob.glob(globStr)
	if len(fileList) == 0:
		print("Not found\nCheck id (globStr = %s)"%(globStr))
		quit()
	print("Found")

	attrList = []
	#Build list of attr dicts from list of files
	for fStr in fileList:
		if 'XXXXX' in fStr:
			continue
		fAttrs = getKVsFromFile(fStr)
		fAttrs['fname'] = fStr
		attrList.append(fAttrs)

	#Print list (time from Gam restarts only, for now)
	attrList.sort(key=sortFn)
	for entry in attrList:
		if doUT:
			formatString = " {}: {}".format(entry['fname'], entry['UT'])
		else:
			formatString = " {}: {:4.2f} [{}]".format(entry['fname'], float(entry['time'])*timeMult, timeFormat)
		print(formatString)


if __name__ == "__main__":
	main()