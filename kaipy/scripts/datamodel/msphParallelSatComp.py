#!/usr/bin/env python

# Standard modules
import sys
import os
import datetime
import subprocess
from xml.dom import minidom
import argparse
from argparse import RawTextHelpFormatter

# Third-party modules
#import spacepy and cdasws
import spacepy
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
import spacepy.datamodel as dm
from cdasws import CdasWs


#import numpy and matplotlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Other
from   astropy.time import Time
import h5py

# Kaipy modules
import kaipy.kaiH5 as kaiH5
import kaipy.kaiViz as kv
import kaipy.kaiTools as kaiTools
import kaipy.kaijson as kj
import kaipy.satcomp.scutils as scutils

def create_command_line_parser():
    """Create the command-line argument parser.

    Returns:
        argparse.ArgumentParser: Command-line argument parser for this script.
    """
    parser = argparse.ArgumentParser(
        description="""Extracts information from satellite trajectory for various
        spacecraft. Spacecraft data is pulled from CDAWeb. Output CDF files
        contain data pulled from CDAWeb along with data extracted from GAMERA.
        Image files of satellite comparisons are also produced.""",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('-id', type=str, metavar='runid', default='msphere',
                        help='RunID of data (default: %(default)s)')
    parser.add_argument('-path', type=str, metavar='path', default='.',
                        help='Path to directory containing REMIX files (default: %(default)s)')
    parser.add_argument('-cmd', type=str, metavar='command', default=None,
                        help='Full path to sctrack.x command')
    parser.add_argument('-numSeg', type=int, metavar='Number of segments',
                        default=1, help='Number of segments to simultaneously process')
    parser.add_argument('--keep', action='store_true',
                        help='Keep intermediate files')
    return parser

def main():
    parser = create_command_line_parser()
    args = parser.parse_args()

    fdir = args.path
    ftag = args.id
    cmd = args.cmd
    keep = args.keep
    numSegments = args.numSeg

    if fdir == '.':
        fdir = os.getcwd()

    if None == cmd:
        my_env = os.environ.copy()
        cmd = os.path.join(os.getenv('KAIJUDIR'),'build','bin','sctrack.x')
    if not (os.path.isfile(cmd) and os.access(cmd, os.X_OK)):
        print(cmd,'either not found or not executable')
        sys.exit()

    satCmd = os.path.join(os.getenv('KAIJUDIR'),'scripts',
        'msphSatComp.py')
    if not (os.path.isfile(satCmd) and os.access(satCmd, os.X_OK)):
        print(satCmd,'either not found or not executable')
        sys.exit()
    scIds = scutils.getScIds()

    process = []
    logs = []
    errs = []
    # Open log files

    for scId in scIds:
        logfile = open(os.path.join(fdir,'log.'+scId+'.txt'),'w')
        errfile = open(os.path.join(fdir,'err.'+scId+'.txt'),'w')
        cmdList = [satCmd,'-id',ftag,'-path',fdir,'-cmd',cmd,'-satId',scId]
        if keep:
            cmdList.append('--keep')
        if numSegments != 1:
            cmdList.append('-numSeg')
            cmdList.append(str(numSegments))
        print(cmdList)
        process.append(subprocess.Popen(cmdList,
            stdout=logfile, stderr=errfile))
        logs.append(logfile)
        errs.append(errfile)
    for proc in process:
        proc.communicate()
    for log in logs:
        log.close()
    for err in errs:
        err.close()

    if not keep:
        subprocess.run(['rm',os.path.join(fdir,'log.*.txt')])
        subprocess.run(['rm',os.path.join(fdir,'err.*.txt')])

    print('All done!')

if __name__ == '__main__':
    main()