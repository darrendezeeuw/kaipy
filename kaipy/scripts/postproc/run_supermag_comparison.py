#!/usr/bin/env python

"""Run a SuperMag comparison for a MAGE magnetosphere run.

Perform a comparison of ground magnetic field perturbations computed for a
MAGE magnetosphere simulation with measured data from SuperMag.

Author
------
Eric Winter (eric.winter@jhuapl.edu)
"""


# Import standard modules.
import argparse
import os
import subprocess

# Import 3rd-party modules.
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# Import project-specific modules.
from kaipy import kaiTools
# import kaipy.supermage as sm


# Program constants and defaults

# Program description.
DESCRIPTION = "Compare MAGE ground delta-B to SuperMag measurements."

# Default SuperMag user name for queries.
DEFAULT_SUPERMAG_USER = os.getlogin()

# Location of template XML file.
XML_TEMPLATE = os.path.join(
    os.environ["KAIPYHOME"], "kaipy", "scripts", "postproc",
    "calcdb.xml.template"
)

# Name of XML file read by calcdb.x.
XML_FILENAME_TEMPLATE = "calcdb_RUNID.xml"

# # Number of microseconds in a second.
# MICROSECONDS_PER_SECOND = 1e6

# # Number of seconds in a day.
# SECONDS_PER_DAY = 86400

# # Location of SuperMag cache folder.
# SUPERMAG_CACHE_FOLDER = os.path.join(os.environ["HOME"], "supermag")


def create_command_line_parser():
    """Create the command-line argument parser.

    Create the parser for command-line arguments.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line argument parser for this script.

    Raises
    ------
    None
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--mpixml", type=str, default=None,
        help="If results from an MPI run, provide XML filename for run "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "--smuser", type=str, default=DEFAULT_SUPERMAG_USER,
        help="SuperMag user ID to use for SuperMag queries "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "mage_results_path",
        help="Path to a result file for a MAGE magnetosphere run."
    )
    return parser


def filename_to_runid(filename: str):
    """Parse the runid from a MAGE results file name.

    Parse the runid from a MAGE results file name.

    The runid is all text before the first period or underscore in the name.

    Parameters
    ----------
    filename : str
        Name of MAGE results file.

    Returns
    -------
    runid : str
        The MAGE runid for the file.

    Raises
    ------
    None
    """
    parts = filename.split(".")
    parts = parts[0].split("_")
    runid = parts[0]
    return runid


def create_xml_file(runid: str, mpixml: str, fdir: str):
    """Create the XML input file for calcdb.x from a template.

    Create the XML input file for calcdb.x from a template.

    Parameters
    ----------
    runid : str
        runid for MAGE results file.
    mpixml : str
        Name of XML run file in results directory.

    Returns
    -------
    xml_file : str
        Name of XML file.

    Raises
    ------
    None
    """
    # Read the template file.
    with open(XML_TEMPLATE, "r", encoding="utf-8") as t:
        lines = t.readlines()

    # Process the template here.
    # <HACK>
    # This should be done with a proper templating package.
    lines[3] = lines[3].replace("RUNID", runid)
    lines[5] = lines[5].replace("EBFILE", runid)
    lines[5] = lines[5].replace("ISMPI", "true")
    if mpixml is not None:
        _, _, Ri, Rj, Rk = kaiTools.getRunInfo(fdir, runid)
        lines[6] = lines[6].replace("RI", str(Ri))
        lines[6] = lines[6].replace("RJ", str(Rj))
        lines[6] = lines[6].replace("RK", str(Rk))
    else:
        lines[6] = "\n"
    # </HACK>

    # Write out the processed Xcompute_ground_delta_BML.
    xml_file = XML_FILENAME_TEMPLATE.replace("RUNID", runid)
    with open(xml_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Return the name of the XML file.
    return xml_file


def compute_ground_delta_B(runid: str, mpixml: str, fdir: str):
    """Compute ground delta B values for a MAGE run.

    Compute ground delta B values for a MAGE run. The computation is done with
    the program calcdb.x.

    Parameters
    ----------
    runid : str
        runid for MAGE results file.
    mpixml : str
        Name of XML run file in results directory.

    Returns
    -------
    delta_B_file : str
        Name of file containing calcdb.x results.

    Raises
    ------
    None
    """
    # Create the XML file for calcdb.x from the template.
    xml_file = create_xml_file(runid, mpixml, fdir)

    # Run the command to compute ground delta B values.
    cmd = "calcdb.x"
    args = [xml_file]
    # subprocess.run([cmd] + args)

    # Compute the name of the file containing the delta B values.
    delta_B_file = runid + ".deltab.h5"
    return delta_B_file


def run_supermag_comparison(args: dict):
    """Compare MAGE results for ground dB to SuperMag data.

    Compare MAGE results for ground dB to SuperMag data.

    Parameters
    ----------
    args: dict
        Dictionary of command-line options.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Local convenience variables.
    debug = args["debug"]
    mpixml = args["mpixml"]
    smuser = args["smuser"]
    verbose = args["verbose"]
    mage_results_path = args["mage_results_path"]

    # Split the MAGE results path into a directory and a file.
    (mage_results_dir, mage_results_file) = os.path.split(mage_results_path)
    if debug:
        print(f"mage_results_dir = {mage_results_dir}")
        print(f"mage_results_file = {mage_results_file}")

    # Compute the runid from the file name.
    runid = filename_to_runid(mage_results_file)
    if debug:
        print(f"runid = {runid}")

    # Move to the results directory.
    if verbose:
        print(f"Moving to results directory {mage_results_dir}.")
    os.chdir(mage_results_dir)

    # Compute the ground delta B values for this run.
    if verbose:
        print("Computing ground delta B values.")
    delta_B_file = compute_ground_delta_B(runid, mpixml, mage_results_dir)
    if debug:
        print(f"delta_B_file = {delta_B_file}")

#     # Read the delta B values.
#     SIM = sm.ReadSimData(delta_B_file)
#     if debug:
#         print("SIM = %s" % SIM)

#     # Fetch the SuperMag indices for the desired time range.

#     # Fetch the start time (as a datetime object) of simulation data.
#     start = SIM["td"][0]
#     if debug:
#         print("start = %s" % start)

#     # Compute the duration of the simulated data, in seconds, then days.
#     duration = SIM["td"][-1] - SIM["td"][0]
#     duration_seconds = duration.seconds + duration.microseconds/MICROSECONDS_PER_SECOND
#     numofdays = duration_seconds/SECONDS_PER_DAY
#     if debug:
#         print("duration = %s" % duration)
#         print("duration_seconds = %s" % duration_seconds)
#         print("numofdays = %s" % numofdays)

#     # Fetch the SuperMag indices for this time period.
#     if verbose:
#         print("Fetching SuperMag indices.")
#     SMI  = sm.FetchSMIndices(smuser, start, numofdays)
#     if debug:
#         print("SMI = %s" % SMI)

#     # Fetch the SuperMag data for this time period.
#     if verbose:
#         print("Fetching SuperMag data.")
#     SM = sm.FetchSMData(smuser, start, numofdays,
#                         savefolder=SUPERMAG_CACHE_FOLDER)
#     if debug:
#         print("SM = %s" % SM)

#     # Interpolate the simulated delta B to the measurement times from SuperMag.
#     if verbose:
#         print("Interpolating simulated data to SuperMag times.")
#     SMinterp = sm.InterpolateSimData(SIM, SM)
#     if debug:
#         print("SMinterp = %s" % SMinterp)


#     # Create the plots in memory.
#     mpl.use("Agg")

#     # Make the indices plot.
#     if verbose:
#         print("Creating indices comparison plot.")
#     sm.MakeIndicesPlot(SMI, SMinterp, fignumber=1)
#     comparison_plot_file = runid + "_indices.png"
#     plt.savefig(comparison_plot_file)

#     # Make the contour plots.
#     if verbose:
#         print("Creating contour plots.")
#     sm.MakeContourPlots(SM, SMinterp, maxx = 1000, fignumber=2)
#     contour_plot_file = runid + "_contours.png"
#     plt.savefig(contour_plot_file)


def main():
    """Driver for command-line version of code."""
    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Convert the arguments from Namespace to dict.
    args = vars(args)

    # Pass the command-line arguments to the main function as a dict.
    run_supermag_comparison(args)


if __name__ == "__main__":
    main()
