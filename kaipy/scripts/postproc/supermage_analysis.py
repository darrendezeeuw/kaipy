#!/usr/bin/env python


"""Create SuperMAGE plots for ground delta B from a MAGE run.

Create SuperMAGE plots for ground delta B from a MAGE run.

Author
------
Eric Winter (eric.winter@jhuapl.edu)
"""


# Import standard modules.
import argparse
import copy
import os
import subprocess
import sys

# Import 3rd-party modules.
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import project-specific modules.
import kaipy.supermage as sm


# Program constants and defaults

# Program description.
DESCRIPTION = "Create MAGE-SuperMag comparison plots."

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "debug": False,
    "smuser": "",
    "verbose": False,
    "calcdb_results_path": None,
}

# Number of microseconds in a second.
MICROSECONDS_PER_SECOND = 1e6

# Number of seconds in a day.
SECONDS_PER_DAY = 86400

# Location of SuperMag cache folder.
SUPERMAG_CACHE_FOLDER = os.path.join(os.environ["HOME"], "supermag")


def create_command_line_parser():
    """Create the command-line parser.

    Create the parser for the command-line.

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
        "--smuser", type=str,
        default=DEFAULT_ARGUMENTS["smuser"],
        help="SuperMag user ID to use for SuperMag queries "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "calcdb_results_path",
        default=DEFAULT_ARGUMENTS["calcdb_results_path"],
        help="Path to a result file from calcdb.x."
    )
    return parser


def create_dbpic_plot(runid: str, args: dict):
    """Create the dbpic.py (Mercator) plot of the dB values.

    Create the dbpic.py (Mercator) plot of the dB values.

    Parameters
    ----------
    runid : str
        Run ID for calcdb results file.
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
    smuser = args["smuser"]
    verbose = args["verbose"]
    calcdb_results_path = args["calcdb_results_path"]

    # ------------------------------------------------------------------------

    # Split the calcdb results path into a directory and a file.
    (calcdb_results_dir, calcdb_results_file) = os.path.split(
        calcdb_results_path
    )
    if debug:
        print(f"calcdb_results_dir = {calcdb_results_dir}")
        print(f"calcdb_results_file = {calcdb_results_file}")

    # Move to the results directory.
    if verbose:
        print(f"Moving to results directory {calcdb_results_dir}.")
    os.chdir(calcdb_results_dir)

    # Run dbpic.py.
    cmd = f"dbpic.py -d . -id {runid}"
    subprocess.run(cmd, shell=True, check=True)


def create_dbpole_plot(runid: str, args: dict):
    """Create the dbpole.py (polar) plot of the dB values.

    Create the dbpole.py (polar) plot of the dB values.

    Parameters
    ----------
    runid : str
        Run ID for calcdb results file.
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
    smuser = args["smuser"]
    verbose = args["verbose"]
    calcdb_results_path = args["calcdb_results_path"]

    # ------------------------------------------------------------------------

    # Split the calcdb results path into a directory and a file.
    (calcdb_results_dir, calcdb_results_file) = os.path.split(
        calcdb_results_path
    )
    if debug:
        print(f"calcdb_results_dir = {calcdb_results_dir}")
        print(f"calcdb_results_file = {calcdb_results_file}")

    # Move to the results directory.
    if verbose:
        print(f"Moving to results directory {calcdb_results_dir}.")
    os.chdir(calcdb_results_dir)

    # Run dbpic.py.
    cmd = f"dbpole.py -d . -id {runid}"
    subprocess.run(cmd, shell=True, check=True)


def create_supermag_comparison_plots(args: dict):
    """Create plots comparing MAGE ground delta B to SuperMag data.

    Create plots comparing MAGE ground delta B to SuperMag data.

    Parameters
    ----------
    args: dict
        Dictionary of command-line options.

    Returns
    -------
    int 0 on success

    Raises
    ------
    None
    """
    # Set defaults for command-line options, then update with values passed
    # from the caller.
    local_args = copy.deepcopy(DEFAULT_ARGUMENTS)
    local_args.update(args)
    args = local_args

    # Local convenience variables.
    debug = args["debug"]
    smuser = args["smuser"]
    verbose = args["verbose"]
    calcdb_results_path = args["calcdb_results_path"]

    # Make sure a calcdb result file was specified.
    if calcdb_results_path is None:
        raise TypeError("A calcdb result path must be specified!")

    # ------------------------------------------------------------------------

    # Split the calcdb results path into a directory and a file.
    (calcdb_results_dir, calcdb_results_file) = os.path.split(
        calcdb_results_path
    )
    if debug:
        print(f"calcdb_results_dir = {calcdb_results_dir}")
        print(f"calcdb_results_file = {calcdb_results_file}")

    # Extract the runid.
    runid = calcdb_results_file
    runid.trim(".deltab.h5")
    if debug:
        print(f"runid = {runid}")

    # Move to the results directory.
    if verbose:
        print(f"Moving to results directory {calcdb_results_dir}.")
    os.chdir(calcdb_results_dir)

    # Read the delta B values.
    if verbose:
        print("Reading MAGE-derived ground delta B values from "
              f"{calcdb_results_file}.")
    SIM = sm.ReadSimData(calcdb_results_file)
    if debug:
        print(f"SIM = {SIM}")

    # Fetch the start time (as a datetime object) of simulation data.
    start = SIM["td"][0]
    if debug:
        print(f"start = {start}")

    # Compute the duration of the simulated data, in seconds, then days.
    duration = SIM["td"][-1] - SIM["td"][0]
    duration_seconds = (
        duration.seconds + duration.microseconds/MICROSECONDS_PER_SECOND
    )
    numofdays = duration_seconds/SECONDS_PER_DAY
    if debug:
        print(f"duration = {duration}")
        print(f"duration_seconds = {duration_seconds}")
        print(f"numofdays = {numofdays}")

    # Fetch the SuperMag indices for this time period.
    if verbose:
        print("Fetching SuperMag indices.")
    SMI = sm.FetchSMIndices(smuser, start, numofdays)
    if debug:
        print(f"SMI = {SMI}")

    # Fetch the SuperMag data for this time period.
    if verbose:
        print("Fetching SuperMag data.")
    SM = sm.FetchSMData(smuser, start, numofdays,
                        savefolder=SUPERMAG_CACHE_FOLDER)
    if debug:
        print(f"SM = {SM}")

    # Abort if no data was found.
    if len(SM["td"]) == 0:
        raise TypeError("No SuperMag data found for requested time period, "
                        " aborting.")

    # Interpolate the simulated delta B to the measurement times from
    # SuperMag.
    if verbose:
        print("Interpolating simulated data to SuperMag times.")
    SMinterp = sm.InterpolateSimData(SIM, SM)
    if debug:
        print("SMinterp = %s" % SMinterp)

    # ------------------------------------------------------------------------

    # Create the plots in memory.
    mpl.use("Agg")

    # ------------------------------------------------------------------------

    # Make the indices plot.
    if verbose:
        print("Creating indices comparison plot.")
    sm.MakeIndicesPlot(SMI, SMinterp, fignumber=1)
    comparison_plot_file = "indices.png"
    plt.savefig(comparison_plot_file)

    # Make the contour plots.
    if verbose:
        print("Creating contour plots.")
    sm.MakeContourPlots(SM, SMinterp, maxx=1000, fignumber=2)
    contour_plot_file = "contours.png"
    plt.savefig(contour_plot_file)

    # Make the dbpic.py plot (Mercator projection).
    if verbose:
        print("Creating Mercator plot.")
    create_dbpic_plot(runid, args)

    dbpic_plot_file = "dbpic.png"
    plt.savefig(dbpic_plot_file)

    # Make the dbpole.py plot (polar projection).

    # Return normally.
    return 0


def main():
    """Driver for command-line version of code."""
    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command line.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Convert the arguments from Namespace to dict.
    args = vars(args)

    # Call the main program code.
    return_code = create_supermag_comparison_plots(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
