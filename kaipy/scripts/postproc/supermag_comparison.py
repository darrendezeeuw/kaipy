#!/usr/bin/env python


"""Create plots comparing ground delta B from a MAGE run to SuperMag data.

Create plots comparing ground delta B from a MAGE run to SuperMag data.

Author
------
Eric Winter (eric.winter@jhuapl.edu)
"""


# Import standard modules.
import argparse
import copy
import os
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
        "--debug", "-d", default=DEFAULT_ARGUMENTS["debug"],
        action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--smuser", type=str,
        default=DEFAULT_ARGUMENTS["smuser"],
        help="SuperMag user ID to use for SuperMag queries "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", default=DEFAULT_ARGUMENTS["verbose"],
        action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "calcdb_results_path",
        default=DEFAULT_ARGUMENTS["calcdb_results_path"],
        help="Path to a (possibly merged) result file from calcdb.x."
    )
    return parser


def supermag_comparison(args: dict):
    """Create plots comparing MAGE ground delta-B to SuperMag data.

    Create plots comparing MAGE ground-delta B to SuperMag data.

    Parameters
    ----------
    args: dict
        Dictionary of command-line options.

    Returns
    -------
    int 0 on success

    Raises
    ------
    AssertionError
        If an invalid argument is provided.
    """
    # Set defaults for command-line options, then update with values passed
    # from the caller.
    local_args = copy.deepcopy(DEFAULT_ARGUMENTS)
    local_args.update(args)
    args = local_args

    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]

    # Validate arguments.
    assert len(args["calcdb_results_path"]) > 0

    # ------------------------------------------------------------------------

    # Split the calcdb results path into a directory and a file.
    (calcdb_results_dir, calcdb_results_file) = os.path.split(
        args["calcdb_results_path"]
    )
    if calcdb_results_dir == "":
        calcdb_results_dir = "."
    if debug:
        print(f"calcdb_results_dir = {calcdb_results_dir}")
        print(f"calcdb_results_file = {calcdb_results_file}")

    # Save the current directory.
    start_directory = os.getcwd()
    if debug:
        print(f"start_directory = {start_directory}")

    # Move to the results directory.
    if verbose:
        print(f"Moving to calcdb results directory {calcdb_results_dir}.")
    os.chdir(calcdb_results_dir)

    # Extract the runid.
    if verbose:
        print("Computing runid from calcdb results file "
              f"{calcdb_results_file}")
    p = calcdb_results_file.index(".deltab.h5")
    runid = calcdb_results_file[:p]
    if debug:
        print(f"runid = {runid}")

    # Read the delta B values.
    if verbose:
        print("Reading MAGE-derived ground delta-B values from "
              f"{calcdb_results_file}.")
    SIM = sm.ReadSimData(calcdb_results_file)
    if debug:
        print(f"SIM = {SIM}")

    # Fetch the start and stop time (as datetime objects) of simulation data.
    date_start = SIM["td"][0]
    date_stop = SIM["td"][-1]
    if debug:
        print(f"date_start = {date_start}")
        print(f"date_stop = {date_stop}")

    # Compute the duration of the simulated data, in seconds, then days.
    duration = date_stop - date_start
    duration_seconds = duration.total_seconds()
    duration_days = duration_seconds/SECONDS_PER_DAY
    if debug:
        print(f"duration = {duration}")
        print(f"duration_seconds = {duration_seconds}")
        print(f"duration_days = {duration_days}")

    # Fetch the SuperMag indices for this time period.
    if verbose:
        print(f"Fetching SuperMag indices for the period {date_start} to "
              f"{date_stop}.")
    SMI = sm.FetchSMIndices(args["smuser"], date_start, duration_days)
    if debug:
        print(f"SMI = {SMI}")

    # Fetch the SuperMag data for this time period.
    if verbose:
        print(f"Fetching SuperMag data for the period {date_start} to "
              f"{date_stop}.")
    SM = sm.FetchSMData(args["smuser"], date_start, duration_days,
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
        print("Interpolating MAGE data to SuperMag times.")
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

    # Move back to the start directory.
    os.chdir(start_directory)

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
    return_code = supermag_comparison(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
