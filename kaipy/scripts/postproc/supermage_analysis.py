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
import re
import sys

# Import 3rd-party modules.

# Import project-specific modules.


# Program constants and defaults

# Program description.
DESCRIPTION = "Create SuperMAGE analysis plots."

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "debug": False,
    "verbose": False,
    "mage_results_path": "",
}


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
        "--verbose", "-v", default=DEFAULT_ARGUMENTS["verbose"],
        action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "mage_results_path",
        default=DEFAULT_ARGUMENTS["mage_results_path"],
        help="Path to a MAGE result file."
    )
    return parser


def mage_filename_to_runid(filename: str):
    """Parse the runid from a MAGE results file name.

    Parse the runid from a MAGE results file name.

    For a result file from an MPI run, the runid is all text before the
    underscore before the set of 6 underscore-separated 4-digit sequences at
    the end of the name and the terminal .gam.h5 string.

    For a result file from a serial run, the runid is the name of the file,
    less the .gam.h5 extension.

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
    # Check to see if the result file is for an MPI run. If not, it must be
    # for a serial run.
    mpi_pattern = (
        r"^(.+)_(\d{4})_(\d{4})_(\d{4})_(\d{4})_(\d{4})_(\d{4})\.gam.h5$"
    )
    serial_pattern = r"^(.+)\.gam.h5$"
    mpi_re = re.compile(mpi_pattern)
    serial_re = re.compile(serial_pattern)
    m = mpi_re.match(filename)
    if not m:
        m = serial_re.match(filename)
    runid = m.groups()[0]
    return runid


def create_dbpic_plots(runid: str, args: dict):
    """Create the Mercator and polar plots of the dB values.

    Create the Mercator and polar plots of the dB values using dbpic.py.

    Parameters
    ----------
    runid : str
        Run ID for MAGE results.
    args: dict
        Dictionary of command-line options.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Run dbpic.py.
    cmd = f"dbpic.py -d . -id {runid} --projection=both"
    subprocess.run(cmd, shell=True, check=True)
    return "dbpic.png"


def supermage_analysis(args: dict):
    """Create SuperMAGE analysis plots for MAGE results.

    Create SuperMAGE analysis plots for MAGE results.

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
    verbose = args["verbose"]

    # Validate arguments.
    assert len(args["mage_results_path"]) > 0

    # ------------------------------------------------------------------------

    # Split the MAGE results path into a directory and a file.
    (mage_results_dir, mage_results_file) = os.path.split(
        args["mage_results_path"]
    )
    if mage_results_dir == "":
        mage_results_dir = "."
    if debug:
        print(f"mage_results_dir = {mage_results_dir}")
        print(f"mage_results_file = {mage_results_file}")

    # Save the current directory.
    start_directory = os.getcwd()
    if debug:
        print(f"start_directory = {start_directory}")

    # Move to the results directory.
    if verbose:
        print(f"Moving to MAGE results directory {mage_results_dir}.")
    os.chdir(mage_results_dir)

    # Compute the runid from the file name.
    if verbose:
        print(f"Computing runid from MAGE results file {mage_results_file}")
    runid = mage_filename_to_runid(mage_results_file)
    if debug:
        print(f"runid = {runid}")

    # ------------------------------------------------------------------------

    # Make the dbpic.py plots (Mercator and polar projection).
    if verbose:
        print("Creating Mercator and polar plots of MAGE ground delta-B "
              "values.")
    dbpic_plot = create_dbpic_plots(runid, args)
    if debug:
        print(f"dbpic_plot = {dbpic_plot}")

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
    return_code = supermage_analysis(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
