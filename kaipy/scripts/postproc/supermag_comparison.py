#!/usr/bin/env python

"""Run a SuperMag comparison for a MAGE magnetosphere run.

Run a comparison of ground magnetic field perturbations (delta B) computed
for a MAGE magnetosphere simulation with measured data from SuperMag. This
script creates a PBS script that can be submitted to perform all steps of
the comparison:

1. Running calcdb.x.
2. Fetching data from SuperMag.
3. Creating comparison plots.

Once generated, the PBS script can also be run on the command line like this:

```
bash scriptname.pbs
```

Author
------
Eric Winter (eric.winter@jhuapl.edu)
"""


# Import standard modules.
import argparse
import copy
import os
import re

# Import 3rd-party modules.
from jinja2 import Template

# Import project-specific modules.
from kaipy import kaiTools


# Program constants and defaults

# Program description.
DESCRIPTION = "Compare MAGE ground delta-B to SuperMag measurements."

# Default values for command-line arguments.
DEFAULT_ARGS = {
    "debug": False,
    "mpi": False,
    "smuser": os.getlogin(),
    "verbose": False,
    "mage_results_path": None,
}

# Location of template XML file.
XML_TEMPLATE = os.path.join(
    os.environ["KAIPYHOME"], "kaipy", "scripts", "postproc",
    "calcdb-template.xml"
)

# Template for name of XML file read by calcdb.x.
XML_FILENAME_TEMPLATE = "calcdb-RUNID.xml"

# Location of template PBS file.
PBS_TEMPLATE = os.path.join(
    os.environ["KAIPYHOME"], "kaipy", "scripts", "postproc",
    "supermag_comparison-template.pbs"
)

# Template for name of PBS file to run calcdb.x.
PBS_FILENAME_TEMPLATE = "supermag_comparison-RUNID.pbs"

# Default options for filling in PBS template.
default_pbs_options = {
    "job_name": None,
    "account": os.getlogin(),
    "queue": "main",
    "job_priority": "economy",
    "select": "1:ncpus=128",
    "walltime": "12:00:00",
    "modules": [
                "ncarenv/23.06",
                "craype/2.7.20",
                "intel/2023.0.0",
                "ncarcompilers/1.0.0",
                "cray-mpich/8.1.25",
                "hdf5-mpi/1.12.2",
    ],
    "conda_environment": "kaiju-3.8",
    "kaipy_home": os.environ["KAIPYHOME"],
    "kaiju_home": os.environ["KAIJUHOME"],
    "mpiexec_cmd": None,
    "calcdb_cmd": None,
    "calcdb_xml": None,
}


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
        default=DEFAULT_ARGS["debug"],
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--mpi", action="store_true",
        default=DEFAULT_ARGS["mpi"],
        help="Use MPI to parallelize computations (default: %(default)s)."
    )
    parser.add_argument(
        "--smuser", type=str,
        default=DEFAULT_ARGS["smuser"],
        help="SuperMag user ID to use for SuperMag queries "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        default=DEFAULT_ARGS["verbose"],
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "mage_results_path",
        default=DEFAULT_ARGS["mage_results_path"],
        help="Path to a result file for a MAGE magnetosphere run."
    )
    return parser


def filename_to_runid(filename: str):
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


def create_calcdb_xml_file(runid: str):
    """Create the XML input file for calcdb.x from a template.

    Create the XML input file for calcdb.x from a template.

    Parameters
    ----------
    runid : str
        runid for MAGE results file.

    Returns
    -------
    xml_file : str
        Name of XML file.

    Raises
    ------
    None
    """
    # Read the template for the XML file.
    with open(XML_TEMPLATE, "r", encoding="utf-8") as f:
        template_content = f.read()
    template = Template(template_content)

    # Fill in the template information.
    options = {}
    options["runid"] = runid
    options["ebfile"] = runid
    _, ismpi, Ri, Rj, Rk = kaiTools.getRunInfo(".", runid)
    options["ismpi"] = ismpi
    options["Ri"] = Ri
    options["Rj"] = Rj
    options["Rk"] = Rk

    # Render the template.
    xml_file = XML_FILENAME_TEMPLATE.replace("RUNID", runid)
    xml_content = template.render(options)
    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Return the name of the XML file.
    return xml_file


def create_calcdb_pbs_file(runid: str, args: dict):
    """Create the PBS file for calcdb.x from a template.

    Create the PBS file for calcdb.x from a template.

    Parameters
    ----------
    runid : str
        runid for MAGE results file.
    args : dict
        Dictionary of command-line and other arguments.

    Returns
    -------
    pbs_file : str
        Name of PBS file.

    Raises
    ------
    None
    """
    # Read the template for the PBS file.
    with open(PBS_TEMPLATE, "r", encoding="utf-8") as f:
        template_content = f.read()
    template = Template(template_content)

    # Fill in the template information.
    options = copy.deepcopy(default_pbs_options)
    options["job_name"] = f"calcdb-{runid}"
    if args["mpi"]:
        # options["mpiexec_cmd"] = "mpiexec"
        options["mpiexec_cmd"] = ""
        options["calcdb_cmd"] = os.path.join(
            os.environ["KAIJUHOME"], "build_mpi", "bin", "calcdb.x"
        )
    else:
        options["mpiexec_cmd"] = ""
        options["calcdb_cmd"] = os.path.join(
            os.environ["KAIJUHOME"], "build_serial", "bin", "calcdb.x"
        )
    options["calcdb_xml"] = args["xml_file"]

    # Render the template.
    pbs_file = PBS_FILENAME_TEMPLATE.replace("RUNID", runid)
    pbs_content = template.render(options)
    with open(pbs_file, "w", encoding="utf-8") as f:
        f.write(pbs_content)

    # Return the name of the PBS file.
    return pbs_file


def run_supermag_comparison(args: dict):
    """Compare MAGE results for ground delta B to SuperMag measurements.

    Compare MAGE results for ground delta B to SuperMag measurements. This
    function can be called directly from other python code.

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
    # Merge caller and default args.
    local_args = copy.deepcopy(DEFAULT_ARGS)
    if args is not None:
        local_args.update(args)
    args = local_args

    # Local convenience variables.
    debug = args["debug"]
    # mpi = args["mpi"]
    # smuser = args["smuser"]
    verbose = args["verbose"]
    mage_results_path = args["mage_results_path"]

    # Make sure a MAGE result file was specified.
    if mage_results_path is None:
        raise TypeError("A MAGE result path must be specified!")

    # ------------------------------------------------------------------------

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

    # Create the XML input file for calcdb.x.
    if verbose:
        print("Creating XML input file for calcdb.x.")
    xml_file = create_calcdb_xml_file(runid)
    if debug:
        print(f"xml_file = {xml_file}")
    args["xml_file"] = xml_file

    # Create the PBS file for calcdb.x.
    if verbose:
        print("Creating PBS script file for calcdb.x.")
    pbs_file = create_calcdb_pbs_file(runid, args)
    if debug:
        print(f"pbs_file = {pbs_file}")


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
