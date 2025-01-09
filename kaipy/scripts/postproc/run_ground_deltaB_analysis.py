#!/usr/bin/env python


"""Analyze the ground delta-B values from a MAGE run.

Perform an analysis of ground magnetic field perturbations computed
for a MAGE magnetosphere simulation. The computed values are compared
with measured data from SuperMag. Then a set of SuperMAGE ground
delta-B maps is created. This is all done by a linked set of PBS jobs
running the MAGE results directory.

NOTE: Before running this script, the user must run the setupEnvironment.sh
scripts for kaipy and kaiju.

NOTE: If the calcdb.x binary was built with a module set other than those
listed below, change the module set in the PBS scripts appropriately.

Author
------
Eric Winter (eric.winter@jhuapl.edu)

"""


# Import standard modules.
import argparse
import copy
import os
import pathlib
import re
import shutil
import sys

# Import 3rd-party modules.
from jinja2 import Template

# Import project-specific modules.
from kaipy import kaiH5
from kaipy import kaiTools


# Program constants and defaults

# Program description.
DESCRIPTION = (
    "Compare MAGE ground delta-B to SuperMag measurements, and create\n"
    "SuperMAGE analysis maps."
)

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "calcdb": "calcdb.x",
    "debug": False,
    "dt": 60.0,
    "hpc": "pleiades",
    "parintime": 1,
    "pbs_account": None,
    "smuser": None,
    "verbose": False,
    "mage_results_path": None,
}

# Valid values for command-line arguments.
VALID_HPC = ["derecho", "pleiades"]

# Directory containing this script.
SCRIPT_DIRECTORY = pathlib.Path(__file__).parent.resolve()
TEMPLATE_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "templates")

# Location of template XML file for calcdb.x.
CALCDB_XML_TEMPLATE = os.path.join(TEMPLATE_DIRECTORY, "calcdb-template.xml")

# # Locations of template PBS file for running calcdb.x.
CALCDB_PBS_TEMPLATE = os.path.join(TEMPLATE_DIRECTORY, "calcdb-template.pbs")

# Location of template PBS file for pitmerge.py.
PITMERGE_PBS_TEMPLATE = os.path.join(
    TEMPLATE_DIRECTORY, "pitmerge-template.pbs"
)

# Location of template PBS file for running the ground delta-B analysis.
GROUND_DELTAB_ANALYSIS_PBS_TEMPLATE = os.path.join(
    TEMPLATE_DIRECTORY, "ground_deltaB_analysis-template.pbs"
)


def create_command_line_parser():
    """Create the command-line parser.

    Create the parser for the command line.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line parser for this script.

    Raises
    ------
    None
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--calcdb", type=str, default=DEFAULT_ARGUMENTS["calcdb"],
        help="Path to calcdb.x binary (default: %(default)s)."
    )
    parser.add_argument(
        "--debug", "-d", default=DEFAULT_ARGUMENTS["debug"],
        action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--dt", type=float, default=DEFAULT_ARGUMENTS["dt"],
        help="Time interval for delta-B computation (seconds) "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "--hpc", type=str, default=DEFAULT_ARGUMENTS["hpc"],
        choices=VALID_HPC,
        help="HPC system to run analysis (default: %(default)s)."
    )
    parser.add_argument(
        "--parintime", type=int, default=DEFAULT_ARGUMENTS["parintime"],
        help="Split the calculation into this many parallel chunks of MAGE "
        "simulation steps, one chunk per node (default: %(default)s)."
    )
    parser.add_argument(
        "--pbs_account", type=str, default=DEFAULT_ARGUMENTS["pbs_account"],
        help="PBS account to use for job accounting (default: %(default)s)."
    )
    parser.add_argument(
        "--smuser", type=str, default=DEFAULT_ARGUMENTS["smuser"],
        help="Account name for SuperMag database access "
        "(default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", default=DEFAULT_ARGUMENTS["verbose"],
        action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "mage_results_path", type=str,
        default=DEFAULT_ARGUMENTS["mage_results_path"],
        help="Path to a result file for a MAGE magnetosphere run."
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


def create_calcdb_xml_file(runid: str, args: dict):
    """Create the XML input file for calcdb.x from a template.

    Create the XML input file for calcdb.x from a template. The file is
    created in the current directory, which much contain the specified MAGE
    results file. Note that the contents of the file depends on wheter
    parintime is 1 (use a single multi-threaded job to run calcdb.x on all
    simulation steps) or > 1 (use multiple jobs to split the calcdb.x work
    into an array of multi-threaded jobs which run on multiple nodes in
    parallel).

    Parameters
    ----------
    runid : str
        Run ID for MAGE results file to use in calcdb.x calculations.
    args : dict
        Dictionary of command-line options.

    Returns
    -------
    xml_file : str
        Name of XML file.

    Raises
    ------
    AssertionError
        If the MAGE result file contains no time steps.
    """
    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]

    # Fetch run information from the MAGE result file.
    if verbose:
        print(f"Fetching MAGE run information for runid {runid}.")
    filename, isMPI, Ri, Rj, Rk = kaiTools.getRunInfo(".", runid)
    if debug:
        print(f"filename = {filename}")
        print(f"isMPI = {isMPI}")
        print(f"Ri = {Ri}")
        print(f"Rj = {Rj}")
        print(f"Rk = {Rk}")

    # Get the number of steps and the step IDs from the MAGE results file.
    if verbose:
        print(f"Counting time steps for run {runid}.")
    nSteps, sIds = kaiH5.cntSteps(filename)
    if debug:
        print(f"nSteps = {nSteps}")
        print(f"sIds = {sIds}")
    assert nSteps > 0

    # Find the time for the last step.
    if verbose:
        print(f"Finding time for last step for run {runid}.")
    tFin = kaiH5.tStep(filename, sIds[-1], aID="time")
    if debug:
        print(f"tFin = {tFin}")

    # Read the template XML file.
    if verbose:
        print("Reading calcdb XML template.")
    with open(CALCDB_XML_TEMPLATE, "r", encoding="utf-8") as f:
        template_content = f.read()
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    options = {
        "runid": runid,
        "dt": args["dt"],
        "tFin": tFin,
        "ebfile": runid,
        "isMPI": isMPI,
        "Ri": Ri,
        "Rj": Rj,
        "Rk": Rk,
        "parintime": args["parintime"],
    }
    if debug:
        print(f"options = {options}")

    # Render the template.
    xml_file = f"calcdb-{runid}.xml"
    if verbose:
        print("Rendering template.")
    if verbose:
        print(f"Creating {xml_file}.")
    xml_content = template.render(options)
    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Return the name of the XML file.
    return xml_file


def create_calcdb_pbs_script(runid: str, args: dict):
    """Create the PBS script to run calcdb.x from a template.

    Create the PBS script to run calcdb.x from a template. The PBS script is
    created in the current directory, which must contain the MAGE results. The
    script will set up and run calcdb.x on the specified set of MAGE results,
    in the current directory.

    Parameters
    ----------
    runid : str
        Run ID for MAGE results to use in calcdb.x calculations.
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    calcdb_pbs_script : str
        Path to PBS script.

    Raises
    ------
    None
    """
    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]

    # Create the XML input file for calcdb.x.
    if verbose:
        print(f"Creating XML file for calcdb.x for runid {runid}.")
    calcdb_xml_file = create_calcdb_xml_file(runid, args)
    if debug:
        print(f"calcdb_xml_file = {calcdb_xml_file}")

    # Copy the calcdb.x binary to the results directory, then make it
    # executable.
    if verbose:
        print(f"Copying {args['calcdb']} to results directory.")
    shutil.copyfile(args["calcdb"], "calcdb.x")
    os.chmod("calcdb.x", 0o755)

    # Read the PBS script template for running calcdb.x.
    if verbose:
        print("Reading calcdb.x PBS template.")
    with open(CALCDB_PBS_TEMPLATE, "r", encoding="utf-8") as f:
        template_content = f.read()
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    options = {
        "hpc": args["hpc"],
        "job_name": f"calcdb-{runid}",
        "pbs_account": args["pbs_account"],
        "conda_prefix": os.environ["CONDA_PREFIX"],
        "kaipyhome": os.environ["KAIPYHOME"],
        "kaijuhome": os.environ["KAIJUHOME"],
        "parintime": args["parintime"],
        "calcdb_xml_file": calcdb_xml_file,
    }
    if debug:
        print(f"options = {options}")

    # Render the template.
    if verbose:
        print("Rendering template.")
    calcdb_pbs_script = f"calcdb-{runid}.pbs"
    calcdb_pbs_content = template.render(options)
    with open(calcdb_pbs_script, "w", encoding="utf-8") as f:
        f.write(calcdb_pbs_content)

    # Return the name of the PBS script.
    return calcdb_pbs_script


def create_pitmerge_pbs_script(runid: str, args: dict):
    """Create the PBS script for pitmerge.py to stitch calcdb output.

    Create the PBS script for stitching together calcdb output using the
    pitmerge.py script.

    Parameters
    ----------
    runid : str
        Run ID for MAGE results used in calcdb.x calculations.
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    stitching_pbs_script : str
        Path to PBS script.

    Raises
    ------
    None
    """
    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]

    # Read the PBS script template for pitmerge.py.
    if verbose:
        print("Reading pitmerge.py PBS template.")
    with open(PITMERGE_PBS_TEMPLATE, "r", encoding="utf-8") as f:
        template_content = f.read()
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    options = {
        "hpc": args["hpc"],
        "job_name": f"pitmerge-{runid}",
        "pbs_account": args["pbs_account"],
        "conda_prefix": os.environ["CONDA_PREFIX"],
        "kaipyhome": os.environ["KAIPYHOME"],
        "kaijuhome": os.environ["KAIJUHOME"],
        "runid": runid,
    }
    if debug:
        print(f"options = {options}")

    # Render the template.
    if verbose:
        print("Rendering template.")
    pitmerge_pbs_script = f"pitmerge-{runid}.pbs"
    pitmerge_pbs_content = template.render(options)
    with open(pitmerge_pbs_script, "w", encoding="utf-8") as f:
        f.write(pitmerge_pbs_content)

    # Return the name of the PBS script.
    return pitmerge_pbs_script


def create_ground_deltab_analysis_pbs_script(runid: str, args: dict):
    """Create the PBS script for the ground delta-B analysis from a template.

    Create the PBS script for the ground delta-B analysis from a template.

    Parameters
    ----------
    runid : str
        Run ID for MAGE results used in calcdb.x calculations.
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    ground_deltab_analysis_pbs_script : str
        Path to PBS script.

    Raises
    ------
    None
    """
    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]

    # Read the PBS script template for ground_deltab_analysis.py.
    if verbose:
        print("Reading ground delta-B analysis PBS template.")
    with open(GROUND_DELTAB_ANALYSIS_PBS_TEMPLATE, "r", encoding="utf-8") as f:
        template_content = f.read()
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    options = {
        "hpc": args["hpc"],
        "job_name": f"ground_deltaB_analysis-{runid}",
        "pbs_account": args["pbs_account"],
        "conda_prefix": os.environ["CONDA_PREFIX"],
        "kaipyhome": os.environ["KAIPYHOME"],
        "kaijuhome": os.environ["KAIJUHOME"],
        "runid": runid,
        "smuser": args["smuser"],
        "calcdb_results_path": f"{runid}.deltab.h5",
        "mage_results_path": args["mage_results_path"],
    }
    if debug:
        print(f"options = {options}")

    # Render the template.
    if verbose:
        print("Rendering template.")
    ground_deltab_analysis_pbs_script = f"ground_deltab_analysis-{runid}.pbs"
    xml_content = template.render(options)
    with open(ground_deltab_analysis_pbs_script, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Return the name of the PBS script.
    return ground_deltab_analysis_pbs_script


def create_submit_script(
        runid: str, calcdb_pbs_script: str, pitmerge_pbs_script: str,
        ground_deltab_analysis_pbs_script: str, args: dict):
    """Create the bash script to submit all of the PBS jobs.

    Create the bash script to submit all of the PBS jobs. The submit script
    submits the job to run calcdb.x, then the job to run pitmerge.py on the
    results (if needed), then the analysis job. Each job only runs if the
    previous job completes successfully.

    Parameters
    ----------
    runid : str
        Run ID for MAGE results used in calcdb.x calculations.
    calcdb_pbs_script : str
        Path to calcdb.x PBS script.
    pitmerge_pbs_script : str
        Path to pitmerge.py PBS script (or None if not needed).
    ground_deltab_analysis_pbs_script : str
        Path to analysis PBS script.
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    submit_script : str
        Path to bash script to submit PBS jobs.

    Raises
    ------
    None
    """
    # Submit the scripts in dependency order.
    submit_script = f"submit-{runid}.sh"
    with open(submit_script, "w", encoding="utf-8") as f:

        # calcdb.x job
        if args["parintime"] > 1:
            cmd = (f"job_id=`qsub -J 1-{args['parintime']} "
                   f"{calcdb_pbs_script}`\n")
        else:
            cmd = f"job_id=`qsub {calcdb_pbs_script}`\n"
        f.write(cmd)
        cmd = "echo $job_id\n"
        f.write(cmd)
        cmd = "old_job_id=$job_id\n"
        f.write(cmd)

        # pitmerge.py job (if needed).
        if pitmerge_pbs_script is not None:
            cmd = (
                "job_id=`qsub -W depend=afterok:$old_job_id "
                f"{pitmerge_pbs_script}`\n"
            )
            f.write(cmd)
            cmd = "echo $job_id\n"
            f.write(cmd)
            cmd = "old_job_id=$job_id\n"
            f.write(cmd)

        # Analysis job.
        cmd = (
            "job_id=`qsub -W depend=afterok:$old_job_id "
            f"{ground_deltab_analysis_pbs_script}`\n"
        )
        f.write(cmd)
        cmd = "echo $job_id\n"
        f.write(cmd)

    # Return the name of the PBS script.
    return submit_script


def run_ground_deltab_analysis(args: dict):
    """Compute and analyze ground delta-B values for a MAGE run.

    Compute and analyze ground delta-B values for a MAGE run.

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
    assert len(args["calcdb"]) > 0
    assert args["dt"] > 0.0
    assert args["hpc"] in VALID_HPC
    assert args["parintime"] > 0
    assert len(args["mage_results_path"]) > 0

    # ------------------------------------------------------------------------

    # Split the MAGE results path into a directory and a file.
    (mage_results_dir, mage_results_file) = os.path.split(
        args["mage_results_path"]
    )
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

    # Create the PBS script to run calcdb.x.
    if verbose:
        print("Creating PBS script to run calcdb.x to compute ground delta-B "
              f"from MAGE results for runid {runid}.")
    calcdb_pbs_script = create_calcdb_pbs_script(runid, args)
    if debug:
        print(f"calcdb_pbs_script = {calcdb_pbs_script}")

    # Create the PBS script to stitch together the output from calcdb.x
    # using pitmerge.py. This script is only needed if parintime > 1.
    if args["parintime"] > 1:
        if verbose:
            print("Creating PBS script to stitch together the calcdb.x output "
                  f" for MAGE runid {runid}")
        pitmerge_pbs_script = create_pitmerge_pbs_script(runid, args)
    else:
        pitmerge_pbs_script = None
    if debug:
        print(f"pitmerge_pbs_script = {pitmerge_pbs_script}")

    # Create the PBS script to compare the calcdb.x results with SuperMag
    # data.
    if verbose:
        print("Creating PBS script to run the ground delta-B analysis "
              f"for MAGE runid {runid}.")
    ground_deltab_analysis_pbs_script = (
        create_ground_deltab_analysis_pbs_script(runid, args)
    )
    if debug:
        print(
            "ground_deltab_analysis_pbs_script = "
            f"{ground_deltab_analysis_pbs_script}"
        )

    # Create the bash script to submit the PBS scripts in the proper order.
    if verbose:
        print("Creating bash script to submit the PBS jobs.")
    submit_script = create_submit_script(
        runid, calcdb_pbs_script, pitmerge_pbs_script,
        ground_deltab_analysis_pbs_script, args
    )
    if debug:
        print(f"submit_script = {submit_script}")

    if verbose:
        print(f"Please run {submit_script} (in the MAGE result directory) to "
              "submit the PBS jobs to perform the MAGE-SuperMag comparison. "
              "If you are not using PBS, then you can manually run the "
              f"scripts individually in the order listed in {submit_script}.")

    # Move back to the start directory.
    os.chdir(start_directory)

    # Return normally.
    return 0


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
    return_code = run_ground_deltab_analysis(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
