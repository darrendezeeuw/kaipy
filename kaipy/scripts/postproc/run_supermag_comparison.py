#!/usr/bin/env python


"""Run a SuperMag comparison for a MAGE magnetosphere run.

Perform a comparison of ground magnetic field perturbations computed for a
MAGE magnetosphere simulation with measured data from SuperMag. This is done
in a PBS job.

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
# import kaipy.supermage as sm


# Program constants and defaults

# Program description.
DESCRIPTION = "Compare MAGE ground delta-B to SuperMag measurements."

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {}
DEFAULT_ARGUMENTS["calcdb"] = "calcdb.x"
DEFAULT_ARGUMENTS["debug"] = False
DEFAULT_ARGUMENTS["parintime"] = 1
DEFAULT_ARGUMENTS["smuser"] = os.getlogin()
DEFAULT_ARGUMENTS["verbose"] = False

# Location of template XML file for calcdb.x.
CALCDB_XML_TEMPLATE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "calcdb-template.xml"
)

# Location of template PBS file for calcdb.x.
CALCDB_PBS_TEMPLATE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "calcdb-template.pbs"
)

# Default options for filling in the calcdb.x PBS template.
DEFAULT_CALCDB_PBS_OPTIONS = {
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
    "kaipyhome": os.environ["KAIPYHOME"],
    "kaijuhome": os.environ["KAIJUHOME"],
}

# Location of template PBS file for pitmerge.py.
PITMERGE_PBS_TEMPLATE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "pitmerge-template.pbs"
)

# Default options for filling in the pitmerge.py PBS template.
DEFAULT_PITMERGE_PBS_OPTIONS = {
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
    "kaipyhome": os.environ["KAIPYHOME"],
    "kaijuhome": os.environ["KAIJUHOME"],
}

# Location of template PBS file for pitmerge.py.
SUPERMAG_COMPARISON_PBS_TEMPLATE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "supermag_comparison-template.pbs"
)

# Default options for filling in the pitmerge.py PBS template.
DEFAULT_SUPERMAG_COMPARISON_PBS_OPTIONS = {
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
    "kaipyhome": os.environ["KAIPYHOME"],
    "kaijuhome": os.environ["KAIJUHOME"],
}


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
        "--calcdb", default=DEFAULT_ARGUMENTS["calcdb"],
        help="Path to calcdb.x binary (default: %(default)s)."
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--parintime", type=int, default=DEFAULT_ARGUMENTS["parintime"],
        help="Split the calculation into this many parallel chunks"
             " (default: %(default)s)."
    )
    parser.add_argument(
        "--smuser", type=str, default=DEFAULT_ARGUMENTS["smuser"],
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


def create_calcdb_xml_file(runid: str,
                           parintime: int = DEFAULT_ARGUMENTS["parintime"],
                           debug: bool = False):
    """Create the XML input file for calcdb.x from a template.

    Create the XML input file for calcdb.x from a template. The file is
    created in the current directory.

    Parameters
    ----------
    runid : str
        runid for MAGE results file.
    parintime : int, default DEFAULT_ARGUMENTS["parintime"]
        Number of threads to use for calcdb.x computation.
    debug : bool, default False
        Set to True to produce debugging output.

    Returns
    -------
    xml_file : str
        Name of XML file.

    Raises
    ------
    TypeError
        If the MAGE result file containts no steps for time >= 0.
    """
    # Fetch run information from the MAGE result file.
    filename, isMPI, Ri, Rj, Rk = kaiTools.getRunInfo(".", runid)
    if debug:
        print(f"filename = {filename}")
        print(f"isMPI = {isMPI}")
        print(f"Ri = {Ri}")
        print(f"Rj = {Rj}")
        print(f"Rk = {Rk}")

    # Get the number of steps and the step IDs from the MAGE results file.
    nSteps, sIds = kaiH5.cntSteps(filename)
    if debug:
        print(f"nSteps = {nSteps}")
        print(f"sIds = {sIds}")

    # Determine the start and end time of the MAGE results.

    # Find the step for the first value of time >= 0, and that time.
    T0 = None
    for i_step in sIds:
        t_step = kaiH5.tStep(filename, sIds[i_step], aID="time")
        if t_step >= 0.0:
            T0 = t_step
            break
    if debug:
        print(f"T0 = {T0}")
    if T0 is None:
        raise TypeError("MAGE results contain no steps for t >= 0!")

    # Find the time for the last step.
    tFin = kaiH5.tStep(filename, sIds[-1], aID="time")
    if debug:
        print(f"T0 = {T0}")
        print(f"tFin = {tFin}")

    # Read the template XML file.
    with open(CALCDB_XML_TEMPLATE, "r", encoding="utf-8") as f:
        template_content = f.read()
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    options = {}
    options["runid"] = runid
    options["T0"] = T0
    options["dt"] = 60.0
    options["tFin"] = tFin
    options["ebfile"] = runid
    if isMPI:
        options["ismpi"] = "true"
    else:
        options["ismpi"] = "false"
    options["Ri"] = Ri
    options["Rj"] = Rj
    options["Rk"] = Rk
    options["NumB"] = parintime
    if debug:
        print(f"options = {options}")

    # Render the template.
    xml_file = f"calcdb-{runid}.xml"
    xml_content = template.render(options)
    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Return the name of the XML file.
    return xml_file


def create_calcdb_pbs_script(args: dict):
    """Create the PBS script for calcdb.x from a template.

    Create the PBS script for calcdb.x from a template. The PBS script is
    created in the directory containing the MAGE results to compare to
    SuperMag data. The script will set up and run calcdb.x on the specified
    set of MAGE results, in the results directory.

    Parameters
    ----------
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
    mage_results_path = args["mage_results_path"]

    # Split the MAGE results path into a directory and a file.
    (mage_results_dir, mage_results_file) = os.path.split(mage_results_path)
    if debug:
        print(f"mage_results_dir = {mage_results_dir}")
        print(f"mage_results_file = {mage_results_file}")

    # Save the current directory.
    start_directory = os.getcwd()
    if debug:
        print(f"start_directory = {start_directory}")

    # Move to the results directory.
    os.chdir(mage_results_dir)

    # Compute the runid from the file name.
    runid = filename_to_runid(mage_results_file)
    if debug:
        print(f"runid = {runid}")

    # Create the XML input file for calcdb.x.
    calcdb_xml_file = create_calcdb_xml_file(runid, args["parintime"], debug)
    if debug:
        print(f"calcdb_xml_file = {calcdb_xml_file}")

    # Copy the calcdb.x binary to the results directory.
    shutil.copyfile(args["calcdb"], "./calcdb.x")

    # Read the PBS script template for calcdb.x.
    with open(CALCDB_PBS_TEMPLATE, "r", encoding="utf-8") as f:
        template_content = f.read()
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    options = copy.deepcopy(DEFAULT_CALCDB_PBS_OPTIONS)
    options["job_name"] = f"calcdb-{runid}"
    options["select"] = f"{options['select']}:ompthreads={args['parintime']}"
    options["omp_num_threads"] = args["parintime"]
    options["calcdb_xml_file"] = calcdb_xml_file
    options["runid"] = runid
    if debug:
        print(f"options = {options}")

    # Render the template.
    calcdb_pbs_script = f"calcdb-{runid}.pbs"
    xml_content = template.render(options)
    with open(calcdb_pbs_script, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Move back to the start directory.
    os.chdir(start_directory)

    # Return the path to the PBS script.
    return calcdb_pbs_script


def create_pitmerge_pbs_script(args: dict):
    """Create the PBS script for pitmerge.py to stitch calcdb output.

    Create the PBS script for stitching together calcdb output using the
    pitmerge.py script.

    Parameters
    ----------
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
    mage_results_path = args["mage_results_path"]

    # Split the MAGE results path into a directory and a file.
    (mage_results_dir, mage_results_file) = os.path.split(mage_results_path)
    if debug:
        print(f"mage_results_dir = {mage_results_dir}")
        print(f"mage_results_file = {mage_results_file}")

    # Save the current directory.
    start_directory = os.getcwd()
    if debug:
        print(f"start_directory = {start_directory}")

    # Move to the results directory.
    os.chdir(mage_results_dir)

    # Compute the runid from the file name.
    runid = filename_to_runid(mage_results_file)
    if debug:
        print(f"runid = {runid}")

    # Read the PBS script template for pitmerge.py.
    with open(PITMERGE_PBS_TEMPLATE, "r", encoding="utf-8") as f:
        template_content = f.read()
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    options = copy.deepcopy(DEFAULT_PITMERGE_PBS_OPTIONS)
    options["job_name"] = f"pitmerge-{runid}"
    options["select"] = f"{options['select']}:ncpus=128"
    options["runid"] = runid
    if debug:
        print(f"options = {options}")

    # Render the template.
    pitmerge_pbs_script = f"pitmerge-{runid}.pbs"
    xml_content = template.render(options)
    with open(pitmerge_pbs_script, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Move back to the start directory.
    os.chdir(start_directory)

    # Return the path to the PBS script.
    return pitmerge_pbs_script


def create_supermag_comparison_pbs_script(args: dict):
    """Create the PBS script for the MAGE-SuperMag comparison from a template.

    Create the PBS script for the MAGE-SuperMag comparison from a template.

    Parameters
    ----------
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    supermag_comparison_pbs_script : str
        Path to PBS script.

    Raises
    ------
    None
    """
    # Local convenience variables.
    debug = args["debug"]
    mage_results_path = args["mage_results_path"]

    # Split the MAGE results path into a directory and a file.
    (mage_results_dir, mage_results_file) = os.path.split(mage_results_path)
    if debug:
        print(f"mage_results_dir = {mage_results_dir}")
        print(f"mage_results_file = {mage_results_file}")

    # Save the current directory.
    start_directory = os.getcwd()
    if debug:
        print(f"start_directory = {start_directory}")

    # Move to the results directory.
    os.chdir(mage_results_dir)

    # Compute the runid from the file name.
    runid = filename_to_runid(mage_results_file)
    if debug:
        print(f"runid = {runid}")

    # Read the PBS script template for supermag_comparison.py.
    with open(SUPERMAG_COMPARISON_PBS_TEMPLATE, "r", encoding="utf-8") as f:
        template_content = f.read()
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    options = copy.deepcopy(DEFAULT_SUPERMAG_COMPARISON_PBS_OPTIONS)
    options["job_name"] = f"supermag_comparison-{runid}"
    options["select"] = f"{options['select']}:ncpus=128"
    options["runid"] = runid
    if debug:
        print(f"options = {options}")

    # Render the template.
    supermag_comparison_pbs_script = f"supermag_comparison-{runid}.pbs"
    xml_content = template.render(options)
    with open(supermag_comparison_pbs_script, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Move back to the start directory.
    os.chdir(start_directory)

    # Return the path to the PBS script.
    return supermag_comparison_pbs_script


def create_submit_script(
        calcdb_pbs_script: str, pitmerge_pbs_script: str,
        supermag_comparison_pbs_script: str, args: dict):
    """Create the PBS script for the MAGE-SuperMag comparison from a template.

    Create the PBS script for the MAGE-SuperMag comparison from a template.
    The submit script submits the job to run calcdb.x, then submits the
    follow-up comparison job to run only if the calcdb.x job completes
    successfully.

    Parameters
    ----------
    calcdb_pbs_script : str
        Path to calcdb.x PBS script.
    pitmerge_pbs_script : str
        Path to pitmerge.py PBS script.
    supermag_comparison_pbs_script : str
        Path to comparison PBS script.
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
    # Local convenience variables.
    debug = args["debug"]
    mage_results_path = args["mage_results_path"]

    # Split the MAGE results path into a directory and a file.
    (mage_results_dir, mage_results_file) = os.path.split(mage_results_path)
    if debug:
        print(f"mage_results_dir = {mage_results_dir}")
        print(f"mage_results_file = {mage_results_file}")

    # Save the current directory.
    start_directory = os.getcwd()
    if debug:
        print(f"start_directory = {start_directory}")

    # Move to the results directory.
    os.chdir(mage_results_dir)

    # Compute the runid from the file name.
    runid = filename_to_runid(mage_results_file)
    if debug:
        print(f"runid = {runid}")

    # Submit the scripts in dependency order.
    submit_script = f"submit-{runid}.sh"
    with open(submit_script, "w", encoding="utf-8") as f:
        cmd = f"job_id=`qsub {calcdb_pbs_script}`\n"
        f.write(cmd)
        cmd = "echo $job_id\n"
        f.write(cmd)
        cmd = "old_job_id=$job_id\n"
        f.write(cmd)
        cmd = (
            "job_id=`qsub -W depend=afterok:$old_job_id "
            f"{pitmerge_pbs_script}`\n"
        )
        f.write(cmd)
        cmd = "echo $job_id\n"
        f.write(cmd)
        cmd = "old_job_id=$job_id\n"
        f.write(cmd)
        cmd = (
            "job_id=`qsub -W depend=afterok:$old_job_id "
            f"{supermag_comparison_pbs_script}`\n"
        )
        f.write(cmd)
        cmd = "echo $job_id\n"
        f.write(cmd)

    # Return the name of the PBS script.
    return submit_script


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
    # Set defaults for command-line options, then update with values passed
    # from the caller.
    local_args = copy.deepcopy(DEFAULT_ARGUMENTS)
    local_args.update(args)
    args = local_args

    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]

    # ------------------------------------------------------------------------

    # Create the PBS script to run calcdb.x.
    if verbose:
        print("Creating PBS script to run the calcdb.x job.")
    calcdb_pbs_script = create_calcdb_pbs_script(args)
    if debug:
        print(f"calcdb_pbs_script = {calcdb_pbs_script}")

    # Create the PBS script to stitch together the output from calcdb.x.
    if verbose:
        print("Creating PBS script to stitch together the calcdb.x output.")
    pitmerge_pbs_script = create_pitmerge_pbs_script(args)
    if debug:
        print(f"pitmerge_pbs_script = {pitmerge_pbs_script}")

    # Create the PBS script to compare the calcdb.x results with SuperMag
    # data.
    if verbose:
        print("Creating PBS script to run the MAGE-SuperMag comparison job.")
    supermag_comparison_pbs_script = create_supermag_comparison_pbs_script(args)
    if debug:
        print(f"supermag_comparison_pbs_script = {supermag_comparison_pbs_script}")

    # Create the bash script to submit the PBS scripts in the proper order.
    if verbose:
        print("Creating bash script to submit the PBS jobs.")
    submit_script = create_submit_script(
        calcdb_pbs_script, pitmerge_pbs_script,
        supermag_comparison_pbs_script, args
    )
    if debug:
        print(f"submit_script = {submit_script}")

    # if verbose:
    #     print(f"Please run {submit_script} to submit the PBS jobs to run "
    #           "calcdb.x and perform the MAGE-SuperMag comparison.")

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
    return_code = run_supermag_comparison(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
