#!/usr/bin/env python


"""Analyze the ground delta-B values from a MAGE run.

Perform an analysis of ground magnetic field perturbations computed
for a MAGE magnetosphere simulation. The computed values are compared
with measured data from SuperMag. Then a set of SuperMAGE ground
delta-B maps is created. This is all done by a linked set of PBS jobs
running the MAGE results directory.

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
DESCRIPTION = "Compare MAGE ground delta-B to SuperMag measurements, and create SuperMAGE analysis maps."

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "calcdb": "calcdb.x",
    "debug": False,
    "dt": 60.0,
    "hpc": "pleiades",
    "parintime": 1,
    "pbs_account": "",
    "smuser": "",
    "verbose": False,
    "mage_results_path": None,
}

# Location of template XML file for calcdb.x.
CALCDB_XML_TEMPLATE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "templates",
    "calcdb-template.xml"
)

# Locations of template PBS files for calcdb.x for derecho and pleiades.
CALCDB_PBS_TEMPLATE_DERECHO = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "templates",
    "calcdb-template-derecho.pbs"
)
CALCDB_PBS_TEMPLATE_PLEIADES = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "templates",
    "calcdb-template-pleiades.pbs"
)

# Default options for filling in the calcdb.x PBS template on derecho.
DEFAULT_CALCDB_PBS_OPTIONS_DERECHO = {
    "job_name": None,
    "account": "",
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

# Default options for filling in the calcdb.x PBS template on pleiades.
DEFAULT_CALCDB_PBS_OPTIONS_PLEIADES = {
    "job_name": None,
    "queue": "normal",
    "select": "1:ncpus=28:model=bro",
    "walltime": "01:00:00",
    "modules": [
        "nas",
        "pkgsrc/2022Q1-rome",
        "comp-intel/2020.4.304",
        "mpi-hpe/mpt.2.23",
        "hdf5/1.8.18_mpt",
    ],
    "conda_environment": "kaiju-3.8",
    "kaipyhome": os.environ["KAIPYHOME"],
    "kaijuhome": os.environ["KAIJUHOME"],
}

# Locations of template PBS files for pitmerge.py for derecho and pleiades.
PITMERGE_PBS_TEMPLATE_DERECHO = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "templates",
    "pitmerge-template-derecho.pbs"
)
PITMERGE_PBS_TEMPLATE_PLEIADES = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "templates",
    "pitmerge-template-pleiades.pbs"
)

# Default options for filling in the pitmerge.py PBS template for
# derecho.
DEFAULT_PITMERGE_PBS_OPTIONS_DERECHO = {
    "job_name": None,
    "account": "",
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

# Default options for filling in the pitmerge.py PBS template for
# pleiades.
DEFAULT_PITMERGE_PBS_OPTIONS_PLEIADES = {
    "job_name": None,
    "queue": "normal",
    "select": "1:ncpus=28:model=bro",
    "walltime": "01:00:00",
    "modules": [
        "nas",
        "pkgsrc/2022Q1-rome",
        "comp-intel/2020.4.304",
        "mpi-hpe/mpt.2.23",
        "hdf5/1.8.18_mpt",
    ],
    "conda_environment": "kaiju-3.8",
    "kaipyhome": os.environ["KAIPYHOME"],
    "kaijuhome": os.environ["KAIJUHOME"],
}

# Locations of template PBS files for running the ground delta-B
# analysis on derecho and pleiades
GROUND_DELTAB_ANALYSIS_PBS_TEMPLATE_DERECHO = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "templates",
    "ground_deltab_analysis-template-derecho.pbs"
)
GROUND_DELTAB_ANALYSIS_PBS_TEMPLATE_PLEIADES = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "templates",
    "ground_deltab_analysis-template-pleiades.pbs"
)

# Default options for filling in the ground delta-B analysis PBS
# template on pleiades.
DEFAULT_GROUND_DELTAB_ANALYSIS_PBS_OPTIONS_PLEIADES = {
    "job_name": None,
    "queue": "normal",
    "select": "1:ncpus=28:model=bro",
    "walltime": "01:00:00",
    "modules": [
        "nas",
        "pkgsrc/2022Q1-rome",
        "comp-intel/2020.4.304",
        "mpi-hpe/mpt.2.23",
        "hdf5/1.8.18_mpt",
    ],
    "conda_environment": "kaiju-3.8",
    "kaipyhome": os.environ["KAIPYHOME"],
    "kaijuhome": os.environ["KAIJUHOME"],
}

# Default options for filling in the ground delta-B analysis PBS
# template on derecho.
DEFAULT_GROUND_DELTAB_ANALYSIS_PBS_OPTIONS_DERECHO = {
    "job_name": None,
    "account": "",
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
        "--dt", type=float, default=DEFAULT_ARGUMENTS["dt"],
        help="Time interval for delta-B computation (seconds)"
             " (default: %(default)s)."
    )
    parser.add_argument(
        "--hpc", type=str, default=DEFAULT_ARGUMENTS["hpc"],
        help="HPC system to run analysis (derecho|pleiades)"
             " (default: %(default)s)."
    )
    parser.add_argument(
        "--parintime", type=int, default=DEFAULT_ARGUMENTS["parintime"],
        help="Split the calculation into this many parallel chunks"
             " (default: %(default)s)."
    )
    parser.add_argument(
        "--pbs_account", type=str, default=DEFAULT_ARGUMENTS["pbs_account"],
        help="Split the calculation into this many parallel chunks"
             " (default: %(default)s)."
    )
    parser.add_argument(
        "--smuser", type=str, default=DEFAULT_ARGUMENTS["smuser"],
        help="Account name for SuperMag database access"
             " (default: %(default)s)."
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


def create_calcdb_xml_file(runid: str, args: dict):
    """Create the XML input file for calcdb.x from a template.

    Create the XML input file for calcdb.x from a template. The file is
    created in the current directory.

    Parameters
    ----------
    runid : str
        runid for MAGE results file.
    args : dict
        Dictionary of command-line options.

    Returns
    -------
    xml_file : str
        Name of XML file.

    Raises
    ------
    TypeError
        If the MAGE result file contains no steps for time >= 0.
    """
    # Local convenience variables.
    debug = args["debug"]
    dt = args["dt"]
    parintime = args["parintime"]
    verbose = args["verbose"]

    # Fetch run information from the MAGE result file.
    if verbose:
        print(f"Fetching run information for run {runid}.")
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

    # Determine the start and end time of the MAGE results.

    # Find the step for the first value of time >= 0, and that time.
    if verbose:
        print(f"Finding time for first step with t >= 0 for run {runid}.")
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
    ismpi = "false"
    if isMPI:
        ismpi = "true"
    options = {
        "runid": runid,
        "T0": T0,
        "dt": dt,
        "tFin": tFin,
        "ebfile": runid,
        "ismpi": ismpi,
        "Ri": Ri,
        "Rj": Rj,
        "Rk": Rk,
        "NumB": parintime,
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
    TypeError
        If an invalid HPC system is specified.
    """
    # Local convenience variables.
    calcdb = args["calcdb"]
    debug = args["debug"]
    hpc = args["hpc"]
    verbose = args["verbose"]
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
    if verbose:
        print(f"Moving to directory {mage_results_dir}.")
    os.chdir(mage_results_dir)

    # Compute the runid from the file name.
    runid = filename_to_runid(mage_results_file)
    if debug:
        print(f"runid = {runid}")

    # Create the XML input file for calcdb.x.
    if verbose:
        print("Creating XML file for calcdb.x.")
    calcdb_xml_file = create_calcdb_xml_file(runid, args)
    if debug:
        print(f"calcdb_xml_file = {calcdb_xml_file}")

    # Copy the calcdb.x binary to the results directory, then make it
    # executable.
    if verbose:
        print(f"Copying {calcdb} to results directory.")
    shutil.copyfile(calcdb, "./calcdb.x")
    os.chmod("./calcdb.x", 0o755)

    # Read the PBS script template for calcdb.x.
    if verbose:
        print(f"Reading calcdb.x PBS template for {hpc}.")
    if hpc == "derecho":
        with open(CALCDB_PBS_TEMPLATE_DERECHO, "r", encoding="utf-8") as f:
            template_content = f.read()
    elif hpc == "pleiades":
        with open(CALCDB_PBS_TEMPLATE_PLEIADES, "r", encoding="utf-8") as f:
            template_content = f.read()
    else:
        raise TypeError(f"Invalid hpc ({hpc})!")
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    if hpc == "derecho":
        options = copy.deepcopy(DEFAULT_CALCDB_PBS_OPTIONS_DERECHO)
        options.update({
            "job_name": f"calcdb-{runid}",
            "account": args["pbs_account"],
            "select": f"{options['select']}:ompthreads={args['parintime']}",
            "omp_num_threads": args["parintime"],
            "calcdb_xml_file": calcdb_xml_file,
            "runid": runid,
        })
    elif hpc == "pleiades":
        options = copy.deepcopy(DEFAULT_CALCDB_PBS_OPTIONS_PLEIADES)
        options.update({
            "job_name": f"calcdb-{runid}",
            "select": f"{options['select']}:ompthreads={args['parintime']}",
            "omp_num_threads": args["parintime"],
            "calcdb_xml_file": calcdb_xml_file,
            "runid": runid,
        })
    else:
        raise TypeError(f"Invalid hpc ({hpc})!")
    if debug:
        print(f"options = {options}")

    # Render the template.
    if verbose:
        print("Rendering template.")
    calcdb_pbs_script = f"calcdb-{runid}.pbs"
    calcdb_pbs_content = template.render(options)
    with open(calcdb_pbs_script, "w", encoding="utf-8") as f:
        f.write(calcdb_pbs_content)

    # Move back to the start directory.
    if verbose:
        print(f"Moving to directory {start_directory}.")
    os.chdir(start_directory)

    # Return the name of the PBS script.
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
    TypeError
        If an invalid HPC system is specified.
    """
    # Local convenience variables.
    debug = args["debug"]
    hpc = args["hpc"]
    verbose = args["verbose"]
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
    if verbose:
        print(f"Moving to directory {mage_results_dir}.")
    os.chdir(mage_results_dir)

    # Compute the runid from the file name.
    runid = filename_to_runid(mage_results_file)
    if debug:
        print(f"runid = {runid}")

    # Read the PBS script template for pitmerge.py.
    if verbose:
        print(f"Reading pitmerge.py PBS template for {hpc}.")
    if hpc == "derecho":
        with open(PITMERGE_PBS_TEMPLATE_DERECHO, "r", encoding="utf-8") as f:
            template_content = f.read()
    elif hpc == "pleiades":
        with open(PITMERGE_PBS_TEMPLATE_PLEIADES, "r", encoding="utf-8") as f:
            template_content = f.read()
    else:
        raise TypeError(f"Invalid hpc ({hpc})!")
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    if hpc == "derecho":
        options = copy.deepcopy(DEFAULT_PITMERGE_PBS_OPTIONS_DERECHO)
        options.update({
            "job_name": f"pitmerge-{runid}",
            "account": args["pbs_account"],
            "runid": runid,
        })
    elif hpc == "pleiades":
        options = copy.deepcopy(DEFAULT_PITMERGE_PBS_OPTIONS_PLEIADES)
        options.update({
            "job_name": f"pitmerge-{runid}",
            "runid": runid,
        })
    else:
        raise TypeError(f"Invalid hpc ({hpc})!")
    if debug:
        print(f"options = {options}")

    # Render the template.
    if verbose:
        print("Rendering template.")
    pitmerge_pbs_script = f"pitmerge-{runid}.pbs"
    pitmerge_pbs_content = template.render(options)
    with open(pitmerge_pbs_script, "w", encoding="utf-8") as f:
        f.write(pitmerge_pbs_content)

    # Move back to the start directory.
    if verbose:
        print(f"Moving to directory {start_directory}.")
    os.chdir(start_directory)

    # Return the name of the PBS script.
    return pitmerge_pbs_script


def create_ground_deltab_analysis_pbs_script(args: dict):
    """Create the PBS script for the ground delta-B analysis from a template.

    Create the PBS script for the ground delta-B analysis from a template.

    Parameters
    ----------
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    ground_deltab_analysis_pbs_script : str
        Path to PBS script.

    Raises
    ------
    TypeError
        If an invalid HPC system is specified.
    """
    # Local convenience variables.
    debug = args["debug"]
    hpc = args["hpc"]
    verbose = args["verbose"]
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
    if verbose:
        print(f"Moving to {mage_results_dir}.")
    os.chdir(mage_results_dir)

    # Compute the runid from the file name.
    runid = filename_to_runid(mage_results_file)
    if debug:
        print(f"runid = {runid}")

    # Read the PBS script template for ground_deltab_analysis.py.
    if verbose:
        print(f"Reading analysis PBS template for {hpc}.")
    if hpc == "derecho":
        with open(GROUND_DELTAB_ANALYSIS_PBS_TEMPLATE_DERECHO, "r",
                  encoding="utf-8") as f:
            template_content = f.read()
    elif hpc == "pleiades":
        with open(GROUND_DELTAB_ANALYSIS_PBS_TEMPLATE_PLEIADES, "r",
                  encoding="utf-8") as f:
            template_content = f.read()
    else:
        raise TypeError(f"Invalid hpc ({hpc})!")
    if debug:
        print(f"template_content = {template_content}")
    template = Template(template_content)
    if debug:
        print(f"template = {template}")

    # Fill in the template options.
    if hpc == "derecho":
        options = copy.deepcopy(
            DEFAULT_GROUND_DELTAB_ANALYSIS_PBS_OPTIONS_DERECHO
        )
        options.update({
            "job_name": f"ground_deltab_analysis-{runid}",
            "account": args["pbs_account"],
            "runid": runid,
            "smuser": args["smuser"],
            "calcdb_results_path": f"./{runid}.deltab.h5",
        })
    elif hpc == "pleiades":
        options = copy.deepcopy(
            DEFAULT_GROUND_DELTAB_ANALYSIS_PBS_OPTIONS_PLEIADES
        )
        options.update({
            "job_name": f"ground_deltab_analysis-{runid}",
            "runid": runid,
            "smuser": args["smuser"],
            "calcdb_results_path": f"./{runid}.deltab.h5",
        })
    if debug:
        print(f"options = {options}")

    # Render the template.
    if verbose:
        print("Rendering template.")
    ground_deltab_analysis_pbs_script = f"ground_deltab_analysis-{runid}.pbs"
    xml_content = template.render(options)
    with open(ground_deltab_analysis_pbs_script, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Move back to the start directory.
    if verbose:
        print(f"Moving to {start_directory}.")
    os.chdir(start_directory)

    # Return the name of the PBS script.
    return ground_deltab_analysis_pbs_script


def create_submit_script(
        calcdb_pbs_script: str, pitmerge_pbs_script: str,
        ground_deltab_analysis_pbs_script: str, args: dict):
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
    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]
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
    if verbose:
        print(f"Moving to {mage_results_dir}.")
    os.chdir(mage_results_dir)

    # Compute the runid from the file name.
    runid = filename_to_runid(mage_results_file)
    if debug:
        print(f"runid = {runid}")

    # Submit the scripts in dependency order.
    submit_script = f"submit-{runid}.sh"
    with open(submit_script, "w", encoding="utf-8") as f:
        cmd = f"job_id=`qsub -J 1-{args['parintime']} {calcdb_pbs_script}`\n"
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
            f"{ground_deltab_analysis_pbs_script}`\n"
        )
        f.write(cmd)
        cmd = "echo $job_id\n"
        f.write(cmd)

    # Return the name of the PBS script.
    return submit_script


def run_ground_deltab_analysis(args: dict):
    """Compare MAGE results for ground dB to SuperMag data.

    Compare MAGE results for ground dB to SuperMag data.

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

    # ------------------------------------------------------------------------

    # Create the PBS script to run calcdb.x.
    if verbose:
        print("Creating PBS script to run calcdb.x.")
    calcdb_pbs_script = create_calcdb_pbs_script(args)
    if debug:
        print(f"calcdb_pbs_script = {calcdb_pbs_script}")

    # Create the PBS script to stitch together the output from calcdb.x
    # using pitmerge.py.
    if verbose:
        print("Creating PBS script to stitch together the calcdb.x output.")
    pitmerge_pbs_script = create_pitmerge_pbs_script(args)
    if debug:
        print(f"pitmerge_pbs_script = {pitmerge_pbs_script}")

    # Create the PBS script to compare the calcdb.x results with SuperMag
    # data.
    if verbose:
        print("Creating PBS script to run the ground delta-B analysis.")
    ground_deltab_analysos_pbs_script = create_ground_deltab_analysis_pbs_script(args)
    if debug:
        print(f"ground_deltab_analysos_pbs_script = {ground_deltab_analysos_pbs_script}")

    # Create the bash script to submit the PBS scripts in the proper order.
    if verbose:
        print("Creating bash script to submit the PBS jobs.")
    submit_script = create_submit_script(
        calcdb_pbs_script, pitmerge_pbs_script,
        ground_deltab_analysos_pbs_script, args
    )
    if debug:
        print(f"submit_script = {submit_script}")

    if verbose:
        print(f"Please run {submit_script} (in the MAGE result directory) to "
              "submit the PBS jobs to run perform the MAGE-SuperMag "
              "comparison.")

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
