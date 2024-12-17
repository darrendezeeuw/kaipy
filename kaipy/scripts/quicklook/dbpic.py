#!/usr/bin/env python


"""Plot the ground magnetic field perturbations from a magnetosphere run.

Plot the ground magnetic field perturbations from a magnetosphere run.

Author
------
Kareem Sorathia (kareem.sorathia@jhuapl.edu)
Eric Winter (eric.winter@jhuapl.edu)
"""


# Import standard modules.
import argparse
import copy
import os
import sys

# Import 3rd-party modules.
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# Import project-specific modules.
import kaipy.cdaweb_utils as cdaweb_utils
import kaipy.cmaps.kaimaps as kmaps
import kaipy.gamera.deltabViz as dbViz
import kaipy.gamera.gampp as gampp
import kaipy.kaiH5 as kh5
import kaipy.kaiTools as ktools
import kaipy.kaiViz as kv


# Program constants and defaults

# Program description.
DESCRIPTION = (
    "Plot the ground magnetic field perturbations for a MAGE magnetosphere "
    "run."
)

# Default values for command-line arguments.
DEFAULT_ARGUMENTS = {
    "d": os.getcwd(),
    "debug": False,
    "id": "msphere",
    "Jr": False,
    "k0": 0,
    "n": -1,
    "projection": "both",
    "spacecraft": None,
    "verbose": False,
}

# Default output filename.
default_output_filename = "qkdbpic.png"

# Size of figure in inches (width x height).
figSz = (12, 6)

# Color to use for magnetic footprint positions.
FOOTPRINT_COLOR = "red"


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
        "-d", type=str, metavar="directory", default=DEFAULT_ARGUMENTS["d"],
        help="Directory containing data to read (default: %(default)s)"
    )
    parser.add_argument(
        "--debug", default=DEFAULT_ARGUMENTS["debug"],
        action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "-id", type=str, metavar="runid", default=DEFAULT_ARGUMENTS["id"],
        help="Run ID of data (default: %(default)s)"
    )
    parser.add_argument(
        "-Jr", action="store_true", default=DEFAULT_ARGUMENTS["Jr"],
        help="Show radial component of anomalous current (default: %(default)s)."
    )
    parser.add_argument(
        "-k0", type=int, metavar="layer", default=DEFAULT_ARGUMENTS["k0"],
        help="Vertical layer to plot (default: %(default)s)")
    parser.add_argument(
        "-n", type=int, metavar="step", default=DEFAULT_ARGUMENTS["n"],
        help="Time slice to plot (default: %(default)s)"
    )
    parser.add_argument(
        "--projection", type=str, metavar="projection",
        default=DEFAULT_ARGUMENTS["projection"],
        help="Map projection to use for plots (mercator|polar|both)"
        " (default: %(default)s)"
    )
    parser.add_argument(
        "--spacecraft", type=str, metavar="spacecraft",
        default=DEFAULT_ARGUMENTS["spacecraft"],
        help="Names of spacecraft to plot magnetic footprints, separated by "
        "commas (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose", "-v", default=DEFAULT_ARGUMENTS["verbose"],
        action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    return parser


def create_mercator_plot(args: dict):
    """Make a Mercator plot of the ground magnetic field perturbations.

    Make a Mercator plot of the ground magnetic field perturbations.

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
    fdir = args["d"]
    debug = args["debug"]
    runid = args["id"]
    doJr = args["Jr"]
    k0 = args["k0"]
    nStp = args["n"]
    verbose = args["verbose"]

    # ------------------------------------------------------------------------

    # Compute the tag of the file containing the ground magnetic field perturbations.
    ftag = f"{runid}.deltab"
    if debug:
        print(f"ftag = {ftag}")

    # Read the ground magnetic field perturbations.
    fname = os.path.join(fdir, f"{ftag}.h5")
    if debug:
        print(f"fname = {fname}")
    dbdata = gampp.GameraPipe(fdir, ftag)
    if debug:
        print(f"dbdata = {dbdata}")
    print("---")

    # Get the ID of the coordinate system, and the Earth radius.
    CoordID, Re = dbViz.GetCoords(fname)
    print(f"Found {CoordID} coordinate data ...")
    if debug:
        print(f"CoordID = {CoordID}")
        print(f"Re = {Re}")

    # If the last simulation step was requested, get the step number.
    if nStp < 0:
        nStp = dbdata.sFin
        print(f"Using Step {nStp}")

    # Check the vertical level.
    Z0 = dbViz.CheckLevel(dbdata, k0, Re)
    if debug:
        print(f"Z0 = {Z0}")

    # If currents were requested, read them. Otherwise, read the ground
    # magnetic field perturbations.
    if doJr:
        print("Reading Jr ...")
        Jr = dbdata.GetVar("dbJ", nStp, doVerb=False)[:, :, k0]
        Q = Jr
    else:
        dBn = dbdata.GetVar("dBn", nStp, doVerb=True)[:, :, k0]
        Q = dBn

    # Convert MJD to UT.
    MJD = kh5.tStep(fname, nStp, aID="MJD")
    if debug:
        print(f"MJD = {MJD}")
    utS = ktools.MJD2UT([MJD])
    if debug:
        print(f"utS = {utS}")
    utDT= utS[0]
    if debug:
        print(f"utDT = {utDT}")

    # Create the mapping grid.
    crs = ccrs.PlateCarree()
    if debug:
        print(f"ccrs = {ccrs}")
    LatI, LonI, LatC, LonC = dbViz.GenUniformLL(dbdata, k0)
    if debug:
        print(f"LatI = {LatI}")
        print(f"LonI = {LonI}")
        print(f"LatC = {LatC}")
        print(f"LonC = {LonC}")

    # Fetch the color map.
    cmap = kmaps.cmDiv
    if debug:
        print(f"cmap = {cmap}")

    # Determine color bar settings.
    if doJr:
        vQ = kv.genNorm(dbViz.jMag)
        cbStr = "Anomalous current"
    else:
        vQ = kv.genNorm(dbViz.dbMag, doSymLog=True, linP=dbViz.dbLin)
        cbStr = r"$\Delta B_N$ [nT]"
    if debug:
        print(f"vQ = {vQ}")
        print(f"cbStr = {cbStr}")

    # Create plot in memory.
    mpl.use("Agg")

    # Create the figure to hold the plot.
    fig = plt.figure(figsize=figSz)

    # Specify the grid for the subplots.
    gs = gridspec.GridSpec(3, 1, height_ratios=[20, 1.0, 1.0], hspace=0.025)

    # Create the subplots.
    AxM = fig.add_subplot(gs[0, 0], projection=crs)
    AxCB = fig.add_subplot(gs[-1, 0])

    # Make the plot.
    AxM.pcolormesh(LonI, LatI, Q, norm=vQ, cmap=cmap)

    # If requested, overlay the spacecraft magnetic footprints.
    if args["spacecraft"]:
        print(f"Overplotting magnetic footprints of {args['spacecraft']}")

        # Split the list into individual spacecraft names.
        spacecraft = args["spacecraft"].split(",")

        # Fetch the position of each footprint pair from CDAWeb.
        for sc in spacecraft:

            # Fetch the northern footprint position.
            fp_nlat, fp_nlon = cdaweb_utils.fetch_satellite_magnetic_northern_footprint_position(
                sc, utS[0]
            )
            if debug:
                print(f"fp_nlat, fp_nlon = {fp_nlat}, {fp_nlon}")

            # Fetch the southern footprint position.
            fp_slat, fp_slon = cdaweb_utils.fetch_satellite_magnetic_southern_footprint_position(
                sc, utS[0]
            )
            if debug:
                print(f"fp_slat, fp_slon = {fp_slat}, {fp_slon}")

            # Plot a labelled dot at the location of each footprint.
            # Skip if no footprint position found.
            if fp_nlon is not None:
                AxM.plot(fp_nlon, fp_nlat, "o", c=FOOTPRINT_COLOR)
                lon_nudge = 2.0
                lat_nudge = 2.0
                AxM.text(fp_nlon + lon_nudge, fp_nlat + lat_nudge, sc + " (N)")
            else:
                print(f"No northern footprint found for spacecraft {sc}.")
            if fp_slon is not None:
                AxM.plot(fp_slon, fp_slat, "o", c=FOOTPRINT_COLOR)
                lon_nudge = 2.0
                lat_nudge = 2.0
                AxM.text(fp_slon + lon_nudge, fp_slat + lat_nudge, sc + " (S)")
            else:
                print(f"No southern footprint found for spacecraft {sc}.")

    # Make the colorbar.
    kv.genCB(AxCB, vQ, cbStr, cM=cmap)

    # Add labels and other decorations.
    tStr = dbViz.GenTStr(AxM, fname, nStp)
    if debug:
        print(f"tStr = {tStr}")
    dbViz.DecorateDBAxis(AxM, crs, utDT)

    # Save the figure.
    fOut = os.path.join(args["d"], default_output_filename)
    if debug:
        print(f"fOut = {fOut}")
    kv.savePic(fOut)


def create_polar_plot(args: dict):
    """Make a polar plot of the ground magnetic field perturbations.

    Make a polar plot of the ground magnetic field perturbations.

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


def dbpic(args: dict):
    """Plot the ground magnetic field perturbations.

    Plot the ground magnetic field perturbations.

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

    # If requested, create the Mercator plot.
    if args["projection"] in ["mercator", "both"]:
        if verbose:
            print("Creating Mercator plot.")
        create_mercator_plot(args)

    # If requested, create the polar plot.
    if args["projection"] in ["polar", "both"]:
        if verbose:
            print("Creating polar plot.")
        create_polar_plot(args)

    # ------------------------------------------------------------------------

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
    return_code = dbpic(args)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
