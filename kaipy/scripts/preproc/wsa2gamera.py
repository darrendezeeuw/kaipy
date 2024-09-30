#! /usr/bin/env python

"""Convert data from a WSA model run to gamhelio format.

This script converts a FITS file created by a WSA run to a HDF5 format
suitable for use by gamhelio.

Authors
-------
Elena Provornikova
Eric Winter

"""


# Import standard modules.
import argparse
import os
# import os,sys,glob
# import time

# Import 3rd-party modules.
from astropy.convolution import convolve, Gaussian2DKernel
import h5py
import numpy as np
from scipy import interpolate
# import matplotlib.pyplot as plt

# Import project modules.
from kaipy.gamera import gamGrids as gg
from kaipy.gamhelio.wsa2gamera import params as params
from kaipy.gamhelio.lib import wsa
from kaipy.kdefs import JD2MJD, Mp_cgs, kbltz, Tsolar, Day2s, Rsolar, vc_cgs


# Program constants

# Program description.
DESCRIPTION = "Convert WSA FITS output to gamhelio format."


def create_command_line_parser(description: str):
    """Create the command-line argument parser.

    Create the parser for command-line arguments.

    Parameters
    ----------
    description : str
        Description of script

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line argument parser for this script.

    Raises
    ------
    None
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "ConfigFileName", default="startup.config",
        help="Name of the configuration file to use (default: %(default)s).")
    return parser


def wsa2gamera(args):
    """Convert WSA FITS output files to gamhelio format.

    Convert WSA FITS output files to gamhelio format.

    Parameters
    ----------
    args : dict
        Dictionary of command-line and other options.

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # Local convenience variables.
    debug = args["debug"]
    verbose = args["verbose"]

    # ------------------------------------------------------------------------

    # Read the configuration file.
    prm = params.params(args["ConfigFileName"])

    # Fetch individual parameters.
    Ng = prm.Nghost
    gamma = prm.gamma
    # Temperature in the current sheet for pressure balance calculation
    TCS = prm.TCS
    # Density in the current sheet for pressure balance calculation
    nCS = prm.nCS

    # Grid parameters
    tMin = prm.tMin
    tMax = prm.tMax
    Rin = prm.Rin
    Rout = prm.Rout
    Ni = prm.Ni
    Nj = prm.Nj
    Nk = prm.Nk

    # Conversions from wsa to gamera units.
    cms2kms = 1.e-5 # cm/s => km/s
    Gs2nT = 1.e5    # Gs => nT
    # Conversion for E field  1 statV/cm = 3.e7 mV/m
    eScl = 3.e7

    # Fetch the name of the WSA FITS file.
    ffits = prm.wsaFile

    # Generate spherical helio grid
    print(f"Generating gamera-helio grid Ni = {Ni}, Nj  = {Nj}, Nk = {Nk}")
    X3, Y3, Z3 = gg.GenKSph(
        Ni=Ni, Nj=Nj, Nk=Nk, Rin=Rin, Rout=Rout, tMin=tMin, tMax=tMax
    )
    gg.WriteGrid(
        X3, Y3, Z3, fOut=os.path.join(prm.GridDir, prm.gameraGridFile)
    )
    if os.path.exists(prm.gameraGridFile):
        print("Grid file heliogrid.h5 is ready!")

    # Read the WSA FITS file.
    (jd_c, phi_wsa_v, theta_wsa_v, phi_wsa_c, theta_wsa_c, bi_wsa, v_wsa,
     n_wsa, T_wsa) = wsa.read(ffits, prm.densTempInfile, prm.normalized)
    # Units of WSA input
    # bi_wsa in [Gs] 
    # v_wsa in [cm/s]
    # n_wsa in [g cm-3]
    # T_wsa in [K]

    # Convert the Julian day in the center of the WSA map into modified
    # Julian day.
    mjd_c = jd_c - JD2MJD

    # Get the GAMERA grid for further interpolation.
    with h5py.File(os.path.join(prm.GridDir, prm.gameraGridFile), "r") as f:
        x = f["X"][:]
        y = f["Y"][:]
        z = f["Z"][:]

    # Compute the cell centers, note order of indexes [k,j,i].
    xc = 0.125*(x[:-1, :-1, :-1] + x[:-1, 1:, :-1] + x[:-1, :-1, 1:] +
                x[:-1, 1:, 1:] + x[1:, :-1, :-1] + x[1:, 1:, :-1] +
                x[1:, :-1, 1:] + x[1:, 1:, 1:])
    yc = 0.125*(y[:-1, :-1, :-1] + y[:-1, 1:, :-1] + y[:-1, :-1, 1:] +
                y[:-1, 1:, 1:] + y[1:, :-1, :-1] + y[1:, 1:, :-1] +
                y[1:, :-1, 1:] + y[1:, 1:, 1:])
    zc = 0.125*(z[:-1, :-1, :-1] + z[:-1, 1:, :-1] + z[:-1, :-1, 1:] +
                z[:-1, 1:, 1:] + z[1:, :-1, :-1] + z[1:, 1:, :-1] +
                z[1:, :-1, 1:] + z[1:, 1:, 1:])

    # Compute the radius of the inner boundary.
    R0 = np.sqrt(x[0, 0, Ng]**2 + y[0, 0, Ng]**2 + z[0, 0, Ng]**2)

    # Compute the heliocentric radius at each grid vertex.
    r = np.sqrt(x**2 + y**2 + z**2)

    # Calculate phi and theta in physical domain (excluding ghost cells).
    P = np.arctan2(y[Ng:-Ng, Ng:-Ng, :], x[Ng:-Ng, Ng:-Ng, :])
    P[P < 0] = P[P < 0] + 2*np.pi
    # P = P % (2*np.pi)  # sometimes the very first point may be a very
                        # small negative number, which the above call sets
                        # to 2*pi. This takes care of it.
    T = np.arccos(z[Ng:-Ng, Ng:-Ng, :]/r[Ng:-Ng, Ng:-Ng, :])

    # Compute the grid for inner i-ghost region; output to innerbc.h5.
    P_out = P[:, :, 0:Ng + 1]
    T_out = T[:, :, 0:Ng + 1]
    R_out = r[Ng:-Ng, Ng:-Ng, 0:Ng + 1]

    # Calculate r, phi and theta coordinates of cell centers in physical
    # domain (excluding ghost cells)
    Rc = np.sqrt(xc[Ng:-Ng, Ng:-Ng, :]**2 + yc[Ng:-Ng, Ng:-Ng, :]**2 +
                 zc[Ng:-Ng, Ng:-Ng, :]**2)
    Pc = np.arctan2(yc[Ng:-Ng, Ng:-Ng, :], xc[Ng:-Ng, Ng:-Ng, :])
    Pc[Pc < 0] = Pc[Pc < 0] + 2*np.pi
    Tc = np.arccos(zc[Ng:-Ng, Ng:-Ng, :]/Rc)

    # This is fast and better than griddata in that it nicely extrapolates
    # boundaries:
    fbi = interpolate.RectBivariateSpline(
        phi_wsa_c, theta_wsa_c, bi_wsa.T, kx=1, ky=1
    )
    br = fbi(Pc[:, 0, 0], Tc[0, :, 0])

    # Smoothing
    if not prm.gaussSmoothWidth == 0:
        gauss = Gaussian2DKernel(width=prm.gaussSmoothWidth)
        br = convolve(br, gauss, boundary="extend")

    # Interpolate to Gamera grid
    fv = interpolate.RectBivariateSpline(
        phi_wsa_c, theta_wsa_c, v_wsa.T, kx=1, ky=1
    )
    vr = fv(Pc[:, 0, 0], Tc[0, :, 0])

    f = interpolate.RectBivariateSpline(
        phi_wsa_c, theta_wsa_c, n_wsa.T, kx=1, ky=1)
    rho = f(Pc[:, 0, 0], Tc[0, :, 0])

    # Not interpolating temperature, but calculating from the total pressure
    # balance AFTER interpolating br and rho to the gamera grid
    # n_CS*k*T_CS = n*k*T + Br^2/8pi  
    temp = (nCS*kbltz*TCS - br**2/8./np.pi)*Mp_cgs/rho/kbltz
    # temperature in [K]

    # check
    # print ("Max and min of temperature in MK")
    # print (np.amax(temp)*1.e-6, np.amin(temp)*1.e-6)

    # note, redefining interpolation functions we could also
    # interpolate from bi_wsa as above, but then we would have to
    # smooth bk, if necessary. The way we're doing it here, bk will be
    # smoothed or not, dependent on whether br has been smoothed.
    # note also, this has to extrapolate
    fbi = interpolate.RectBivariateSpline(
        Pc[:, 0, 0], Tc[0, :, 0], br, kx=1, ky=1
    )
    fv = interpolate.RectBivariateSpline(
        Pc[:, 0, 0], Tc[0, :, 0], vr, kx=1, ky=1
    )

    br_kface = fbi(P[:-1, 0, 0], Tc[0, :, 0])  #(Nk,Nj)
    vr_kface = fv(P[:-1, 0, 0], Tc[0, :, 0])   #(Nk,Nj)

    # before applying scaling inside ghost region
    # get br values to the left of an edge for E_theta calculation
    br_kedge = np.roll(br, 1, axis=1)

    # Scale inside ghost region
    (vr, vr_kface, rho, temp, br, br_kface) = [
        np.dstack(Ng*[var]) for var in (vr, vr_kface, rho, temp, br, br_kface)
    ]
    rho *= (R0/Rc[0, 0, :Ng])**2
    br *= (R0/Rc[0, 0, :Ng])**2
    br_kface *= (R0/Rc[0, 0, :Ng])**2

    # Calculating E-field component on k_edges in [mV/m]
    # E_theta = B_phi*Vr/c = - Omega*R*sin(theta)/Vr*Br * Vr/c =
    # - Omega*R*sin(theta)*Br/c
    omega = 2*np.pi/(Tsolar*Day2s) # [1/s]
    # Theta at centers of k-faces (== theta at kedges)
    Tcf = 0.25*(T[:, :-1, :-1] + T[:, 1:, 1:] + T[:, :-1, 1:] + T[:, 1:,: -1])
    et_kedge = -omega*R0*Rsolar*np.sin(
        Tcf[:-1, :, Ng-1])*br_kedge/vc_cgs  #[statV/cm]

    # Unit conversion agreement. Input to GAMERA innerbc.h5 has units V[km/s], Rho[cm-3], T[K], B[nT], E[mV/m]
    vr *= cms2kms
    vr_kface *= cms2kms
    rho /= Mp_cgs
    br *= Gs2nT
    br_kface *= Gs2nT
    et_kedge *= eScl

    # Create the output HDF5 file.
    with h5py.File(os.path.join(prm.IbcDir, prm.gameraIbcFile), "w") as hf:
        hf.create_dataset("X", data=P_out)
        hf.create_dataset("Y", data=T_out)
        hf.create_dataset("Z", data=R_out)
        grname = "Step#0"
        grp = hf.create_group(grname)
        grp.attrs.create("MJD", mjd_c)
        grp.create_dataset("vr",data=vr)
        grp.create_dataset("vr_kface",data=vr_kface) # size (Nk,Nj,Ng)
        grp.create_dataset("rho",data=rho)
        grp.create_dataset("temp",data=temp)
        grp.create_dataset("br",data=br)
        grp.create_dataset("br_kface",data=br_kface) # size (Nk,Nj,Ng)
        grp.create_dataset("et_kedge",data=et_kedge)  # size (Nk, Nj)
    hf.close()


def main():
    """Driver for command-line version of code."""
    # Set up the command-line parser.
    parser = create_command_line_parser(DESCRIPTION)

    # Parse the command-line arguments.
    args = parser.parse_args()
    if args.debug:
        print(f"args = {args}")

    # Convert the arguments from Namespace to dict.
    args = vars(args)
    if args["debug"]:
        print(f"args = {args}")

    # Pass the command-line arguments to the main function as a dict.
    wsa2gamera(args)


if __name__ == "__main__":
    main()
