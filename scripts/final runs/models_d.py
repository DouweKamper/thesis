# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:39:10 2024

@author: Douwe

a function that returns hydraulic heads for the four MODFLOW6 models I designed
"""

# external modules
import os
from tempfile import TemporaryDirectory
from pprint import pformat
import numpy as np
import flopy

# modules created by Douwe:
from constants_d import (
    UNIVERSAL,
    UNIQUE,
)  # define MODFLOW6 model dimensions etc.


def model_output(
    MODEL_NR,
    LOCATIONS=None,
    K=None,
    OUTER_DVCLOSE=None,
    INNER_DVCLOSE=None,
    OUTER_MAXIMUM=1000,
    INNER_MAXIMUM=1000,
    SILENT=True,
):
    """
    Runs the MODFLOW model with optionally user-specified hydraulic conductivities.

        Parameters:
        -----------
        MODEL_NR : int
            The identifier number of the specific model.

        LOCATIONS : tuple
            Specifies the measurement location(s). Where each location is a tuple
            such as (1,15,14), or for e.g. 3 locations: ((0, 6, 2), (0, 7, 4), (0, 12, 20)),
            where the dimensions are as follows: (z, x, y)
            If not provided, hydraulic heads at all possible locations will be returned

        K : custom, optional
            Hydraulic conductivity value(s). This can be a single float or int, or any custom object that represents hydraulic conductivity.
            If not provided, default values will be used.

        OUTER_DVCLOSE : float, optional
            Convergence criterion for the outer (non-linear) solver

        INNER_DVCLOSE : float, optional
            Convergence criterion for the inner (linear) solver

        OUTER_MAXIMUM : int, optional
            Maximum number of iterations for the outer (non-linear) solver

        INNER_MAXIMUM : int, optional
            Maximum number of iterations for the outer (linear) solver

        SILENT : bool, optional
            whether MODFLOW prints info about the simulation

        Returns:
        --------
        Array of hydraulic heads for the given measurement locations and model
        setup at the end of the simulation period.
    """
    # universal constants
    ZTOP = UNIVERSAL["ZTOP"]  # highest point in z direction
    NROW = UNIVERSAL["NROW"]  # number of rows (for discretization in x)
    NCOL = UNIVERSAL["NCOL"]  # number of rows (for discretization in y)
    DELR = UNIVERSAL["DELR"]  # length of a single cell in x direction
    DELC = UNIVERSAL["DELC"]  # length of a single cell in y direction
    HB = UNIVERSAL["HB"]  # head at boundary (Dirichlet boundary condition)
    WEL_SPECS = UNIVERSAL["WEL_SPECS"]  # well location and pumping rate
    # unique constants
    NAME = UNIQUE["NAME"][MODEL_NR - 1]  # name of model
    NLAY = UNIQUE["NLAY"][MODEL_NR - 1]  # number of layers in model
    BOTM = UNIQUE["BOTM"][MODEL_NR - 1]  # vertical discretization all layers
    ISOTROPIC = UNIQUE["ISOTROPIC"][MODEL_NR - 1]
    if K is None:
        HK = UNIQUE["HK"][MODEL_NR - 1]  # horizontal conductivity
        VK = UNIQUE["VK"][MODEL_NR - 1]  # vertical conductivity
    else:
        if ISOTROPIC is True:
            HK = VK = K  # in this case the layer is asumed isotropic
        else:  # currently no models are anisotropic
            print("oof need to assign HK and VK somehow for anisotropic model")

    # For this model we will set up a temporary workspace.
    # Model input files and output files will reside here.
    with TemporaryDirectory() as temp_dir:
        workspace = os.path.join(temp_dir, f"workspace_{NAME}")

        # %% create flopy model objects
        # Create the Flopy simulation object
        sim = flopy.mf6.MFSimulation(
            sim_name=NAME, exe_name="mf6", version="mf6", sim_ws=workspace
        )
        # , 3 is informative

        # Create the Flopy temporal discretization object
        tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
            sim,
            pname="tdis",
            time_units="DAYS",
            nper=1,  # one stress period in total
            perioddata=[
                (1.0, 1, 1.0)
            ],  # stress period of 1 day, in steps of 1 day
        )

        # Create the Flopy groundwater flow (gwf) model object
        model_nam_file = f"{NAME}.nam"
        gwf = flopy.mf6.ModflowGwf(
            sim, modelname=NAME, model_nam_file=model_nam_file
        )

        # Create the Flopy iterative model solver (ims) Package object, specifies solver
        # use MODERATE if SIMPLE does not result in convergence, similarly use COMPLEX
        # if MODERATE does not result in succesful convergence
        ims = flopy.mf6.modflow.mfims.ModflowIms(
            sim,
            pname="ims",
            complexity="SIMPLE",
            outer_dvclose=OUTER_DVCLOSE,
            outer_maximum=OUTER_MAXIMUM,
            inner_dvclose=INNER_DVCLOSE,
            inner_maximum=INNER_MAXIMUM,
        )

        # %% create packages
        # discretization package
        dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
            gwf,
            pname="dis",
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=ZTOP,
            botm=BOTM,
        )

        # initial conditions package
        start = HB * np.ones((NLAY, NROW, NCOL))
        ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(
            gwf, pname="ic", strt=start
        )

        # node property flow package
        npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
            gwf, pname="npf", icelltype=1, k=HK, k33=VK, save_flows=True
        )  # don't know whether icelltype=1 is correct.

        # constant head package
        # create constant heads at all boundaries of 10.0
        # must be entered as a tuple as the first entry.
        # required structure: ((layer, row, column), constant_head)
        chd_rec = (
            []
        )  # contains all tuples defining location and value of constant heads
        for i in range(NROW):  # 25 equals nrow and ncol (square)
            chd_rec.append(((0, 0, i), HB))
            chd_rec.append(((0, 24, i), HB))
            if i not in (0, NROW - 1):
                chd_rec.append(((0, i, 0), HB))
                chd_rec.append(((0, i, 24), HB))

        chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
            gwf,
            pname="chd",
            maxbound=len(chd_rec),
            stress_period_data=chd_rec,
            save_flows=True,
        )

        # Create the well package
        wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=WEL_SPECS)

        # output control package
        head_filerecord = [f"{NAME}.hds"]
        saverecord = [("HEAD", "LAST")]  # later only save specific locations?
        printrecord = [("HEAD", "LAST")]
        oc = flopy.mf6.ModflowGwfoc(
            gwf,
            saverecord=saverecord,
            head_filerecord=head_filerecord,
            printrecord=printrecord,
        )

        # %% run simulation
        sim.write_simulation(
            silent=SILENT
        )  # silent influences what is printed
        # try running simulation, if it fails
        try:
            success, buff = sim.run_simulation(silent=SILENT)
            assert success, pformat(buff)  # so I assert whether succesfull
        except AssertionError as e:  # excecuted if Not succesfull
            print(
                f"MODFLOW Simulation failed for MODEL_NR={MODEL_NR}, LOCATIONS={LOCATIONS}, K={K}"
            )
            print(f"Error: {e}")
            return None

        # output
        h = gwf.output.head().get_data(kstpkper=(0, 0))  # all heads

    if LOCATIONS is None:  # not specified, so return all possible locations
        return h
    else:  # obtain hydraulic head at all measured locations
        h_locations = np.empty(len(LOCATIONS))
        for index, location in enumerate(LOCATIONS):
            h_locations[index] = h[location]
        return h_locations
