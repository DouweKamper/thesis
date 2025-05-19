# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:20:53 2024

@author: Douwe

all universal and unique constants for the MODFLOW6 models I designed 
"""
import numpy as np

UNIVERSAL = {
    "LX": 1000.0,  # length in x direction
    "LY": 1000.0,  # length in y direction
    "ZTOP": 0.0,  # highest point in z direction
    "ZBOT": -50.0,  # lowest point in z direction
    "NROW": 25,  # number of rows (relevant for discretization in x direction)
    "NCOL": 25,  # number of rows (relevant for discretization in x direction)
    "DELR": 40.0,  # LX / NCOL, length of a single cell in x direction
    "DELC": 40.0,  # LX / NROW, length of a single cell in y direction
    "HB": 10.0,  # head at boundary (Dirichlet boundary condition)
    "WEL_SPECS": [
        (0, int(25 / 2), int(25 / 2), -500.0)
    ],  # well loc and pump rate
}

UNIQUE = {
    "NAME": ("model_1", "model_2", "model_3", "model_4"),
    "MODEL_NR": (1, 2, 3, 4),
    "NDIM": (1, 2, 3, 5),  # dimensionality of the problem
    "NLAY": (1, 2, 3, 5),  # number of layers in the vertical direction
    "BOTM": (  # vertical discretication of layers
        np.array([-50]),
        np.array([-25, -50]),
        np.array([-24, -26, -50]),
        np.array([-15, -22, -30, -35, -50]),
    ),
    "ISOTROPIC": (True, True, True, True),
    "HK": (  # horizontal conductivity
        [5.0],
        [2.0, 1.0],
        [1.0, 0.01, 10.0],
        [1.0, 0.1, 4.0, 0.01, 3.0],
    ),
    "VK": (  # vertical conductivity
        [5.0],
        [2.0, 1.0],
        [1.0, 0.01, 10.0],
        [1.0, 0.1, 4.0, 0.01, 3.0],
    ),
    "LITHOLOGY": (  # type of lithology present in layer
        ["clean sand"],
        ["clean sand", "silty sand"],
        ["silty sand", "silt, loess", "clean sand"],
        [
            "silty sand",
            "silt, loess",
            "clean sand",
            "silt, loess",
            "clean sand",
        ],
    ),
}
