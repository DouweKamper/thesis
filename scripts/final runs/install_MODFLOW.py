# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:29:31 2024

@author: Douwe

installs MODFLOW, including MODFLOW6
"""

import flopy

# install MODFLOW
flopy.utils.get_modflow(":flopy")
