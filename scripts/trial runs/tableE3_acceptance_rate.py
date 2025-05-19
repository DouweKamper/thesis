# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:59:52 2025

@author: Douwe

Calculates acceptance rates for a specific chain, for each sampler.
"""

import pandas as pd
from diagnostics_d import acceptance_fraction

# %% acceptance fraction
# Create empty DataFrame
columns = [
    "parameter 1",
    "parameter 2",
    "parameter 3",
    "parameter 4",
    "parameter 5",
]
index = ["Stretch", "DE", "DEsnooker"]
df = pd.DataFrame(index=index, columns=columns)

# calculate acceptance fraction and fill DataFrame
for move in ["Stretch", "DE", "DEsnooker"]:
    for parameter in [1, 2, 3, 4, 5]:
        df.at[move, f"parameter {parameter}"] = acceptance_fraction(
            move=move,
            MODEL_NR=4,
            ensemble=3,
            chain=2,
            parameter=parameter,
        )
        # print(parameter)
        # print(df)

# convert DataFrame to latex table
latex_table = df.to_latex(float_format="%.3f")
print(latex_table)
