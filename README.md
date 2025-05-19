# MSc thesis 
This repository contains the data and Python scripts used for my MSc thesis: A comparison of Markov Chain Monte Carlo algorithms for parameter calibration in hydrology. Additionally, the 'thesis' folder contains the required (LaTeX) files to compile my thesis report as a pdf.  

## Data
The 'data' folder is organized as follows:
- 'final runs' contains the data used for the main analysis
- 'trial runs' contains the data used for testing runs, which are discussed in Appendix E of the report

## Scripts
The 'scripts' folder is organized similarly to the 'data' folder'.
- 'final runs' contains the scripts used to generate the synthetic data and figures for the main analysis 
- 'trial runs' contains the scripts used to generate the synthetic data and figures for the testing runs


### Prerequisites
- Python 3.8 or higher
- Python packages listed in the relevant requirements.txt file. The folders 'final runs' and 'trial runs' both contain a unique requirements file (during the thesis a separate virtual environment was created for both). 

### Running scripts
If you would like to run the scripts to generate the synthetic data. Then it is important to first install MODFLOW 6. Run `install_MODFLOW.py` from either `final runs` or `trial runs`.

If you would like to create tables or figures, then you don't need to install MODFLOW 6. However, it is important to make sure the required data files are in the same folder as the respective python script. 

File names are quite informative, so it should (hopefully) be self explanatory for what they are used.




 



