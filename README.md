# Observed eddy-internal wave interactions in the Southern Ocean

[![DOI](https://zenodo.org/badge/275956774.svg)](https://zenodo.org/badge/latestdoi/275956774)

Jesse M Cusack, J. James Alexander Brearley, Alberto C. Naveira Garabato, David A. Smeed, Kurt L. Polzin, Nick Velzeboer, Callum J. Shakespeare

This repository contains the code for reproducing the analysis and figures in the publication.

Please follow the steps below to be able to run the code.

## Installation

First install miniconda (https://docs.conda.io/en/latest/miniconda.html), a python package manager.

Download or clone this git repository using the terminal command:

`git clone https://github.com/jessecusack/DIMES_eddy_wave_interactions.git`

which will crease a folder called `DIMES_eddy_wave_interactions`.

Then create a conda environment with the packages necessary to run the code using the provided environment file. Using the terminal, run:

`cd DIMES_eddy_wave_interactions`

to step into the directory and then

`conda env create -f environment.yml`

to create the python environment.

## Downloading additional data

For the code to run properly, all data must be downloaded and extracted into the `data` directory. Most of it is already there. See the `README.md` file inside the data directory for more information. The one additional file required is `topo_19.1.img`, which is a global bathymetry dataset that can be downloaded from https://topex.ucsd.edu/marine_topo/.

This repository contains a conveniently packaged subset of the DIMES data used in the publication. For the original data, please visit: https://www.bodc.ac.uk/projects/data_management/international/dimes/

## Running code

The python code is contained in `code` directory.

For the scripts to run correctly you must activate the conda environment created in the installation step.

`conda activate dimes-eiw`

Step into the code directory,

`cd code`

and run the data post-processing script (this takes a minute or so on my laptop):

`python post_process_data.py`

Most of the computationally intensive analysis is done in the post-processing script which will save some intermediate output to the `data` directory.

The remaining jupyter notebook files are for generating figures and additional smaller bits of analysis. Each notebook is paired with a .py file of the same name using jupytext (https://github.com/mwouts/jupytext). It is possible to run the .py files from the command line, e.g.

`python <INSERT_SCRIPT_NAME>.py`

Doing so will save figures.