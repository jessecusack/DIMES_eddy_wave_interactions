# Observed eddy internal wave interactions in the Southern Ocean

Jesse M Cusack, J. James Alexander Brearley, Alberto C. Naveira Garabato, David A. Smeed, Kurt L. Polzin, Nick Velzeboer, Callum J. Shakespeare

This repository contains the code for reproducing the analysis and figures in the publication.

Please follow the steps below to be able to run the code.

## Installation

First install miniconda (https://docs.conda.io/en/latest/miniconda.html), a python package manager.

Download or clone this git repository using the terminal command:

`git clone https://`

which will crease a folder called `DIMES_eddy_wave_interactions`.

Then create a conda environment with the packages necessary to run the code using the provided environment file. Using the terminal, run:

`cd DIMES_eddy_wave_interactions`

to step into the directory and then

`conda env create -f environment.yml`

to create the python environment.

## Downloading data

For the code to run properly, all data must be downloaded and extracted into the `data` directory.

The oceanographic observations data are stored in a convenient format on Zenodo (link) and should contain the following files:
* topo (high resolution local bathymetry)
* moorings (mooring data)
* adcp (ADCP data from C mooring)
* VMP (vertical microstructure profiler data)
* Model data

The original data are stored here: https://cchdo.ucsd.edu/cruise/740H20101130

All DIMES data is available here: http://dimes.ucsd.edu/en/fieldwork-and-cruise-reports/index.html


Other data that must be downloaded include:
* AVISO (satellite altimetry dataset)
* Global bathymetry dataset (Smith and Sandwell)


## Running code

The python code is contained in `code` directory.

For the scripts to run correctly you must activate the conda environment created in the installation step.

`conda activate dimes-eiw`

First run the data post-processing script (this takes a minute or so on my laptop):

`cd code`

`python code/post_process_data.py`

Most of the computationally intensive analysis is done in the post-processing script which will save more intermediate output to the `data` directory.

The remaining jupyter notebook files are for generating figures and additional smaller bits of analysis. Each notebook is paired with a .py file of the same name using jupytext (https://github.com/mwouts/jupytext). It is possible to run the .py files from the command line, e.g.

`python code/<INSERT_SCRIPT_NAME>.py`

Doing so will save figures.