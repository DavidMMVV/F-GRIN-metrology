# F-GRIN-metrology
This repo is created for developing tools for determining the index of refraction distribution of F-GRIN materials.

## Getting started

This is a python based project which uses conda, what means that the guest system must have [anaconda](https://www.anaconda.com/download) installed with python in a version greater or equal to 3.13.5.

To download this project usethe following command:
- for http
```bash
git clone https://github.com/DavidMMVV/F-GRIN-metrology.git
```
- for ssh

```bash
git clone git@github.com:DavidMMVV/F-GRIN-metrology.git
```

To create the conda environment, in anaconda promp run:

```bash
conda env create -f .\wenvironment.yml # For Windows
conda env create -f .\lenvironment.yml # For Linux
```

and activate it with:

```bash
conda activate f-grin-metrology 
```

In case you would like to run any of the scripts of the project, run the following command to install the module locally:

```bash
pip install -e .
```
