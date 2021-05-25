## Sea level trends across the Bahamas constrain peak Last Interglacial ice melt
Repository for Dyer et al. 2021 PNAS

### Associated Web Application

[Follow this link](http://bahamas-lig.herokuapp.com/) to explore field photos and outcrop locations associated with this project.

### Installing GIA model outputs and inversion results

There are four archives that should be downloaded from the most recent release of this repository:
 - The gridded GIA model outputs: **get_GIA.zip**
     - extract archive contents into `/model_outputs/`
  - Posterior traces for the LIG data: **lig_inversion.zip**
     - extract archive contents into `/model_outputs/`
 - Posterior traces for the synthetic data: **synthetic_test.zip**
     - extract archive contents into `/model_outputs/`
 - Posterior traces for the holocene data: **holocene.zip**
     - extract archive contents into `/model_outputs/`

### Tensorflow

To run the CNN code using tensorflow follow the steps here, paying special attention to the steps to install CUDA for your machine if you want to use a CUDE enabled GPU for the calculations.
<https://www.tensorflow.org/install>

### Create virtual environment for dependencies

Use the .yml files with anaconda to create an environment to run the contained code:

`conda env create -f environment.yml`

(or alternatively with pip)

`pip install -r requirements.txt`


