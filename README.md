## Sea level trends across the Bahamas constrain peak Last Interglacial ice melt
Repository for Dyer et al. 2021 PNAS

The seas are rising as the planet warms, and reconstructions of past sea level provide critical insight into the sensitivity of ice sheets to warmer temperatures. Past sea level is reconstructed from the geologic record by measuring the elevations of fossilized marine sediments and coral reefs. However, the elevations of these features also record local uplift or subsidence due to the growth and decay of ice sheets since the time of deposition. We compare new observations of paleo sea level across the Bahamian archipelago to a range of Earth deformation models to revise estimates of last interglacial global mean sea level. Our results suggest that polar ice sheets may be less sensitive to high latitude warming than previously thought.

### Associated Web Application

[Follow this link](http://bahamas-lig.herokuapp.com/) to explore field photos and outcrop locations associated with this project.

### How to use this repository

`bahamas_lig/utils.py` contains all of the helper functions defined for the analysis in this paper. These functions are used throughout the included notebooks. The PyMC3 GP Regression model is created in the `inference_model()` function. In the notebooks folder, there are three GP Regression notebooks that illustrate how to use these functions to estimate past sea level.

### Installing GIA model outputs and inversion results

There are four archives that should be downloaded from the most recent [release](https://github.com/blakedyer/bahamas_lig/releases) of this repository:
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


