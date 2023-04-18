## Constraining the contribution of the Antarctic Ice Sheet to Last Interglacial sea-level

Repository for **Barnett et al. 2023**, _Science Advances_

Polar temperatures during the Last Interglacial (LIG; ~129-116 ka) were warmer than today, making this time period an important testing ground to better understand how ice sheets respond to warming. Yet it remains debated how much and when the Antarctic and Greenland ice sheets changed during this period. Here we present a combination of new and existing absolutely dated LIG sea-level observations from southwest Britain, northern France, and Denmark. Due to glacial isostatic adjustment (GIA), the LIG Greenland ice melt contribution to sea-level change in this region is small, which allows us to constrain Antarctic ice melt. Combining data and GIA modelling, we find that the Antarctic contribution to LIG global mean sea level peaked early in the interglacial (prior to 125 ka), with a maximum contribution of 5.6 m (50th percentile, 3.3–8.8 m central 68% probability) before declining. Our results support an asynchronous melt history over the LIG, with an early Antarctic contribution followed by later Greenland ice-sheet mass loss.

### How to use this repository

This branch adds or modifies a few methods from the main branch of this repository to analyze last interglacial sea level records from Europe. Run inversions from the [GP Regression EU LIG](https://github.com/blakedyer/bahamas_lig/blob/europe_lig/notebooks/GP%20Regression%20EU%20LIG.ipynb) notebook.

### Installing GIA model outputs and inversion results

There are two archives that should be downloaded from the most recent [release](https://github.com/blakedyer/bahamas_lig/releases) of this repository (on the `europe_lig` branch):

- The gridded GIA model outputs: **GIA_models_EU.zip**
  - extract archive contents into `/model_outputs/`
- Posterior traces for the LIG data: **EU_lig_inversion.zip**
  - extract archive contents into `/model_outputs/`

### Create virtual environment for dependencies

Use the .yml files with anaconda to create an environment to run the contained code:

`conda env create -f environment.yml`

(or alternatively with pip)

`pip install -r requirements.txt`
