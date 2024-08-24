# Grid-Point Neural Network (GPNN)

## Overview

- GPNN is a package for ML prediction of charge density based on the methods described in [Solving the electronic structure problem with machine learning](https://www.nature.com/articles/s41524-019-0162-7) by Ramprasad et. al.
- The [NMC](https://data.dtu.dk/articles/dataset/NMC_Li-ion_Battery_Cathode_Energies_and_Charge_Densities/16837721) cathode dataset is used for demonstration. A pretrained model for this dataset is included and can be used for inference as is. 
- The workflow is constructed as a (build -> train -> predict) pipeline.
    - **Build:** Construct an HDF5 dataset file given a directory of CHGCAR (VASP) files.
    - **Train:** Train an ML model on the preprocessed data (HDF5 file genereated from the BUILD step).
    - **Predict:** Use the trained ML model to run inference on new structures (CIF -> Predicted CHGCAR)
- Charge density can be predicted orders of magnitude faster (1 to 3 minutes in total) than DFT and for larger systems. 
- GPNN's core dependencies are:
    - [Hydra](https://hydra.cc/docs/intro/): Used for config management of hyperparameters during the build, train and predict stages (hydra-core==1.3.2).
    - [PyRho](https://materialsproject.github.io/pyrho/index.html): Used for charge density management and interpolation (mp-pyrho==0.3.0).
    - [HDF5](https://docs.h5py.org/en/stable/quick.html): Used for efficient data storage and organization (h5py==3.8.0).
    - [Dask](https://docs.dask.org/en/stable/10-minutes-to-dask.html): Used for distributed massive array manipulation (dask==2023.11.0).
    - [CuPy](https://cupy.dev/): Used in conjunction with Dask for GPU-accelerated array operations (cupy-cuda11x==13.0.0)


## Installation

1. Clone repo
3. Create new conda env
5. Run `pip install -e .`

## Prepare Configs

In order to simplify interdependent configurations, configs are organized into basic categories (data, model, train, inference) and composed by [Hydra](https://hydra.cc/docs/intro/) when scripts are run. Configs should be modified in place (not moved to other locations).

NOTE: Hydra includes a CLI that can be used to modify parameters on-the-fly instead of manually modifying the yaml configs. See "Run Experiments" for examples. 

1. Main Config:
    - Navigate to `GPNN/configs/config.yaml`
    - This is the high level config that orchestrates the lower level configs (data, inference and training).
    - If you will be training your own model, change "expname". Otherwise, leave it as is. This variable is used to select from pretrained models or name new models. 
    - Change "project_root" to the global path where you cloned the repo (path/to/GPNN). 
2. Inference:
    - Navigate to `GPNN/configs/inference/default.yaml`
    - Update the indicated variables
3. Training
    - Navigate to `GPNN/configs/train/default.yaml`
    - Update the indicated variables
4. Data.
    - If training on NMC, navigate to `GPNN/configs/data/NMC.yaml` and follow the instructions.
    - If defining a custom dataset, navigate to `GPNN/configs/data/template.yaml` and follow the instructions.

## Run Experiments
**Navigate to GPNN/scripts before running all scripts**

1. Inference with Pretrained Model (included model is trained on NMC)
    - If you completed step 2 under "Prepare Configs", simply run `python predict.py`
    - Otherwise, run `python predict.py inference.cif_path=<path/to/CIFs> inference.save_dir=<path/to/save/chgcars> inference.shape=<grid_shape> inference.device=<gpu|cpu>`

2. Train NMC from scratch
    - Navigate to `GPNN/configs/config.yaml`
    - Change "expname" to a name of your choice. This will become the name of your model.
    - Run `python train.py`
    - Your trained model will be stored in `GPNN/models/<expname>`

3. Train on custom dataset
    - Navigate to `GPNN/configs/data/template.yaml` and follow the instructions.
    - You will be loading your CHGCAR files as described, and defining the relevant config parameters. 
    - Navigate to `GPNN/configs/config.yaml`
    - Change "expname" to a name of your choice, that best describes the dataset you want to train on. This will become the name of your model. Note, new experiments run under this name, will use the last model checkpoint under ../models/expname,
    - Update line 10 of this file GPNN/configs/config.yaml to the name of your dataset.
    - Run `python build.py` to build the dataset file.
    - Run `python train.py` to train a model on this data.
    - Run `python predict.py` to use your model for inference on CIF files. 

## Technical Summary 
*Charge density* is represented as a set of scalar values defined on a regularly distributed discrete grid spanning the volume of a unit cell.

A *neural network* is trained to predict the *charge density* values based on a representation of the local atomic environment around each grid point. This representation (called the *fingerprint*) contains information on the location and types of atoms relative to each grid point. 

The fingerprint is constructed as a combination of rotationally invariant *scalar* and *vector* components. The *scalar* component captures radial distance information while the *vector* component captures angular information.

The authors propose a predefined set of Gaussian functions (k) of varying widths (σ<sub>k</sub>) centered about every grid-point (g) to determine these fingerprints. The *scalar* fingerprint (S<sub>k</sub>) for a particular grid-point, g, and Gaussian, k, in an N-atom, single-elemental system is defined as
$$S_k = C_k \sum_{i=1}^{N} \exp \left( \frac{-r_{gi}^2}{2\sigma_k^2} \right) f_c(r_{gi})$$

where *r<sub>gi</sub>* is the distance between the reference grid-point, *g*, and the atom, *i*, and *f<sub>c</sub>(r<sub>gi</sub>)* is a cutoff function, which decays to zero for atoms beyond the cutoff radius (default 6Å) from the grid-point. C<sub>k</sub> is a normalizing constant (see paper for definition).

The *vector* component is defined as 
$$V_k^\alpha = C_k \sum_{i=1}^{N} \frac{r_{gi}^\alpha}{2\sigma_k^2} \exp \left( \frac{-r_{gi}^2}{2\sigma_k^2} \right) f_c(r_{gi})$$
where where, α and β represent the x, y, or z directions. In order to maintain rotational invariance, this result is then composed like
$$V_k = \sqrt{(V_k^x)^2 + (V_k^y)^2 + (V_k^z)^2}$$

NOTE: The tensor component described in the paper is ommitted due to the marginal performance improvement at significantly increased computational cost. 
