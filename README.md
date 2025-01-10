# MPI-MNIST

This repository provides download code for the MPI-MNIST dataset from Zenodo. Furthermore, it contains scripts for converting it into a [PyTorch](https://pytorch.org/) dataset and preparing for usage with PyTorch-based modules such as [PyTorch Lightning](https://lightning.ai/).

## Dataset

MPI-MNIST is a dataset for magnetic particle imaging with MNIST phantoms. It contains simulated MPI measurements along with ground truth phantoms selected from the MNIST database of handwritten digits. To achieve realistic simulations, noise samples measured from an MPI scanner are incorporated into the simulated measurements.

The dataset is provided on [Zenodo](https://doi.org/10.5281/zenodo.12799417) with a short description. 


## Repository Description 
```
MPI-MNIST/
│
├── utils/
│ └── download_zenodo.py
│
├── data/
│ ├── datamodules.py
│ └── datasets.py
│
├── demo_customdata.ipynb
├── demo_data.ipynb
├── README.md
└── requirements.txt

```

- ```utils/download_zenodo.py``` provides auxilliary functions such as ```download_and_unpack_mpimnist(extract_folder)``` in order to download the MPI-MNIST zenodo dataset from its specified Zenodo ID to a given directory ```extract_folder```.
- ```data/datasets.py``` contains the ```MPIMNISTDataset``` and ```CustomMPIMNISTDataset``` classes. These download the data from the Zenodo website and subsequently integrate it as a PyTorch dataset. The ```CustomMPIMNISTDataset``` comes with a customized setting, where the MPIMNISTDataset adapts to the settings of the Zenodo version. 
- ```data/datamodules.py``` contains the classes ```MPIMNISTDataModule``` and ```CustomMPIMNISTDataModule```, which integrate the two datasets as a PyTorch Lightning data module. 
- ```demo_data.ipynb``` contains a demonstration of the usage of the data module ```MPIMNISTDatamodule```. 
- ```demo_customdata.ipynb``` contains a demonstration of the usage of the data module ```CustomMPIMNISTDatamodule``` for custom features. 

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch
- PyTorch Lightning (optional, for using DataModules)
- Requests library (for downloading the dataset)

### Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/meiraiske/MPI-MNIST.git
cd MPI-MNIST
pip install -r requirements.txt
```
### Usage 
#### Download Only

To download the data from the website to your local file directory ```/your/directory``` run 

```
from utils.download_zenodo import download_and_unpack_mpimnist

directory = "/your/directory"
download_and_unpack_mpimnist(directory)
```

#### Usage as PyTorch Lightning Data Module

A code demonstration of this case is given in ```demo_data.ipynb```. The ```MPIMNISTDataModule``` class provides a convenient way to handle the MPI-MNIST dataset in a PyTorch Lightning project. Below, we describe its usage, parameters, and key attributes.

##### Overview

The ```MPIMNISTDataModule``` organizes the dataset into batches of the form ```[x, obs, obs_noisy]```, where:

  - ```x``` is the preprocessed ground truth MNIST phantom.
  - ```obs``` is the simulated noise-free MPI measurement.
  - ```obs_noisy``` is the noise-perturbed simulated MPI measurement.

##### Key Attributes

   - ```A```: The system matrix used for data generation, featuring high spatial resolution.
   - ```A_rec```: The system matrix used for data reconstruction, featuring lower spatial resolution.

##### Parameters
- ```batch_size```: Specifies the size of the batches for the DataLoader.
- ```num_workers```: Determines the number of subprocesses used for data loading.
- ```datapath```: Path to the dataset.
- ```noise_dev```: Whether to add pixel-wise noise to the reconstruction system matrix (```True``` or ```False```).
- ```model_dev```: Specifies the physical model deviation for reconstruction. It can be ```None```, ```'mono_small'```, ```'mono_large'```, or ```'equilibrium'```.
- ```freq_selection```: A list defining the frequency selection range for the dataset.
- ```flattened```: If ```True```, the Tensors are concatenated with respect to the receive channel dimension.

All parameters allow for flexibility in the reconstruction scenario. The parameters ```model_dev``` and ```noise_dev``` are responsible for determining the reconstruction matrix ```A_rec```, determined by the underlying physical model as well as potential pixelwise noise perturbations on the matrix. This enables a setting, in which reconstruction from inexact forward operators can be performed. 

The ```CustomMPIMNISTDataModule``` extends the ```MPIMNISTDataModule``` by loading the ```CustomMPIMNISTDataset```. This module enables customized settings such as resolution, modeltype for data generation and concentration of the phantoms. These might differ from the settings provided in the Zenodo MPI-MNIST version. 

##### Additional Parameters/Attributes of ```CustomMPIMNISTDataModule```

- ```exact_modeltype```: Selects the data generation modeltype.
- ```concentration```: Selects the pixelrange of the ground truth phantoms.
- ```resolution```: Selects the resolution of the ground truth phantoms.

## License
This project is licensed under the MIT License.

## References
- Zenodo dataset: [https://doi.org/10.5281/zenodo.12799417](https://doi.org/10.5281/zenodo.12799417)
