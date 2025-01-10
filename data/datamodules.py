import torch
import pytorch_lightning as pl

from data.datasets import MPIMNISTDataset, CustomMPIMNISTDataset

class MPIMNISTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the MPIMNIST dataset.

    This DataModule handles the setup and loading of training and testing datasets for the MPIMNIST dataset.
    
    Attributes:
        batch_size (int): The size of the batches for the DataLoader.
        num_workers (int): The number of subprocesses to use for data loading.
        train_set (MPIMNISTDataset): The training dataset.
        test_set (MPIMNISTDataset): The testing dataset.
        A (torch.Tensor): System matrix used for data generation.
        A_rec (torch.Tensor): System matrix used for data reconstruction.

    Methods:
        train_dataloader(): Returns the DataLoader for the training dataset.
        test_dataloader(): Returns the DataLoader for the testing dataset.
    """
    def __init__(self, batch_size, num_workers, datapath, noise_dev = False, model_dev = None,
                 freq_selection = [0, 817], flattened = False, dim_phantom = 1):
        """
        Parameters:
            batch_size (int): The size of the batches for the DataLoader.
            num_workers (int): The number of subprocesses to use for data loading.
            datapath (str): The path to the dataset.
            noise_dev (bool, optional): Whether to add noise to the reconstruction system matrix. Defaults to True.
            model_dev (str or None, optional): The selected model deviation for reconstruction. 
                                               Options are [None, 'poly', 'mono_small', 'mono_large', 'equilibrium']. 
                                               Defaults to None.
            freq_selection (list, optional): The frequency selection range for the dataset. Defaults to [0, 817].
            flattened (bool, optional): Whether to flatten over the receive channels. Defaults to False.
            dim_phantom (int, optional): Dimension of MNIST phantom. 
                                   If dim_phantom = 1, x is flattened w.r.t. the last dimension to a vector. 
                                   Shape: [LEN[dataset], NUM_CHANNELS, total pixels] 
                                   If dim_phantom = 2, last two dimensions correspond to a 2D image. 
                                   Shape: [LEN[dataset], NUM_CHANNELS, pixels dimension 1, pixels dimension 2].
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datapath = datapath

        print('Train set...')
        self.train_set  = MPIMNISTDataset(datapath=datapath,
                                          dataset='train',
                                          noise_dev=noise_dev,
                                          model_dev=model_dev,
                                          freq_selection=freq_selection,
                                          flattened=flattened,
                                          dim_phantom=dim_phantom
                                          )
        print('Test set...')
        self.test_set  = MPIMNISTDataset(datapath=datapath,
                                         dataset='test',
                                         noise_dev=noise_dev,
                                         model_dev=model_dev,
                                         freq_selection=freq_selection,
                                         flattened = flattened,
                                         dim_phantom=dim_phantom
                                         )
        self.A = self.train_set.A
        self.A_rec = self.train_set.A_rec

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size = self.batch_size, shuffle = True,
                                            num_workers = self.num_workers, drop_last = True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size = self.batch_size, shuffle = False,
                                             num_workers = self.num_workers, drop_last = True)
    
class CustomMPIMNISTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the CustomMPIMNIST dataset.

    This DataModule handles the setup and loading of training and testing datasets for the CustomMPIMNIST dataset.
    In addition to the MPIMNISTDataModule class, with this class is possible to select a customized resolution, 
    modeltype for data generation and concentration.

    Attributes:
        batch_size (int): The size of the batches for the DataLoader.
        num_workers (int): The number of subprocesses to use for data loading.
        datapath (str): Path to data location
        exact_modeltype (str): Selects the data generation modeltype.
        concentration (str): Selects the pixelrange of the ground truth phantoms.
        resolution (str): Selects the resolution of the ground truth phantoms.
        train_set (MPIMNISTDataset): The training dataset.
        test_set (MPIMNISTDataset): The testing dataset.
        A (torch.Tensor): System matrix used for data generation.
        A_rec (torch.Tensor): System matrix used for data reconstruction.

    Methods:
        train_dataloader(): Returns the DataLoader for the training dataset.
        test_dataloader(): Returns the DataLoader for the testing dataset.
    """
    def __init__(self, batch_size, num_workers, datapath, exact_modeltype = 'poly', 
                 resolution = 'coarse', concentration = 10, noise_dev = False, model_dev = None,
                 freq_selection = [0, 817], flattened = False, dim_phantom = 1):
        """
        Parameters:
            batch_size (int): The size of the batches for the DataLoader.
            num_workers (int): The number of subprocesses to use for data loading.
            datapath (str): The path to the dataset.
            dataset (str, optional): The subset of the dataset to use ('train' or 'test'). Defaults to 'train'.
            exact_modeltype (str, optional): The underlying physical model of the generation system matrix. 
                                             Options are ['poly', 'mono_small', 'mono_large', 'equilibrium'].
                                             Defaults to 'poly'.
            resolution (str, optional): Resolution of the reconstructed data. 
                                        Options are ['coarse' (15x17), 'int' (45x51), 'fine' (75,85)]. 
                                        Defaults to 'coarse'.
            concentration (int, optional): Pixelrange of ground truth data. Defaults to 10.
            noise_dev (bool, optional): Whether to add noise to the reconstruction system matrix. Defaults to True.
            model_dev (str or None, optional): The selected model deviation for reconstruction. 
                                               Options are [None, 'poly', 'mono_small', 'mono_large', 'equilibrium']. 
                                               Defaults to None.
            freq_selection (list, optional): The frequency selection range for the dataset. Defaults to [0, 817].
            flattened (bool, optional): Whether to flatten over the receive channels. Defaults to False.
            dim_phantom (int, optional): Dimension of MNIST phantom. 
                                   If dim_phantom = 1, x is flattened w.r.t. the last dimension to a vector. 
                                   Shape: [LEN[dataset], NUM_CHANNELS, total pixels] 
                                   If dim_phantom = 2, last two dimensions correspond to a 2D image. 
                                   Shape: [LEN[dataset], NUM_CHANNELS, pixels dimension 1, pixels dimension 2].
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.exact_modeltype = exact_modeltype
        self.resolution = resolution
        self.concentration = concentration
        self.datapath = datapath

        print('Train set...')
        self.train_set  = CustomMPIMNISTDataset(datapath=datapath,
                                                dataset='train',
                                                exact_modeltype=exact_modeltype, 
                                                resolution=resolution,
                                                concentration=concentration,
                                                noise_dev=noise_dev,
                                                model_dev=model_dev,
                                                freq_selection=freq_selection,
                                                flattened=flattened,
                                                dim_phantom=dim_phantom
                                                )
        print('Test set...')
        self.test_set  = CustomMPIMNISTDataset(datapath=datapath,
                                               dataset='test',
                                               exact_modeltype=exact_modeltype, 
                                               resolution=resolution,
                                               concentration=concentration,
                                               noise_dev=noise_dev,
                                               model_dev=model_dev,
                                               freq_selection=freq_selection,
                                               flattened = flattened,
                                               dim_phantom=dim_phantom
                                               )
        self.A = self.train_set.A
        self.A_rec = self.train_set.A_rec

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size = self.batch_size, shuffle = True,
                                            num_workers = self.num_workers, drop_last = True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size = self.batch_size, shuffle = False,
                                             num_workers = self.num_workers, drop_last = True)