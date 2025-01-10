import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.download_zenodo import input_yes_no, download_and_unpack_mpimnist

LEN = {'train' : 60000, 'test' : 10000}
XY_DIMS = {'coarse' : [15, 17], 'int' : [45, 51], 'fine' : [75, 85]}
NUM_CHANNELS = 3

class MPIMNISTDataset(Dataset):
    """
    Dataset class for the MPIMNIST dataset.

    This class handles the loading, preprocessing, and retrieval of samples from the MPIMNIST dataset.

    Attributes:
        extract_path (str): The path where the MPIMNIST dataset is extracted.
        dataset (str): The subset of the dataset to use ('train' or 'test').
        noise_dev (bool): Whether to add noise to the dataset.
        model_dev (str or None): The device to load the model on.
        freq_selection (list): The frequency selection range for the dataset.
        flattened (bool): Whether to flatten the dataset.
        A (tensor): The system matrix for data generation.
        A_rec (torch.Tensor): The system matrix for data reconstruction.
        path_A (str): The path to the system matrix file.
        sm_noise (torch.Tensor): System matrix noise.
        x (torch.Tensor): Ground truth phantoms in desired resolution.
        obs (torch.Tensor): Noise-free measurements.
        obs_noisy (torch.Tensor): Noise-perturbed measurements.

    Methods:
        check_for_mpimnist(): Checks if the MPIMNIST dataset exists at the specified path.
        load_system_matrix(): Loads the generation system matrix from the specified path.
        load_reco_system_matrix(sm_noise): Loads the system matrix used for reconstruction.
        load_noise(): Loads the noise component for the reconstruction system matrix.
        load_data(): Loads the ground truth phantoms and (noise-pertubed) MPI measurements. 
        __getitem__(idx): Retrieves the data sample at the specified index.
        __len__(): Returns the total number of samples in the dataset.
    """
    def __init__(self, datapath, dataset = 'train', noise_dev = False, model_dev = None, 
                    freq_selection = [0, 817], flattened = False, dim_phantom = 1, load_matrices = True):
        """
        Parameters:
            datapath (str): The path to the dataset.
            dataset (str, optional): The subset of the dataset to use ('train' or 'test'). Defaults to 'train'.
            noise_dev (bool, optional): Whether to add noise to the reconstruction system matrix. Defaults to True.
            model_dev (str or None, optional): The selected model deviation for reconstruction. Defaults to None
            freq_selection (list, optional): The frequency selection range for the dataset. Defaults to [0, 817].
            flattened (bool, optional): Whether to flatten with respect to the receive channels. Defaults to False.
            dim_phantom (int): Dimension of MNIST phantom. 
                         If dim_phantom = 1, x is flattened w.r.t. the last dimension to a vector. 
                         Shape: [LEN[dataset], total pixels] 
                         If dim_phantom = 2, last two dimensions correspond to 2D image. 
                         Shape: [LEN[dataset], pixels dimension 1, pixels dimension 2].
            load_matrices (bool): Whether to load the system and noise matrices. Defaults to True.
        """

        super().__init__()

        self.extract_path = datapath

        if not self.check_for_mpimnist():
            print(f'The MPI-MNIST dataset could not be found under the path {datapath}.')
            print('Do you want to download it now? (y: download, n: input other path)')
            download = input_yes_no()
            if download:
                download_and_unpack_mpimnist(self.extract_path)
            else:
                print('Path to MPIMNIST dataset:')
                DATA_PATH = input()
                self.extract_path = DATA_PATH

        self.dataset = dataset
        self.noise_dev = noise_dev
        self.model_dev = model_dev
        self.freq_selection = freq_selection
        self.flattened = flattened
        self.c0 = 10

        if load_matrices:
            self.path_A = os.path.join(self.extract_path, 'MPI-MNIST/SM/SM_fluid_opt_fine.mdf')

            if self.model_dev == 'mono small':
                self.path_A_rec = os.path.join(self.extract_path, 'MPI-MNIST/SM/SM_fluid_small_params_coarse.mdf')
            elif self.model_dev == 'mono large':
                self.path_A_rec = os.path.join(self.extract_path, 'MPI-MNIST/SM/SM_fluid_large_params_coarse.mdf')
            elif self.model_dev == 'equilibrium':
                self.path_A_rec = os.path.join(self.extract_path, 'MPI-MNIST/SM/SM_fluid_equilibrium_coarse.mdf')
            else:
                print('Physical models of generative and reconstruction system matrices are identical.')
                self.path_A_rec = os.path.join(self.extract_path, 'MPI-MNIST/SM/SM_fluid_opt_coarse.mdf')
        
            self.A = self.load_system_matrix()
            print("Loaded SM for data generation.")
            self.sm_noise = self.load_noise()
            print("Loaded Noise.")
            self.A_rec = self.load_reco_system_matrix(self.sm_noise)
            print("Loaded SM for data reconstruction.")

        self.x, self.obs, self.obs_noisy =  self.load_data()

        self.x = torch.reshape(self.x, 
                               (LEN[self.dataset], XY_DIMS['coarse'][0]*XY_DIMS['coarse'][1]))
        
        if dim_phantom == 2:
            self.x = torch.reshape(self.x,
                                   (LEN[self.dataset], XY_DIMS['coarse'][0],XY_DIMS['coarse'][1]))

    def check_for_mpimnist(self):
        """Fast check whether first and last file of each dataset part exist
        under the configured data path.

        Returns:
            exists (bool): Whether MPIMNIST seems to exist.
        """
        for part in ['train', 'test']:
            file = os.path.join(
                self.extract_path, 
                f'MPI-MNIST/{part}_obs/{part}_obs.mdf')
            if not os.path.exists(file):
                return False
        return True

    def load_system_matrix(self):
        """Loads system matrix from datapath given in the class constructor.

        Returns:
            torch.Tensor: generation system matrix of shape [NUM_CHANNELS, frequencies, pixels] 
                          or [NUM_CHANNELS*frequencies, pixels]
        """
        A_file = h5py.File(self.path_A)
        A = torch.from_numpy(np.array(A_file['/measurement/data'])).cfloat()
        A = A.squeeze(0)
        A = A[:, self.freq_selection[0]:self.freq_selection[1], :]
        if self.flattened:
            A = A.contiguous().view(-1, A.shape[-1])
        return A
    
    def load_reco_system_matrix(self, sm_noise):
        """Load the system matrix used for reconstruction purposes. Depending on the class initialization 
           it will be identical or different from the generation system matrix. In either case, the spatial
           dimension of generation and reconstruction matrix will vary.

        Args:
            sm_noise (torch.Tensor): Noise matrix. 
                               If noise_dev == True, noise will be added pixelwise to the system matrix.

        Returns:
            torch.Tensor: reconstruction system matrix of shape [NUM_CHANNELS, frequencies, pixels] 
                          or [NUM_CHANNELS*frequencies, pixels]
        """
        A_rec_file =  h5py.File(self.path_A_rec)
        A_rec = torch.from_numpy(np.array(A_rec_file['/measurement/data'])).cfloat()
        A_rec = A_rec[0, :, self.freq_selection[0]:self.freq_selection[1], :]
        if self.noise_dev:
            A_rec = A_rec + 1/self.c0 * torch.permute(sm_noise,(1, 2, 0)) 

        if self.flattened:
            A_rec = A_rec.contiguous().view(-1, A_rec.shape[-1])
        return A_rec
    
    def load_noise(self):
        """Loads noise component required for noise on reconstruction system matrix.

        Returns:
            torch.Tensor: pixelwise noise matrix of shape [pixels, NUM_CHANNELS, frequencies]
        """
        sm_noise_path = f'MPI-MNIST/{self.dataset}_noise/NoiseMeas_SM_{self.dataset}.mdf'
        with h5py.File(os.path.join(self.extract_path, sm_noise_path), 'r') as f_noise:
            sm_noise = torch.from_numpy(f_noise['/measurement/data'][:]).cfloat()
            sm_noise = sm_noise[:XY_DIMS['coarse'][0]*XY_DIMS['coarse'][1], 0, :, :]
            sm_noise = sm_noise[:,:,self.freq_selection[0]:self.freq_selection[1]]
        return sm_noise
    
    def load_data(self):
        with h5py.File(os.path.join(self.extract_path, f"MPI-MNIST/{self.dataset}_obs/{self.dataset}_obs.mdf"),"r") as f_obs: 
            obs_batch = torch.from_numpy(f_obs["/measurement/data"][()]).cfloat() # shape: [LEN[dataset], 1, NUM_CHANNELS, 817]
            obs = obs_batch.squeeze(1)
        with h5py.File(os.path.join(self.extract_path, f"MPI-MNIST/{self.dataset}_obsnoisy/{self.dataset}_obsnoisy.mdf"),"r") as f_obs_noisy: 
            obs_noisy_batch = torch.from_numpy(f_obs_noisy["/measurement/data"][()]).cfloat() # shape: [LEN[dataset], 1, NUM_CHANNELS, 817]
            obs_noisy = obs_noisy_batch.squeeze(1)
        with h5py.File(os.path.join(self.extract_path, f"MPI-MNIST/{self.dataset}_gt/{self.dataset}_gt.hdf5"),"r") as f_x: 
            x_batch = torch.from_numpy(f_x["phantom_data"][()]).cfloat() # shape: [LEN[dataset], 255]
            x = x_batch.squeeze(1)
        return x, obs, obs_noisy
    
    def __getitem__(self, idx):
        """
        Retrieves the data sample at the specified index.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing (x, obs, obs_noisy) where 'x' is the input data and 
                   'obs' and 'obs_noisy' are the noise free and noise contained measurements.
        """       
        x_idx = self.x[idx]
        obs_idx = self.obs[idx, :, self.freq_selection[0]:self.freq_selection[1]]
        obs_noisy_idx = self.obs_noisy[idx, :, self.freq_selection[0]:self.freq_selection[1]]

        if self.flattened:
            obs_idx = obs_idx.contiguous().view(-1)
            obs_noisy_idx = obs_noisy_idx.contiguous().view(-1)

        return x_idx, obs_idx, obs_noisy_idx
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return LEN[self.dataset]
    

class CustomMPIMNISTDataset(MPIMNISTDataset):
    """
    Dataset class for the Customized MPIMNIST dataset.

    This class handles the loading, preprocessing, and retrieval of samples from the MPIMNIST dataset. 
    In addition to the MPIMNISTDataset class, with this class is possible to select a customized resolution, 
    modeltype for data generation and concentration. 

    Attributes:
        exact_modeltype (str): Selects the data generation modeltype.
        resolution (str): Selects the resolution of the ground truth phantoms.
        concentration (str): Selects the pixelrange of the ground truth phantoms.
        phantom_noise (torch.Tensor): Measures Bruker noise samples used as additive measurement noise.
        sm_noise (torch.Tensor): System matrix noise.
        extract_path (str): The path where the MPIMNIST dataset is extracted.
        dataset (str): The subset of the dataset to use ('train' or 'test').
        noise_dev (bool): Whether to add noise to the dataset.
        model_dev (str or None): The device to load the model on.
        freq_selection (list): The frequency selection range for the dataset.
        flattened (bool): Whether to flatten the dataset.
        A (tensor): The system matrix for data generation.
        A_rec (torch.Tensor): The system matrix for data reconstruction.
        path_A (str): The path to the system matrix file.
        path_A_rec (str): The path to the reconstruction system matrix file.
        x (torch.Tensor): Ground truth phantoms in desired resolution.
        x_fine (torch.Tensor): Ground truth phantoms in fine resolution.
        obs (torch.Tensor): Noise-free measurements.
        obs_noisy (torch.Tensor): Noise-perturbed measurements.

    Methods:
        load_noise(): Loads the noise component for the reconstruction system matrix.
        compute_measurements(): Computes simulated MPI measurements.
        __getitem__(idx): Retrieves the data sample at the specified index.
    """
    def __init__(self, datapath, dataset = 'train', exact_modeltype = 'poly', resolution = 'coarse', 
                 concentration = 10, noise_dev = False, model_dev = None, freq_selection = [0, 817], 
                 flattened = False, dim_phantom = 1):
        """
        Parameters:
            datapath (str): The path to the dataset.
            dataset (str, optional): The subset of the dataset to use ('train' or 'test'). Defaults to 'train'.
            exact_modeltype (str, optional): The underlying physical model of the generation system matrix. 
                                             Options are ['poly', 'mono_small', 'mono_large', 'equilibrium'].
                                             Defaults to 'poly'.
            resolution (str, optional): Resolution of the reconstructed data. Options are ['coarse', 'int', 'fine']. 
                                        Defaults to 'coarse'.
            concentration (int, optional): Pixelrange of ground truth data. Defaults to 10.
            noise_dev (bool, optional): Whether to add noise to the reconstruction system matrix. Defaults to True.
            model_dev (str or None, optional): The selected model deviation for reconstruction. Defaults to None.
            freq_selection (list, optional): The frequency selection range for the dataset. Defaults to [0, 817].
            flattened (bool, optional): Whether to flatten with respect to the receive channels. Defaults to False.
            dim_phantom (int): Dimension of MNIST phantom. 
                         If dim_phantom = 1, x is flattened w.r.t. the last dimension to a vector. 
                         Shape: [LEN[dataset], total pixels] 
                         If dim_phantom = 2, last two dimensions correspond to 2D image. 
                         Shape: [LEN[dataset], pixels dimension 1, pixels dimension 2].
        """
        self.resolution = resolution
        self.exact_modeltype = exact_modeltype
        self.concentration = concentration

        if self.resolution not in ['coarse', 'int', 'fine']:
            raise ValueError('Invalid resolution. Please select between [coarse, int, fine].')

        super().__init__(datapath=datapath, dataset=dataset, noise_dev=noise_dev, model_dev=model_dev, 
                         freq_selection=freq_selection, flattened=False, load_matrices=False)

        # system matrix path
        if self.exact_modeltype == 'poly':
            self.path_A = os.path.join(self.extract_path, 'MPI-MNIST/SM/SM_fluid_opt_fine.mdf')
        elif self.exact_modeltype == 'mono_small':
            self.path_A = os.path.join(self.extract_path, 'MPI-MNIST/SM/SM_fluid_small_params_fine.mdf')
        elif self.exact_modeltype == 'mono_large': 
            self.path_A = os.path.join(self.extract_path, 'MPI-MNIST/SM/SM_fluid_large_params_fine.mdf')
        elif self.exact_modeltype == 'equilibrium':
            self.path_A = os.path.join(self.extract_path, 'MPI-MNIST/SM_fluid_equilibrium_fine.mdf')
        else:
            raise ValueError('Invalid modeltype. Please select between [poly, mono_small, mono_large, equilibrium].')
        
        # reconstruction system matrix path
        if self.model_dev == None or self.model_dev == self.exact_modeltype:
           print('Physical models of generative and reconstruction system matrices are identical.')
           self.path_A_rec = self.path_A.replace('_fine.mdf', f'_{self.resolution}.mdf')
        elif self.model_dev == 'mono_small':
            self.path_A_rec = os.path.join(self.extract_path, f'MPI-MNIST/SM/SM_fluid_small_params_{self.resolution}.mdf')
        elif self.model_dev == 'mono_large':
            self.path_A_rec = os.path.join(self.extract_path, f'MPI-MNIST/SM/SM_fluid_large_params_{self.resolution}.mdf')
        elif self.model_dev == 'equiliibrium':
            self.path_A_rec = os.path.join(self.extract_path, f'MPI-MNIST/SM/SM_fluid_equilibrium_{self.resolution}.mdf')
        else:
            raise ValueError('Invalid model deviation. Please select between [None, poly, mono_small, mono_large, equilibrium].')
        
        self.A = self.load_system_matrix()
        print("Loaded SM for data generation.")
        self.sm_noise, self.phantom_noise = self.load_noise()
        print("Loaded Noise.")
        self.A_rec = self.load_reco_system_matrix(self.sm_noise)
        print("Loaded SM for data reconstruction.")
        
        # upsample if spatial dimension is not coarse
        x_reshaped = torch.real(torch.reshape(self.x, (LEN[self.dataset], 1, XY_DIMS['coarse'][0]*XY_DIMS['coarse'][1])))
        x_reshaped = torch.reshape(x_reshaped, (LEN[self.dataset], 1, XY_DIMS['coarse'][0], XY_DIMS['coarse'][1]))
        self.x_fine = torch.nn.functional.interpolate(input=x_reshaped, scale_factor=(5, 5), mode='nearest').squeeze(1)
        self.x_fine = torch.flatten(self.x_fine, start_dim=-2).cfloat()
        if self.resolution == 'int':
            x_reshaped = torch.nn.functional.interpolate(input=x_reshaped, scale_factor=(3, 3), mode='nearest').squeeze(1)
            self.x = torch.flatten(x_reshaped, start_dim=-2).cfloat()
        elif self.resolution == 'fine':
            self.x = self.x_fine
        
        # verify whether it is required to recompute (noisy) observations
        if self.concentration == 10 or self.exact_modeltype == 'poly':
            change_in_observations = True
        else:
            change_in_observations = False
        
        if change_in_observations:
            # rescale for varying concentration
            if self.concentration != 10:
                x_real = torch.real(self.x)
                self.x = x_real*self.concentration/10
                self.x = self.x.cfloat()
                x_fine_real = torch.real(self.x_fine)
                self.x_fine = x_fine_real*self.concentration/10
                self.x_fine = self.x_fine.cfloat()
            self.obs, self.obs_noisy = self.compute_measurements()

        if dim_phantom == 2:
            self.x = torch.reshape(self.x,
                    (LEN[self.dataset], XY_DIMS[f'{self.resolution}'][0], XY_DIMS[f'{self.resolution}'][1]))  
            
        if flattened: 
            self.phantom_noise = self.phantom_noise.contiguous().view(self.phantom_noise.shape[0], -1)
            self.sm_noise = self.sm_noise.contiguous().view(self.sm_noise.shape[0], -1)
            self.obs = self.obs.contiguous().view(self.obs.shape[0], -1)
            self.obs_noisy = self.obs_noisy.contiguous().view(self.obs_noisy.shape[0], -1)
            self.A = self.A.contiguous().view(-1, self.A.shape[-1])
            self.A_rec = self.A_rec.contiguous().view(-1, self.A_rec.shape[-1])
    
    def load_noise(self):
        """Loads phantom noise required for additive noise on simulated measurements.

        Returns:
            torch.Tensor: pixelwise noise matrix of shape [LEN[dataset], NUM_CHANNELS, frequencies]
        """
        sm_noise_path = f'MPI-MNIST/{self.dataset}_noise/NoiseMeas_SM_{self.dataset}.mdf'
        with h5py.File(os.path.join(self.extract_path, sm_noise_path), 'r') as f_noise:
            sm_noise = torch.from_numpy(f_noise['/measurement/data'][:]).cfloat()
            if self.resolution == 'fine' and self.noise_dev == True:
                raise NotImplementedError('The combination of noise_dev == True together with resolution == fine is not yet implemented.')
            else:
                sm_noise = sm_noise[:XY_DIMS[f'{self.resolution}'][0]*XY_DIMS[f'{self.resolution}'][1], 0, :, :]
                sm_noise = sm_noise[:,:,self.freq_selection[0]:self.freq_selection[1]]
                
        phantom_noise_path = f'MPI-MNIST/{self.dataset}_noise/NoiseMeas_phantom_{self.dataset}.mdf'
        with h5py.File(os.path.join(self.extract_path, phantom_noise_path), 'r') as f_noise:
            phantom_noise = torch.from_numpy(f_noise['/measurement/data'][:]).cfloat()
            phantom_noise = phantom_noise[:, 0, :,self.freq_selection[0]:self.freq_selection[1]]
        
        return sm_noise, phantom_noise

    def compute_measurements(self):
        """Computes measurements of phantoms with the exact operator A.

        Returns:
            y (torch.Tensor); shape: [LEN[dataset], receive coils, frequency]: measurement vectors
            y_noisy (torch.Tensor); shape: [LEN[dataset], receive coils, frequency]: noise-perturbed measurement vectors
        """
        x_fine_channelwise = self.x_fine.unsqueeze(1).expand(self.x_fine.shape[0], NUM_CHANNELS, self.x_fine.shape[1])
        y = torch.matmul(self.A, torch.permute(x_fine_channelwise, (1, 2, 0))) #batchwise matrix-matrix multiplication
        y = torch.permute(y, (2, 0, 1))
        y_noisy = y + self.phantom_noise
        return y, y_noisy
    
    def __getitem__(self, idx):
        """
        Retrieves the data sample at the specified index.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing (x, obs, obs_noisy) where 'x' is the input data and 
                   'obs' and 'obs_noisy' are the noise free and noise contained measurements.
        """       
        x_idx = self.x[idx]
        obs_idx = self.obs[idx]
        obs_noisy_idx = self.obs_noisy[idx]

        return x_idx, obs_idx, obs_noisy_idx