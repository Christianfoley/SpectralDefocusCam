import numpy as np
import torch
import os
import glob
import scipy.io as io
from datetime import datetime
import tqdm

from defocuscam.utils.diffuser_utils import *
import defocuscam.utils.psf_utils as psf_utils


class ForwardModel(torch.nn.Module):
    def __init__(
        self,
        mask,
        params,
        psf_dir=None,
        passthrough=False,
        device=torch.device("cpu"),
    ):
        """
        Initializes Forward_Model() object, used for forward physics simulation of
        imaging system.

        For details pertaining to the physics model, see precompute_data_lri.ipynb.
        For base (default) parameters see configs/base_config.py

        Parameters
        ----------
        mask : torch.Tensor
            spectral response (transmittance) matrix for the spectral filter array
        params : dict
            dictionary of params, must contain {'stack_depth', 'operations', 'psf'}
        psf_dir : str
            path to directory containing LRI or LSI psf data, by default None
        passthrough : bool
            option to make model "passthrough", by default False
        device : torch.Device
            device to perform operations & hold tensors on, by default "cpu"
        """
        super(ForwardModel, self).__init__()
        self.device = device
        self.passthrough = passthrough
        self.psf_dir, self.psfs = psf_dir, None
        self.num_ims = params["stack_depth"]

        # Which forward operations to perform
        self.operations = params["operations"]

        # what psfs to use/simulate
        self.psf = params["psf"]

        ## Initialize constants
        self.DIMS0 = mask.shape[0]  # Image Dimensions
        self.DIMS1 = mask.shape[1]  # Image Dimensions
        self.PAD_SIZE0 = int((self.DIMS0))  # Pad size
        self.PAD_SIZE1 = int((self.DIMS1))  # Pad size

        # Initialize simulated psfs
        self.w_init = self.psf.get("w_init", None)
        if self.w_init is None:
            self.w_init = np.linspace(0.002, 0.035, self.num_ims)
        if not self.psf["symmetric"]:
            self.w_init = np.linspace(0.002, 0.01, self.num_ims)
            self.w_init = np.repeat(np.array(self.w_init)[np.newaxis], self.num_ims, axis=0).T
            self.w_init[:, 1] *= 0.5
        x = np.linspace(-1, 1, self.DIMS1)
        y = np.linspace(-1, 1, self.DIMS0)
        X, Y = np.meshgrid(x, y)

        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.Y = torch.tensor(Y, dtype=torch.float32, device=self.device)

        self.w_blur = torch.nn.Parameter(
            torch.tensor(
                self.w_init,
                dtype=torch.float32,
                device=self.device,
                requires_grad=self.psf["optimize"],
            ),
            requires_grad=self.psf["optimize"],
        )

        mask = np.transpose(mask, (2, 0, 1))
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        self.mask = mask.to(self.device).to(torch.float32).unsqueeze(0)

        # initialize calibration noise (mask read noise) parameter defaults
        self.mask_noise_params = params.get("mask_noise", {})
        self.mask_noise_intensity = self.mask_noise_params.get("intensity", 0.05)
        self.mask_noise_stopband = self.mask_noise_params.get("stopband_only", True)
        self.mask_noise_type = self.mask_noise_params.get("type", "gaussian")

        # initialize sample noise (photon & read noise) parameters
        self.sample_noise_params = params.get("sample_noise", {})
        self.read_noise_intensity = self.sample_noise_params.get("intensity", 0.001)
        self.shot_noise_photon_count = self.sample_noise_params.get("photon_count", 10e4)

        # determine forward and adjoint methods
        self.fwd = self.Hfor
        self.adj = self.Hadj
        if self.psf["lri"]:
            self.fwd = self.Hfor_varying
            self.adj = self.Hadj_varying

    def simulate_lsi_psf(self):
        """
        Simulates gaussian-distributed PSFS for an LSI system. Initial psf variance
        specified by self.w_init

        Returns
        -------
        torch.Tensor
            stack of increasingly defocused psfs (n, y, x)
        """
        exposures = self.psf.get("exposures", [])
        psfs = []
        for i in range(0, self.num_ims):
            var_yx = (self.w_blur[i], self.w_blur[i])
            if not self.psf["symmetric"]:
                var_yx = (self.w_blur[i, 0], self.w_blur[i, 1])

            psf = torch.exp(-((self.X / var_yx[0]) ** 2 + (self.Y / var_yx[1]) ** 2))
            psf = psf / torch.linalg.norm(psf, ord=float("inf"))

            psfs.append((psf / torch.max(psf)) * exposures[i])

        if exposures:
            psfs = psf_utils.even_exposures(psfs, self.num_ims, exposures)
        return torch.stack(psfs, 0)

    def sim_mask_noise(self, mask):
        """
        Returns a version of this model's mask abberated with some type of noise (either
        DC or absolute gaussian noise of variance "variance").

        Class Parameters
        ----------
        noise_intensity : float or tuple(int, int), optional
            value or range of values of DC noise variance of gaussian noise (maxval = 1),
            if None, taken randomly from options in initialization, by default None
        noise_type : str, optional
            one of {"DC", "gaussian"}, default is 'DC'
        noise_stopband : bool or None
            Whether to apply the noise to only low-signal pixels in the mask,
            default is None

        Returns
        -------
        torch.Tensor
            noise-abberated mask (n, y, x)
        """
        intensity = self.mask_noise_intensity
        if not isinstance(self.mask_noise_intensity, float):
            intensity = np.random.uniform(*self.mask_noise_intensity)

        noise = torch.abs(torch.randn(*mask.shape)) * intensity
        if self.mask_noise_type == "DC":
            noise = torch.ones_like(noise) * intensity
        noise = noise.to(mask.device)

        if not self.mask_noise_stopband:
            mask = torch.where(mask < intensity, mask, mask + noise)
        else:
            mask = torch.where(mask > intensity, mask, mask + noise)

        del noise
        return mask

    def sim_read_noise(self, b):
        """
        Simulates read noise sampling from a gaussian normal distribution, with an
        intensity varying randomly within the range selected.

        Parameters
        ----------
        b : torch.Tensor
            Input measurement (b, n, 1, y, x)
        self.intensity : tuple(low,high)
            intensity range for the noise

        Returns
        -------
        torch.Tensor
            noised measurement (b, n, 1, y, x)
        """
        intensity = self.read_noise_intensity

        # for synthetic data generation, intensity is sampled randomly from a
        # specified range i.e. as an augmentation
        if not isinstance(self.read_noise_intensity, float):
            intensity = np.random.uniform(*self.read_noise_intensity)

        noise = torch.randn_like(b) * intensity * b.max()
        return torch.clip(b, min=0) + noise.to(self.device)

    def sim_shot_noise(self, b):
        """
        Simulates shot noise by sampling from a Poisson distribution with a mean
        equal to the input measurement. The intensity of the noise is determined
        by the photon count specified in the model parameters.

        Parameters
        ----------
        b : torch.Tensor
            Input measurement (b, n, 1, y, x)

        Returns
        -------
        torch.Tensor
            noised measurement (b, n, 1, y, x)
        """
        photon_count = self.shot_noise_photon_count

        # for synthetic data generation, photon count is sampled randomly from a
        # specified range i.e. as an augmentation
        if not isinstance(self.shot_noise_photon_count, int | float):
            photon_count = np.random.uniform(*self.shot_noise_photon_count)

        return torch.poisson(torch.clip(b, min=0) * photon_count) / photon_count

    def init_psfs(self):
        """
        Initializes psfs depending on the specified parameters.
            - If simulating psfs (LSI only), generates initial psf values
            - If using LRI psfs, loads in calibrated psf data
            - If using measured LSI psfs, loads and preprocesses psf stack
        """
        if self.operations["sim_blur"]:
            self.psfs = self.simulate_lsi_psf().to(self.device).to(torch.float32)

        elif self.operations.get("load_npy_psfs", False):
            self.psfs = psf_utils.load_psf_npy(self.psf_dir, self.psf.get("norm", None))
            self.psfs = torch.tensor(
                psf_utils.center_pad_to_shape(self.psfs, (self.DIMS0, self.DIMS1)),
                device=self.device,
            )

        elif self.psfs is None:
            if self.psf["lri"]:
                psfs = psf_utils.load_lri_psfs(
                    self.psf_dir,
                    self.num_ims,
                )
            else:
                psfs = psf_utils.get_lsi_psfs(
                    self.psf_dir,
                    self.num_ims,
                    self.psf["padded_shape"],
                    self.mask.shape[-2:],
                    ksizes=self.psf.get("ksizes", []),
                    exposures=self.psf.get("exposures", []),
                    start_idx=self.psf.get("idx_offset", 0),
                    blurstride=self.psf.get("stride", 1),
                    threshold=self.psf.get("threshold", 0.7),
                    zero_outside=self.psf.get("largest_psf_diam", 128),
                    use_first=self.psf.get("use_first", True),
                    norm=self.psf.get("norm", ""),
                )
            self.psfs = torch.tensor(psfs, device=self.device)

    def spectral_pad(self, x, spec_dim=2, size=-1):
        """
        Zero pads spectral dimension of input tensor x to a power of 2

        Parameters
        ----------
        x : torch.Tensor
            input tensor to be padded (b, n, c, y, x)
        spec_dim : int, optional
            index of spectral dimension in tensor, by default 2
        size : int, optional
            size of padding, if specific size is requested, by default -1

        Returns
        -------
        torch.Tensor
            padded output tensor
        """
        spec_channels = x.shape[spec_dim]
        padsize = 0
        while spec_channels & (spec_channels - 1) != 0:
            spec_channels += 1
            padsize += 1
        padsize = size if size >= 0 else padsize
        return F.pad(x, (0, 0, 0, 0, padsize // 2, padsize // 2 + padsize % 2), "constant", 0)

    def Hfor(self, v, h, mask):
        """
        LSI spatially invariant system forward method

        Parameters
        ----------
        v : torch.Tensor
            ground truth hyperspectral batch (b, 1, c, y, x)
        h : torch.Tensor
            measured or simulated psf data (n, y, x)
        mask : torch.Tensor
            spectral calibration matrix/mask (c, y, x)

        Returns
        -------
        torch.Tensor
            simulated measurements (b, n, 1, y, x)
        """
        V = fft_im(my_pad(self, v))
        H = fft_psf(self, h)
        b = torch.sum(mask * crop_forward(self, torch.fft.ifft2(H * V).real), 2, keepdim=True)
        return b

    def Hadj(self, b, h, mask):
        """
        LSI spatially invariant convolution adjoint method

        Parameters
        ----------
        b : torch.Tensor
            simulated (or not) measurement (b, n, 1, y, x)
        h : torch.Tensor
            measured or simulated psf data (n, y, x)
        mask : torch.Tensor
            spectral calibration matrix/mask (c, y, x)

        Returns
        -------
        torch.Tensor
            LSI system adjoint
        """
        Hconj = torch.conj(fft_psf(self, h))
        B_adj = fft_im(pad_zeros_torch(self, b * mask))
        v_hat = crop_forward(self, torch.fft.ifft2(Hconj * B_adj).real)
        return v_hat

    def Hfor_varying(self, v, h, mask):
        """
        LRI spatially varying system forward method

        Parameters
        ----------
        v : torch.Tensor
            ground truth hyperspectral batch (b, 1, c, y, x)
        h : torch.Tensor
            calibrated psf data (n, y, theta, x)
        mask : torch.Tensor
            spectral calibration matrix/mask (c, y, x)

        Returns
        -------
        torch.Tensor
            simulated measurements (b, n, 1, y, x)
        """
        b = torch.sum(mask * batch_ring_convolve(v, h, self.device), 2, keepdim=True)
        return b  # quantize(b)

    def Hadj_varying(self, b, h, mask):
        """
        LRI spatially varying convolution adjoint method

        Parameters
        ----------
        b : torch.Tensor
            simulated (or not) measurement (b, n, 1, y, x)
        h : torch.Tensor
            calibrated psf data (n, y, theta, x)
        mask : torch.Tensor
            spectral calibration matrix/mask (c, y, x)

        Returns
        -------
        torch.Tensor
            LRI system adjoint (b, n, c, y, x)
        """
        v_hat = batch_ring_convolve(b * mask, torch.conj(h), self.device).real
        return v_hat

    def forward(self, v):
        """
        Applies forward model operations, as specified in initialization

        Parameters
        ----------
        v : torch.tensor
            5d tensor (b, n, c, y, x)

        Returns
        -------
        torch.Tensor
            5d tensor (b, n, 1, y, x)
        """
        # Option to pass through forward model if data is precomputed
        if self.passthrough:
            self.b = v
            return self.b

        self.init_psfs()

        if self.operations["sim_meas"]:
            if self.operations.get("fwd_mask_noise", False):
                self.b = self.fwd(v, self.psfs, self.sim_mask_noise(self.mask))
            else:
                self.b = self.fwd(v, self.psfs, self.mask)
        else:
            self.b = v

        if self.operations.get("shot_noise", False):
            self.b = self.sim_shot_noise(self.b)
        if self.operations.get("read_noise", False):
            self.b = self.sim_read_noise(self.b)

        if self.operations["adjoint"]:
            if self.operations.get("adj_mask_noise", False):
                self.b = self.adj(self.b, self.psfs, self.sim_mask_noise(self.mask))
            else:
                self.b = self.adj(self.b, self.psfs, self.mask)

        if self.operations["spectral_pad"]:
            self.b = self.spectral_pad(self.b, spec_dim=2, size=2)

        return self.b.to(torch.float32)


def build_data_pairs(data_path, model, batchsize=1):
    """
    Compute simulated system measurements for each .mat file's ground truth sample
    using the given forward model. Simulated measurements are stored in the same .mat
    file under a key defined by the parameters of the forward model.

    Automatically uses the same device as the given model.

    Parameters
    ----------
    data_path : str
        path to data dir containing preprocessed .mat data files
    model : torch.nn.Module
        forward simulation model. Instance of ForwardModel
    batchsize : int, optional
        size for batching, by default 1
    """
    data_files = glob.glob(os.path.join(data_path, "*.mat"))
    if len(data_files) == 0:
        print(f"No preprocessed patches at {os.path.basename(data_path)}... Skipping.")
        return

    model_params = {
        "stack_depth": model.num_ims,
        "psf": model.psf,
        "operations": model.operations,
        "timestamp": datetime.now().strftime("%H,%M,%d,%m,%y"),
    }
    key = str(model_params)
    desc = f"Generating Pair {os.path.basename(data_path)}"
    batch = []
    img_shape = io.loadmat(data_files[0])["image"].shape
    for i, sample_file in tqdm.tqdm(list(enumerate(data_files)), desc=desc):
        try:
            sample = io.loadmat(sample_file)
        except Exception as e:
            print(f"Exception {e} \n File: {sample_file}")
            continue

        if len(list(sample.keys())) > 4:  # if sample already in sample_file, skip
            continue

        batch.append((sample_file, sample))
        if len(batch) == batchsize or i == len(data_files) - 1:
            img_batch = torch.zeros((len(batch), 1, *img_shape), device=model.device)
            for j, (_, sample) in enumerate(batch):
                img_batch[j] = torch.tensor(np.expand_dims(sample["image"], 0)).to(model.device)

            out_batch = model(img_batch.permute(0, 1, 4, 2, 3)).detach().cpu().numpy()

            for j, (samp_f, sample) in enumerate(batch):
                sample[key] = out_batch[j]
                io.savemat(samp_f, sample)

            batch = []
