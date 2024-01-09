import numpy as np
import torch
import os, glob
import scipy.io as io
from datetime import datetime
import tqdm

from utils.diffuser_utils import *
import SpectralDefocusCam.utils.psf_utils as psf_utils


class ForwardModel(torch.nn.Module):
    def __init__(
        self,
        mask,
        params,
        psf_dir=None,
        w_init=None,
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
        w_init : list, optional
            specific initialization for the blur kernels, by default None
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
            self.w_init = np.repeat(
                np.array(self.w_init)[np.newaxis], self.num_ims, axis=0
            ).T
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

        self.mask = torch.tensor(
            np.transpose(mask, (2, 0, 1)), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

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

            psfs.append((psf / np.max(psf)) * exposures)

        if exposures:
            psfs = psf_utils.even_exposures(psfs, self.num_ims, exposures)
        return torch.stack(psfs, 0)

    def init_psfs(self):
        """
        Initializes psfs depending on the specified parameters.
            - If simulating psfs (LSI only), generates initial psf values
            - If using LRI psfs, loads in calibrated psf data
            - If using measured LSI psfs, loads and preprocesses psf stack
        """
        if self.passthrough:
            return

        if self.operations["sim_blur"]:
            self.psfs = self.simulate_lsi_psf().to(self.device).astype(torch.float32)
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
                    self.mask.shape[-2],
                    self.psf["padded_shape"],
                    ksizes=self.psf.get("ksizes", []),
                    exposures=self.psf.get("exposures", []),
                    blurstride=self.psf.get("stride", 1),
                    threshold=self.psf.get("threshold", 0.7),
                    zero_outside=self.psf.get("largest_psf_diam", 0),
                    use_first=self.psf.get("use_first", True),
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
        return F.pad(
            x, (0, 0, 0, 0, padsize // 2, padsize // 2 + padsize % 2), "constant", 0
        )

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
        b = torch.sum(
            mask * crop_forward(self, torch.fft.ifft2(H * V).real), 2, keepdim=True
        )
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
        return b

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
            5d tensor (b, n, 1, y x)
        """
        # Option to pass through forward model if data is precomputed
        if self.passthrough:
            self.b = v
            return self.b

        self.init_psfs()

        if self.operations["sim_meas"]:
            self.b = self.fwd(v, self.psfs, self.mask)
        else:
            self.b = v

        if self.operations["adjoint"]:
            self.b = self.adj(self.b, self.psfs, self.mask)

        # applies global normalization
        self.b = (self.b - torch.min(self.b)) / torch.max(self.b - torch.min(self.b))

        if self.operations["spectral_pad"]:
            self.b = self.spectral_pad(self.b, spec_dim=2, size=2)

        if self.operations["roll"]:
            self.b = torch.flip(self.b, dims=(1,))
            self.b = torch.roll(self.b, -1, dims=1)

        return self.b


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
                img_batch[j] = torch.tensor(np.expand_dims(sample["image"], 0)).to(
                    model.device
                )

            out_batch = model(img_batch.permute(0, 1, 4, 2, 3)).detach().cpu().numpy()

            for j, (samp_f, sample) in enumerate(batch):
                sample[key] = out_batch[j]
                io.savemat(samp_f, sample)

            batch = []
