# Spectral Defocuscam
*Authors: Christian Foley, Kristina Monakhova, Kyrollos Yanny*

This software was developed in partnership with the Department of Electrical Engineering and Computer Sciences at UC Berkeley and the UCB/UCSF Joint Graduate Program in Bioengineering at UC Berkeley. <br> <br>

The Spectral DefocusCam is a novel type of snapshot hyperspectral camera which captures multiple measurements at differing levels of focus, and fuses them into a single high-resolution hyperspectral volume. The camera design is implemented using a focus-tunable lense to take images in quick succession, so as to preserve the temporal resolution of the camera. <br>

![camera_design.png](data/meta_images/camera_design.png "Camera design") <br>

The camera iterates on previous approaches (https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-10-1298&id=440114) by using physics-inspired signals processing in tandem with a 3d Unet to perform image reconstruction. By leveraging multiple slightly defocused measurements a deep-learning approach we are able to achieve vastly higher simulation resolution, and overcome the sparsity constraint of previous methods. <br>

![result_comparison.png](data/meta_images/result_comparison.png "Result comparison") <br>

If you would like to read more, you may find the paper at the publishers website: https://opg.optica.org/abstract.cfm?uri=COSI-2022-CF2C.1 <br>
Copies for personal use can be provided upon request.

### Running the codebase
If you plan to use this software, please cite the paper (see link above)

The spectral_defocus_tutorial_learning notebook will take you through the focus of this software (Spectral DefocusCam) and demonstrate how our reconstructive neural network is trained and applied. Keep in mind that the software only simulates a forward imaging model to perform the computational reconstruction.


### Dependencies:
Notebooks are dependent upon a properly configured Anaconda virtual work environment with the following:
* Anaconda (https://docs.anaconda.com/anaconda/install/index.html)
* PyTorch (https://pytorch.org/)
* Cuda (https://numpy.org/install/
* NumPy (https://numpy.org/install/)
* openCV/CV2 (https://pypi.org/project/opencv-python/) <br> <br>

All dependencies can be installed through creating a new anaconda environment from the requirements.txt file:
> conda create -n defocuscam python=3.9 

activating the environment: <br>
> conda activate defocuscam <br>
> conda install pip

and installing the requirements through pip <br>
> pip install -r requirements.txt 
