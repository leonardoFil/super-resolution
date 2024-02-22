"""
This Python script performs super-resolution on an example satellite image using two different models: EDSR and WDSR.
It loads pre-trained weights for both models and applies them to a set of input images at different resolutions.
The resulting super-resolved images are then displayed using matplotlib.

Author: Leonardo Filipe
Email: leonardo.filipe@wavec.org
Date: 20 February 2024

The script follows the following steps:
1. Imports necessary modules and functions from the project.
2. Defines the EDSR and WDSR models and loads their pre-trained weights.
3. Specifies the root directory and image name for the input images.
4. Defines a list of resolutions for the input images.
5. Initializes empty lists to store the low-resolution (lr) and super-resolution (sr) images for each model.
6. Loops over each resolution and each model, loading the input image, storing it in the lr list, and applying the model to obtain the super-resolved image, which is stored in the sr list.
7. Loops over each model and displays the lr and sr images side by side using matplotlib.
8. Shows the plot.

Note: The script assumes that the necessary model weights and input images are available in the specified directories.
"""

from model import resolve_single
from model.wdsr import wdsr_b
from model.edsr import edsr
from utils import load_image
import os
import matplotlib.pyplot as plt

# Define the EDSR and WDSR models and load their pre-trained weights
models = [edsr(scale=4, num_res_blocks=16), wdsr_b(scale=4, num_res_blocks=32)]
models[0].load_weights('weights/edsr-16-x4/weights.h5')
models[1].load_weights('weights/wdsr-b-32-x4/weights.h5')

# Specify the root directory and image name for the input images
root = r"C:\Users\LFilipe\OneDrive - WavEC Offshore Renewables\Pictures\GEOSAT"
img_name = r"DE2_PSH_L1C_000000_20230926T103740_20230926T103743_DE2_50243_31E0_T1.png"

# Define a list of resolutions for the input images (each element represents a subdirectory in the root directory)
resolutions = ['pngFull/crops', 'pngFull/downsampledCrops1m', 'pngFull/downsampledCrops3m', 'pngFull/downsampledCrops6m']

# Initialize empty lists to store the low-resolution (lr) and super-resolution (sr) images for each model
lrs = {'edsr':[],'wdsr':[]}
srs = {'edsr':[],'wdsr':[]}

# Loop over each resolution and each model, loading the input image, storing it in the lr list,
# and applying the model to obtain the super-resolved image, which is stored in the sr list
for resolution in resolutions:
    for model in models:
        lr = load_image(os.path.join(root,resolution,img_name))
        lrs[model.name].append(lr)
        srs[model.name].append(resolve_single(model,lr))

# Loop over each model and display the lr and sr images side by side using matplotlib
for model in models:
    fig, axs = plt.subplots(len(lrs['edsr']), 2, figsize=(10, 10))
    fig.suptitle(str(model.name).upper(), fontsize=16, fontweight="bold", ha="center")
    for n in range(len(lrs['edsr'])):
        axs[n,0].imshow(lrs['edsr'][n])
        axs[n,0].axis('off')
        axs[n,1].imshow(srs[model.name][n])
        axs[n,1].axis('off')
        if n==0:
            axs[n,0].set_title('Low Resolution - Input')
            axs[n,1].set_title('Super Resolution - Output')
        plt.tight_layout()

# Show the plot
plt.show(block=False)
plt.show()