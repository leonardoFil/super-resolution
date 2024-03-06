from model.srgan import generator, discriminator
import os
from model import resolve_single
from utils import load_image
import matplotlib.pyplot as plt

# Location of model weights (needed for demo)
weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

os.makedirs(weights_dir, exist_ok=True)

pre_generator = generator()
gan_generator = generator()

pre_generator.load_weights(weights_file('pre_generator.h5'))
gan_generator.load_weights(weights_file('gan_generator.h5'))

def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)
    
    pre_sr = resolve_single(pre_generator, lr)
    gan_sr = resolve_single(gan_generator, lr)
    
    plt.figure(figsize=(20, 20))
    
    images = [lr, pre_sr, gan_sr]
    titles = ['LR', 'SR (PRE)', 'SR (GAN)']
    positions = [1, 3, 4]
    
    for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
        plt.subplot(2, 2, pos)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.show(block=False)
    
root = r"C:\Users\LFilipe\OneDrive - WavEC Offshore Renewables\Pictures\GEOSAT"
img_name = r"DE2_PSH_L1C_000000_20230926T103740_20230926T103743_DE2_50243_31E0_T1.png"
resolutions = ['pngFull/crops', 'pngFull/downsampledCrops1m', 'pngFull/downsampledCrops3m', 'pngFull/downsampledCrops6m']    

for resolution in resolutions:
    resolve_and_plot(os.path.join(root,resolution,img_name))
plt.show()