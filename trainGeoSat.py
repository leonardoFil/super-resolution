import os, glob
from pathlib import Path
import tensorflow as tf
from model.edsr import edsr
from train import EdsrTrainer
from data import CustomDataset

## Load Dataset
images_dir = Path(r"C:\Users\LFilipe\OneDrive - WavEC Offshore Renewables\Pictures\GEOSAT\_preProcessed")
train_ds, valid_ds = CustomDataset(images_dir=images_dir, scale=4, downgrade='bicubic').dataset()

# Create a training context for an EDSR x4 model with 16 
# residual blocks.
trainer = EdsrTrainer(model=edsr(scale=4, num_res_blocks=16), 
                      checkpoint_dir=r'.ckpt/edsr-16-x4')
                      
# Train EDSR model for 40 steps and evaluate model
# every step on the first 10 images of the
# validation set. Save a checkpoint only if evaluation
# PSNR has improved.
trainer.train(train_ds,
              valid_ds,
              steps=40, 
              evaluate_every=1, 
              save_best_only=True)
              
# Restore from checkpoint with highest PSNR.
trainer.restore()

# Evaluate model on full validation set.
psnr = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr.numpy():3f}')

# Save weights to separate location.
trainer.model.save_weights('weights/edsr-16-x4/weights_geosat.h5')  