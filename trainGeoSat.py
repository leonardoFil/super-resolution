import os, glob
from pathlib import Path
import tensorflow as tf
from model.edsr import edsr
from train import EdsrTrainer

## Load Dataset

ds_root = os.path.join(Path(os.getcwd()).parents[1],r"C:\Users\LFilipe\OneDrive - WavEC Offshore Renewables\Pictures\GEOSAT\_preProcessed")
lr_images_path = os.path.join(ds_root,'GEOSAT_LR_X4')
hr_images_path = os.path.join(ds_root,'GEOSAT_HR_RESIZED')

file_paths_1 = sorted(sorted(glob.glob(os.path.join(lr_images_path, "*.png"))))
file_paths_2 = sorted(sorted(glob.glob(os.path.join(hr_images_path, "*.png"))))


file_paths_1 = [r"{}".format(path) for path in file_paths_1]
file_paths_2 = [r"{}".format(path) for path in file_paths_2]
# print(os.path.isdir(ds_root))
# print(os.path.isdir(lr_images_path))
# print(f'lr_images_path={lr_images_path}')
# print(f'file_paths1={file_paths_1}')

lr_dataset = tf.data.Dataset.from_tensor_slices(file_paths_1)
hr_dataset = tf.data.Dataset.from_tensor_slices(file_paths_2)

# for x in lr_dataset:
#     print(x)

def load_and_preprocess_image(file_path):
    
    image = tf.io.read_file(file_path)

    image = tf.image.decode_png(image, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)

    return image

lr_dataset = lr_dataset.map(load_and_preprocess_image)
hr_dataset = hr_dataset.map(load_and_preprocess_image)

# for x in lr_dataset:
#     print(x)

combined_dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

batched_dataset = combined_dataset.batch(1)

prefetched_dataset = batched_dataset.prefetch(tf.data.experimental.AUTOTUNE)

DATASET_SIZE= len(prefetched_dataset)

train_size = int(0.97 * DATASET_SIZE)
valid_size = DATASET_SIZE - train_size

train_ds = prefetched_dataset.take(train_size)
valid_ds = prefetched_dataset.skip(train_size)

# Create a training context for an EDSR x4 model with 16 
# residual blocks.
trainer = EdsrTrainer(model=edsr(scale=4, num_res_blocks=16), 
                      checkpoint_dir=r'.ckpt/edsr-16-x4')
                      
# Train EDSR model for 40 steps and evaluate model
# every step on the first 10 images of the
# validation set. Save a checkpoint only if evaluation
# PSNR has improved.
trainer.train(train_ds,
              valid_ds.take(10),
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