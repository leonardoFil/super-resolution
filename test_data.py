import os
import tensorflow as tf
from data import CustomDataset

# Define the test directory path
test_dir = './.custom_data_test'

def test_custom_dataset():
    # Instantiate the CustomDataset class
    dataset = CustomDataset(test_dir, downgrade='bicubic', scale=4)
    
    # Test the __len__ method
    assert len(dataset) == len(os.listdir(os.path.join(test_dir,'HR')))*2

    # Test the dataset method
    ds = dataset.dataset()
    for x in ds:
        assert isinstance(x, tf.data.Dataset)
        assert x.element_spec == (tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32), tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32))

    # Test the produce_downgraded_images method
        
# Run the test
test_custom_dataset()