import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np

seed = 42

def process_batch(x, y):
    cutmix = tf.keras.layers.CutMix(seed=seed)
    return cutmix({"images": x, "labels": y})

x = np.random.randint(0, 255, (8, 224, 224, 3), dtype='uint8')
y = tf.keras.ops.one_hot(np.random.randint(0, 3, (8,)), num_classes=3)
    
print("Processing with input:")
print(x.shape, y.shape)

try:
    res = process_batch(x, y)
    print("Success! Keys returned:", res.keys())
except Exception as e:
    print("Error:", e)
