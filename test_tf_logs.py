import os
import sys

level = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = level
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
print("TF imported with level", level)
