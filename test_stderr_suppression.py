import sys, os
old_stderr = os.dup(2)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)
os.close(devnull)
import tensorflow as tf
os.dup2(old_stderr, 2)
print("TF imported")
