import re
import time
import keras
from timeit import default_timer
start_time = default_timer()
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Input, Dropout, LayerNormalization, Conv1D, Reshape

print("------------------------------------- Importing Config ----------------------------------------------")
print(f"------------------------------------- Tensorflow Version:{tf.__version__} -------------------------------------")
print("------------------------------------- Time Taken: {0:.2f} sec -----------------------------------------".format(default_timer()-start_time))
