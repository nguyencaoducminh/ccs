import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
import pandas as pd

from dataloaderccs import DatasetCCS

# data
dataset = 'meier'
batch_size = 12
ionmod_full = False 
epochs = 10
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

data = DatasetCCS(dataset=dataset, ionmod_full=ionmod_full, batch_size=batch_size, epochs=epochs, device=device, device_type=device_type)