"""
predict using a trained model
"""
import os
from contextlib import nullcontext
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from modelcharge import Config, Transformer
# from dataloader import load_test_data, min_max_scale_rev
from dataloadercharge import load_test_data, min_max_scale_rev, min_max_scale

# to fix cuda problem (don't know why training works by predicting doesn't)
# scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
# RuntimeError: CUDA error: invalid configuration argument
#torch.backends.cuda.enable_mem_efficient_sdp(False)
#torch.backends.cuda.enable_flash_sdp(False)
#torch.backends.cuda.enable_math_sdp(True)

# -----------------------------------------------------------------------------
output_dir = 'out'
model_dir = 'out' 
dataset = 'None'
input = 'None'
seq_header = 'sequence'
rt_header = 'rt'
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device_type = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
output = 'output_py.txt'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# CCS Available datasets
MEIER_DATASETS = ['small', 'train', 'meier']

def to_device(x, y):
    if device_type == 'cuda':
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model. Load from a model saved in a specific directory
ckpt_path = os.path.join(model_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
config = Config(**checkpoint['model_args'])
config.device = device
model = Transformer(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# model's parameters
para = pd.read_csv(os.path.join(model_dir, 'parameters.txt'), sep = '\t', index_col = 0)
print(para)

#predict
input_file = None
if dataset == 'None':
    dataset = para.loc['data', 'value']
    input_file = input
print(dataset)
x_test, y_test, all_peps = load_test_data(data=dataset, input_file=input_file, 
                                          seq_header=seq_header, rt_header=rt_header,
                                          CLS=config.CLS, seq_length=int(para.loc['max_length', 'value']))

# y_test = min_max_scale(y_test, min = float(para.loc['min_val', 'value']), max = float(para.loc['max_val', 'value']))
# x_test, y_test = to_device(x_test, y_test)

# with torch.no_grad():
#     with ctx:
#         y_predict, loss = model(x_test, y_test)
#         print(f"Predict loss: {loss.item():.4f}")

y_test_scaled = min_max_scale(y_test, min = float(para.loc['min_val', 'value']), max = float(para.loc['max_val', 'value']))
x_tensor = TensorDataset(x_test, y_test_scaled)
load_test_data = DataLoader(x_tensor, batch_size = 1, shuffle = False)

with torch.no_grad():
    losses = torch.zeros(len(load_test_data))
    y_predict = torch.zeros(len(load_test_data))
    for i, (x, y) in enumerate(load_test_data):
        x, y = to_device(x, y)
        with ctx:
            logits, loss = model(x, y)
        losses[i] = loss.item()
        y_predict[i] = logits.item()
    loss_predict = losses.mean()
    print(f"Predict loss: {loss_predict.item():.4f}")

def postprocessing(y_predict, y_test, min_val, max_val):
    predict = min_max_scale_rev(y_predict, min = min_val, max = max_val)
    test = y_test.squeeze()
    return predict, test

predict, test = postprocessing(y_predict.detach(), y_test, float(para.loc['min_val', 'value']), float(para.loc['max_val', 'value']))
mae = torch.median(abs(test-predict))
print('\nModel epoch =', checkpoint['epoch'], '; MAE =', mae, '\n\n')

pd.DataFrame({'sequence': all_peps,
                'y': test.cpu().float(),
                'y_pred': predict.cpu().float()}).to_csv(os.path.join(output_dir, output), sep = '\t', index = False)