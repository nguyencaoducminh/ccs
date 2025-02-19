"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import pandas as pd

from model import Config, Transformer
# from dataloader import DatasetRT
from dataloaderccs import DatasetCCS

# -----------------------------------------------------------------------------
# default config values designed to train 
# I/O
out_root = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' 
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'deepdia'
batch_size = 12
ionmod_full = False 
# model
n_layer = 10
n_head = 8
n_embd = 512  # Chau: d_model
dff = 1024
epochs = 10
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed    
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

cp_path = dataset + '-b' + str(batch_size) + '-dm' + str(n_embd) + '-df' + str(dff) + '-nl' + str(n_layer) + '-nh' + str(n_head) + '-dr' + str(dropout) + '-ep' + str(epochs)
out_dir = os.path.join(out_root, cp_path)
print(out_dir)

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
#context managers that allow regions of your script to run in mixed precision. In these regions, CUDA ops run in a dtype chosen by autocast to improve performance while maintaining accuracy.
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print(f"device = {device}, device_type = {device_type}, ddp = {ddp}")
# deepdia data loader # poor man's data loader
# data = DatasetRT(dataset=dataset, batch_size=batch_size, epochs=epochs, device=device, device_type=device_type)
data = DatasetCCS(dataset=dataset, ionmod_full=ionmod_full, batch_size=batch_size, epochs=epochs, device=device, device_type=device_type)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
epoch = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=data.block_size,
                  bias=bias, vocab_size=data.vocab_size, dropout=dropout, device=device, dff=dff,
                  CLS=data.CLS) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = Config(**model_args)
    model = Transformer(gptconf)

    pd.DataFrame({'parameter': ['data', 'batch_size', 'd_model', 'd_ff', 'n_layers', 'n_heads', 'dropout', 
                                        'epochs', 'vocab_size', 'max_length', 'min_val', 'max_val'],
                          'value': [dataset, (batch_size), str(n_embd), str(dff), str(n_layer), str(n_head), str(dropout), 
                                    str(epochs), str(data.vocab_size), str(data.block_size), str(data.min_val), str(data.max_val)]}).to_csv(os.path.join(out_dir, 'parameters.txt'), sep= '\t', index = False)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = Config(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data.get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def to_device(x, y):
    if device_type == 'cuda':
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_val_loss():    
    losses = torch.zeros(len(data.loader_val))
    model.eval()
    for step, (X, y) in enumerate(data.loader_val):        
        X, y = to_device(X, y)
        with ctx:
            logits, loss = model(X, y)
        losses[step] = loss.item()    
    model.train()
    return losses.mean()    

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

raw_model = model.module if ddp else model # unwrap DDP container if needed

def trainer(model, dataloader, epoch, best_val_loss, verbose=True):
    if epoch==0:
        best_val_loss = estimate_val_loss()
    if verbose: print(f"Initial val loss {best_val_loss:.4f}")
    while epoch < epochs:
        losses = torch.zeros(len(dataloader))
        for step, (X, y) in enumerate(dataloader):    
            X, y = to_device(X, y)       
            optimizer.zero_grad(set_to_none=True)  
            with ctx:
                logits, loss = model(X, y)             
            loss.backward()            # Getting gradients w.r.t. parameters
            optimizer.step()           # Update parameters
            losses[step] = loss.item() # Add loss for this batch to running total
        
        loss_train = losses.mean()
        loss_val = estimate_val_loss()
        epoch += 1
        if verbose: print(f"epoch: {epoch}, train loss: {loss_train:.4f}, val loss {loss_val:.4f}")
        if loss_val < best_val_loss or always_save_checkpoint:
            best_val_loss = loss_val
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'config': config,
            }        
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

trainer(model, dataloader=data.loader_train, epoch=epoch, best_val_loss=best_val_loss, verbose=True)

if ddp:
    destroy_process_group()
