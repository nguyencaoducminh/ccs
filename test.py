import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import pandas as pd


import rt
from model import Config, Transformer
from dataloader import DatasetRT
from rt import min_max_scale_rev

def quick_test(y_predict, y_test, min_val, max_val, data):
    if data == 'autort':
        a = min_max_scale_rev(y_test, min = 0.0, max = 101.33)
        b = min_max_scale_rev(y_predict, min = 0.0, max = 101.33)                

    elif data == 'prosit':
        y_predict = min_max_scale_rev(y_predict, min = min_val, max = max_val)
        
        v = 1883.0160689
        m = 56.35363441
        a = y_test.astype(np.float32).reshape(-1) * np.sqrt(v) + m
        b = y_predict * np.sqrt(v) + m
        
    elif data == 'deepdia':
        
        y_predict = min_max_scale_rev(y_predict, min = min_val, max = max_val)

        a = y_test * 100
        b = y_predict * 100

    elif data == 'phospho':
        
        y_predict = min_max_scale_rev(y_predict, min = min_val, max = max_val)

        a = y_test
        b = y_predict

    return a, b

a,b = quick_test(
        y_predict = torch.tensor([0.1, 0.2]),
        y_test = torch.tensor([0.6, 0.7]),        
        min_val = 0.0,
        max_val = 1.0,
        data='deepdia'
)
print('MAE =', np.median(np.abs(a-b)), '\n\n')            

pd.DataFrame({'sequence': ['ABC', 'BBB'],
                    'y': a,
                    'y_pred': b}).to_csv('out-test.txt', sep = '\t', index = False)

def check_dataloader():
        data = DatasetRT('deepdia', 10)
        print(data.block_size)
        x,y = data.get_batch('train')
        print(x.shape, y.shape)

def check_transformer():
     torch.manual_seed(1337)
     block_size = 5
     d_model = 2
     model_args = dict(n_layer=1, n_head=1, n_embd=d_model, block_size=block_size,
                  bias=None, vocab_size=10, dropout=0.0, CLS=0, device='cpu') 
     gptconf = Config(**model_args)
     model = Transformer(gptconf)
     model.to('cpu')     
     X = torch.tensor([[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0]])
     Y = torch.tensor([[2],[3],[4]])
     logits, loss = model(X, Y)
     print(logits)
     print(loss)



def check_scaled_dot_product_attention():
    print("----check scaled_dot_product_attention w/o mask -----------------")

    x_t =tf.constant([[[[-1.0768,  0.4640],
            [ 0.4448,  0.6557],
            [ 1.1747, -0.9382],
            [-1.5625,  0.4215],
            [ 2.5409, -1.4361]]],


            [[[ 0.0285, -0.1528],
            [-0.1481, -0.0250],
            [-0.2336,  0.1521],
            [ 0.6804,  0.4380],
            [-0.7299, -0.9437]]],


            [[[-0.0100, -0.1074],
            [-0.4001, -0.5974],
            [-1.1877,  1.0379],
            [-0.4248, -0.6217],
            [ 0.2859, -1.2051]]]])

    x_p = torch.tensor([[[[-1.0768,  0.4640],
            [ 0.4448,  0.6557],
            [ 1.1747, -0.9382],
            [-1.5625,  0.4215],
            [ 2.5409, -1.4361]]],


            [[[ 0.0285, -0.1528],
            [-0.1481, -0.0250],
            [-0.2336,  0.1521],
            [ 0.6804,  0.4380],
            [-0.7299, -0.9437]]],


            [[[-0.0100, -0.1074],
            [-0.4001, -0.5974],
            [-1.1877,  1.0379],
            [-0.4248, -0.6217],
            [ 0.2859, -1.2051]]]])

    scaled_attention, attention_weights = rt.scaled_dot_product_attention(x_t, x_t, x_t, mask=None)
    y = torch.nn.functional.scaled_dot_product_attention(x_p, x_p, x_p, attn_mask=None, dropout_p=0, is_causal=False)            

    print(scaled_attention)
    print(y)
    b = (y.detach().numpy()-scaled_attention) < 0.001
    print(b)

def check_attention_mask():
    print("----check scaled_dot_product_attention WITH mask to see if mask is correct-----------------")


    x_t =tf.constant([[[[-1.0768,  0.4640],
            [ 0.4448,  0.6557],
            [ 1.1747, -0.9382],
            [-1.5625,  0.4215],
            [ 2.5409, -1.4361]]],


            [[[ 0.0285, -0.1528],
            [-0.1481, -0.0250],
            [-0.2336,  0.1521],
            [ 0.6804,  0.4380],
            [-0.7299, -0.9437]]],


            [[[-0.0100, -0.1074],
            [-0.4001, -0.5974],
            [-1.1877,  1.0379],
            [-0.4248, -0.6217],
            [ 0.2859, -1.2051]]]])

    x_p = torch.tensor([[[[-1.0768,  0.4640],
            [ 0.4448,  0.6557],
            [ 1.1747, -0.9382],
            [-1.5625,  0.4215],
            [ 2.5409, -1.4361]]],


            [[[ 0.0285, -0.1528],
            [-0.1481, -0.0250],
            [-0.2336,  0.1521],
            [ 0.6804,  0.4380],
            [-0.7299, -0.9437]]],


            [[[-0.0100, -0.1074],
            [-0.4001, -0.5974],
            [-1.1877,  1.0379],
            [-0.4248, -0.6217],
            [ 0.2859, -1.2051]]]])

    seq = [[1., 1., 0., 0., 0.], # (batch_size, sequence_length)
                [2., 2., 2., 0., 0.],
                    [3., 3., 3., 3., 0.]]
    mask_t = rt.create_padding_mask(seq)
    src_padding_mask = torch.eq(torch.tensor(seq), 0.)[:, None, None, :]

    scaled_attention, attention_weights = rt.scaled_dot_product_attention(x_t, x_t, x_t, mask=mask_t)
    y = torch.nn.functional.scaled_dot_product_attention(x_p, x_p, x_p, attn_mask=src_padding_mask.logical_not(), dropout_p=0, is_causal=False)            

    print(scaled_attention)
    print(y)
    b = (y.detach().numpy()-scaled_attention) < 0.001
    print(b)
    print("---------------------")

def check_attention():
    x_t = tf.constant([[[ 0.9338,  0.9565],
            [ 0.6994, -1.3554],
            [ 0.7156, -1.6993],
            [-0.7461,  1.1423],
            [ 1.4014, -0.3997]],

            [[ 0.2302, -0.4514],
            [-0.4724,  1.5105],
            [-0.8648,  0.4025],
            [-0.1598,  2.5070],
            [-0.5000, -1.0080]],

            [[ 1.5549, -1.1028],
            [ 0.0629, -0.9278],
            [-0.8199,  0.8228],
            [-1.6971,  2.6501],
            [ 0.5497, -0.5089]]])
    x_p = torch.tensor([[[ 0.9338,  0.9565],
            [ 0.6994, -1.3554],
            [ 0.7156, -1.6993],
            [-0.7461,  1.1423],
            [ 1.4014, -0.3997]],

            [[ 0.2302, -0.4514],
            [-0.4724,  1.5105],
            [-0.8648,  0.4025],
            [-0.1598,  2.5070],
            [-0.5000, -1.0080]],

            [[ 1.5549, -1.1028],
            [ 0.0629, -0.9278],
            [-0.8199,  0.8228],
            [-1.6971,  2.6501],
            [ 0.5497, -0.5089]]])



    seq = [[1., 1., 0., 0., 0.], # (batch_size, sequence_length)
                [2., 2., 2., 0., 0.],
                    [3., 3., 3., 3., 0.]]
    mask_t = rt.create_padding_mask(seq)
    mha = rt.multi_head_attention(d_model=2, num_heads=2)
    attn_output, _ = mha(x_t, x_t, x_t, mask=None)  # (batch_size, input_seq_len, d_model)
    #attn_output, _ = mha(x_t, x_t, x_t, mask_t)  # (batch_size, input_seq_len, d_model)
    print(attn_output)

    src_padding_mask = torch.eq(torch.tensor(seq), 0.)[:, None, None, :]
    model_args = dict(n_head=2, n_embd=2, block_size=5)
    config = model.Config(**model_args)
    attn = model.MultiHeadAttention(config)
    #y_p = attn(x_p, src_padding_mask)
    y_p = attn(x_p)
    print(y_p)

#check_scaled_dot_product_attention()
#check_attention_mask()
#check_attention()
#check_transformer()

"""

from dataloader import DataLoader

data = DataLoader('deepdia', 5)
x_train = data.x_train
print(len(x_train))
print(x_train.shape)
print(x_train[0])
print(x_train[1])

print(data.y_train[0])
print(data.y_train[1])

(x, y) = data.next_batch()
print(x)
print(y)
print(data.block_size)
"""