import os
import sys
import argparse
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

"""
Load CCS datasets

Ionmob
PATH: ./data/ionmob
AVAILABLE: chang, ogata, sara, tenzer, tenzer-phospho, zepeda with 'ionmob' for the combined dataset

Meier 2021
PATH: ./data/meier_2021
AVAILABLE:
    - small : the first 100000 entries of SourceData_Figure_1.csv (80:20 train:val:test)
    - train : all of SourceData_Figure_1.csv (80:20 train:val)
    - meier : all Meier 2021 datasets with SourceData_Figure_1.csv (80:20 train:val) and SourceData_Figure_4.csv as test
"""

# For git repos
DATA_DIR = './data'
IONMOB_DIR = DATA_DIR + '/ionmob'
MEIER_DIR = DATA_DIR + '/meier_2021'

# For server
# DATA_DIR = os.environ['HOME'] + '/data-ccs'
# IONMOB_DIR = DATA_DIR + '/zenodo/unimod'
# MEIER_DIR = DATA_DIR + '/Meier_2021'

MEIER_SMALL_DIR = './data/meier_2021'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Set random seed for spliting train-val-test
random = 1024

# Available datasets
IONMOB_DATASETS = ['chang', 'ogata', 'sara', 'tenzer', 'tenzer-phospho', 'zepeda', 'ionmob']
MEIER_DATASETS = ['small', 'train', 'meier']

def min_max_scale(x, min, max):
    new_x = 1.0 * (x - min) / (max - min)
    return new_x

def min_max_scale_rev(x, min, max):
    old_x = x * (max - min) + min
    return old_x

class data_ionmob():

    @classmethod
    def filter_data(cls, data):
        # Filter data that contains unsupported tokens
        UNSUPPORTED_TOKEN = ['C[UNIMOD:312]', 'C[UNIMOD:4]', 'K[UNIMOD:1]']
        return data[~data['sequence-tokenized'].apply(lambda x: any(token in x for token in UNSUPPORTED_TOKEN))]
    
    @classmethod
    def load_data(cls, dataset, ionmob_full = False):
        # Load data from ionmob dataset
        if dataset == 'chang':
            data = pd.read_parquet(IONMOB_DIR + '/Chang_unique_unimod.parquet')
            data = data.drop(columns=['sequence'])
        elif dataset == 'ogata':
            data = pd.read_parquet(IONMOB_DIR + '/Ogata_unique_unimod.parquet')
            data = data.drop(columns=['name'])
        elif dataset == 'sara':
            data = pd.read_parquet(IONMOB_DIR + '/Sara_unique_unimod.parquet')
            data_c1 = pd.read_parquet(IONMOB_DIR + '/Sara_unique_c1_unimod.parquet')
            data = pd.concat([data, data_c1])
        elif dataset == 'tenzer':
            data = pd.read_parquet(IONMOB_DIR + '/Tenzer_unique_unimod.parquet')
            data_c1 = pd.read_parquet(IONMOB_DIR + '/Tenzer_unique_c1_unimod.parquet')
            data = pd.concat([data, data_c1])
        elif dataset == 'tenzer-phospho':
            data_train = pd.read_parquet(IONMOB_DIR + '/Tenzer-phospho-train_unique_unimod.parquet')
            data_valid = pd.read_parquet(IONMOB_DIR + '/Tenzer-phospho-valid_unique_unimod.parquet')
            if ionmob_full:
                return data_train, data_valid
            else:
                return cls.filter_data(data_train), cls.filter_data(data_valid)
        elif dataset == 'zepeda':
            data = pd.read_parquet(IONMOB_DIR + '/Zepeda_unique_unimod.parquet')
            data_c1 = pd.read_parquet(IONMOB_DIR + '/Zepeda_unique_c1_unimod.parquet')
            data_thunder = pd.read_parquet(IONMOB_DIR + '/Zepeda_thunder_unique_unimod.parquet')
            data = pd.concat([data, data_c1, data_thunder])
        elif dataset == 'ionmob':
            data_chang = pd.read_parquet(IONMOB_DIR + '/Chang_unique_unimod.parquet')
            data_otega = pd.read_parquet(IONMOB_DIR + '/Ogata_unique_unimod.parquet')
            data_sara = pd.read_parquet(IONMOB_DIR + '/Sara_unique_unimod.parquet')
            data_sara_c1 = pd.read_parquet(IONMOB_DIR + '/Sara_unique_c1_unimod.parquet')
            data_tenzer = pd.read_parquet(IONMOB_DIR + '/Tenzer_unique_unimod.parquet')
            data_tenzer_c1 = pd.read_parquet(IONMOB_DIR + '/Tenzer_unique_c1_unimod.parquet')
            data_tenzer_phospho_train = pd.read_parquet(IONMOB_DIR + '/Tenzer-phospho-train_unique_unimod.parquet')
            data_tenzer_phospho_valid = pd.read_parquet(IONMOB_DIR + '/Tenzer-phospho-valid_unique_unimod.parquet')
            data_zepeda = pd.read_parquet(IONMOB_DIR + '/Zepeda_unique_unimod.parquet')
            data_zepeda_c1 = pd.read_parquet(IONMOB_DIR + '/Zepeda_unique_c1_unimod.parquet')
            data_zepeda_thunder = pd.read_parquet(IONMOB_DIR + '/Zepeda_thunder_unique_unimod.parquet')
            data = pd.concat([data_chang, data_otega, data_sara, data_sara_c1, data_tenzer, data_tenzer_c1, data_tenzer_phospho_train, data_tenzer_phospho_valid, data_zepeda, data_zepeda_c1, data_zepeda_thunder])
        if ionmob_full:
            return data
        else:
            return cls.filter_data(data)
    
    @classmethod
    def ionmob_to_integer(cls, sequences, max_sequence_length = 60):
        
        ionmob_ALPHABET = {
            "A": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "K": 9,
            "L": 10,
            "M": 11,
            "N": 12,
            "P": 13,
            "Q": 14,
            "R": 15,
            "S": 16,
            "T": 17,
            "V": 18,
            "W": 19,
            "Y": 20,
            "M[UNIMOD:35]": 21,
            "S[UNIMOD:21]": 22,
            "T[UNIMOD:21]": 23,
            "Y[UNIMOD:21]": 24,
            "<START>[UNIMOD:1]": 25,
            'C[UNIMOD:312]': 26, 
            'C[UNIMOD:4]': 27, 
            'K[UNIMOD:1]': 28
        }

        array = np.zeros([len(sequences), max_sequence_length], dtype=int)
        for i, sequence in enumerate(sequences):
            non_dict = 0
            for j, s in enumerate(sequence):
                if s == '<START>':
                    non_dict += 1
                    continue
                if s == '<END>':
                    continue
                array[i, j-non_dict] = ionmob_ALPHABET[s]
        return array
    
    @classmethod
    def integer_to_sequence_phospho(cls, X):
        # int2seq = '_ACDEFGHIKLMNPQRSTVWY1234*'
        int2seq = '_ACDEFGHIKLMNPQRSTVWY1234*567'
        int2seqf = lambda x : ''.join([int2seq[c] for c in x if c > 0])
        return ([int2seqf(x) for x in X])
    
    @classmethod
    def split_data(cls, data, split = 0.8):
        data = data.sample(frac=1, random_state=random).reset_index(drop=True)
        split_index = int(len(data) * split)
        return data[:split_index], data[split_index:]

    @classmethod
    def load_ionmob(cls, dataset, max_sequence_length = 60, ionmob_full = False):
        if dataset == 'tenzer-phospho':
            if ionmob_full:
                data_train, data_valid = cls.load_data('tenzer-phospho', ionmob_full)
            else:
                data_train, data_valid = cls.load_data('tenzer-phospho')
            x_train = cls.ionmob_to_integer(data_train['sequence-tokenized'], max_sequence_length = max_sequence_length)
            y_train = data_train['ccs'].to_numpy()
            x_valid = cls.ionmob_to_integer(data_valid['sequence-tokenized'], max_sequence_length = max_sequence_length)
            y_valid = data_valid['ccs'].to_numpy()
            return (x_train, y_train), (x_valid, y_valid)

        if ionmob_full:
            data = cls.load_data(dataset, ionmob_full)
        else:
            data = cls.load_data(dataset)
        data_train_val, data_test = cls.split_data(data)
        data_train, data_val = cls.split_data(data_train_val, split=0.875)
        
        x_train = cls.ionmob_to_integer(data_train['sequence-tokenized'], max_sequence_length = max_sequence_length)
        y_train = data_train['ccs'].to_numpy()
        x_val = cls.ionmob_to_integer(data_val['sequence-tokenized'], max_sequence_length = max_sequence_length)
        y_val = data_val['ccs'].to_numpy()
        
        x_test = cls.ionmob_to_integer(data_test['sequence-tokenized'], max_sequence_length = max_sequence_length)
        y_test = data_test['ccs'].to_numpy()
        np.save(f'./data/ionmob/test/{dataset}_x_test.npy', x_test)
        np.save(f'./data/ionmob/test/{dataset}_y_test.npy', y_test)

        return (x_train, y_train), (x_val, y_val)

    @classmethod
    def load_training_transformer(cls, dataset, ionmob_full = False):
        if ionmob_full:
            return cls.load_ionmob(dataset, ionmob_full=ionmob_full)
        else:
            return cls.load_ionmob(dataset)       
        
    @classmethod
    def load_testing_transformer(cls, dataset):
        x_test = np.load(f'./data/ionmob/test/{dataset}_x_test.npy')
        y_test = np.load(f'./data/ionmob/test/{dataset}_y_test.npy')

        return (x_test, y_test)

class data_meier():

    @classmethod
    def split_data(cls, data, split = 0.8):
        data = data.sample(frac=1, random_state=random).reset_index(drop=True)
        split_index = int(len(data) * split)
        return data[:split_index], data[split_index:]

    @classmethod
    def load_data(cls, dataset):
        if dataset == 'small':
            data = pd.read_csv(MEIER_SMALL_DIR + '/meier_small.csv')
        if dataset == 'train':
            data = pd.read_csv(MEIER_DIR + '/SourceData_Figure_1.csv')
        if dataset == 'meier':
            datatrain = pd.read_csv(MEIER_DIR + '/SourceData_Figure_1.csv')
            datatest = pd.read_csv(MEIER_DIR + '/SourceData_Figure_4.csv')
            # datatest = pd.read_csv(MEIER_DIR + '/Figure_4_MassCharge.csv')
            return datatrain, datatest

        return cls.split_data(data=data, split=0.8)
    
    @classmethod
    def meier_to_integer(cls, sequences, max_sequence_length = 60):
        
        MEIER_ALPHABET = {
            "A": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "K": 9,
            "L": 10,
            "M": 11,
            "N": 12,
            "P": 13,
            "Q": 14,
            "R": 15,
            "S": 16,
            "T": 17,
            "V": 18,
            "W": 19,
            "Y": 20,
            "1": 21, # M(ox)
            "*": 22  # (ac)
        }

        array = np.zeros([len(sequences), max_sequence_length], dtype=int)
        for i, sequence in enumerate(sequences):
            # remove '_'
            sequence = sequence[1:-1]
            # if sequence starts with (ac) replace with '*'
            if sequence.startswith('(a'):
                sequence = '*' + sequence[4:]
            sequence = re.sub('M\(ox\)', '1', sequence)
            for j, s in enumerate(sequence):
                array[i, j] = MEIER_ALPHABET[s]
        return array
    
    @classmethod
    def integer_to_sequence(cls, X):
        int2seq = '_ACDEFGHIKLMNPQRSTVWY1*'
        int2seqf = lambda x : ''.join([int2seq[c] for c in x if c > 0])
        return ([int2seqf(x) for x in X])
    
    @classmethod
    def load_meier(cls, dataset, max_sequence_length = 60):
        data_train_val, data_test = cls.load_data(dataset)
        
        if dataset == 'train':
            x_train = cls.meier_to_integer(data_train_val['Modified sequence'], max_sequence_length = max_sequence_length)
            y_train = data_train_val['CCS'].to_numpy()
            x_val = cls.meier_to_integer(data_test['Modified sequence'], max_sequence_length = max_sequence_length)
            y_val = data_test['CCS'].to_numpy()

            return (x_train, y_train), (x_val, y_val)
        
        data_train, data_val = cls.split_data(data_train_val, split=0.875)

        x_train = cls.meier_to_integer(data_train['Modified sequence'], max_sequence_length = max_sequence_length)
        y_train = data_train['CCS'].to_numpy()
        x_val = cls.meier_to_integer(data_val['Modified sequence'], max_sequence_length = max_sequence_length)
        y_val = data_val['CCS'].to_numpy()
        
        if dataset == 'meier':
            x_test = cls.meier_to_integer(data_test['Modified_sequence'], max_sequence_length = max_sequence_length)
            y_test = data_test['CCS'].to_numpy()
        else:
            x_test = cls.meier_to_integer(data_test['Modified sequence'], max_sequence_length = max_sequence_length)
            y_test = data_test['CCS'].to_numpy()
        
        np.save(f'./data/meier_2021/test/{dataset}_x_test.npy', x_test)
        np.save(f'./data/meier_2021/test/{dataset}_y_test.npy', y_test)

        return (x_train, y_train), (x_val, y_val)

    @classmethod
    def load_training_transformer(cls, dataset):
        return cls.load_meier(dataset)

    @classmethod
    def load_testing_transformer(cls, dataset):
        if dataset == 'train': print("No test data for 'train' use 'meier' instead"); exit(0)
        x_test = np.load(f'./data/meier_2021/test/{dataset}_x_test.npy')
        y_test = np.load(f'./data/meier_2021/test/{dataset}_y_test.npy')

        return (x_test, y_test)

def load_test_data(data, input_file, seq_header, rt_header, CLS, seq_length):
    if input_file is not None:
        print('Incomplete')
        exit(0)
    elif data in IONMOB_DATASETS:
        x_test, y_test = data_ionmob.load_testing_transformer(data)
        all_peps = data_ionmob.integer_to_sequence_phospho(x_test)
    elif data in MEIER_DATASETS:
        x_test, y_test = data_meier.load_testing_transformer(data)
        all_peps = data_meier.integer_to_sequence(x_test)               
    else:
        print('Unknown model')
        exit(0)

    print(seq_length, x_test.shape)
    assert seq_length >= x_test.shape[1]
    x_test = np.concatenate((np.full((x_test.shape[0], 1), CLS), x_test), axis = 1)    
    if seq_length > x_test.shape[1]:        
        x_test = np.concatenate((x_test, np.full((x_test.shape[0],  (seq_length - x_test.shape[1])), CLS)), axis = 1)
    assert seq_length == x_test.shape[1]
    return torch.tensor(x_test), torch.tensor(y_test).unsqueeze(1), all_peps
    
class DatasetCCS():

    def __init__(self, dataset, batch_size, epochs, device_type = 'cpu', device = 'cpu', ionmob_full = False):
        if dataset in IONMOB_DATASETS:
            if ionmob_full:
                (x_train, y_train), (x_val, y_val) = data_ionmob.load_training_transformer(dataset, ionmob_full)
            else:
                (x_train, y_train), (x_val, y_val) = data_ionmob.load_training_transformer(dataset)
        elif dataset in MEIER_DATASETS:
            (x_train, y_train), (x_val, y_val) = data_meier.load_training_transformer(dataset)
        else: 
            print('Unknown data')
            exit(0)
        
        min_val = min(y_train.min(), y_val.min()) - 0.01
        max_val = max(y_train.max(), y_val.max()) + 0.01

        y_train = min_max_scale(y_train, min = min_val, max = max_val)
        y_val = min_max_scale(y_val, min = min_val, max = max_val)

        self.CLS = x_train.max() + 1
        x_train = np.concatenate((np.full((x_train.shape[0], 1), self.CLS), x_train), axis = 1)
        x_val = np.concatenate((np.full((x_val.shape[0], 1), self.CLS), x_val), axis = 1)

        train_tensor = TensorDataset(torch.tensor(x_train), torch.tensor(y_train).unsqueeze(1))        
        self.loader_train = DataLoader(train_tensor, batch_size=batch_size, shuffle=True,drop_last=True)
        print(f"Total number of train batches: {len(self.loader_train)}")
        
        val_tensor = TensorDataset(torch.tensor(x_val), torch.tensor(y_val).unsqueeze(1))
        self.loader_val = DataLoader(val_tensor, batch_size=batch_size, shuffle=True,drop_last=True)
        print(f"Total number of val batches: {len(self.loader_val)}")

        print('CLS =', self.CLS)

        self.train_iter = iter(self.loader_train)
        self.val_iter =  iter(self.loader_val)
        self.min_val = min_val
        self.max_val = max_val
        self.block_size = x_train.shape[1]
        self.vocab_size = self.CLS + 1
        self.device_type = device_type
        self.device = device
        self.epoch = epochs

    def get_batch(self, split):
        if split == 'train':
            try:
                x,y = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.loader_train)   
                print(f"end of epoch {self.epoch}")
                self.epoch -= 1         
                x,y = next(self.train_iter)
        else:
            try:
                x,y = next(self.val_iter)
            except StopIteration:
                self.val_iter = iter(self.loader_val) 
                x,y = next(self.val_iter)

        if self.device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y