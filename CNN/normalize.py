#############################################################
# This script does data normalization and saves the 		   #
#  output in a pickle (data_std.pickle) file					   #
# 																			   #
# Required files: refined_set_dataset.csv, features.pickle  #
# 																			   #
# Written by: Mahmudulla Hassan									   #
# Last  modified: 02/17/2018										   #
#############################################################

import pickle
import csv
import numpy as np
from tqdm import *
import multiprocessing as mp


def read_score():
    pdb_ids = []
    score = {}
    with open('refined_set_dataset.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # Skip the header
        for row in reader:
            pdb_ids.append(row[1])
            score[row[1]]= row[5]

    return score

# Read scores
print('Reading scores...')
score = read_score()

print('Loading features...')
with open('features_1.pickle', 'rb') as f:
    features = pickle.load(f)

# Save the scores (that have features) in a dictionary
score = {k: score[k] for k in features.keys()}


# Select a small amount of data
feats = {}
sample_count = 30
for k in list(features.keys())[:sample_count]:
    feats[k] = features[k]
print('Sample size: ', len(feats))

max_dim = np.array([165, 285, 250, 8])

#data_x = np.empty([len(feats)] + list(max_dim))
#data_y = np.empty(len(feats))
data = {}
# Reshaping data to the max dimension
with tqdm(total=len(feats), desc='Reshaping') as pbar:
    for k in feats.keys():
        f = feats[k]
        f_shape = np.array(f.shape)
        dim_diff = max_dim - f_shape
        pad_dim = np.round(dim_diff / 2).astype(int)
        f1 = np.pad(f, 
                    [(pad_dim[0], dim_diff[0]-pad_dim[0]),
                     (pad_dim[1], dim_diff[1]-pad_dim[1]),
                     (pad_dim[2], dim_diff[2]-pad_dim[2]),
                     (pad_dim[3], dim_diff[3]-pad_dim[3])],
                    'constant')
        
        data[k] = f1
        pbar.update()

def get_mean(d, dim):
    values = np.zeros(dim)
    for val in tqdm(d.values(), desc='Mean'):
        values = values + val
    return values/len(d)

def get_sd(d, mean, dim):
    sum_sqrd_diff = np.zeros(dim)
    for val in tqdm(d.values(), desc='SD'):
        sum_sqrd_diff = sum_sqrd_diff + (val - mean)**2
    return np.sqrt(sum_sqrd_diff/len(d))


def normalize(data, dim):
    data_std = {}
    mean = get_mean(data, dim)
    sd = get_sd(data, mean, dim)
    for k in tqdm(data.keys(), desc='Normalizing'):
        data_std[k] = (data[k] - mean) / (sd + 0.00001)
        
    return data_std


data_std = normalize(data, max_dim)

with open('data_std.pickle', 'wb') as f:
    pickle.dump(data_std, f)
