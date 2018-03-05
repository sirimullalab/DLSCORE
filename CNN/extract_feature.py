##########################################################
# This program -                                         #
#  - reads the pdb-bind dataset from a csv file          #
#  - reads the molecule object obtained by htmd library  #
#  - extracts features from molecule object and saves    #
#    in features.pickle file										#
#                                                        #
# Author: Mahmudulla Hassan                              #
# Computer Aided Drug Discovery Lab, UTEP                #
# Date: February 15th, 2018                              #
##########################################################


import htmd.ui as ht
import htmd.molecule.voxeldescriptors as vd
import csv
from tqdm import *
import os
import pickle
import numpy as np
import multiprocessing as mp
import math

print('Available CPUs: ', mp.cpu_count())


# Read the data file and get all the pdb ids
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

# Load the molecule object using HTMLD module
# Descriptors: (‘hydrophobic’, ‘aromatic’, ‘hbond_acceptor’, ‘hbond_donor’, ‘positive_ionizable’, ‘negative_ionizable’, ‘metal’, ‘occupancies’)

def molecules():
    molecules = {}
    molecule_file = 'molecules_refined.pickle'
    if os.path.isfile(molecule_file):
        with open(molecule_file, 'rb') as pickle_file:
            molecules = pickle.load(pickle_file)
    else:
        with tqdm(total=len(pdb_ids)) as pbar:
            for i, _id in enumerate(pdb_ids):
                pbar.update(1)
                try:
                    molecules[_id] = ht.Molecule(_id)
                except:
                    pass

        # Write into a pickle file for future use
        with open(molecule_file, 'wb') as f:
            pickle.dump(molecules, f)
	
    return molecules


def chunks(l, n):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract_features(feat, keys, n):
    text = "Progressor #{:>2}".format(n)
    with tqdm(total=len(keys), desc=text, position=n) as pbar:
        for key in keys:
            try:
                feat[key], centers, N = vd.getVoxelDescriptors(molecules[key], buffer = 0)
                feat[key] = feat[key].reshape(N[0], N[1], N[2], -1)
                pbar.update(1)
            except:
                pass

# Prepare the process manager and the dictionary to save the descriptors
mgr = mp.Manager()
features = mgr.dict()

chunk_size = math.ceil(len(molecules.keys())/mp.cpu_count())
key_chunks = [k for k in chunks(list(molecules.keys()), chunk_size)]
# Comment out the following line to use all the CPUs.
# The lenght of the list is equal to the number of processes to be created.
key_chunks = key_chunks[:10]

# Create processes
jobs = [mp.Process(target=extract_features, args=(features, keys, n)) for (n, keys) in enumerate(key_chunks)]
print('Number of processes: ', len(jobs))

# Start the processes
for j in jobs:
    j.start()

# Collect the outputs and join together
for j in jobs:
    j.join()    

# Convert to a normal dictionary (It's weird that multiprocessing dict. doesn't get saved!!)
feat = {}

with tqdm(total=len(features.keys()), desc='Collecting features', position=len(jobs)+1) as pbar:
    for k in features.keys():
        feat[k] = features[k]
        pbar.update()

del features # No longer needed. Free the memory

# Write to the disk
with open('features.pickle', 'wb') as f:
    pickle.dump(feat, f)

