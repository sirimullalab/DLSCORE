# DLSCORE: A deep learning based scoring function for predicting protein-ligand binding affinity.

Purpose: The main purpose of DLSCORE is to accurately predict binding affinities between a protein (target) and a ligand. 

DLSCORE is an ensemble of neural networks, trained on the recent release of the refined PDBBind data(v2016) using BINding ANAlyzer (BINANA ) descriptors. The dataset was divided into training and cross-validation with a total of 2,792 protein-ligand complexes, while the testing stage was performed with a total of 300.  



DLSCORE obtained a Pearson R2 of 0.95 for our training set and 0.82 for the testing stage.


## Prerequisites

## Installing

## Running the tests

We recommend to use the proteins and ligands (found in a pdbqt format) from [tests](https://github.com/sirimullalab/DLSCORE/tree/master/tests) to try DLSCORE. When running it, it should look like this:

## Built with

## Contributing

## Authors
Mahmudulla Hassan
Olac Fuentes
Suman Sirimulla

See also the list of [contributors](www.google.com) who made this project possible.

## Acknowledgments
We would like to thank [Patrick Walters](https://github.com/PatWalters) for sharing his [metk](https://github.com/PatWalters/metk) code and strengthing our statistical results.

<b>Contributors:</b> <br>
Mahmudulla Hassan <br>
Olac Fuentes <br>
Suman Sirimulla <br>
