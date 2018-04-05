# DLSCORE: A deep learning based scoring function for predicting protein-ligand binding affinity.

Purpose: The main purpose of DLSCORE is to accurately predict binding affinities between a protein (target) and a ligand. 

DLSCORE is an ensemble of neural networks, trained on the recent release of the refined PDBBind data(v2016) using BINding ANAlyzer (BINANA ) descriptors. The dataset was divided into training and cross-validation with a total of 2,792 protein-ligand complexes, while the testing stage was performed with a total of 300.  



DLSCORE obtained a Pearson R2 of 0.95 for our training set and 0.82 for the testing stage.


## Prerequisites

## Installing

## Running the tests

We recommend to use the proteins and ligands (found in a pdbqt format) from [tests](https://github.com/sirimullalab/DLSCORE/tree/master/tests) to try DLSCORE. When running it, it should look like this:

`python dlscore.py -r receptor1/receptor1.pdbqt -l ligand1/ligand1.pdbqt -v path/to/vina -n 10`

Where:
````
-r stands for the receptor or protein (MUST BE in a pdbqt format)
-l stands for the ligand (MUST BE in a pdbqt format)
-v is the path to Autodock Vina
-n is the desired number of networks
````

We recommend setting `-n` to 10, as it has been shown to give the optimum results (highest Pearson, Spearman and Kendall correlation coefficients, and lowest RMSE and MAE values).

## Built with

## Contributing

[contributing]

## Authors
Mahmudulla Hassan
Olac Fuentes
Suman Sirimulla

See also the list of [contributors](https://github.com/sirimullalab/DLSCORE/blob/master/contributors) who made this project possible.

## Funding
This work is supported by Dr. Suman Sirimulla’s startup fund from UTEP School of Pharmacy.

## Acknowledgments
We would like to thank [Patrick Walters](https://github.com/PatWalters) for sharing his [metk](https://github.com/PatWalters/metk) code and strengthing our statistical results.


