# DLSCORE: A deep learning based scoring function for predicting protein-ligand binding affinity.

Purpose: The main purpose of DLSCORE is to accurately predict binding affinities between a protein (target) and a ligand. 

DLSCORE is an ensemble of neural networks, trained on the recent release of the refined PDBBind data(v2016) using BINding ANAlyzer (BINANA ) descriptors. The dataset was divided into training and cross-validation with a total of 2,792 protein-ligand complexes, while the testing stage was performed with a total of 300.  



DLSCORE obtained a Pearson R2 of 0.95 for our training set and 0.82 for the testing stage.


## Prerequisites

## Installing

## Running the tests

We recommend to use the proteins and ligands (in a .pdbqt format) from [tests](https://github.com/sirimullalab/DLSCORE/tree/master/tests) to try DLSCORE. When running it, it should look like this:

`python dlscore.py -r receptor1/receptor1.pdbqt -l ligand1/ligand1.pdbqt -v path/to/vina -n 10`

Where:
````
-r stands for the receptor or protein (MUST BE in a pdbqt format)
-l stands for the ligand (MUST BE in a pdbqt format)
-v is the path to Autodock Vina
-n is the desired number of networks
````

We recommend setting `-n` to 10, as it has been shown to give the optimum results (highest Pearson, Spearman and Kendall correlation coefficients, and lowest RMSE and MAE values).

The command to run DLSCORE should look this:

`
python dlscore.py -r 1erb/1erb_protein.pdbqt -l 1erb/1erb_ligand.pdbqt -v autodock_vina_1_1_2_linux_x86/bin/vina -n 10
`

And the output (with some parameter and warning messages displayed) should be:

`
[{'dlscore': [7.0548329, 6.6804061, 7.331718, 7.5543647, 7.4937844, 7.2915564, 6.8153844, 7.3344746, 6.8370795, 6.8463974], 'nnscore': [7.528647808595601, 6.841528068839551, 8.970434997068462, 8.652536458379778, 7.5314641378295235, 6.567334597110879, 8.493516759548358, 7.6294518800050355, 6.909480876402764, 8.603892203786629, 10.735374727080952, 7.269388390573901, 8.861930933915144, 6.566368370705019, 6.649809561604744, 6.611589875000943, 5.496196964085726, 7.70903543971305, 6.70953530232466, 7.821874753600213], 'pdb_id': '1erb', 'vina_output': [-8.31353, 56.54909, 1495.07589, 0.50669, 68.97156, 0.0]}]
`

DLSCORE will be producing the number of networks specified with in `-n`, NNScore 2.0 will display 20, and vina 6. The output thrown by DLSCORE and NNScore 2.0 are pKd values, while Vina gives delta G (kcal/mol)

The same applies for the rest of the proteins and ligands. To see the rest of the protein-ligand complexes for our dataset (300 from PDBbind v.2016 refined-set), please take a look at [results].


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


