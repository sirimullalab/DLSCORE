# DLSCORE: A deep learning based scoring function for predicting protein-ligand binding affinity.

Purpose: The main purpose of DLSCORE is to accurately predict binding affinities between a protein (target) and a ligand. 

DLSCORE is an ensemble of neural networks, trained on the recent release of the refined PDBBind data(v2016) using BINding ANAlyzer (BINANA ) descriptors. 


## Prerequisites
- Tensorflow
- Keras
- MGLTools


## Running the tests
For a test run, type the following in the terminal: 

`
bash test_run.sh
`

DLSCORE will be producing the number of networks specified with in `-num_networks`, NNScore 2.0 will display 20, and vina 1 (plus 5) . The output thrown by DLSCORE and NNScore 2.0 are pKd values, while Vina gives delta G (kcal/mol)

The same applies for the rest of the proteins and ligands. To see the rest of the protein-ligand complexes for our dataset (300 from PDBbind v.2016 refined-set), please read our [research article](https://doi.org/10.26434/chemrxiv.6159143.v1)



## Contributing

If you wish to contribute, submit pull requests or read more about our code of conduct, please read [CONTRIBUTING.md](https://github.com/sirimullalab/DLSCORE/blob/master/CONTRIBUTING.md)

## Authors

Mahmudulla Hassan , Dr. Olac Fuentes, Dr. Suman Sirimulla, Daniel Castaneda Mogollon

See also the list of [contributors](https://github.com/sirimullalab/DLSCORE/blob/master/contributors) who made this project possible.

## Funding
This work is supported by Dr. Suman Sirimulla’s startup fund from UTEP School of Pharmacy.

## Acknowledgments
We would like to thank [Patrick Walters](https://github.com/PatWalters) for sharing his [metk](https://github.com/PatWalters/metk) code and strengthing our statistical results.

