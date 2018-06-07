# DLSCORE: A deep learning based scoring function for predicting protein-ligand binding affinity.

Purpose: The main purpose of DLSCORE is to accurately predict binding affinities between a protein (target) and a ligand. 

DLSCORE is an ensemble of neural networks, trained on the recent release of the refined PDBBind data(v2016) using BINding ANAlyzer (BINANA ) descriptors. 


## Prerequisites
- Tensorflow
- Keras
- MGLTools

## Running the script
For options, please type of the following in the terminal :

`
python dlscore.py -h
`

The output will be the following:

```
usage: dlscore.py [-h] --ligand LIGAND --receptor RECEPTOR
                  [--vina_executable VINA_EXECUTABLE]
                  [--num_networks NUM_NETWORKS] [--output OUTPUT_FILE]
                  [--network_type {refined,general}] [--verbose {0,1}]

Main script for running DLSCORE

optional arguments:
  -h, --help            show this help message and exit
  --ligand LIGAND       Ligand file. Supported file types: .pdb, .pdbqt, .mol2
  --receptor RECEPTOR   Receptor file. Supported file types: .pdb, .pdbqt,
                        .mol2
  --vina_executable VINA_EXECUTABLE
                        File path for Vina executable.
  --num_networks NUM_NETWORKS
                        Number of networks to use for prediction. Default:10
  --output OUTPUT_FILE  Name of the output file. Dafault: out.csv
  --network_type {refined,general}
                        DLSCORE has two types of weights trained on PDB-Bind
                        2016 refined set and general set (including refined).
                        Any of these two variants can be used. Possible
                        options are: general and refined. Dafault is set to
                        general.
  --verbose {0,1}       Verbose mode. False if 0, True if 1. Default is set to
                        False.
```

For a test run, type: 


`
bash test_run.sh
`

DLSCORE will be producing the number of networks specified with in `-num_networks`, NNScore 2.0 will display 20, and vina 1 (plus 5) . The output thrown by DLSCORE and NNScore 2.0 are pKd values, while Vina gives delta G (kcal/mol)

The same applies for the rest of the proteins and ligands. To see the rest of the protein-ligand complexes for our dataset (300 from PDBbind v.2016 refined-set), please read our [research article](https://doi.org/10.26434/chemrxiv.6159143.v1)



## Contributing

If you wish to contribute, submit pull requests or read more about our code of conduct, please read [CONTRIBUTING.md](https://github.com/sirimullalab/DLSCORE/blob/master/CONTRIBUTING.md)

## Authors

[Mahmudulla Hassan](https://github.com/hassanmohsin) , Dr. Olac Fuentes, Dr. Suman Sirimulla, Daniel Castaneda Mogollon

See also the list of [contributors](https://github.com/sirimullalab/DLSCORE/blob/master/contributors) who made this project possible.

## Funding
This work is supported by Dr. Suman Sirimulla’s startup fund from UTEP School of Pharmacy.

## Acknowledgments
We would like to thank [Patrick Walters](https://github.com/PatWalters) for sharing his [metk](https://github.com/PatWalters/metk) code and strengthing our statistical results.

