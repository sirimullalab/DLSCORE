# DLSCORE: A deep learning based scoring function for predicting protein-ligand binding affinity.

Purpose: The main purpose of DLSCORE is to accurately predict binding affinities between a protein (target) and a ligand. 

DLSCORE is an ensemble of neural networks, trained on the recent release of the refined PDBBind data(v2016) using BINding ANAlyzer (BINANA ) descriptors. The dataset was divided into training and cross-validation with a total of 2,792 protein-ligand complexes, while the testing stage was performed with a total of 300.  

DLSCORE obtained a Pearson R2 of 0.95 for our training set and 0.82 for the testing stage.
 



<b>Requirements:</b> <br>
python 3.5 <br>
tensorflow <br>
keras <br>
numpy <br>
pandas <br>
json <br>
pickle <br>
scv <br>
scipy <br>

<b>Contributors:</b> <br>
Mahmudulla Hassan <br>
Olac Fuentes <br>
Suman Sirimulla <br>
