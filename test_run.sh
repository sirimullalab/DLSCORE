# Single ligand
python dlscore2.py -r samples/10gs/10gs_protein.pdb -l samples/10gs/10gs_ligand.pdbqt -v /home/mhassan/autodock_vina_1_1_2_linux_x86/bin/vina -n 10 -w out.csv

# Multiple ligands
#python dlscore.py -r samples/ampc/receptor.pdbqt -l samples/ampc/ligand.pdbqt -v /home/mhassan/autodock_vina_1_1_2_linux_x86/bin/vina -n 10 -o out
#python NNScore2.01.02.py -receptor samples/10gs/10gs_protein.pdbqt -ligand samples/10gs/10gs_ligand.pdbqt -vina_executable  /opt/autodock_vina_1_1_2_linux_x86/bin/vina
