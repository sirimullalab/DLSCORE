#######################################################
# A Launcher to run NNScore in parallel               #
#                                                     #
# Please edit the variable names according to the     #
#  requirements.                                      #
#######################################################

import glob
import re
import os
import sys

script_name = 'NNScore2.01_binana.py'
vina_executable = '/opt/autodock_vina_1_1_2_linux_x86/bin/vina'
ligands = sorted(glob.glob("/home/mhassan/proteins2.0/**/*_ligand.pdbqt", recursive=True))
receptors = sorted(glob.glob("/home/mhassan/proteins2.0/**/*_protein.pdbqt", recursive=True))
process_count = 100

# Get the pdb ids that have both ligand and protein files available.
lig_files = {}
rec_files = {}
for lig in ligands:
    f = re.findall('\w\w\w\w/', lig)
    pdb_id = f[len(f)-1].strip('/')
    lig_files[pdb_id] = lig

for rec in receptors:
    f = re.findall('\w\w\w\w/', rec)
    pdb_id = f[len(f)-1].strip('/')
    rec_files[pdb_id] = rec

pdb_ids = [k for k in lig_files if k in rec_files]
#print('Total pdb ids: ', len(pdb_ids))

# Comment the following line to execute NNScore for all the files
#pdb_ids = pdb_ids[:10]

result_count = 0
run_count = 0
counter = 0

commands = []

for id in pdb_ids:
    commands.append('python -W ignore ' + script_name + ' -receptor ' + rec_files[id] + ' -ligand ' + lig_files[id] + ' -vina_executable ' + vina_executable + ' & \n')
    counter = counter + 1
    if counter%process_count == 0:
        commands.append('wait\n')
f = open('batch_run.sh', 'w')

for com in commands:
    f.write(com)
#print('python launcher.py > batch_run.sh')
f.close()
os.system('bash batch_run.sh')

# for id in pdb_ids:
    # if not os.path.isfile('output/' + id + '_predicted_values.csv'):
        # commands.append('python -W ignore ' + script_name + ' -receptor ' + rec_files[id] + ' -ligand ' + lig_files[id] + ' -vina_executable ' + vina_executable + ' & \n')
        # counter = counter + 1
        # if counter%16 == 0:
            # commands.append('wait\n')
    # else:
        # result_count = result_count + 1


# if result_count < len(pdb_ids):
    # f = open('batch_run.sh', 'w')
    # for com in commands:
        # f.write(com)
    # #print('python launcher.py > batch_run.sh')
    # f.close()
    # os.system('bash batch_run.sh')
    # run_count = run_count + 1
    # if run_count > 3: # Total number of session 
        # sys.exit()
    
