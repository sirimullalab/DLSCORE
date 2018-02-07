#Creator: Daniel Castaneda Mogollon
#Date: 02/06/2018 (date created)
#Purpose: This code was created to print all the statements to run the DLS score function. The reason behind this
#is to run launcher in a .sh file, using a 'for loop' would have crashed launcher, and thus the reason of creating
#4,000 statements.

import os
import re

path_to = '/home/exx/Desktop/all/general-set-except-refined'                            #Specifying the path to access to the set
os.chdir(path_to)                                                                       #Moving into that path
content = os.listdir(path_to)                                                           #Storing all the content in that folder
for f in content:                                                                       #Moving along all the content in a loop
    match = re.findall('\w{4}$',f)                                                      #This regex looks for all files that have exactly four alphanumeric characters (pdbID in my general-set folder
    if match:                                                                           #Conditional statement to print a folder with four characters (for some reason it was reading empty folders)
        path_to2 = path_to+'/'+str(match[0])                                            #New path inside every protein folder
        print('python NNScore2.01.01_csv.py -receptor ' +path_to2+'/'+str(match[0])+    #Prints the statement to run DLS score in the terminal.
              '_protein.pdbqt'+' -ligand ' +path_to2+'/'+str(match[0]+
              '_ligand.pdbqt -vina_executable /home/exx/Hassan/autodock_vina/bin/vina'))