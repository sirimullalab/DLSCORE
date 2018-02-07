#Creator: Daniel Castaneda Mogollon
#Date: 02/02/2018 (first created)
#Purpose of this code: This code will get inside the DUDE molecule folder, and it will also get inside each protein folder. After that, it will extract
#the value given by the 21st network (the DLS score network), and the value of the average given by the 20 networks from NNScore. FInally, it will
#write a file to store the 60 proteins' (in this case they were 60 readable proteins with ligands) scores.

import os
import shutil
import re
path = '/home/exx/Downloads/all/'
os.chdir('/home/exx/Downloads/all')                                                                 #Changing path where the files are
folders = os.listdir(path)                                                                          #Listing all the folders in the directory
list_names=[]                                                                                       #A list for all the proteins analyzed by DLS score
list_DLS_scores=[]                                                                                  #DLS scores
list_scores=[]                                                                                      #A list for the average score of the 20 networks given by NNScore
for f in folders:
    if f.endswith('sh') or f.endswith('.pdb') or f.endswith('.py') or f.endswith('pdbqt'):          #Excluding the files that are not .txt with the scores
        y=5                                                                                         #It does nothing
    else:
        try:
            os.chdir(path+f)                                                                        #Getting inside a specific folder
            #os.rename('score_output_DLS2.01.01.csv',f+'_score.txt')                                #Converting from .csv to .txt
            #shutil.copy(f+'_score.txt','/home/exx/PycharmProjects/DLS_score')                      #Copying the files to Pycharm
            with open(f+'_score.txt') as x:
                for line in x:
                    if line.startswith('	Network #21'):                                          #Reading the line with the DLS score
                        split = re.split(' ',line)
                    if line.startswith('	Average Score:'):                                       #Reading the line wit the average score from NNScore
                        split2 = re.split(' ',line)
                        print (f + ' has a score of ' + str(split[6]) + ' and the average of the other networks is ' + str(split2[8]))
                        list_names.append(f)
                        list_DLS_scores.append(split[6])
                        list_scores.append(split2[8])
        except:
            print(str(f)+'_score.txt not found')

file2 = open('scores_DLS.txt','w')                                                                  #A file where I will store all the scores from the 60 proteins analyzed by DLS and NNScore
os.chdir('/home/exx/PycharmProjects/DLS_score')
for i in range(0,len(list_names),1):
    file2.write(list_names[i]+'\t')
    file2.write(list_DLS_scores[i]+'\t')
    file2.write(list_scores[i]+'\t\n')