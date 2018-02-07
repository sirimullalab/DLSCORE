#Creator: Daniel Castaneda Mogollon
#Date: 02/07/2018 (date created)
#Purpose: The purpose of this code is to copy and paste the experimental values from PDB bind data and put them into the
#new generated file created by running the NNScore2.01.01_csv.py (with the DLS score)
import os
import shutil
import xlrd
from xlutils.copy import copy


path = '/home/exx/launcher-master/tests'                                                                        #Setting the path where the output csv file is
path2 = '/home/exx/PycharmProjects/DLS_score'                                                                   #Path to pycharm
os.chdir(path)                                                                                                  #Heading to launcher path
shutil.copy('predicted_values.xlsx',path2)                                                                      #Copying file to pycharm directory
book = xlrd.open_workbook('predicted_values.xlsx')                                                              #Opening the output file from DLS score
shutil.copy('/home/exx/Desktop/all/general-set-except-refined/experimental_pdb_binddata.xlsx',path2)            #Copying the experimental data from pdb bind to pycharm
os.chdir(path2)                                                                                                 #Setting path to pycharm and work from there
book2 = xlrd.open_workbook('experimental_pdb_binddata.xlsx')                                                    #Opening the experimental pdb bind data
worksheet1 = book.sheet_by_index(0)                                                                             #Setting the sheet to work from in the output Excel file
worksheet2 = book2.sheet_by_index(0)                                                                            #Setting the sheet to work from in pdb bind
book_out = copy(book)                                                                                           #This command copies the original workbook so we can edit on it
sheet_out = book_out.get_sheet(0)                                                                               #We open the first sheet of our output file
id_list = []                                                                                                    #Creating a list for the pdb bind ID
id_list2 = []                                                                                                   #ID list from the predicted_value csv file
value_list=[]                                                                                                   #Creating a list for the experimental values from pdb bind ID
for i in range(1,4632,1):                                                                                       #Loop that reads all the 4632 IDs from the file
    id_list.append(worksheet2.cell_value(i,0))                                                                  #Storing IDs from the file into the list
    value_list.append(worksheet2.cell_value(i,1))                                                               #Storing values from the IDs, and making sure they match to each one of them

for j in range(1,1692,1):                                                                                       #In this case we use this for loop to walk through the 1692 IDs from the output file
    id_list2.append(worksheet1.cell_value(j,0))                                                                 #We append the IDs into the list variable
    if id_list2[j-1] in id_list:                                                                                #We look if experimental file has the same ID from the output file
        index_number = (id_list.index(id_list2[j-1]))                                                           #If so, we look for the index from our previous list
        sheet_out.write(j,24,value_list[index_number])                                                          #We store the score from the experimental values into the csv output file from DLS score
book_out.save('predicted_values.xlsx')                                                                          #We save our edited file with the same name


#THIS PART IS OPTIONAL
for k in range(1,1692,1):
    if id_list2[k-1] in id_list:
        index_number2 = (id_list.index(id_list2[k-1]))
