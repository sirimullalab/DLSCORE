#######################################################################
# DLSCORE.py                                                          #
#                                                                     #
# First version of DLSCORE                                            #
# Required deep learning libraries: Tensorflow and Keras              #
#                                                                     # 
# Output: A list of dictionaries                                      #
#                                                                     # 
# To run in a terminal, type "python dlscore.py -h" for options.      #
#                                                                     #
# To run within a script:                                             #
#  from dlscore import *                                              #
#  ds = dlscore(ligand='ligand_file.pdbqt',                           #
#       receptor='protein_file.pdbqt',                                #
#       vina_executable='vina exectuable file'                        #
#       nb_nets = 2 (Optional. Default value is 10))                  #
#  output = ds.get_output()                                           #
#                                                                     #
# Author: Mahmudulla Hassan                                           #
# Department of Computer Science and School of Pharmacy               #
# The University of Texas at El Paso, TX, USA                         #
# Last modified: 06/06/2018                                           #
#                                                                     #
#######################################################################

import numpy as np
import textwrap
import math
import os
import sys
import textwrap
import glob
import pickle
import csv
import re
import keras
from keras import metrics
from keras.models import Sequential, model_from_json
import keras.backend as K
import pickle
import h5py

# ignore warning from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get the directories
current_dir = os.path.dirname(os.path.realpath(__file__))
networks_dir = os.path.join(current_dir, "networks/refined")
mgltools_dir = os.path.join(current_dir, "mgltools")
vina_path = os.path.join(current_dir, "autodock_vina_1_1_2_linux_x86/bin/vina")
verbose = False

# The ffnet class was derived from the ffnet python package developed by Marek Wojciechowski (http://ffnet.sourceforge.net/).
# As Mr. Wojciechowski's program was released under the GPL license, NNScore is likewise GPL licensed.
class ffnet:

    def load(self, net_array):
       
        self.outno = net_array['outno']
        self.eni = net_array['eni']
        self.weights = net_array['weights']
        self.conec = net_array['conec']
        self.deo = net_array['deo']
        self.inno = net_array['inno']
        self.units = {}
        
        self.output = {}

    def normcall(self, input):
        #Takes single input pattern and returns network output
        #This have the same functionality as recall but now input and
        #output are normalized inside the function.
        #eni = [ai, bi], eno = [ao, bo] - parameters of linear mapping

        self.input = input

        #set input units
        self.setin()

        #print("FROM FFNET PROP: MAX INPUT: ", np.max(self.input))
        #print("INPUT: ", input)
        #print("UNITS: ", self.units)
    
        #propagate signals
        self.prop()
    
        #get output
        self.getout()

        return self.output[1]

    def setin(self):
        #normalize and set input units
    
        for k in range(1,len(self.inno) + 1):
            self.units[self.inno[k]] = self.eni[k][1] * self.input[k-1] + self.eni[k][2] # because self.input is a python list, and the others were inputted from a fortran-format file
            #print("K: ", k, " len(self.inno): ", len(self.inno), " self.inno[k]: ", self.inno[k], " self.eni[k][1]: ", self.eni[k][1], " self.eni[k][2]: ", self.eni[k][2])
            #print("input[", k-1, "] = ", self.input[k-1])

    def prop(self):
        #Gets conec and units with input already set 
        #and calculates all activations.
        #Identity input and sigmoid activation function for other units
        #is assumed
    
        #propagate signals with sigmoid activation function
        if len(self.conec) > 0:
            ctrg = self.conec[1][2]
            self.units[ctrg] = 0.0
            for xn in range(1,len(self.conec) + 1):
                src = self.conec[xn][1]
                trg = self.conec[xn][2]
                # if next unit
                if trg != ctrg:
                    self.units[ctrg] = 1.0/(1.0+math.exp(-self.units[ctrg]))
                    ctrg = trg
                    if src == 0: # handle bias
                        self.units[ctrg] = self.weights[xn]
                    else:
                        self.units[ctrg] = self.units[src] * self.weights[xn]
                else:
                    if src == 0: # handle bias
                        self.units[ctrg] = self.units[ctrg] + self.weights[xn]
                    else:
                        self.units[ctrg] = self.units[ctrg] + self.units[src] * self.weights[xn]
            self.units[ctrg] = 1.0/(1.0+math.exp(-self.units[ctrg])) # for last unit
    
    def getout(self):
        #get and denormalize output units
    
        for k in range(1,len(self.outno)+1):
            self.output[k] = self.deo[k][1] * self.units[self.outno[k]] + self.deo[k][2]


def dl_nets(nb_nets):
    """ Yields feed forward nerual nets from the network directory """
    # Read the networks
    with open(os.path.join(networks_dir, "sorted_models.pickle"), 'rb') as f:
        model_files = pickle.load(f)
    with open(os.path.join(networks_dir, "sorted_weights.pickle"), 'rb') as f:
        weight_files = pickle.load(f)
    
    assert(len(model_files) == len(weight_files)),         'Number of model files and the weight files are not the same.'
    for i, (model, weight) in enumerate(zip(model_files, weight_files)):
        if i==nb_nets:
            break
        # Load the network
        with open(os.path.join(networks_dir, model), 'r') as json_file:
            loaded_model = model_from_json(json_file.read())
            
        # Load the weights
        loaded_model.load_weights(os.path.join(networks_dir, weight))
        
        # Compile the network
        #loaded_model.compile(
        #    loss='mean_squared_error',
        #    optimizer=keras.optimizers.Adam(lr=0.001),
        #    metrics=[metrics.mse])
        
        yield loaded_model


class point:
    x=99999.0
    y=99999.0
    z=99999.0
    
    def __init__ (self, x, y ,z):
        self.x = x
        self.y = y
        self.z = z

    def copy_of(self):
        return point(self.x, self.y, self.z)

    def dist_to(self,apoint):
        return math.sqrt(math.pow(self.x - apoint.x,2) + math.pow(self.y - apoint.y,2) + math.pow(self.z - apoint.z,2))

    def Magnitude(self):
        return self.dist_to(point(0,0,0))
        
    def CreatePDBLine(self, index):

        output = "ATOM "
        output = output + str(index).rjust(6) + "X".rjust(5) + "XXX".rjust(4)
        output = output + ("%.3f" % self.x).rjust(18)
        output = output + ("%.3f" % self.y).rjust(8)
        output = output + ("%.3f" % self.z).rjust(8)
        output = output + "X".rjust(24) 
        return output

class atom:
        
    def __init__ (self):
        self.atomname = ""
        self.residue = ""
        self.coordinates = point(99999, 99999, 99999)
        self.element = ""
        self.PDBIndex = ""
        self.line=""
        self.atomtype=""
        self.IndeciesOfAtomsConnecting=[]
        self.charge = 0
        self.resid = 0
        self.chain = ""
        self.structure = ""
        self.comment = ""
        
    def copy_of(self):
        theatom = atom()
        theatom.atomname = self.atomname 
        theatom.residue = self.residue 
        theatom.coordinates = self.coordinates.copy_of()
        theatom.element = self.element 
        theatom.PDBIndex = self.PDBIndex 
        theatom.line= self.line
        theatom.atomtype= self.atomtype
        theatom.IndeciesOfAtomsConnecting = self.IndeciesOfAtomsConnecting[:]
        theatom.charge = self.charge 
        theatom.resid = self.resid 
        theatom.chain = self.chain 
        theatom.structure = self.structure 
        theatom.comment = self.comment
        
        return theatom

    def CreatePDBLine(self, index):
        
        output = "ATOM "
        output = output + str(index).rjust(6) + self.atomname.rjust(5) + self.residue.rjust(4)
        output = output + ("%.3f" % self.coordinates.x).rjust(18)
        output = output + ("%.3f" % self.coordinates.y).rjust(8)
        output = output + ("%.3f" % self.coordinates.z).rjust(8)
        output = output + self.element.rjust(24) 
        return output

    def NumberOfNeighbors(self):
        return len(self.IndeciesOfAtomsConnecting)

    def AddNeighborAtomIndex(self, index):
        if not (index in self.IndeciesOfAtomsConnecting):
            self.IndeciesOfAtomsConnecting.append(index)
    
    def SideChainOrBackBone(self): # only really applies to proteins, assuming standard atom names
        if self.atomname.strip() == "CA" or self.atomname.strip() == "C" or self.atomname.strip() == "O" or self.atomname.strip() == "N":
            return "BACKBONE"
        else:
            return "SIDECHAIN"
    
    def ReadPDBLine(self, Line):
        self.line = Line
        self.atomname = Line[11:16].strip()
        
        if len(self.atomname)==1:
            self.atomname = self.atomname + "  "
        elif len(self.atomname)==2:
            self.atomname = self.atomname + " "
        elif len(self.atomname)==3:
            self.atomname = self.atomname + " " # This line is necessary for babel to work, though many PDBs in the PDB would have this line commented out
        
        self.coordinates = point(float(Line[30:38]), float(Line[38:46]), float(Line[46:54]))
        
        # now atom type (for pdbqt)
        self.atomtype = Line[77:79].strip().upper()

        if Line[69:76].strip() != "":
            self.charge = float(Line[69:76])
        else:
            self.charge = 0.0
        
        if self.element == "": # try to guess at element from name
            two_letters = self.atomname[0:2].strip().upper()
            if two_letters=='BR':
                self.element='BR'
            elif two_letters=='CL':
                self.element='CL'
            elif two_letters=='BI':
                self.element='BI'
            elif two_letters=='AS':
                self.element='AS'
            elif two_letters=='AG':
                self.element='AG'
            elif two_letters=='LI':
                self.element='LI'
            #elif two_letters=='HG':
            #    self.element='HG'
            elif two_letters=='MG':
                self.element='MG'
            elif two_letters=='MN':
                self.element='MN'
            elif two_letters=='RH':
                self.element='RH'
            elif two_letters=='ZN':
                self.element='ZN'
            elif two_letters=='FE':
                self.element='FE'
            else: #So, just assume it's the first letter.
                # Any number needs to be removed from the element name
                self.element = self.atomname
                self.element = self.element.replace('0','')
                self.element = self.element.replace('1','')
                self.element = self.element.replace('2','')
                self.element = self.element.replace('3','')
                self.element = self.element.replace('4','')
                self.element = self.element.replace('5','')
                self.element = self.element.replace('6','')
                self.element = self.element.replace('7','')
                self.element = self.element.replace('8','')
                self.element = self.element.replace('9','')
                self.element = self.element.replace('@','')

                self.element = self.element[0:1].strip().upper()
                
        self.PDBIndex = Line[6:12].strip()
        self.residue = Line[16:20]
        self.residue = " " + self.residue[-3:] # this only uses the rightmost three characters, essentially removing unique rotamer identification

        if Line[23:26].strip() != "": self.resid = int(Line[23:26])
        else: self.resid = 1

        self.chain = Line[21:22]
        if self.residue.strip() == "": self.residue = " MOL"

class PDB:

    def __init__ (self):
        self.AllAtoms={}
        self.NonProteinAtoms = {}
        self.max_x = -9999.99
        self.min_x = 9999.99
        self.max_y = -9999.99
        self.min_y = 9999.99
        self.max_z = -9999.99
        self.min_z = 9999.99
        self.rotateable_bonds_count = 0
        self.functions = MathFunctions()
        self.protein_resnames = ["ALA", "ARG", "ASN", "ASP", "ASH", "ASX", "CYS", "CYM", "CYX", "GLN", "GLU", "GLH", "GLX", "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
        self.aromatic_rings = []
        self.charges = [] # a list of points
        self.OrigFileName = ""

    def LoadPDB_from_file(self, FileName, line_header=""):
    
        self.line_header=line_header
        
        # Now load the file into a list
        file = open(FileName,"r")
        lines = file.readlines()
        file.close()
        self.LoadPDB_from_list(lines, self.line_header)

    def LoadPDB_from_list(self, lines, line_header=""):

        self.line_header=line_header

        autoindex = 1

        self.__init__()

        atom_already_loaded = [] # going to keep track of atomname_resid_chain pairs, to make sure redundants aren't loaded. This basically
                                 # gets rid of rotomers, I think.

        for t in range(0,len(lines)):
            line=lines[t]

            if "between atoms" in line and " A " in line:
                    self.rotateable_bonds_count = self.rotateable_bonds_count + 1

            if len(line) >= 7:
                if line[0:4]=="ATOM" or line[0:6]=="HETATM": # Load atom data (coordinates, etc.)
                    TempAtom = atom()
                    TempAtom.ReadPDBLine(line)

                    key = TempAtom.atomname.strip() + "_" + str(TempAtom.resid) + "_" + TempAtom.residue.strip() + "_" + TempAtom.chain.strip() # this string unique identifies each atom

                    if key in atom_already_loaded and TempAtom.residue.strip() in self.protein_resnames: # so this is a receptor atom that has already been loaded once
                        print(self.line_header + "WARNING: Duplicate receptor atom detected: \"" + TempAtom.line.strip()+ "\". Not loading this duplicate.")
                        #print ""

                    if not key in atom_already_loaded or not TempAtom.residue.strip() in self.protein_resnames: # so either the atom hasn't been loaded, or else it's a non-receptor atom
                                                                                                        # so note that non-receptor atoms can have redundant names, but receptor atoms cannot.
                                                                                                        # This is because protein residues often contain rotamers
                        atom_already_loaded.append(key) # so each atom can only be loaded once. No rotamers.
                        self.AllAtoms[autoindex] = TempAtom # So you're actually reindexing everything here.
                        if not TempAtom.residue[-3:] in self.protein_resnames: self.NonProteinAtoms[autoindex] = TempAtom

                        autoindex = autoindex + 1

        self.CheckProteinFormat()

        self.CreateBondsByDistance() # only for the ligand, because bonds can be inferred based on atomnames from PDB
        self.assign_aromatic_rings()
        self.assign_charges()

    def printout(self, thestring):
        lines = textwrap.wrap(thestring, 80)
        for line in lines:
            print(line)
            
    def SavePDB(self, filename):
        f = open(filename, 'w')
        towrite = self.SavePDBString()
        if towrite.strip() == "": towrite = "ATOM      1  X   XXX             0.000   0.000   0.000                       X" # just so no PDB is empty, VMD will load them all
        f.write(towrite)
        f.close()
    
    def SavePDBString(self):

        ToOutput = ""
        
        # write coordinates
        for atomindex in self.AllAtoms:
            ToOutput = ToOutput + self.AllAtoms[atomindex].CreatePDBLine(atomindex) + "\n"
            
        return ToOutput
    
    def AddNewAtom(self, atom):
        
        # first get available index
        t = 1
        while t in self.AllAtoms.keys():
            t = t + 1
    
        # now add atom
        self.AllAtoms[t] = atom
    
    def connected_atoms_of_given_element(self, index, connected_atom_element):
        atom = self.AllAtoms[index]
        connected_atoms = []
        for index2 in atom.IndeciesOfAtomsConnecting:
            atom2 = self.AllAtoms[index2]
            if atom2.element == connected_atom_element:
                connected_atoms.append(index2)
        return connected_atoms
    
    def connected_heavy_atoms(self, index):
        atom = self.AllAtoms[index]
        connected_atoms = []
        for index2 in atom.IndeciesOfAtomsConnecting:
            atom2 = self.AllAtoms[index2]
            if atom2.element != "H": connected_atoms.append(index2)
        return connected_atoms

    def CheckProteinFormat(self):
        curr_res = ""
        first = True
        residue = []
        
        for atom_index in self.AllAtoms:
            atom = self.AllAtoms[atom_index]
            
            key = atom.residue + "_" + str(atom.resid) + "_" + atom.chain
            
            if first == True:
                curr_res = key
                first = False
                
            if key != curr_res: 

                self.CheckProteinFormat_process_residue(residue, last_key)
                
                residue = []
                curr_res = key
            
            residue.append(atom.atomname.strip())
            last_key = key
        
        self.CheckProteinFormat_process_residue(residue, last_key)


    def CheckProteinFormat_process_residue(self, residue, last_key): 
        temp = last_key.strip().split("_")
        resname = temp[0]
        real_resname = resname[-3:]
        resid = temp[1]
        chain = temp[2]
                
        if real_resname in self.protein_resnames: # so it's a protein residue
            
            if not "N" in residue:
                self.printout('WARNING: There is no atom named "N" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine secondary structure. If this residue is far from the active site, this warning may not affect the NNScore.')
                print("")
            if not "C" in residue:
                self.printout('WARNING: There is no atom named "C" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine secondary structure. If this residue is far from the active site, this warning may not affect the NNScore.')
                print("")
            if not "CA" in residue:
                self.printout('WARNING: There is no atom named "CA" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine secondary structure. If this residue is far from the active site, this warning may not affect the NNScore.')
                print("")
            
            if real_resname == "GLU" or real_resname == "GLH" or real_resname == "GLX":
                if not "OE1" in residue:
                    self.printout('WARNING: There is no atom named "OE1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine salt-bridge interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "OE2" in residue:
                    self.printout('WARNING: There is no atom named "OE2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine salt-bridge interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")

            if real_resname == "ASP" or real_resname == "ASH" or real_resname == "ASX":
                if not "OD1" in residue:
                    self.printout('WARNING: There is no atom named "OD1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine salt-bridge interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "OD2" in residue:
                    self.printout('WARNING: There is no atom named "OD2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine salt-bridge interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
            
            if real_resname == "LYS" or real_resname == "LYN":
                if not "NZ" in residue:
                    self.printout('WARNING: There is no atom named "NZ" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-cation and salt-bridge interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
            
            if real_resname == "ARG":
                if not "NH1" in residue:
                    self.printout('WARNING: There is no atom named "NH1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-cation and salt-bridge interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "NH2" in residue:
                    self.printout('WARNING: There is no atom named "NH2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-cation and salt-bridge interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
            
            if real_resname == "HIS" or real_resname == "HID" or real_resname == "HIE" or real_resname == "HIP":
                if not "NE2" in residue:
                    self.printout('WARNING: There is no atom named "NE2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-cation and salt-bridge interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "ND1" in residue:
                    self.printout('WARNING: There is no atom named "ND1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-cation and salt-bridge interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
            
            if real_resname == "PHE":
                if not "CG" in residue:
                    self.printout('WARNING: There is no atom named "CG" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CD1" in residue:
                    self.printout('WARNING: There is no atom named "CD1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CD2" in residue:
                    self.printout('WARNING: There is no atom named "CD2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CE1" in residue:
                    self.printout('WARNING: There is no atom named "CE1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CE2" in residue:
                    self.printout('WARNING: There is no atom named "CE2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CZ" in residue:
                    self.printout('WARNING: There is no atom named "CZ" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
            
            if real_resname == "TYR":
                if not "CG" in residue:
                    self.printout('WARNING: There is no atom named "CG" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CD1" in residue:
                    self.printout('WARNING: There is no atom named "CD1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CD2" in residue:
                    self.printout('WARNING: There is no atom named "CD2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CE1" in residue:
                    self.printout('WARNING: There is no atom named "CE1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CE2" in residue:
                    self.printout('WARNING: There is no atom named "CE2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CZ" in residue:
                    self.printout('WARNING: There is no atom named "CZ" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
            
            if real_resname == "TRP":
                if not "CG" in residue:
                    self.printout('WARNING: There is no atom named "CG" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CD1" in residue:
                    self.printout('WARNING: There is no atom named "CD1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CD2" in residue:
                    self.printout('WARNING: There is no atom named "CD2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "NE1" in residue:
                    self.printout('WARNING: There is no atom named "NE1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CE2" in residue:
                    self.printout('WARNING: There is no atom named "CE2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CE3" in residue:
                    self.printout('WARNING: There is no atom named "CE3" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CZ2" in residue:
                    self.printout('WARNING: There is no atom named "CZ2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CZ3" in residue:
                    self.printout('WARNING: There is no atom named "CZ3" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CH2" in residue:
                    self.printout('WARNING: There is no atom named "CH2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
            
            if real_resname == "HIS" or real_resname == "HID" or real_resname == "HIE" or real_resname == "HIP":
                if not "CG" in residue:
                    self.printout('WARNING: There is no atom named "CG" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "ND1" in residue:
                    self.printout('WARNING: There is no atom named "ND1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CD2" in residue:
                    self.printout('WARNING: There is no atom named "CD2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "CE1" in residue:
                    self.printout('WARNING: There is no atom named "CE1" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
                if not "NE2" in residue:
                    self.printout('WARNING: There is no atom named "NE2" in the protein residue ' + last_key + '. Please use standard naming conventions for all protein residues. This atom is needed to determine pi-pi and pi-cation interactions. If this residue is far from the active site, this warning may not affect the NNScore.')
                    print("")
            
                        
    # Functions to determine the bond connectivity based on distance
    # ==============================================================
    
    def CreateBondsByDistance(self):
        for AtomIndex1 in self.NonProteinAtoms:
            atom1 = self.NonProteinAtoms[AtomIndex1]
            if not atom1.residue[-3:] in self.protein_resnames: # so it's not a protein residue            
                for AtomIndex2 in self.NonProteinAtoms:
                    if AtomIndex1 != AtomIndex2:
                        atom2 = self.NonProteinAtoms[AtomIndex2]
                        if not atom2.residue[-3:] in self.protein_resnames: # so it's not a protein residue
                            dist = self.functions.distance(atom1.coordinates, atom2.coordinates)
                            
                            if (dist < self.BondLength(atom1.element, atom2.element) * 1.2):
                                atom1.AddNeighborAtomIndex(AtomIndex2)
                                atom2.AddNeighborAtomIndex(AtomIndex1)

    def BondLength(self, element1, element2):
        
        '''Bond lengths taken from Handbook of Chemistry and Physics. The information provided there was very specific,
        so I tried to pick representative examples and used the bond lengths from those. Sitautions could arise where these
        lengths would be incorrect, probably slight errors (<0.06) in the hundreds.'''
        
        distance = 0.0
        if element1 == "C" and element2 == "C": distance = 1.53
        if element1 == "N" and element2 == "N": distance = 1.425
        if element1 == "O" and element2 == "O": distance = 1.469
        if element1 == "S" and element2 == "S": distance = 2.048
        if (element1 == "C" and element2 == "H") or (element1 == "H" and element2 == "C"): distance = 1.059
        if (element1 == "C" and element2 == "N") or (element1 == "N" and element2 == "C"): distance = 1.469
        if (element1 == "C" and element2 == "O") or (element1 == "O" and element2 == "C"): distance = 1.413
        if (element1 == "C" and element2 == "S") or (element1 == "S" and element2 == "C"): distance = 1.819
        if (element1 == "N" and element2 == "H") or (element1 == "H" and element2 == "N"): distance = 1.009
        if (element1 == "N" and element2 == "O") or (element1 == "O" and element2 == "N"): distance = 1.463
        if (element1 == "O" and element2 == "S") or (element1 == "S" and element2 == "O"): distance = 1.577
        if (element1 == "O" and element2 == "H") or (element1 == "H" and element2 == "O"): distance = 0.967
        if (element1 == "S" and element2 == "H") or (element1 == "H" and element2 == "S"): distance = 2.025/1.5 # This one not from source sited above. Not sure where it's from, but it wouldn't ever be used in the current context ("AutoGrow")
        if (element1 == "S" and element2 == "N") or (element1 == "N" and element2 == "S"): distance = 1.633
    
        if (element1 == "C" and element2 == "F") or (element1 == "F" and element2 == "C"): distance = 1.399
        if (element1 == "C" and element2 == "CL") or (element1 == "CL" and element2 == "C"): distance = 1.790
        if (element1 == "C" and element2 == "BR") or (element1 == "BR" and element2 == "C"): distance = 1.910
        if (element1 == "C" and element2 == "I") or (element1 == "I" and element2 == "C"): distance=2.162
    
        if (element1 == "S" and element2 == "BR") or (element1 == "BR" and element2 == "S"): distance = 2.321
        if (element1 == "S" and element2 == "CL") or (element1 == "CL" and element2 == "S"): distance = 2.283
        if (element1 == "S" and element2 == "F") or (element1 == "F" and element2 == "S"): distance = 1.640
        if (element1 == "S" and element2 == "I") or (element1 == "I" and element2 == "S"): distance= 2.687
    
        if (element1 == "P" and element2 == "BR") or (element1 == "BR" and element2 == "P"): distance = 2.366
        if (element1 == "P" and element2 == "CL") or (element1 == "CL" and element2 == "P"): distance = 2.008
        if (element1 == "P" and element2 == "F") or (element1 == "F" and element2 == "P"): distance = 1.495
        if (element1 == "P" and element2 == "I") or (element1 == "I" and element2 == "P"): distance= 2.490
        if (element1 == "P" and element2 == "O") or (element1 == "O" and element2 == "P"): distance= 1.6 # estimate based on eye balling Handbook of Chemistry and Physics
    
        if (element1 == "N" and element2 == "BR") or (element1 == "BR" and element2 == "N"): distance = 1.843
        if (element1 == "N" and element2 == "CL") or (element1 == "CL" and element2 == "N"): distance = 1.743
        if (element1 == "N" and element2 == "F") or (element1 == "F" and element2 == "N"): distance = 1.406
        if (element1 == "N" and element2 == "I") or (element1 == "I" and element2 == "N"): distance= 2.2
    
        if (element1 == "SI" and element2 == "BR") or (element1 == "BR" and element2 == "SI"): distance = 2.284
        if (element1 == "SI" and element2 == "CL") or (element1 == "CL" and element2 == "SI"): distance = 2.072
        if (element1 == "SI" and element2 == "F") or (element1 == "F" and element2 == "SI"): distance = 1.636
        if (element1 == "SI" and element2 == "P") or (element1 == "P" and element2 == "SI"): distance= 2.264
        if (element1 == "SI" and element2 == "S") or (element1 == "S" and element2 == "SI"): distance= 2.145
        if (element1 == "SI" and element2 == "SI") or (element1 == "SI" and element2 == "SI"): distance= 2.359
        if (element1 == "SI" and element2 == "C") or (element1 == "C" and element2 == "SI"): distance= 1.888
        if (element1 == "SI" and element2 == "N") or (element1 == "N" and element2 == "SI"): distance= 1.743
        if (element1 == "SI" and element2 == "O") or (element1 == "O" and element2 == "SI"): distance= 1.631
        
        return distance;
    
    # Functions to identify positive charges
    # ======================================
    
    def assign_charges(self):
        # Get all the quartinary amines on non-protein residues (these are the only non-protein groups that will be identified as positively charged)
        AllCharged = []
        for atom_index in self.NonProteinAtoms:
            atom = self.NonProteinAtoms[atom_index]
            if atom.element == "MG" or atom.element == "MN" or atom.element == "RH" or atom.element == "ZN" or atom.element == "FE" or atom.element == "BI" or atom.element == "AS" or atom.element == "AG":
                    chrg = self.charged(atom.coordinates, [atom_index], True)
                    self.charges.append(chrg)
            
            if atom.element == "N":
                if atom.NumberOfNeighbors() == 4: # a quartinary amine, so it's easy
                    indexes = [atom_index]
                    indexes.extend(atom.IndeciesOfAtomsConnecting) 
                    chrg = self.charged(atom.coordinates, indexes, True) # so the indicies stored is just the index of the nitrogen and any attached atoms
                    self.charges.append(chrg)
                elif atom.NumberOfNeighbors() == 3: # maybe you only have two hydrogen's added, by they're sp3 hybridized. Just count this as a quartinary amine, since I think the positive charge would be stabalized.
                    nitrogen = atom
                    atom1 = self.AllAtoms[atom.IndeciesOfAtomsConnecting[0]]
                    atom2 = self.AllAtoms[atom.IndeciesOfAtomsConnecting[1]]
                    atom3 = self.AllAtoms[atom.IndeciesOfAtomsConnecting[2]]
                    angle1 = self.functions.angle_between_three_points(atom1.coordinates, nitrogen.coordinates, atom2.coordinates) * 180.0 / math.pi
                    angle2 = self.functions.angle_between_three_points(atom1.coordinates, nitrogen.coordinates, atom3.coordinates) * 180.0 / math.pi
                    angle3 = self.functions.angle_between_three_points(atom2.coordinates, nitrogen.coordinates, atom3.coordinates) * 180.0 / math.pi
                    average_angle = (angle1 + angle2 + angle3) / 3
                    if math.fabs(average_angle - 109.0) < 5.0:
                        indexes = [atom_index]
                        indexes.extend(atom.IndeciesOfAtomsConnecting)
                        chrg = self.charged(nitrogen.coordinates, indexes, True) # so indexes added are the nitrogen and any attached atoms.
                        self.charges.append(chrg)
            
            if atom.element == "C": # let's check for guanidino-like groups (actually H2N-C-NH2, where not CN3.)
                if atom.NumberOfNeighbors() == 3: # the carbon has only three atoms connected to it
                    nitrogens = self.connected_atoms_of_given_element(atom_index,"N")
                    if len(nitrogens) >= 2: # so carbon is connected to at least two nitrogens
                        # now we need to count the number of nitrogens that are only connected to one heavy atom (the carbon)
                        nitrogens_to_use = []
                        all_connected = atom.IndeciesOfAtomsConnecting[:]
                        not_isolated = -1
                        
                        for atmindex in nitrogens:
                            if len(self.connected_heavy_atoms(atmindex)) == 1:
                                nitrogens_to_use.append(atmindex)
                                all_connected.remove(atmindex)
                            
                        if len(all_connected) > 0: not_isolated = all_connected[0] # get the atom that connects this charged group to the rest of the molecule, ultimately to make sure it's sp3 hybridized

                        if len(nitrogens_to_use) == 2 and not_isolated != -1: # so there are at two nitrogens that are only connected to the carbon (and probably some hydrogens)

                            # now you need to make sure not_isolated atom is sp3 hybridized
                            not_isolated_atom = self.AllAtoms[not_isolated]
                            if (not_isolated_atom.element == "C" and not_isolated_atom.NumberOfNeighbors()==4) or (not_isolated_atom.element == "O" and not_isolated_atom.NumberOfNeighbors()==2) or not_isolated_atom.element == "N" or not_isolated_atom.element == "S" or not_isolated_atom.element == "P":
                            
                                pt = self.AllAtoms[nitrogens_to_use[0]].coordinates.copy_of()
                                pt.x = pt.x + self.AllAtoms[nitrogens_to_use[1]].coordinates.x
                                pt.y = pt.y + self.AllAtoms[nitrogens_to_use[1]].coordinates.y
                                pt.z = pt.z + self.AllAtoms[nitrogens_to_use[1]].coordinates.z
                                pt.x = pt.x / 2.0
                                pt.y = pt.y / 2.0
                                pt.z = pt.z / 2.0
                                
                                indexes = [atom_index]
                                indexes.extend(nitrogens_to_use)
                                indexes.extend(self.connected_atoms_of_given_element(nitrogens_to_use[0],"H"))
                                indexes.extend(self.connected_atoms_of_given_element(nitrogens_to_use[1],"H"))
                                
                                chrg = self.charged(pt, indexes, True) # True because it's positive
                                self.charges.append(chrg)
            
            if atom.element == "C": # let's check for a carboxylate
                if atom.NumberOfNeighbors() == 3: # a carboxylate carbon will have three items connected to it.
                    oxygens = self.connected_atoms_of_given_element(atom_index,"O")
                    if len(oxygens) == 2: # a carboxylate will have two oxygens connected to it.
                        # now, each of the oxygens should be connected to only one heavy atom (so if it's connected to a hydrogen, that's okay)
                        if len(self.connected_heavy_atoms(oxygens[0])) == 1 and len(self.connected_heavy_atoms(oxygens[1])) == 1:
                            # so it's a carboxylate! Add a negative charge.
                            pt = self.AllAtoms[oxygens[0]].coordinates.copy_of()
                            pt.x = pt.x + self.AllAtoms[oxygens[1]].coordinates.x
                            pt.y = pt.y + self.AllAtoms[oxygens[1]].coordinates.y
                            pt.z = pt.z + self.AllAtoms[oxygens[1]].coordinates.z
                            pt.x = pt.x / 2.0
                            pt.y = pt.y / 2.0
                            pt.z = pt.z / 2.0
                            chrg = self.charged(pt, [oxygens[0], atom_index, oxygens[1]], False)
                            self.charges.append(chrg)
            
            if atom.element == "P": # let's check for a phosphate or anything where a phosphorus is bound to two oxygens where both oxygens are bound to only one heavy atom (the phosphorus). I think this will get several phosphorus substances.
                oxygens = self.connected_atoms_of_given_element(atom_index,"O")
                if len(oxygens) >=2: # the phosphorus is bound to at least two oxygens
                    # now count the number of oxygens that are only bound to the phosphorus
                    count = 0
                    for oxygen_index in oxygens:
                        if len(self.connected_heavy_atoms(oxygen_index)) == 1: count = count + 1
                    if count >=2: # so there are at least two oxygens that are only bound to the phosphorus
                        indexes = [atom_index]
                        indexes.extend(oxygens)
                        chrg = self.charged(atom.coordinates, indexes, False)
                        self.charges.append(chrg)
            
            if atom.element == "S": # let's check for a sulfonate or anything where a sulfur is bound to at least three oxygens and at least three are bound to only the sulfur (or the sulfur and a hydrogen).
                oxygens = self.connected_atoms_of_given_element(atom_index,"O")
                if len(oxygens) >=3: # the sulfur is bound to at least three oxygens
                    # now count the number of oxygens that are only bound to the sulfur
                    count = 0
                    for oxygen_index in oxygens:
                        if len(self.connected_heavy_atoms(oxygen_index)) == 1: count = count + 1
                    if count >=3: # so there are at least three oxygens that are only bound to the sulfur
                        indexes = [atom_index]
                        indexes.extend(oxygens)
                        chrg = self.charged(atom.coordinates, indexes, False)
                        self.charges.append(chrg)
            
        # Now that you've found all the positive charges in non-protein residues, it's time to look for aromatic rings in protein residues
        curr_res = ""
        first = True
        residue = []
        
        for atom_index in self.AllAtoms:
            atom = self.AllAtoms[atom_index]
            
            key = atom.residue + "_" + str(atom.resid) + "_" + atom.chain
            
            if first == True:
                curr_res = key
                first = False
                
            if key != curr_res: 

                self.assign_charged_from_protein_process_residue(residue, last_key)
                
                residue = []
                curr_res = key
            
            residue.append(atom_index)
            last_key = key
        
        self.assign_charged_from_protein_process_residue(residue, last_key)

    def assign_charged_from_protein_process_residue(self, residue, last_key): 
        temp = last_key.strip().split("_")
        resname = temp[0]
        real_resname = resname[-3:]
        resid = temp[1]
        chain = temp[2]

        if real_resname == "LYS" or real_resname == "LYN": # regardless of protonation state, assume it's charged.
            for index in residue: 
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "NZ":
                    
                    # quickly go through the residue and get the hydrogens attached to this nitrogen to include in the index list
                    indexes = [index]
                    for index2 in residue:
                        atom2 = self.AllAtoms[index2]
                        if atom2.atomname.strip() == "HZ1": indexes.append(index2)
                        if atom2.atomname.strip() == "HZ2": indexes.append(index2)
                        if atom2.atomname.strip() == "HZ3": indexes.append(index2)
                    
                    chrg = self.charged(atom.coordinates, indexes, True)
                    self.charges.append(chrg)
                    break

        if real_resname == "ARG":
            charge_pt = point(0.0,0.0,0.0)
            count = 0.0
            indices = []
            for index in residue: 
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "NH1": 
                    charge_pt.x = charge_pt.x + atom.coordinates.x
                    charge_pt.y = charge_pt.y + atom.coordinates.y
                    charge_pt.z = charge_pt.z + atom.coordinates.z
                    indices.append(index)
                    count = count + 1.0
                if atom.atomname.strip() == "NH2": 
                    charge_pt.x = charge_pt.x + atom.coordinates.x
                    charge_pt.y = charge_pt.y + atom.coordinates.y
                    charge_pt.z = charge_pt.z + atom.coordinates.z
                    indices.append(index)
                    count = count + 1.0
                if atom.atomname.strip() == "2HH2": indices.append(index)
                if atom.atomname.strip() == "1HH2": indices.append(index)
                if atom.atomname.strip() == "CZ": indices.append(index)
                if atom.atomname.strip() == "2HH1": indices.append(index)
                if atom.atomname.strip() == "1HH1": indices.append(index)
                
            if count != 0.0:
                
                charge_pt.x = charge_pt.x / count
                charge_pt.y = charge_pt.y / count
                charge_pt.z = charge_pt.z / count
                
                if charge_pt.x != 0.0 or charge_pt.y != 0.0 or charge_pt.z != 0.0:
                    chrg = self.charged(charge_pt, indices, True)
                    self.charges.append(chrg)

        if real_resname == "HIS" or real_resname == "HID" or real_resname == "HIE" or real_resname == "HIP": # regardless of protonation state, assume it's charged. This based on "The Cation-Pi Interaction," which suggests protonated state would be stabalized. But let's not consider HIS when doing salt bridges.
            charge_pt = point(0.0,0.0,0.0)
            count = 0.0
            indices = []
            for index in residue: 
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "NE2": 
                    charge_pt.x = charge_pt.x + atom.coordinates.x
                    charge_pt.y = charge_pt.y + atom.coordinates.y
                    charge_pt.z = charge_pt.z + atom.coordinates.z
                    indices.append(index)
                    count = count + 1.0
                if atom.atomname.strip() == "ND1": 
                    charge_pt.x = charge_pt.x + atom.coordinates.x
                    charge_pt.y = charge_pt.y + atom.coordinates.y
                    charge_pt.z = charge_pt.z + atom.coordinates.z
                    indices.append(index)
                    count = count + 1.0
                if atom.atomname.strip() == "HE2": indices.append(index)
                if atom.atomname.strip() == "HD1": indices.append(index)
                if atom.atomname.strip() == "CE1": indices.append(index)
                if atom.atomname.strip() == "CD2": indices.append(index)
                if atom.atomname.strip() == "CG": indices.append(index)

            if count != 0.0:
                charge_pt.x = charge_pt.x / count
                charge_pt.y = charge_pt.y / count
                charge_pt.z = charge_pt.z / count
                if charge_pt.x != 0.0 or charge_pt.y != 0.0 or charge_pt.z != 0.0:
                    chrg = self.charged(charge_pt, indices, True)
                    self.charges.append(chrg)
    
        if real_resname == "GLU" or real_resname == "GLH" or real_resname == "GLX": # regardless of protonation state, assume it's charged. This based on "The Cation-Pi Interaction," which suggests protonated state would be stabalized.
            charge_pt = point(0.0,0.0,0.0)
            count = 0.0
            indices = []
            for index in residue: 
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "OE1": 
                    charge_pt.x = charge_pt.x + atom.coordinates.x
                    charge_pt.y = charge_pt.y + atom.coordinates.y
                    charge_pt.z = charge_pt.z + atom.coordinates.z
                    indices.append(index)
                    count = count + 1.0
                if atom.atomname.strip() == "OE2": 
                    charge_pt.x = charge_pt.x + atom.coordinates.x
                    charge_pt.y = charge_pt.y + atom.coordinates.y
                    charge_pt.z = charge_pt.z + atom.coordinates.z
                    indices.append(index)
                    count = count + 1.0
                if atom.atomname.strip() == "CD": indices.append(index)

            if count != 0.0:
                charge_pt.x = charge_pt.x / count
                charge_pt.y = charge_pt.y / count
                charge_pt.z = charge_pt.z / count
                if charge_pt.x != 0.0 or charge_pt.y != 0.0 or charge_pt.z != 0.0:
                    chrg = self.charged(charge_pt, indices, False) # False because it's a negative charge
                    self.charges.append(chrg)
    
        if real_resname == "ASP" or real_resname == "ASH" or real_resname == "ASX": # regardless of protonation state, assume it's charged. This based on "The Cation-Pi Interaction," which suggests protonated state would be stabalized.
            charge_pt = point(0.0,0.0,0.0)
            count = 0.0
            indices = []
            for index in residue: 
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "OD1": 
                    charge_pt.x = charge_pt.x + atom.coordinates.x
                    charge_pt.y = charge_pt.y + atom.coordinates.y
                    charge_pt.z = charge_pt.z + atom.coordinates.z
                    indices.append(index)
                    count = count + 1.0
                if atom.atomname.strip() == "OD2": 
                    charge_pt.x = charge_pt.x + atom.coordinates.x
                    charge_pt.y = charge_pt.y + atom.coordinates.y
                    charge_pt.z = charge_pt.z + atom.coordinates.z
                    indices.append(index)
                    count = count + 1.0
                if atom.atomname.strip() == "CG": indices.append(index)

            if count != 0.0:
                charge_pt.x = charge_pt.x / count
                charge_pt.y = charge_pt.y / count
                charge_pt.z = charge_pt.z / count
                if charge_pt.x != 0.0 or charge_pt.y != 0.0 or charge_pt.z != 0.0:
                    chrg = self.charged(charge_pt, indices, False) # False because it's a negative charge
                    self.charges.append(chrg)
    
    class charged():
        def __init__(self, coordinates, indices, positive):
            self.coordinates = coordinates
            self.indices = indices
            self.positive = positive # true or false to specifiy if positive or negative charge

    # Functions to identify aromatic rings
    # ====================================

    def add_aromatic_marker(self, indicies_of_ring):
        # first identify the center point
        points_list = []
        total = len(indicies_of_ring)
        x_sum = 0.0
        y_sum = 0.0
        z_sum = 0.0
        
        for index in indicies_of_ring:
            atom = self.AllAtoms[index]
            points_list.append(atom.coordinates)
            x_sum = x_sum + atom.coordinates.x
            y_sum = y_sum + atom.coordinates.y
            z_sum = z_sum + atom.coordinates.z
        
        if total == 0: return # to prevent errors in some cases
        
        center = point(x_sum / total, y_sum / total, z_sum / total)
 
        # now get the radius of the aromatic ring
        radius = 0.0
        for index in indicies_of_ring:
            atom = self.AllAtoms[index]
            dist = center.dist_to(atom.coordinates)
            if dist > radius: radius = dist
            
        # now get the plane that defines this ring
        if len(indicies_of_ring) < 3:
            return # to prevent an error in some cases. If there aren't three point, you can't define a plane
        elif len(indicies_of_ring) == 3:
            A = self.AllAtoms[indicies_of_ring[0]].coordinates
            B = self.AllAtoms[indicies_of_ring[1]].coordinates
            C = self.AllAtoms[indicies_of_ring[2]].coordinates
        elif len(indicies_of_ring) == 4:
            A = self.AllAtoms[indicies_of_ring[0]].coordinates
            B = self.AllAtoms[indicies_of_ring[1]].coordinates
            C = self.AllAtoms[indicies_of_ring[3]].coordinates
        else: # best, for 5 and 6 member rings
            A = self.AllAtoms[indicies_of_ring[0]].coordinates
            B = self.AllAtoms[indicies_of_ring[2]].coordinates
            C = self.AllAtoms[indicies_of_ring[4]].coordinates
        
        AB = self.functions.vector_subtraction(B,A)
        AC = self.functions.vector_subtraction(C,A)
        ABXAC = self.functions.CrossProduct(AB,AC)
        
        # formula for plane will be ax + by + cz = d
        x1 = self.AllAtoms[indicies_of_ring[0]].coordinates.x
        y1 = self.AllAtoms[indicies_of_ring[0]].coordinates.y
        z1 = self.AllAtoms[indicies_of_ring[0]].coordinates.z
        
        a = ABXAC.x
        b = ABXAC.y
        c = ABXAC.z
        d = a*x1 + b*y1 + c*z1
        
        ar_ring = self.aromatic_ring(center, indicies_of_ring, [a,b,c,d], radius)
        self.aromatic_rings.append(ar_ring)
                
    class aromatic_ring():
        def __init__(self, center, indices, plane_coeff, radius):
            self.center = center
            self.indices = indices
            self.plane_coeff = plane_coeff # a*x + b*y + c*z = dI think that
            self.radius = radius

    def assign_aromatic_rings(self):
        # Get all the rings containing each of the atoms in the ligand
        AllRings = []
        for atom_index in self.NonProteinAtoms:
            AllRings.extend(self.all_rings_containing_atom(atom_index))
        
        for ring_index_1 in range(len(AllRings)):
            ring1 = AllRings[ring_index_1]
            if len(ring1) != 0:
                for ring_index_2 in range(len(AllRings)):
                    if ring_index_1 != ring_index_2:
                        ring2 = AllRings[ring_index_2]
                        if len(ring2) != 0:
                            if self.set1_is_subset_of_set2(ring1, ring2) == True: AllRings[ring_index_2] = []

        while [] in AllRings: AllRings.remove([])
        
        # Now we need to figure out which of these ligands are aromatic (planar)
        
        for ring_index in range(len(AllRings)):
            ring = AllRings[ring_index]
            is_flat = True
            for t in range(-3, len(ring)-3):
                pt1 = self.NonProteinAtoms[ring[t]].coordinates
                pt2 = self.NonProteinAtoms[ring[t+1]].coordinates
                pt3 = self.NonProteinAtoms[ring[t+2]].coordinates
                pt4 = self.NonProteinAtoms[ring[t+3]].coordinates
                
                # first, let's see if the last atom in this ring is a carbon connected to four atoms. That would be a quick way of telling this is not an aromatic ring
                cur_atom = self.NonProteinAtoms[ring[t+3]]
                if cur_atom.element == "C" and cur_atom.NumberOfNeighbors() == 4:
                    is_flat = False
                    break
                
                # now check the dihedral between the ring atoms to see if it's flat
                angle = self.functions.dihedral(pt1, pt2, pt3, pt4) * 180 / math.pi
                if (angle > -165 and angle < -15) or (angle > 15 and angle < 165): # 15 degress is the cutoff #, ring[t], ring[t+1], ring[t+2], ring[t+3] # range of this function is -pi to pi
                    is_flat = False
                    break

                # now check the dihedral between the ring atoms and an atom connected to the current atom to see if that's flat too.
                for substituent_atom_index in cur_atom.IndeciesOfAtomsConnecting:
                    pt_sub = self.NonProteinAtoms[substituent_atom_index].coordinates
                    angle = self.functions.dihedral(pt2, pt3, pt4, pt_sub) * 180 / math.pi
                    if (angle > -165 and angle < -15) or (angle > 15 and angle < 165): # 15 degress is the cutoff #, ring[t], ring[t+1], ring[t+2], ring[t+3] # range of this function is -pi to pi
                        is_flat = False
                        break

            if is_flat == False: AllRings[ring_index] = []
            if len(ring) < 5: AllRings[ring_index] = [] # While I'm at it, three and four member rings are not aromatic
            if len(ring) > 6: AllRings[ring_index] = [] # While I'm at it, if the ring has more than 6, also throw it out. So only 5 and 6 member rings are allowed.



        while [] in AllRings: AllRings.remove([])
        
        for ring in AllRings:
            self.add_aromatic_marker(ring)
            
        # Now that you've found all the rings in non-protein residues, it's time to look for aromatic rings in protein residues
        curr_res = ""
        first = True
        residue = []
        
        for atom_index in self.AllAtoms:
            atom = self.AllAtoms[atom_index]
            
            key = atom.residue + "_" + str(atom.resid) + "_" + atom.chain
            
            if first == True:
                curr_res = key
                first = False
                
            if key != curr_res: 

                self.assign_aromatic_rings_from_protein_process_residue(residue, last_key)
                
                residue = []
                curr_res = key
            
            residue.append(atom_index)
            last_key = key
        
        self.assign_aromatic_rings_from_protein_process_residue(residue, last_key)

    def assign_aromatic_rings_from_protein_process_residue(self, residue, last_key): 
        temp = last_key.strip().split("_")
        resname = temp[0]
        real_resname = resname[-3:]
        resid = temp[1]
        chain = temp[2]

        if real_resname == "PHE":
            indicies_of_ring = []

            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CG": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CD1": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CE1": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CZ": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CE2": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CD2": indicies_of_ring.append(index)
                
            self.add_aromatic_marker(indicies_of_ring)

        if real_resname == "TYR": 
            indicies_of_ring = []

            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CG": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CD1": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CE1": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CZ": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CE2": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CD2": indicies_of_ring.append(index)
                
            self.add_aromatic_marker(indicies_of_ring)

        if real_resname == "HIS" or real_resname == "HID" or real_resname == "HIE" or real_resname == "HIP": 
            indicies_of_ring = []

            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CG": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "ND1": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CE1": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "NE2": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CD2": indicies_of_ring.append(index)
                
            self.add_aromatic_marker(indicies_of_ring)
        
        if real_resname == "TRP": 
            indicies_of_ring = []

            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CG": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CD1": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "NE1": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CE2": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CD2": indicies_of_ring.append(index)
            
            self.add_aromatic_marker(indicies_of_ring)

            indicies_of_ring = []

            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CE2": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CD2": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CE3": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CZ3": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CH2": indicies_of_ring.append(index)
            for index in residue: # written this way because order is important
                atom = self.AllAtoms[index]
                if atom.atomname.strip() == "CZ2": indicies_of_ring.append(index)
                
            self.add_aromatic_marker(indicies_of_ring)
        
    def set1_is_subset_of_set2(self, set1, set2):
        is_subset = True
        for item in set1:
            if not item in set2:
                is_subset = False
                break
        return is_subset
            
    def all_rings_containing_atom(self, index):
        
        AllRings = []
        
        atom = self.AllAtoms[index]
        for conneceted_atom in atom.IndeciesOfAtomsConnecting:
            self.ring_recursive(conneceted_atom, [index], index, AllRings)
 
        return AllRings
            
    def ring_recursive(self, index, AlreadyCrossed, orig_atom, AllRings):

        if len(AlreadyCrossed) > 6: return # since you're only considering aromatic rings containing 5 or 6 members anyway, save yourself some time.

        atom = self.AllAtoms[index]

        temp = AlreadyCrossed[:]
        temp.append(index)

        for conneceted_atom in atom.IndeciesOfAtomsConnecting:
            if not conneceted_atom in AlreadyCrossed:
                self.ring_recursive(conneceted_atom, temp, orig_atom, AllRings)
            if conneceted_atom == orig_atom and orig_atom != AlreadyCrossed[-1]:
                AllRings.append(temp)
    
    # Functions to assign secondary structure to protein residues
    # ===========================================================
    
    def assign_secondary_structure(self):
        # first, we need to know what resid's are available
        resids = []
        last_key = "-99999_Z"
        for atom_index in self.AllAtoms:
            atom = self.AllAtoms[atom_index]
            key = str(atom.resid) + "_" + atom.chain
            if key != last_key:
                last_key = key
                resids.append(last_key)
        
        structure = {}
        for resid in resids:
            structure[resid] = "OTHER"
        
        atoms = []
        
        for atom_index in self.AllAtoms:
            atom = self.AllAtoms[atom_index]
            if atom.SideChainOrBackBone() == "BACKBONE":
                if len(atoms) < 8:
                    atoms.append(atom)
                else:
                    atoms.pop(0)
                    atoms.append(atom)
                
                    # now make sure the first four all have the same resid and the last four all have the same resid
                    if atoms[0].resid == atoms[1].resid and atoms[0].resid == atoms[2].resid and atoms[0].resid == atoms[3].resid and atoms[0] != atoms[4].resid and atoms[4].resid == atoms[5].resid and atoms[4].resid == atoms[6].resid and atoms[4].resid == atoms[7].resid and atoms[0].resid + 1 == atoms[7].resid and atoms[0].chain == atoms[7].chain:
                        resid1 = atoms[0].resid
                        resid2 = atoms[7].resid
                        
                        # Now give easier to use names to the atoms
                        for atom in atoms:
                            if atom.resid == resid1 and atom.atomname.strip() == "N": first_N = atom
                            if atom.resid == resid1 and atom.atomname.strip() == "C": first_C = atom
                            if atom.resid == resid1 and atom.atomname.strip() == "CA": first_CA = atom
                        
                            if atom.resid == resid2 and atom.atomname.strip() == "N": second_N = atom
                            if atom.resid == resid2 and atom.atomname.strip() == "C": second_C = atom
                            if atom.resid == resid2 and atom.atomname.strip() == "CA": second_CA = atom
                        
                        # Now compute the phi and psi dihedral angles
                        phi = self.functions.dihedral(first_C.coordinates, second_N.coordinates, second_CA.coordinates, second_C.coordinates) * 180.0 / math.pi
                        psi = self.functions.dihedral(first_N.coordinates, first_CA.coordinates, first_C.coordinates, second_N.coordinates) * 180.0 / math.pi

                        # Now use those angles to determine if it's alpha or beta
                        if phi > -145 and phi < -35 and psi > -70 and psi < 50:
                            key1 = str(first_C.resid) + "_" + first_C.chain
                            key2 = str(second_C.resid) + "_" + second_C.chain
                            structure[key1] = "ALPHA"
                            structure[key2] = "ALPHA"
                        if (phi >= -180 and phi < -40 and psi <= 180 and psi > 90) or (phi >= -180 and phi < -70 and psi <= -165): # beta. This gets some loops (by my eye), but it's the best I could do.
                            key1 = str(first_C.resid) + "_" + first_C.chain
                            key2 = str(second_C.resid) + "_" + second_C.chain
                            structure[key1] = "BETA"
                            structure[key2] = "BETA"
                
        # Now update each of the atoms with this structural information
        for atom_index in self.AllAtoms:
            atom = self.AllAtoms[atom_index]
            key = str(atom.resid) + "_" + atom.chain
            atom.structure = structure[key]
            
        # Some more post processing. 
        CA_list = [] # first build a list of the indices of all the alpha carbons
        for atom_index in self.AllAtoms:
            atom = self.AllAtoms[atom_index]
            if atom.residue.strip() in self.protein_resnames and atom.atomname.strip() == "CA": CA_list.append(atom_index)
            
        # some more post processing. 
        change = True
        while change == True:
            change = False
            
            # A residue of index i is only going to be in an alpha helix its CA is within 6 A of the CA of the residue i + 3
            for CA_atom_index in CA_list:
                CA_atom = self.AllAtoms[CA_atom_index]
                if CA_atom.structure == "ALPHA": # so it's in an alpha helix
                    another_alpha_is_close = False
                    for other_CA_atom_index in CA_list: # so now compare that CA to all the other CA's
                        other_CA_atom = self.AllAtoms[other_CA_atom_index]
                        if other_CA_atom.structure == "ALPHA": # so it's also in an alpha helix
                            if other_CA_atom.resid - 3 == CA_atom.resid or other_CA_atom.resid + 3 == CA_atom.resid: # so this CA atom is one of the ones the first atom might hydrogen bond with
                                if other_CA_atom.coordinates.dist_to(CA_atom.coordinates) < 6.0: # so these two CA atoms are close enough together that their residues are probably hydrogen bonded
                                    another_alpha_is_close = True
                                    break
                    if another_alpha_is_close == False:
                        self.set_structure_of_residue(CA_atom.chain, CA_atom.resid, "OTHER")
                        change = True

            # Alpha helices are only alpha helices if they span at least 4 residues (to wrap around and hydrogen bond). I'm going to require them to span at least 5 residues, based on examination of many structures.
            for index_in_list in range(len(CA_list)-5): 
                
                index_in_pdb1 = CA_list[index_in_list]
                index_in_pdb2 = CA_list[index_in_list+1]
                index_in_pdb3 = CA_list[index_in_list+2]
                index_in_pdb4 = CA_list[index_in_list+3]
                index_in_pdb5 = CA_list[index_in_list+4]
                index_in_pdb6 = CA_list[index_in_list+5]

                atom1 = self.AllAtoms[index_in_pdb1]
                atom2 = self.AllAtoms[index_in_pdb2]
                atom3 = self.AllAtoms[index_in_pdb3]
                atom4 = self.AllAtoms[index_in_pdb4]
                atom5 = self.AllAtoms[index_in_pdb5]
                atom6 = self.AllAtoms[index_in_pdb6]
                
                if atom1.resid + 1 == atom2.resid and atom2.resid + 1 == atom3.resid and atom3.resid + 1 == atom4.resid and atom4.resid + 1 == atom5.resid and atom5.resid + 1 == atom6.resid: # so they are sequential
                    
                    if atom1.structure != "ALPHA" and atom2.structure == "ALPHA" and atom3.structure != "ALPHA":
                        self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
                        change = True
                    if atom2.structure != "ALPHA" and atom3.structure == "ALPHA" and atom4.structure != "ALPHA":
                        self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
                        change = True
                    if atom3.structure != "ALPHA" and atom4.structure == "ALPHA" and atom5.structure != "ALPHA":
                        self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
                        change = True
                    if atom4.structure != "ALPHA" and atom5.structure == "ALPHA" and atom6.structure != "ALPHA":
                        self.set_structure_of_residue(atom5.chain, atom5.resid, "OTHER")
                        change = True
                        
                    if atom1.structure != "ALPHA" and atom2.structure == "ALPHA" and atom3.structure == "ALPHA" and atom4.structure != "ALPHA":
                        self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
                        self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
                        change = True
                    if atom2.structure != "ALPHA" and atom3.structure == "ALPHA" and atom4.structure == "ALPHA" and atom5.structure != "ALPHA":
                        self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
                        self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
                        change = True
                    if atom3.structure != "ALPHA" and atom4.structure == "ALPHA" and atom5.structure == "ALPHA" and atom6.structure != "ALPHA":
                        self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
                        self.set_structure_of_residue(atom5.chain, atom5.resid, "OTHER")
                        change = True

                    if atom1.structure != "ALPHA" and atom2.structure == "ALPHA" and atom3.structure == "ALPHA" and atom4.structure == "ALPHA" and atom5.structure != "ALPHA":
                        self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
                        self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
                        self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
                        change = True
                    if atom2.structure != "ALPHA" and atom3.structure == "ALPHA" and atom4.structure == "ALPHA" and atom5.structure == "ALPHA" and atom6.structure != "ALPHA":
                        self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
                        self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
                        self.set_structure_of_residue(atom5.chain, atom5.resid, "OTHER")
                        change = True

                    if atom1.structure != "ALPHA" and atom2.structure == "ALPHA" and atom3.structure == "ALPHA" and atom4.structure == "ALPHA" and atom5.structure == "ALPHA" and atom6.structure != "ALPHA":
                        self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
                        self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
                        self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
                        self.set_structure_of_residue(atom5.chain, atom5.resid, "OTHER")
                        change = True

            # now go through each of the BETA CA atoms. A residue is only going to be called a beta sheet if CA atom is within 6.0 A of another CA beta, same chain, but index difference > 2.
            for CA_atom_index in CA_list:
                CA_atom = self.AllAtoms[CA_atom_index]
                if CA_atom.structure == "BETA": # so it's in a beta sheet
                    another_beta_is_close = False
                    for other_CA_atom_index in CA_list:
                        if other_CA_atom_index != CA_atom_index: # so not comparing an atom to itself
                            other_CA_atom = self.AllAtoms[other_CA_atom_index]
                            if other_CA_atom.structure == "BETA": # so you're comparing it only to other BETA-sheet atoms
                                if other_CA_atom.chain == CA_atom.chain: # so require them to be on the same chain. needed to indecies can be fairly compared
                                    if math.fabs(other_CA_atom.resid - CA_atom.resid) > 2: # so the two residues are not simply adjacent to each other on the chain
                                        if CA_atom.coordinates.dist_to(other_CA_atom.coordinates) < 6.0: # so these to atoms are close to each other
                                            another_beta_is_close = True
                                            break
                    if another_beta_is_close == False:
                        self.set_structure_of_residue(CA_atom.chain, CA_atom.resid, "OTHER")
                        change = True
                
            # Now some more post-processing needs to be done. Do this again to clear up mess that may have just been created (single residue beta strand, for example)
            # Beta sheets are usually at least 3 residues long
            
            for index_in_list in range(len(CA_list)-3):
                
                index_in_pdb1 = CA_list[index_in_list]
                index_in_pdb2 = CA_list[index_in_list+1]
                index_in_pdb3 = CA_list[index_in_list+2]
                index_in_pdb4 = CA_list[index_in_list+3]
                
                atom1 = self.AllAtoms[index_in_pdb1]
                atom2 = self.AllAtoms[index_in_pdb2]
                atom3 = self.AllAtoms[index_in_pdb3]
                atom4 = self.AllAtoms[index_in_pdb4]
                
                if atom1.resid + 1 == atom2.resid and atom2.resid + 1 == atom3.resid and atom3.resid + 1 == atom4.resid: # so they are sequential
                    
                    if atom1.structure != "BETA" and atom2.structure == "BETA" and atom3.structure != "BETA":
                        self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
                        change = True
                    if atom2.structure != "BETA" and atom3.structure == "BETA" and atom4.structure != "BETA":
                        self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
                        change = True
                    if atom1.structure != "BETA" and atom2.structure == "BETA" and atom3.structure == "BETA" and atom4.structure != "BETA":
                        self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
                        self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
                        change = True
            
    def set_structure_of_residue(self, chain, resid, structure):
        for atom_index in self.AllAtoms:
            atom = self.AllAtoms[atom_index]
            if atom.chain == chain and atom.resid == resid:
                atom.structure = structure


# In[10]:


class MathFunctions:
    
    def vector_subtraction(self, vector1, vector2): # vector1 - vector2
        return point(vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z)
    
    def CrossProduct(self, Pt1, Pt2): # never tested
        Response = point(0,0,0)
    
        Response.x = Pt1.y * Pt2.z - Pt1.z * Pt2.y
        Response.y = Pt1.z * Pt2.x - Pt1.x * Pt2.z
        Response.z = Pt1.x * Pt2.y - Pt1.y * Pt2.x
    
        return Response;
    
    def vector_scalar_multiply(self, vector, scalar):
        return point(vector.x * scalar, vector.y * scalar, vector.z * scalar)
    
    def dot_product(self, point1, point2):
        return point1.x * point2.x + point1.y * point2.y + point1.z * point2.z
    
    def dihedral(self, point1, point2, point3, point4): # never tested
    
        b1 = self.vector_subtraction(point2, point1)
        b2 = self.vector_subtraction(point3, point2)
        b3 = self.vector_subtraction(point4, point3)
    
        b2Xb3 = self.CrossProduct(b2,b3)
        b1Xb2 = self.CrossProduct(b1,b2)
    
        b1XMagb2 = self.vector_scalar_multiply(b1,b2.Magnitude())
        radians = math.atan2(self.dot_product(b1XMagb2,b2Xb3), self.dot_product(b1Xb2,b2Xb3))
        return radians
    
    def angle_between_three_points(self, point1, point2, point3): # As in three connected atoms
        vector1 = self.vector_subtraction(point1, point2)
        vector2 = self.vector_subtraction(point3, point2)
        return self.angle_between_points(vector1, vector2)
    
    def angle_between_points(self, point1, point2):
        new_point1 = self.return_normalized_vector(point1)
        new_point2 = self.return_normalized_vector(point2)
        dot_prod = self.dot_product(new_point1, new_point2)
        if dot_prod > 1.0: dot_prod = 1.0 # to prevent errors that can rarely occur
        if dot_prod < -1.0: dot_prod = -1.0
        return math.acos(dot_prod)
    
    def return_normalized_vector(self, vector):
        dist = self.distance(point(0,0,0), vector)
        return point(vector.x/dist, vector.y/dist, vector.z/dist)
    
    def distance(self, point1, point2):
        deltax = point1.x - point2.x
        deltay = point1.y - point2.y
        deltaz = point1.z - point2.z
    
        return math.sqrt(math.pow(deltax,2) + math.pow(deltay,2) + math.pow(deltaz,2))
        
    def project_point_onto_plane(self, apoint, plane_coefficients): # essentially finds the point on the plane that is closest to the specified point
        # the plane_coefficients are [a,b,c,d], where the plane is ax + by + cz = d

        # First, define a plane using cooeficients a, b, c, d such that ax + by + cz = d
        a = plane_coefficients[0]
        b = plane_coefficients[1]
        c = plane_coefficients[2]
        d = plane_coefficients[3]
        
        # Now, define a point in space (s,u,v)
        s = apoint.x
        u = apoint.y
        v = apoint.z
        
        # the formula of a line perpendicular to the plan passing through (s,u,v) is:
        #x = s + at
        #y = u + bt
        #z = v + ct
        
        t = (d - a*s - b*u - c*v) / (a*a + b*b + c*c)
        
        # here's the point closest on the plane
        x = s + a*t
        y = u + b*t
        z = v + c*t
        
        return point(x,y,z)

def getCommandOutput2(command):
    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        raise RuntimeError('%s failed w/ exit code %d' % (command, err))
    return data

class binana:

    functions = MathFunctions()
    
    # supporting functions
    def list_alphebetize_and_combine(self, list):
        list.sort()
        return '_'.join(list)

    def hashtable_entry_add_one(self, hashtable, key, toadd = 1): # note that dictionaries (hashtables) are passed by reference in python
        if key in hashtable:
            hashtable[key] = hashtable[key] + toadd
        else:
            hashtable[key] = toadd

    def extend_list_by_dictionary(self, list1, dictionary):
        # first, sort the dictionary by the key
        keys = list(dictionary.keys())
        keys.sort()

        # now make a list of the values
        vals = []
        for key in keys:
            vals.append( dictionary[key])

        # now append vals to the list
        newlist = list1[:]
        newlist.extend(vals)

        # return the extended list
        return newlist

    def center(self, string, length):
        while len(string) < length:
            string = " " + string
            if len(string) < length:
                string = string + " "
        return string
    
    # The meat of the class
    #def __init__(self, ligand_pdbqt_filename, receptor_pdbqt_filename, parameters, line_header, actual_filename_if_ligand_is_list="", actual_filename_if_receptor_is_list=""): # must be a more elegant way of doing this
    def __init__(self, ligand_pdbqt_filename, receptor, parameters, line_header, actual_filename_if_ligand_is_list="", actual_filename_if_receptor_is_list=""): # must be a more elegant way of doing this
        
        receptor_pdbqt_filename = receptor.OrigFileName
        
        ligand = PDB()
        if actual_filename_if_ligand_is_list!="": # so a list was passed as the ligand 
            ligand.LoadPDB_from_list(ligand_pdbqt_filename, line_header)

            # now write the file so when VINA is run it has a ligand file for input
            f = open(actual_filename_if_ligand_is_list,'w')
            for line in ligand_pdbqt_filename:
                if not "MODEL" in line and not "ENDMDL" in line: f.write(line)
            f.close()

            ligand_pdbqt_filename = actual_filename_if_ligand_is_list
        else: # so a filename was passed as the ligand
            ligand.LoadPDB_from_file(ligand_pdbqt_filename, line_header)
        

        #receptor = PDB()
        #if actual_filename_if_ligand_is_list=="": # receptor is a filename instead of a list
        #    receptor.LoadPDB_from_file(receptor_pdbqt_filename, line_header)
        #else: # so it's a list that was passed, not a filename
        #    receptor.LoadPDB_from_list(receptor_pdbqt_filename, line_header)
        #    receptor_pdbqt_filename = actual_filename_if_receptor_is_list

        receptor.assign_secondary_structure()

        # Get distance measurements between protein and ligand atom types, as well as some other measurements

        ligand_receptor_atom_type_pairs_less_than_two_half = {}
        ligand_receptor_atom_type_pairs_less_than_four = {}
        ligand_receptor_atom_type_pairs_electrostatic = {}
        active_site_flexibility = {}
        hbonds = {}
        hydrophobics = {}
        ligand.rotateable_bonds_count
        functions = MathFunctions()
        
        pdb_close_contacts = PDB()
        pdb_contacts = PDB()
        pdb_contacts_alpha_helix = PDB()
        pdb_contacts_beta_sheet = PDB()
        pdb_contacts_other_2nd_structure = PDB()
        pdb_side_chain = PDB()
        pdb_back_bone = PDB()
        pdb_hydrophobic = PDB()
        pdb_hbonds = PDB()
        
        for ligand_atom_index in ligand.AllAtoms:
            for receptor_atom_index in receptor.AllAtoms:
                ligand_atom = ligand.AllAtoms[ligand_atom_index]
                receptor_atom = receptor.AllAtoms[receptor_atom_index]
                
                dist = ligand_atom.coordinates.dist_to(receptor_atom.coordinates)
                if dist < 2.5: # less than 2.5 A
                    list = [ligand_atom.atomtype, receptor_atom.atomtype]
                    self.hashtable_entry_add_one(ligand_receptor_atom_type_pairs_less_than_two_half, self.list_alphebetize_and_combine(list))
                    pdb_close_contacts.AddNewAtom(ligand_atom.copy_of())
                    pdb_close_contacts.AddNewAtom(receptor_atom.copy_of())
                elif dist < 4.0: # less than 4 A
                    list = [ligand_atom.atomtype, receptor_atom.atomtype]
                    self.hashtable_entry_add_one(ligand_receptor_atom_type_pairs_less_than_four, self.list_alphebetize_and_combine(list))
                    pdb_contacts.AddNewAtom(ligand_atom.copy_of())
                    pdb_contacts.AddNewAtom(receptor_atom.copy_of())

                if dist < 4.0:
                    # calculate electrostatic energies for all less than 4 A
                    ligand_charge = ligand_atom.charge
                    receptor_charge = receptor_atom.charge
                    coulomb_energy = (ligand_charge * receptor_charge / dist) * 138.94238460104697e4 # to convert into J/mol # might be nice to double check this
                    list = [ligand_atom.atomtype, receptor_atom.atomtype]
                    self.hashtable_entry_add_one(ligand_receptor_atom_type_pairs_electrostatic, self.list_alphebetize_and_combine(list), coulomb_energy)
                    
                if dist < 4.0:
                    # Now get statistics to judge active-site flexibility
                    flexibility_key = receptor_atom.SideChainOrBackBone() + "_" + receptor_atom.structure # first can be sidechain or backbone, second back be alpha, beta, or other, so six catagories
                    if receptor_atom.structure == "ALPHA": pdb_contacts_alpha_helix.AddNewAtom(receptor_atom.copy_of())
                    elif receptor_atom.structure == "BETA": pdb_contacts_beta_sheet.AddNewAtom(receptor_atom.copy_of())
                    elif receptor_atom.structure == "OTHER": pdb_contacts_other_2nd_structure.AddNewAtom(receptor_atom.copy_of())

                    if receptor_atom.SideChainOrBackBone() == "BACKBONE": pdb_back_bone.AddNewAtom(receptor_atom.copy_of())
                    elif receptor_atom.SideChainOrBackBone() == "SIDECHAIN": pdb_side_chain.AddNewAtom(receptor_atom.copy_of())

                    self.hashtable_entry_add_one(active_site_flexibility, flexibility_key)
                    
                if dist < 4.0:
                    # Now see if there's hydrophobic contacts (C-C contacts)
                    if ligand_atom.element == "C" and receptor_atom.element == "C":
                        hydrophobic_key = receptor_atom.SideChainOrBackBone() + "_" + receptor_atom.structure
                        pdb_hydrophobic.AddNewAtom(ligand_atom.copy_of())
                        pdb_hydrophobic.AddNewAtom(receptor_atom.copy_of())
                        
                        self.hashtable_entry_add_one(hydrophobics, hydrophobic_key)
                    
                if dist < 4.0:
                    # Now see if there's some sort of hydrogen bond between these two atoms. distance cutoff = 4, angle cutoff = 40. Note that this is liberal.
                    if (ligand_atom.element == "O" or ligand_atom.element == "N") and (receptor_atom.element == "O" or receptor_atom.element == "N"):
                        
                        # now build a list of all the hydrogens close to these atoms
                        hydrogens = []
                        
                        for atm_index in ligand.AllAtoms:
                            if ligand.AllAtoms[atm_index].element == "H": # so it's a hydrogen
                                if ligand.AllAtoms[atm_index].coordinates.dist_to(ligand_atom.coordinates) < 1.3: # O-H distance is 0.96 A, N-H is 1.01 A. See http://www.science.uwaterloo.ca/~cchieh/cact/c120/bondel.html
                                    ligand.AllAtoms[atm_index].comment = "LIGAND"
                                    hydrogens.append(ligand.AllAtoms[atm_index])
                            
                        for atm_index in receptor.AllAtoms:
                            if receptor.AllAtoms[atm_index].element == "H": # so it's a hydrogen
                                if receptor.AllAtoms[atm_index].coordinates.dist_to(receptor_atom.coordinates) < 1.3: # O-H distance is 0.96 A, N-H is 1.01 A. See http://www.science.uwaterloo.ca/~cchieh/cact/c120/bondel.html
                                    receptor.AllAtoms[atm_index].comment = "RECEPTOR"
                                    hydrogens.append(receptor.AllAtoms[atm_index])
                        
                        # now we need to check the angles
                        for hydrogen in hydrogens:
                            if math.fabs(180 - functions.angle_between_three_points(ligand_atom.coordinates, hydrogen.coordinates, receptor_atom.coordinates) * 180.0 / math.pi) <= 40.0:
                                hbonds_key = "HDONOR_" + hydrogen.comment + "_" + receptor_atom.SideChainOrBackBone() + "_" + receptor_atom.structure
                                pdb_hbonds.AddNewAtom(ligand_atom.copy_of())
                                pdb_hbonds.AddNewAtom(hydrogen.copy_of())
                                pdb_hbonds.AddNewAtom(receptor_atom.copy_of())
                                self.hashtable_entry_add_one(hbonds, hbonds_key)
                                                    
        # Get the total number of each atom type in the ligand
        ligand_atom_types = {}
        for ligand_atom_index in ligand.AllAtoms:
            ligand_atom = ligand.AllAtoms[ligand_atom_index]
            self.hashtable_entry_add_one(ligand_atom_types, ligand_atom.atomtype)
            
        pi_padding = 0.75 # This is perhaps controversial. I noticed that often a pi-cation interaction or other pi interaction was only slightly off, but looking at the structure, it was clearly supposed to be a
        # pi-cation interaction. I've decided then to artificially expand the radius of each pi ring. Think of this as adding in a VDW radius, or accounting for poor crystal-structure resolution, or whatever you want
        # to justify it.
        
        # Count pi-pi stacking and pi-T stacking interactions
        PI_interactions = {}
        pdb_pistack = PDB()
        pdb_pi_T = PDB()
        # "PI-Stacking Interactions ALIVE AND WELL IN PROTEINS" says distance of 7.5 A is good cutoff. This seems really big to me, except that pi-pi interactions (parallel) are actuall usually off centered. Interesting paper.
        # Note that adenine and tryptophan count as two aromatic rings. So, for example, an interaction between these two, if positioned correctly, could count for 4 pi-pi interactions.
        #print ligand.aromatic_rings, "****"
        #print receptor.aromatic_rings,"****"
        for aromatic1 in ligand.aromatic_rings:
            #print "dude"
            for aromatic2 in receptor.aromatic_rings:
                dist = aromatic1.center.dist_to(aromatic2.center)
                if dist < 7.5: # so there could be some pi-pi interactions.
                    # first, let's check for stacking interactions. Are the two pi's roughly parallel?
                    aromatic1_norm_vector = point(aromatic1.plane_coeff[0], aromatic1.plane_coeff[1], aromatic1.plane_coeff[2])
                    aromatic2_norm_vector = point(aromatic2.plane_coeff[0], aromatic2.plane_coeff[1], aromatic2.plane_coeff[2])
                    angle_between_planes = self.functions.angle_between_points(aromatic1_norm_vector, aromatic2_norm_vector) * 180.0/math.pi

                    if math.fabs(angle_between_planes-0) < 30.0 or math.fabs(angle_between_planes-180) < 30.0: # so they're more or less parallel, it's probably pi-pi stackingoutput_dir
                        # now, pi-pi are not usually right on top of each other. They're often staggared. So I don't want to just look at the centers of the rings and compare. Let's look at each of the atoms.
                        # do atom of the atoms of one ring, when projected onto the plane of the other, fall within that other ring?
                        
                        pi_pi = False # start by assuming it's not a pi-pi stacking interaction
                        for ligand_ring_index in aromatic1.indices:
                            # project the ligand atom onto the plane of the receptor ring
                            pt_on_receptor_plane = self.functions.project_point_onto_plane(ligand.AllAtoms[ligand_ring_index].coordinates, aromatic2.plane_coeff)
                            if pt_on_receptor_plane.dist_to(aromatic2.center) <= aromatic2.radius + pi_padding:
                                pi_pi = True
                                break
                        
                        if pi_pi == False: # if you've already determined it's a pi-pi stacking interaction, no need to keep trying
                            for receptor_ring_index in aromatic2.indices:
                                # project the ligand atom onto the plane of the receptor ring
                                pt_on_ligand_plane = self.functions.project_point_onto_plane(receptor.AllAtoms[receptor_ring_index].coordinates, aromatic1.plane_coeff)
                                if pt_on_ligand_plane.dist_to(aromatic1.center) <= aromatic1.radius + pi_padding:
                                    pi_pi = True
                                    break
                        
                        if pi_pi == True:
                            structure = receptor.AllAtoms[aromatic2.indices[0]].structure
                            if structure == "": structure = "OTHER" # since it could be interacting with a cofactor or something
                            key = "STACKING_" + structure
                            
                            for index in aromatic1.indices: pdb_pistack.AddNewAtom(ligand.AllAtoms[index].copy_of())
                            for index in aromatic2.indices: pdb_pistack.AddNewAtom(receptor.AllAtoms[index].copy_of())
                            
                            self.hashtable_entry_add_one(PI_interactions, key)
                            
                    elif math.fabs(angle_between_planes-90) < 30.0 or math.fabs(angle_between_planes-270) < 30.0: # so they're more or less perpendicular, it's probably a pi-edge interaction
                        #print "dude"
                        # having looked at many structures, I noticed the algorithm was identifying T-pi reactions when the two rings were in fact quite distant, often with other atoms
                        # in between. Eye-balling it, requiring that at their closest they be at least 5 A apart seems to separate the good T's from the bad
                        min_dist = 100.0
                        for ligand_ind in aromatic1.indices:
                            ligand_at = ligand.AllAtoms[ligand_ind]
                            for receptor_ind in aromatic2.indices:
                                receptor_at = receptor.AllAtoms[receptor_ind]
                                dist = ligand_at.coordinates.dist_to(receptor_at.coordinates)
                                if dist < min_dist: min_dist = dist
                                
                        if min_dist <= 5.0: # so at their closest points, the two rings come within 5 A of each other.
                        
                            # okay, is the ligand pi pointing into the receptor pi, or the other way around?
                            # first, project the center of the ligand pi onto the plane of the receptor pi, and vs. versa
                            
                            # This could be directional somehow, like a hydrogen bond.
                            
                            pt_on_receptor_plane = self.functions.project_point_onto_plane(aromatic1.center, aromatic2.plane_coeff)
                            pt_on_lignad_plane = self.functions.project_point_onto_plane(aromatic2.center, aromatic1.plane_coeff)
                            
                            # now, if it's a true pi-T interaction, this projected point should fall within the ring whose plane it's been projected into.
                            if (pt_on_receptor_plane.dist_to(aromatic2.center) <= aromatic2.radius + pi_padding) or (pt_on_lignad_plane.dist_to(aromatic1.center) <= aromatic1.radius + pi_padding): # so it is in the ring on the projected plane.
                                structure = receptor.AllAtoms[aromatic2.indices[0]].structure
                                if structure == "": structure = "OTHER" # since it could be interacting with a cofactor or something
                                key = "T-SHAPED_" + structure
    
                                for index in aromatic1.indices: pdb_pi_T.AddNewAtom(ligand.AllAtoms[index].copy_of())
                                for index in aromatic2.indices: pdb_pi_T.AddNewAtom(receptor.AllAtoms[index].copy_of())
    
                                self.hashtable_entry_add_one(PI_interactions, key)
                            
        # Now identify pi-cation interactions
        pdb_pi_cat = PDB()
        
        for aromatic in receptor.aromatic_rings:
            for charged in ligand.charges:
                if charged.positive == True: # so only consider positive charges
                    if charged.coordinates.dist_to(aromatic.center) < 6.0: # distance cutoff based on "Cation-pi interactions in structural biology."
                        # project the charged onto the plane of the aromatic
                        charge_projected = self.functions.project_point_onto_plane(charged.coordinates,aromatic.plane_coeff)
                        if charge_projected.dist_to(aromatic.center) < aromatic.radius + pi_padding:
                            structure = receptor.AllAtoms[aromatic.indices[0]].structure
                            if structure == "": structure = "OTHER" # since it could be interacting with a cofactor or something
                            key = "PI-CATION_LIGAND-CHARGED_" + structure
                            
                            for index in aromatic.indices: pdb_pi_cat.AddNewAtom(receptor.AllAtoms[index].copy_of())
                            for index in charged.indices: pdb_pi_cat.AddNewAtom(ligand.AllAtoms[index].copy_of())
                            
                            self.hashtable_entry_add_one(PI_interactions, key)
                    
        for aromatic in ligand.aromatic_rings: # now it's the ligand that has the aromatic group
            for charged in receptor.charges:
                if charged.positive == True: # so only consider positive charges
                    if charged.coordinates.dist_to(aromatic.center) < 6.0: # distance cutoff based on "Cation-pi interactions in structural biology."
                        # project the charged onto the plane of the aromatic
                        charge_projected = self.functions.project_point_onto_plane(charged.coordinates,aromatic.plane_coeff)
                        if charge_projected.dist_to(aromatic.center) < aromatic.radius + pi_padding:
                            structure = receptor.AllAtoms[charged.indices[0]].structure
                            if structure == "": structure = "OTHER" # since it could be interacting with a cofactor or something
                            key = "PI-CATION_RECEPTOR-CHARGED_" + structure
    
                            for index in aromatic.indices: pdb_pi_cat.AddNewAtom(ligand.AllAtoms[index].copy_of())
                            for index in charged.indices: pdb_pi_cat.AddNewAtom(receptor.AllAtoms[index].copy_of())
    
                            self.hashtable_entry_add_one(PI_interactions, key)

        # now count the number of salt bridges
        pdb_salt_bridges = PDB()
        salt_bridges = {}
        for receptor_charge in receptor.charges:
            for ligand_charge in ligand.charges:
                if ligand_charge.positive != receptor_charge.positive: # so they have oppositve charges
                    if ligand_charge.coordinates.dist_to(receptor_charge.coordinates) < 5.5: # 4  is good cutoff for salt bridges according to "Close-Range Electrostatic Interactions in Proteins", but looking at complexes, I decided to go with 5.5 A
                        structure = receptor.AllAtoms[receptor_charge.indices[0]].structure
                        if structure == "": structure = "OTHER" # since it could be interacting with a cofactor or something
                        key = "SALT-BRIDGE_" + structure
                        
                        for index in receptor_charge.indices: pdb_salt_bridges.AddNewAtom(receptor.AllAtoms[index].copy_of())
                        for index in ligand_charge.indices: pdb_salt_bridges.AddNewAtom(ligand.AllAtoms[index].copy_of())
                        
                        self.hashtable_entry_add_one(salt_bridges, key)

        # Now save the files
        preface ="REMARK "
            
        # if an output directory is specified, and it doesn't exist, create it
        #if parameters.params['output_dir'] != "":
        #    if not os.path.exists(parameters.params['output_dir']):
        #        os.mkdir(parameters.params['output_dir'])

        # Now get vina
        vina_output = getCommandOutput2(parameters.params['vina_executable'] + ' --score_only --receptor ' + receptor_pdbqt_filename + ' --ligand ' + ligand_pdbqt_filename)
        #vina_output = getCommandOutput2('vina --score_only --receptor ' + receptor_pdbqt_filename + ' --ligand ' + ligand_pdbqt_filename)

        # if ligand_pdbqt_filename was originally passed as a list instead of a filename, delete the temporary file that was created to accomodate vina
        #if "MODEL" in ligand_pdbqt_filename: 
        #        os.remove(ligand_pdbqt_filename)
        #        print "REMOVE THIS SECTION FOR PRODUCTION!!!"
        #os.remove(ligand_pdbqt_filename)

        #print vina_output
        vina_output = vina_output.split("\n")
        vina_affinity = 0.0
        vina_gauss_1 = 0.0
        vina_gauss_2 = 0.0
        vina_repulsion = 0.0
        vina_hydrophobic = 0.0
        vina_hydrogen = 0.0
        for item in vina_output:
            item = item.strip()
            if "Affinity" in item: vina_affinity = float(item.replace("Affinity: ","").replace(" (kcal/mol)",""))
            if "gauss 1" in item: vina_gauss_1 = float(item.replace("gauss 1     : ",""))
            if "gauss 2" in item: vina_gauss_2 = float(item.replace("gauss 2     : ",""))
            if "repulsion" in item: vina_repulsion = float(item.replace("repulsion   : ",""))
            if "hydrophobic" in item: vina_hydrophobic = float(item.replace("hydrophobic : ",""))
            if "Hydrogen" in item: vina_hydrogen = float(item.replace("Hydrogen    : ",""))
    
        vina_output = [vina_affinity, vina_gauss_1, vina_gauss_2, vina_repulsion, vina_hydrophobic, vina_hydrogen]

        stacking = []
        t_shaped = []
        pi_cation = []
        for key in PI_interactions:
            value = PI_interactions[key]
            together = key + "_" + str(value) # not that this is put together strangely!!!
            if "STACKING" in together: stacking.append(together)
            if "CATION" in together: pi_cation.append(together)
            if "SHAPED" in together: t_shaped.append(together)

        # now create a single descriptor object
        data = {}
        data['vina_output'] = vina_output
        data['ligand_receptor_atom_type_pairs_less_than_two_half'] = ligand_receptor_atom_type_pairs_less_than_two_half
        data['ligand_receptor_atom_type_pairs_less_than_four'] = ligand_receptor_atom_type_pairs_less_than_four
        data['ligand_atom_types'] = ligand_atom_types
        data['ligand_receptor_atom_type_pairs_electrostatic'] = ligand_receptor_atom_type_pairs_electrostatic
        data['rotateable_bonds_count'] = ligand.rotateable_bonds_count
        data['active_site_flexibility'] = active_site_flexibility
        data['hbonds'] = hbonds
        data['hydrophobics'] = hydrophobics
        data['stacking'] = stacking
        data['pi_cation'] = pi_cation
        data['t_shaped'] = t_shaped
        data['salt_bridges'] = salt_bridges

        self.vina_output = data['vina_output']
        
        self.rotateable_bonds_count = {'rot_bonds':data['rotateable_bonds_count']}

        self.ligand_receptor_atom_type_pairs_less_than_two_half = {"A_A": 0, "A_C": 0, "A_CL": 0, "A_F": 0, "A_FE": 0, "A_MG": 0, "A_MN": 0, "A_NA": 0, "A_SA": 0, "BR_C": 0, "BR_OA": 0, "C_CL": 0, "CD_OA": 0, "CL_FE": 0, "CL_MG": 0, "CL_N": 0, "CL_OA": 0, "CL_ZN": 0, "C_MN": 0, "C_NA": 0, "F_N": 0, "F_SA": 0, "F_ZN": 0, "HD_MN": 0, "MN_N": 0, "NA_SA": 0, "N_SA": 0, "A_HD": 0, "A_N": 0, "A_OA": 0, "A_ZN": 0, "BR_HD": 0, "C_C": 0, "C_F": 0, "C_HD": 0, "CL_HD": 0, "C_MG": 0, "C_N": 0, "C_OA": 0, "C_SA": 0, "C_ZN": 0, "FE_HD": 0, "FE_N": 0, "FE_OA": 0, "F_HD": 0, "F_OA": 0, "HD_HD": 0, "HD_I": 0, "HD_MG": 0, "HD_N": 0, "HD_NA": 0, "HD_OA": 0, "HD_P": 0, "HD_S": 0, "HD_SA": 0, "HD_ZN": 0, "MG_NA": 0, "MG_OA": 0, "MN_OA": 0, "NA_OA": 0, "NA_ZN": 0, "N_N": 0, "N_NA": 0, "N_OA": 0, "N_ZN": 0, "OA_OA": 0, "OA_SA": 0, "OA_ZN": 0, "SA_ZN": 0, "S_ZN": 0}
        for key in data['ligand_receptor_atom_type_pairs_less_than_two_half']:
            if not key in self.ligand_receptor_atom_type_pairs_less_than_two_half: 
                  print ("\tWARNING: Atoms of types " + key.replace("_"," and ") + " come within 2.5 angstroms of each other.")
                  print ("\t  The neural networks were not trained to deal with this juxtaposition,")
                  print ("\t  so it will be ignored.")
                  self.error = True
            else:
                  self.ligand_receptor_atom_type_pairs_less_than_two_half[key] = data['ligand_receptor_atom_type_pairs_less_than_two_half'][key]
        
        self.ligand_receptor_atom_type_pairs_less_than_four = {"A_CU": 0, "A_MG": 0, "A_MN": 0, "BR_SA": 0, "C_CD": 0, "CL_FE": 0, "CL_MG": 0, "CL_MN": 0, "CL_NA": 0, "CL_P": 0, "CL_S": 0, "CL_ZN": 0, "CU_HD": 0, "CU_N": 0, "FE_NA": 0, "FE_SA": 0, "MG_N": 0, "MG_S": 0, "MG_SA": 0, "MN_NA": 0, "MN_S": 0, "MN_SA": 0, "NA_P": 0, "P_S": 0, "P_SA": 0, "S_SA": 0, "A_A": 0, "A_BR": 0, "A_C": 0, "A_CL": 0, "A_F": 0, "A_FE": 0, "A_HD": 0, "A_I": 0, "A_N": 0, "A_NA": 0, "A_OA": 0, "A_P": 0, "A_S": 0, "A_SA": 0, "A_ZN": 0, "BR_C": 0, "BR_HD": 0, "BR_N": 0, "BR_OA": 0, "C_C": 0, "C_CL": 0, "C_F": 0, "C_FE": 0, "C_HD": 0, "C_I": 0, "CL_HD": 0, "CL_N": 0, "CL_OA": 0, "CL_SA": 0, "C_MG": 0, "C_MN": 0, "C_N": 0, "C_NA": 0, "C_OA": 0, "C_P": 0, "C_S": 0, "C_SA": 0, "C_ZN": 0, "FE_HD": 0, "FE_N": 0, "FE_OA": 0, "F_HD": 0, "F_N": 0, "F_OA": 0, "F_SA": 0, "HD_HD": 0, "HD_I": 0, "HD_MG": 0, "HD_MN": 0, "HD_N": 0, "HD_NA": 0, "HD_OA": 0, "HD_P": 0, "HD_S": 0, "HD_SA": 0, "HD_ZN": 0, "I_N": 0, "I_OA": 0, "MG_NA": 0, "MG_OA": 0, "MG_P": 0, "MN_N": 0, "MN_OA": 0, "MN_P": 0, "NA_OA": 0, "NA_S": 0, "NA_SA": 0, "NA_ZN": 0, "N_N": 0, "N_NA": 0, "N_OA": 0, "N_P": 0, "N_S": 0, "N_SA": 0, "N_ZN": 0, "OA_OA": 0, "OA_P": 0, "OA_S": 0, "OA_SA": 0, "OA_ZN": 0, "P_ZN": 0, "SA_SA": 0, "SA_ZN": 0, "S_ZN": 0}
        for key in data['ligand_receptor_atom_type_pairs_less_than_four']:
            if not key in self.ligand_receptor_atom_type_pairs_less_than_four: 
                  print ("\tWARNING: Atoms of types " + key.replace("_"," and ") + " come within 4 angstroms of each other.")
                  print ("\t  The neural networks were not trained to deal with this juxtaposition,")
                  print ("\t  so it will be ignored.")

                  self.error = True
            else:
                  self.ligand_receptor_atom_type_pairs_less_than_four[key] = data['ligand_receptor_atom_type_pairs_less_than_four'][key]

        self.ligand_atom_types = {'A': 0, 'BR': 0, 'C': 0, 'CL': 0, 'F': 0, 'HD': 0, 'I': 0, 'N': 0, 'NA': 0, 'OA': 0, 'P': 0, 'S': 0, 'SA': 0}
        for key in data['ligand_atom_types']:
            if not key in self.ligand_atom_types: 
                  print ("\tWARNING: The ligand contains an atoms of type " + key + ". The neural networks")
                  print ("\t  were not trained to deal with this ligand atom type, so it will be ignored.")

                  self.error = True
            else:
                  self.ligand_atom_types[key] = data['ligand_atom_types'][key]

        self.ligand_receptor_atom_type_pairs_electrostatic = {"A_MG": 0, "A_MN": 0, "BR_SA": 0, "CL_FE": 0, "CL_MG": 0, "CL_MN": 0, "CL_NA": 0, "CL_P": 0, "CL_S": 0, "CL_ZN": 0, "CU_HD": 0, "CU_N": 0, "FE_NA": 0, "FE_SA": 0, "MG_N": 0, "MG_S": 0, "MG_SA": 0, "MN_NA": 0, "MN_S": 0, "MN_SA": 0, "NA_P": 0, "P_S": 0, "P_SA": 0, "S_SA": 0, "A_A": 0.0, "A_BR": 0.0, "A_C": 0.0, "A_CL": 0.0, "A_F": 0.0, "A_FE": 0.0, "A_HD": 0.0, "A_I": 0.0, "A_N": 0.0, "A_NA": 0.0, "A_OA": 0.0, "A_P": 0.0, "A_S": 0.0, "A_SA": 0.0, "A_ZN": 0.0, "BR_C": 0.0, "BR_HD": 0.0, "BR_N": 0.0, "BR_OA": 0.0, "C_C": 0.0, "C_CL": 0.0, "C_F": 0.0, "C_FE": 0.0, "C_HD": 0.0, "C_I": 0.0, "CL_HD": 0.0, "CL_N": 0.0, "CL_OA": 0.0, "CL_SA": 0.0, "C_MG": 0.0, "C_MN": 0.0, "C_N": 0.0, "C_NA": 0.0, "C_OA": 0.0, "C_P": 0.0, "C_S": 0.0, "C_SA": 0.0, "C_ZN": 0.0, "FE_HD": 0.0, "FE_N": 0.0, "FE_OA": 0.0, "F_HD": 0.0, "F_N": 0.0, "F_OA": 0.0, "F_SA": 0.0, "HD_HD": 0.0, "HD_I": 0.0, "HD_MG": 0.0, "HD_MN": 0.0, "HD_N": 0.0, "HD_NA": 0.0, "HD_OA": 0.0, "HD_P": 0.0, "HD_S": 0.0, "HD_SA": 0.0, "HD_ZN": 0.0, "I_N": 0.0, "I_OA": 0.0, "MG_NA": 0.0, "MG_OA": 0.0, "MG_P": 0.0, "MN_N": 0.0, "MN_OA": 0.0, "MN_P": 0.0, "NA_OA": 0.0, "NA_S": 0.0, "NA_SA": 0.0, "NA_ZN": 0.0, "N_N": 0.0, "N_NA": 0.0, "N_OA": 0.0, "N_P": 0.0, "N_S": 0.0, "N_SA": 0.0, "N_ZN": 0.0, "OA_OA": 0.0, "OA_P": 0.0, "OA_S": 0.0, "OA_SA": 0.0, "OA_ZN": 0.0, "P_ZN": 0.0, "SA_SA": 0.0, "SA_ZN": 0.0, "S_ZN": 0, "F_ZN": 0}

        for key in data['ligand_receptor_atom_type_pairs_electrostatic']:
            if not key in self.ligand_receptor_atom_type_pairs_electrostatic: 
                 print ("\tWARNING: Atoms of types " + key.replace("_"," and ") + ", which come within 4 angstroms of each")
                 print ("\t  other, may interact electrostatically. However, the neural networks")
                 print ("\t  were not trained to deal with electrostatic interactions between atoms")
                 print ("\t  of these types, so they will be ignored.")
                 
                 self.error = True
            else:
                 self.ligand_receptor_atom_type_pairs_electrostatic[key] = data['ligand_receptor_atom_type_pairs_electrostatic'][key]
            
        self.active_site_flexibility = {'BACKBONE_ALPHA': 0, 'BACKBONE_BETA': 0, 'BACKBONE_OTHER': 0, 'SIDECHAIN_ALPHA': 0, 'SIDECHAIN_BETA': 0, 'SIDECHAIN_OTHER': 0}
        for key in data['active_site_flexibility']:
            self.active_site_flexibility[key] = data['active_site_flexibility'][key]
        
        alpha_tmp = self.active_site_flexibility['BACKBONE_ALPHA'] + self.active_site_flexibility['SIDECHAIN_ALPHA']
        beta_tmp = self.active_site_flexibility['BACKBONE_BETA'] + self.active_site_flexibility['SIDECHAIN_BETA']
        other_tmp = self.active_site_flexibility['BACKBONE_OTHER'] + self.active_site_flexibility['SIDECHAIN_OTHER']
        self.active_site_flexibility_by_structure = {'ALPHA':alpha_tmp, 'BETA':beta_tmp, 'OTHER':other_tmp}

        backbone_tmp = self.active_site_flexibility['BACKBONE_ALPHA'] + self.active_site_flexibility['BACKBONE_BETA'] + self.active_site_flexibility['BACKBONE_OTHER']
        sidechain_tmp = self.active_site_flexibility['SIDECHAIN_ALPHA'] + self.active_site_flexibility['SIDECHAIN_BETA'] + self.active_site_flexibility['SIDECHAIN_OTHER']
        self.active_site_flexibility_by_backbone_or_sidechain = {'BACKBONE':backbone_tmp, 'SIDECHAIN':sidechain_tmp}
        
        all = self.active_site_flexibility['BACKBONE_ALPHA'] + self.active_site_flexibility['BACKBONE_BETA'] + self.active_site_flexibility['BACKBONE_OTHER'] + self.active_site_flexibility['SIDECHAIN_ALPHA'] + self.active_site_flexibility['SIDECHAIN_BETA'] + self.active_site_flexibility['SIDECHAIN_OTHER']
        self.active_site_flexibility_all = {'all': all}
 
        self.hbonds = {'HDONOR-LIGAND_BACKBONE_ALPHA': 0, 'HDONOR-LIGAND_BACKBONE_BETA': 0, 'HDONOR-LIGAND_BACKBONE_OTHER': 0, 'HDONOR-LIGAND_SIDECHAIN_ALPHA': 0, 'HDONOR-LIGAND_SIDECHAIN_BETA': 0, 'HDONOR-LIGAND_SIDECHAIN_OTHER': 0, 'HDONOR-RECEPTOR_BACKBONE_ALPHA': 0, 'HDONOR-RECEPTOR_BACKBONE_BETA': 0, 'HDONOR-RECEPTOR_BACKBONE_OTHER': 0, 'HDONOR-RECEPTOR_SIDECHAIN_ALPHA': 0, 'HDONOR-RECEPTOR_SIDECHAIN_BETA': 0, 'HDONOR-RECEPTOR_SIDECHAIN_OTHER': 0}
        for key in data['hbonds']:
            key2 = key.replace("HDONOR_","HDONOR-")
            self.hbonds[key2] = data['hbonds'][key]
            
        hdonor_ligand = self.hbonds['HDONOR-LIGAND_BACKBONE_ALPHA'] + self.hbonds['HDONOR-LIGAND_BACKBONE_BETA'] + self.hbonds['HDONOR-LIGAND_BACKBONE_OTHER'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_ALPHA'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_BETA'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_OTHER']
        hdonor_receptor = self.hbonds['HDONOR-RECEPTOR_BACKBONE_ALPHA'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_BETA'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_OTHER'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_ALPHA'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_BETA'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_OTHER'] 
        self.hbonds_by_location_of_hdonor = {'LIGAND':hdonor_ligand, 'RECEPTOR':hdonor_receptor}
        
        hbond_backbone = self.hbonds['HDONOR-LIGAND_BACKBONE_ALPHA'] + self.hbonds['HDONOR-LIGAND_BACKBONE_BETA'] + self.hbonds['HDONOR-LIGAND_BACKBONE_OTHER'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_ALPHA'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_BETA'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_OTHER']
        hbond_sidechain = self.hbonds['HDONOR-LIGAND_SIDECHAIN_ALPHA'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_BETA'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_OTHER'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_ALPHA'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_BETA'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_OTHER']
        self.hbonds_by_backbone_or_sidechain = {'BACKBONE':hbond_backbone, 'SIDECHAIN':hbond_sidechain}
        
        hbond_alpha = self.hbonds['HDONOR-LIGAND_BACKBONE_ALPHA'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_ALPHA'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_ALPHA'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_ALPHA']
        hbond_beta = self.hbonds['HDONOR-LIGAND_BACKBONE_BETA'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_BETA'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_BETA'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_BETA']
        hbond_other = self.hbonds['HDONOR-LIGAND_BACKBONE_OTHER'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_OTHER'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_OTHER'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_OTHER']
        self.hbonds_by_structure = {'ALPHA':hbond_alpha, 'BETA':hbond_beta, 'OTHER':hbond_other}

        all = self.hbonds['HDONOR-LIGAND_BACKBONE_ALPHA'] + self.hbonds['HDONOR-LIGAND_BACKBONE_BETA'] + self.hbonds['HDONOR-LIGAND_BACKBONE_OTHER'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_ALPHA'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_BETA'] + self.hbonds['HDONOR-LIGAND_SIDECHAIN_OTHER'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_ALPHA'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_BETA'] + self.hbonds['HDONOR-RECEPTOR_BACKBONE_OTHER'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_ALPHA'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_BETA'] + self.hbonds['HDONOR-RECEPTOR_SIDECHAIN_OTHER']
        self.hbonds_all = {'all':all}

        self.hydrophobics = {'BACKBONE_ALPHA': 0, 'BACKBONE_BETA': 0, 'BACKBONE_OTHER': 0, 'SIDECHAIN_ALPHA': 0, 'SIDECHAIN_BETA': 0, 'SIDECHAIN_OTHER': 0}
        for key in data['hydrophobics']:
            self.hydrophobics[key] = data['hydrophobics'][key]

        alpha_tmp = self.hydrophobics['BACKBONE_ALPHA'] + self.hydrophobics['SIDECHAIN_ALPHA']
        beta_tmp = self.hydrophobics['BACKBONE_BETA'] + self.hydrophobics['SIDECHAIN_BETA']
        other_tmp = self.hydrophobics['BACKBONE_OTHER'] + self.hydrophobics['SIDECHAIN_OTHER']
        self.hydrophobics_by_structure = {'ALPHA':alpha_tmp, 'BETA':beta_tmp, 'OTHER':other_tmp}

        backbone_tmp = self.hydrophobics['BACKBONE_ALPHA'] + self.hydrophobics['BACKBONE_BETA'] + self.hydrophobics['BACKBONE_OTHER']
        sidechain_tmp = self.hydrophobics['SIDECHAIN_ALPHA'] + self.hydrophobics['SIDECHAIN_BETA'] + self.hydrophobics['SIDECHAIN_OTHER']
        self.hydrophobics_by_backbone_or_sidechain = {'BACKBONE':backbone_tmp, 'SIDECHAIN':sidechain_tmp}

        all = self.hydrophobics['BACKBONE_ALPHA'] + self.hydrophobics['BACKBONE_BETA'] + self.hydrophobics['BACKBONE_OTHER'] + self.hydrophobics['SIDECHAIN_ALPHA'] + self.hydrophobics['SIDECHAIN_BETA'] + self.hydrophobics['SIDECHAIN_OTHER']
        self.hydrophobics_all = {'all':all}        

        stacking_tmp = {}
        for item in data['stacking']:
            item = item.split("_")
            stacking_tmp[item[1]] = int(item[2])
        self.stacking = {'ALPHA': 0, 'BETA': 0, 'OTHER': 0}
        for key in stacking_tmp:
            self.stacking[key] = stacking_tmp[key]
        
        all = self.stacking['ALPHA'] + self.stacking['BETA'] + self.stacking['OTHER']
        self.stacking_all = {'all': all}
        
        pi_cation_tmp = {}
        for item in data['pi_cation']:
            item = item.split("_")
            pi_cation_tmp[item[1] + "_" + item[2]] = int(item[3])
        self.pi_cation = {'LIGAND-CHARGED_ALPHA': 0, 'LIGAND-CHARGED_BETA': 0, 'LIGAND-CHARGED_OTHER': 0, 'RECEPTOR-CHARGED_ALPHA': 0, 'RECEPTOR-CHARGED_BETA': 0, 'RECEPTOR-CHARGED_OTHER': 0}
        for key in pi_cation_tmp:
            self.pi_cation[key] = pi_cation_tmp[key]

        pi_cation_ligand_charged = self.pi_cation['LIGAND-CHARGED_ALPHA'] + self.pi_cation['LIGAND-CHARGED_BETA'] + self.pi_cation['LIGAND-CHARGED_OTHER']
        pi_cation_receptor_charged = self.pi_cation['RECEPTOR-CHARGED_ALPHA'] + self.pi_cation['RECEPTOR-CHARGED_BETA'] + self.pi_cation['RECEPTOR-CHARGED_OTHER']
        self.pi_cation_charge_location = {'LIGAND':pi_cation_ligand_charged, 'RECEPTOR':pi_cation_receptor_charged}
        
        pi_cation_alpha = self.pi_cation['LIGAND-CHARGED_ALPHA'] + self.pi_cation['RECEPTOR-CHARGED_ALPHA']
        pi_cation_beta = self.pi_cation['LIGAND-CHARGED_BETA'] + self.pi_cation['RECEPTOR-CHARGED_BETA']
        pi_cation_other = self.pi_cation['LIGAND-CHARGED_OTHER'] + self.pi_cation['RECEPTOR-CHARGED_OTHER']
        self.pi_cation_by_structure = {'ALPHA':pi_cation_alpha, 'BETA':pi_cation_beta, "OTHER":pi_cation_other}

        all = self.pi_cation['LIGAND-CHARGED_ALPHA'] + self.pi_cation['LIGAND-CHARGED_BETA'] + self.pi_cation['LIGAND-CHARGED_OTHER'] + self.pi_cation['RECEPTOR-CHARGED_ALPHA'] + self.pi_cation['RECEPTOR-CHARGED_BETA'] + self.pi_cation['RECEPTOR-CHARGED_OTHER']
        self.pi_cation_all = {'all': all}

        t_shaped_tmp = {}
        for item in data['t_shaped']:
            item = item.split("_")
            t_shaped_tmp[item[1]] = int(item[2])
        self.t_shaped = {'ALPHA': 0, 'BETA': 0, 'OTHER': 0}
        for key in t_shaped_tmp:
            self.t_shaped[key] = t_shaped_tmp[key]

        all = self.t_shaped['ALPHA'] + self.t_shaped['BETA'] + self.t_shaped['OTHER']
        self.t_shaped_all = {'all': all}

        self.salt_bridges = {'ALPHA': 0, 'BETA': 0, 'OTHER': 0}
        for key in data['salt_bridges']:
            key2 = key.replace("SALT-BRIDGE_","")
            self.salt_bridges[key2] = data['salt_bridges'][key]
        
        all = self.salt_bridges['ALPHA'] + self.salt_bridges['BETA'] + self.salt_bridges['OTHER']
        self.salt_bridges_all = {'all': all}

        self.input_vector = []
        self.input_vector.extend(self.vina_output) # a list
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.ligand_receptor_atom_type_pairs_less_than_four)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.ligand_receptor_atom_type_pairs_electrostatic)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.ligand_atom_types)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.ligand_receptor_atom_type_pairs_less_than_two_half)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.hbonds)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.hydrophobics)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.stacking)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.pi_cation)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.t_shaped)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.active_site_flexibility)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.salt_bridges)
        self.input_vector = self.extend_list_by_dictionary(self.input_vector, self.rotateable_bonds_count)

class command_line_parameters:
    
    params = {}
    
    def __init__(self, parameters):
        
        #global vina_executable
        
        # first, set defaults
        self.params['receptor'] = ''
        self.params['ligand'] = ''
        self.params['vina_executable'] = '' #vina_executable
        self.params['check_vina_version'] = "TRUE" # TRUE by default, but setting to false will speed up execution. Good when rescoring many poses.

        # now get user inputed values

        for index in range(len(parameters)):
            item = parameters[index]
            if item[:1] == '-': # so it's a parameter key value
                key = item.replace('-','').lower()

                if key == "help":
                    print ("INTRODUCTION")
                    print ("============\n")
                    print (textwrap.fill("NNScore 2.01 (NNScore2.py) is a python script for predicting the binding affinity of receptor-ligand complexes. It is particularly well suited for rescoring ligands that have been docked using AutoDock Vina.") + "\n")
                    print ("REQUIREMENTS")
                    print ("============\n")
                    print (textwrap.fill("Python: NNScore 2.01 has been tested using Python versions 2.6.5, 2.6.1, and 2.5.2 on Ubuntu 10.04.1 LTS, Mac OS X 10.6.8, and Windows XP Professional, respectively. A copy of the Python interpreter can be downloaded from http://www.python.org/getit/.") + "\n")
                    print (textwrap.fill("AutoDock Vina 1.1.2: NNScore 2.01 uses AutoDock Vina 1.1.2 to obtain some information about the receptor-ligand complex. Note that previous versions of AutoDock Vina are not suitble. AutoDock Vina 1.1.2 can be downloaded from http://vina.scripps.edu/download.html.") + "\n")
                    print (textwrap.fill("MGLTools: As receptor and ligand inputs, NNScore 2.01 accepts models in the PDBQT format. Files in the more common PDB format can be converted to the PDBQT format using scripts included in MGLTools (prepare_receptor4.py and prepare_ligand4.py). MGLTools can be obtained from http://mgltools.scripps.edu/downloads.") + "\n")
                    print ("COMMAND-LINE PARAMETERS")
                    print ("=======================\n")
                    print (textwrap.fill("-receptor: File name of the receptor PDBQT file.") + "\n")
                    print (textwrap.fill("-ligand: File name of the ligand PDBQT file. AutoDock Vina output files, typically containing multiple poses, are also permitted.") + "\n")
                    print (textwrap.fill("-vina_executable: The location of the AutoDock Vina 1.1.2 executable. If you don't wish to specify the location of this file every time you use NNScore 2.01, simply edit the vina_executable variable defined near the begining of the NNScore2.py script.") + "\n")
                    print ("PROGRAM OUTPUT")
                    print ("==============\n")
                    print (textwrap.fill("NNScore 2.01 evaluates each of the ligand poses contained in the file specified by the -ligand tag using 20 distinct neural-network scoring functions. The program then seeks to identify which of the poses has the highest predicted affinity using several metrics:") + "\n")
                    print (textwrap.fill("1) Each of the 20 networks are considered separately. The poses are ranked in 20 different ways by the scores assigned by each network.") + "\n")
                    print (textwrap.fill("2) The poses are ranked by the best score given by any of the 20 networks.") + "\n")
                    print (textwrap.fill("3) The poses are ranked by the average of the scores given by the 20 networks. This is the recommended metric.") + "\n")


                    sys.exit(0)

                value = parameters[index+1]
                if key in self.params.keys():
                    self.params[key] = value
                    parameters[index] = ""
                    parameters[index + 1] = ""
        
        if self.okay_to_proceed() == True:
            print ("Command-line parameters used:")
            print ("\tReceptor:        " + self.params["receptor"])
            print ("\tLigand:          " + self.params["ligand"])
            print ("\tVina executable: " + self.params["vina_executable"] + "\n")
        
        # make a list of all the command-line parameters not used
        error = ""
        for index in range(1,len(parameters)):
            item = parameters[index]
            if item != "": error = error + item + " "
        
        if error != "":
            print ("WARNING: The following command-line parameters were not used:")
            print ("\t" + error + "\n")
    

        # do a check to make sure the autodock vina is version 1.1.2
        if self.params["check_vina_version"] == "TRUE":
            vina_version_output = ""
            if not os.path.exists(self.params['vina_executable']):
                vina_version_output = ""
            else:
                try:
                    vina_version_output = getCommandOutput2(self.params['vina_executable'] + ' --version')
                except:
                    vina_version_output = ""
    
            if not " 1.1.2 " in vina_version_output:
                print ("ERROR: NNScore 2.01 is designed to work with AutoDock Vina 1.1.2.\nOther versions of AutoDock may not work properly. Please download\nAutoDock Vina 1.1.2 from http://vina.scripps.edu/download.html.\n")
                print ("Once this executable is downloaded, you can use the -vina_executable\ntag to indicate its location. Alternatively, you can simply modify\nthe vina_executable variable defined near the beginning of\nthe NNScore2.py file.\n")
                sys.exit(0)


    def okay_to_proceed(self):
        if self.params['receptor'] != '' and self.params['ligand'] != '' and self.params['vina_executable'] != '':
            return True
        else: return False

def score_to_kd(score):
    kd = math.pow(10,-score)
    if score <= 0: return "Kd = " + str(round(kd,2)) + " M"
    temp = kd * pow(10,3)
    if temp >=1 and temp <1000: return "Kd = " + str(round(temp,2)) + " mM"
    temp = temp * pow(10,3)
    if temp >=1 and temp <1000: return "Kd = " + str(round(temp,2)) + " uM"
    temp = temp * pow(10,3)
    if temp >=1 and temp <1000: return "Kd = " + str(round(temp,2)) + " nM"
    temp = temp * pow(10,3)
    if temp >=1 and temp <1000: return "Kd = " + str(round(temp,2)) + " pM"
    temp = temp * pow(10,3)
    #if temp >=1 and temp <1000:
    return "Kd = " + str(round(temp,2)) + " fM"
    #return "Kd = " + str(kd) + " M"


def calculate_score(lig, rec, cmd_params, nb_nets, actual_filename_if_lig_is_list="", actual_filename_if_rec_is_list="", line_header = ""):

        d = binana(lig, rec, cmd_params, line_header, actual_filename_if_lig_is_list, actual_filename_if_rec_is_list)
        # now load in neural networks
        scores = []
        total = 0.0
        nets = []
        #with open(os.path.join(networks_dir, "networks.pickle"), "rb") as pickle_file:
        #        nets = pickle.load(pickle_file) 
        
        try:      
            with open(os.path.join(networks_dir, "networks.pickle"), "rb") as f:
                nets = pickle.load(f)
        except IOError:
            print("ERROR: 'Networks.pickle' FILE NOT FOUND")
            sys.exit(1)
        
        output_dict = {}
        # Save the pdb id
        #f = re.findall('\w\w\w\w/', actual_filename_if_lig_is_list)
        #pdb_id = f[len(f)-1].strip('/')
        #output_dict['pdb_id'] = pdb_id
        # Add vina output 
        output_dict['vina_output'] = d.input_vector[0]
        nnscores = []
        dlscores = []
        for net_array in nets:
                try:
                    net = ffnet()
                    net.load(net_array)
                    val = net.normcall(d.input_vector)
                    nnscores.append(val)
                except OverflowError:
                    print (line_header + "The output of network #" + str(len(scores) + 1) + " could not be determined because of an overflow error!")

        #try:
        # Session and memory management (Requires for GPU instance of tensorflow)
        #config = K.tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #config.gpu_options.visible_device_list = "0"
        #K.set_session(K.tf.Session(config=config))
                        
        # Data processing
        input_data = np.array(d.input_vector)
        
        try:
            with open(os.path.join(networks_dir, "transform.pickle"), "rb") as f:
                transform = pickle.load(f)
        except IOError:
            print("ERROR: 'transform.pickle' FILE NOT FOUND")
            sys.exit(1)
                
        input_data = (input_data - transform['mean'])/ (transform['std'] + 0.0001)
                
        # Load the neural networks
        count = 1
        #nb_nets = 10
        for dl_net in dl_nets(nb_nets):
            val = dl_net.predict(input_data.reshape(1, -1))[0][0]
            dlscores.append(val)
            count = count + 1
            # Delete the model and clear the session. 
            del dl_net
            K.clear_session()               
                
        #except IOError:
        #        print("Could not load the network file")
        #        sys.exit(1)	
        
        if (len(nnscores) > 0):
            output_dict['nnscore'] = nnscores
            output_dict['dlscore'] = dlscores
            return output_dict
        else:
            print (line_header + "Could not compute the score of this receptor-ligand complex because none of the networks returned a valid score.")
            return {}


class dlscore():
    parameters = []
    ligand = ''
    receptor = ''
    vina_executable = ''
    nb_nets = 10
    
    def __init__(self, ligand, receptor, vina_executable, nb_nets=10):
        self.ligand = ligand
        self.receptor = receptor
        self.vina_executable = vina_executable
        self.nb_nets = nb_nets
    
    def which(self, program):
        """ Find and return the path of the specified program."""
        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file

        return None
        
    def check_inputs(self):
        """
        Checks the input files. Convert mol2 and pdb files into pdbqt files if necessary.
        :return: New files with pdbqt extension.
        """
        l_filename, l_file_ext = os.path.splitext(self.ligand)
        r_filename, r_file_ext = os.path.splitext(self.receptor)
        if l_file_ext == '.pdb' or l_file_ext == '.mol2':
            # Check if pythonsh is available or not
            pythonsh = self.which("pythonsh")
            if pythonsh == None:
                print("ERROR: pythonsh commmand not found. Please install mgltools and define pythonsh in the ~/.bashrc")
            
            # Convert files
            out_file = l_filename + '.pdbqt'
            cmd = "pythonsh " + os.path.join(mgltools_dir, "prepare_ligand4.py") + " -l " + self.ligand + " -o " + out_file
            os.system(cmd)
            self.ligand = out_file

        if r_file_ext == '.pdb' or r_file_ext == '.mol2':
            # Check if pythonsh is available or not
            pythonsh = self.which("pythonsh")
            if pythonsh == None:
                print("ERROR: pythonsh commmand not found. Please install mgltools and define pythonsh in the ~/.bashrc")
            
            # Convert files
            out_file = r_filename +'.pdbqt'
            cmd = "pythonsh " + os.path.join(mgltools_dir, "prepare_receptor4.py") + " -U  nphs_lps_waters -r " + \
                  self.receptor + " -o " + out_file
            os.system(cmd)
            self.receptor = out_file


    def get_output(self):
        """
        Program output
        :return: A list of dictionaries for each of the ligand files
        """
        # Checking the input file format
        self.check_inputs()
        self.parameters = ['-ligand', self.ligand,
                           '-receptor', self.receptor,
                           '-vina_executable', self.vina_executable]

        input_parameters = command_line_parameters(self.parameters)
        if input_parameters.okay_to_proceed() == False:
            return {}
        
        lig = input_parameters.params['ligand']
        rec = input_parameters.params['receptor']
        receptor = PDB()
        receptor.LoadPDB_from_file(rec)
        receptor.OrigFileName = rec
        f = open(lig,'r')
        lig_array = []
        line = "NULL"
        scores = []
        model_id = 1
        while len(line) != 0:
            line = f.readline()
            if line[:6] != "ENDMDL": lig_array.append(line)
            if line[:6] == "ENDMDL" or len(line) == 0:
                if len(lig_array) != 0 and lig_array != ['']:
                    temp_filename = lig + ".MODEL_" + str(model_id) + ".pdbqt"
                    temp_f = open(temp_filename, 'w')
                    for ln in lig_array: temp_f.write(ln)
                    temp_f.close()
                    model_name = "MODEL " + str(model_id)
                    score=calculate_score(lig_array, receptor, input_parameters, self.nb_nets, temp_filename, rec, "\t")
                    score['dlscore'] = sum(score['dlscore']) / len(score['dlscore'])
                    score['nnscore'] = sum(score['nnscore']) / len(score['nnscore'])
                    scores.append(score)
                    os.remove(temp_filename)
                    lig_array = []
                    model_id = model_id + 1

        f.close()
        return scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Main script for running DLSCORE")
    parser.add_argument('--ligand', action='store', dest='ligand', required=True,
                        help='Ligand file. Supported file types: .pdb, .pdbqt, .mol2')
    parser.add_argument('--receptor', action='store', dest='receptor', required=True,
                        help='Receptor file. Supported file types: .pdb, .pdbqt, .mol2')
    parser.add_argument('--vina_executable', action='store', dest='vina_executable',
                        default=vina_path,
                        help='File path for Vina executable.')
    parser.add_argument('--num_networks', action='store', dest='num_networks', type=int,
                        default=10,
                        help='Number of networks to use for prediction. Default:10')
    parser.add_argument('--output', action='store', dest='output_file',
                        default='out',
                        help='Name of the output file. Dafault: out.csv')
    parser.add_argument('--network_type', action='store', dest='network_type',
                        choices=['refined', 'general'],
                        default='refined',
                        help='DLSCORE has two types of weights trained on PDB-Bind 2016 refined set and general'\
                        ' set (including refined). Any of these two variants can be used. Possible options are: '\
                        'general and refined. Dafault is set to general.')
    parser.add_argument('--verbose', action='store', dest='verbose', type=int,
                        choices=[0, 1],
                        default=0,
                        help='Verbose mode. False if 0, True if 1. Default is set to False.')
    
    arg_dict = vars(parser.parse_args())
    
    # Get the arguments
    ligand = arg_dict['ligand']
    receptor = arg_dict['receptor']
    vina_executable = arg_dict['vina_executable']
    num_networks = arg_dict['num_networks']
    output_file = arg_dict['output_file']
    network_type = arg_dict['network_type']
    verbose = True if arg_dict['verbose']==1 else False
    
    if network_type == 'general':
        networks_dir = os.path.join(current_dir, "networks/general")
    else:
        networks_dir = os.path.join(current_dir, "networks/refined")
    
    # Get an instance of dlscore
    ds = dlscore(ligand, receptor, vina_executable, num_networks)
    
    # Get the output
    output = ds.get_output()
    
    # Write the output file if requested
    if output_file != '':
        with open(output_file + '.csv', 'w') as f:
            w = csv.DictWriter(f, fieldnames=["vina_output", "nnscore", "dlscore"])
            w.writeheader()
            for d in output:
                w.writerow(d)

    if verbose: print("DLSCORE OUTPUT: " + str(output))
    
