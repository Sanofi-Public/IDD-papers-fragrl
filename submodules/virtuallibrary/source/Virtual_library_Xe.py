'''
@author: Christoph Grebner
@version 1.0
@since: 07.11.2019
'''

import re
import glob
import os
import openeye.oechem as oechem
import openeye.oeomega as oeomega
from rdkit import Chem
from rdkit.Chem import Lipinski
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.rdMolDescriptors as MolDescriptors


class Virtual_Library_Xe():
    
    def __init__(self):
        self.folders_for_reagents = True
        self.reactions_dict = dict()  
        self.reagents_dict = dict()   
        self.reagent_folder = "./BBlocks"
        self.reaction_list = "./reactions.txt"  


    def init_reactions(self, reaction_file_name):
        with open(reaction_file_name) as reaction_file:
            for line in reaction_file:
                if(line[0]=="#"):  # ignore all lines starting with #
                    continue
                line = line.rstrip('\n')
                line = line.split()
                if line[0] not in self.reactions_dict:
                    reaction_info = []
                    reaction_info.append(line[1])
                    reaction_info.append(line[2].split(','))
                    reaction_info.append(line[3].split(','))
                    reaction_info.append(line[4].split(','))
                    reaction_info.append(line[5].split(','))
                    self.reactions_dict[line[0]] = reaction_info
                else:
                    print("Already added: ", line)    
        
    def init_reagents(self, reagent_folder):
        folders = os.listdir(reagent_folder)
        for reagent_class in folders:
            reagent_file_name = glob.glob(reagent_folder+"/"+reagent_class+"/"+"*_final_R1.smi")
            if len(reagent_file_name) == 1:
                try:
                    with open(reagent_file_name[0]) as reagent_file:
                        for line in reagent_file:
                            line = line.rstrip('\n')
                            line = line.split()
                            if len(line) == 2:
                                reagent_smiles = line[0]
                                reagent_id = line[1]
                                if reagent_class not in self.reagents_dict:
                                    self.reagents_dict[reagent_class] = {}
                                    self.reagents_dict[reagent_class][reagent_id] = reagent_smiles
                                else:
                                    self.reagents_dict[reagent_class][reagent_id] = reagent_smiles
                            elif len(line) < 2:
                                print("Warning! Either molecule name or SMILES missing in reagent file.")
                            else:
                                print("Too much information in reagent file. Ignoring line.")
                except:
                    pass
            elif len(reagent_file_name)>1:
                print("More than one R1-file for: ", reagent_class)

    def MergeXeSmiles(self, reaction_smiles, reagent_smiles, xe_flag, convert):
        linker_number = "99"
        if xe_flag == "Xe":
            linker_number = "92"
        elif xe_flag == "Kr":
            linker_number = "93"
        elif xe_flag == "Ar":
            linker_number = "94"
        elif xe_flag == "Ne":
            linker_number = "95"

        # just linking of two compounds
        if reaction_smiles == "[Kr][Xe]":
            product_smiles = reagent_smiles.replace("Xe", "Kr")
        elif reaction_smiles == "[Xe][Kr]":
            product_smiles = reagent_smiles.replace("Xe", "Kr")
        elif reaction_smiles == "[Xe]=[Kr]":
            product_smiles = reagent_smiles.replace("[Xe]", "=[Kr]")
        # usual, standard case
        else:  
            smiles1 = self.rep_smi(reaction_smiles, xe_flag, linker_number)
            smiles2 = self.rep_smi(reagent_smiles, "Xe", linker_number)
            product_smiles = smiles1+"."+smiles2
        
        if convert: 
            mol = oechem.OEMol()
            oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Warning)

            try: 
                oechem.OEParseSmiles(mol, product_smiles)
            except:
                print("Some error/warning occured: OEParseSmiles")
            oechem.OEPerceiveChiral(mol)
            for flipmol in oeomega.OEFlipper(mol, 0, False, False, False):
                mol = flipmol          
            try:
                product_smiles = oechem.OECreateIsoSmiString(mol)
            except:
                print("Some error/warning occured: OECreateIsoSmiString")       
            del mol
        return product_smiles


    def rep_smi(self, smiles, xe_flag, linker_number):
        smiles1=""
        regex=re.compile('^\['+xe_flag+']')
        if re.match(regex, smiles):
            xe_pos = smiles.find("["+xe_flag+"]")
            smiles1 = smiles.replace("["+xe_flag+"]", "")
            if smiles1[xe_pos:(xe_pos+1)].isalpha():
                if len(smiles1) > 1 and smiles1[(xe_pos+1)].isdigit():
                        smiles1 = smiles1[xe_pos:(xe_pos+2)]+"%"+linker_number+smiles1[(xe_pos+2):]
                else:
                    smiles1 = smiles1[xe_pos:(xe_pos+1)]+"%"+linker_number+smiles1[(xe_pos+1):]
            else:
                 print("please remove stereoinfo from inputmolecules and building blocks")
                 exit()
        elif "(["+xe_flag+"])" in smiles:
            smiles1 = smiles.replace("(["+xe_flag+"])", "%"+linker_number)
        elif "["+xe_flag+"]" in smiles:
            smiles1 = smiles.replace("["+xe_flag+"]", "%"+linker_number)
        return smiles1
