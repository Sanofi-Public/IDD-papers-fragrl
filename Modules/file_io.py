import os, shutil
import ast
from rdkit import Chem
import Modules.global_parameters as gl

######################### MODULE FOR READING AND WRITING FILES ######################

# deletes a file if it exists
def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)

# deletes a folder if it exists        
def delete_folder(foldername):
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
        
# creates a folder if it doesn't already exists        
def create_folder(foldername):
    if not os.path.exists("{}/".format(foldername)):
        os.makedirs(foldername)
        
# check for a list of files if they all exist        
def check_files(filelist):
    ret = True
    for filename in filelist:
        if os.path.exists(filename) == False:
            ret = False
            print("Error! File {} does not exist!".format(filename))
    return ret
        

# Read a file containing SMILES
# The file should be a .smi or a .csv where the first column should contain a SMILES string
# file_type = MOL if file contains molecules, FRAG if it contains fragments
# as_smiles: if True a list of smiles is returned, otherwise a list of molecules
# if it reads in molecules, chiral information (@) is removed
def read_molfile(file_name, file_type="MOL", as_smiles=False, remove_chiral=True):
    
    # if file is empty
    if os.stat(file_name).st_size == 0:
        return []
    
    # Drop salt from SMILES string
    def drop_salt(s):
        s = s.split(".")
        return [x for _, x in sorted(zip(map(len,s), s), reverse=True)][0]
    
    # checks if there are elements with atomic number > MOL_SPLIT_START in molecule
    # if yes: this molecule cannot be used
    def check_for_heavy_elements(s):
        mol = Chem.MolFromSmiles(s)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() >= gl.PARAMS["MOL_SPLIT_START"]:
                return False
        return True
    
    # function that removes chiral centers from a molecule
    def remove_chiral(mol):
        smi = Chem.MolToSmiles(mol)
        smi = smi.replace("@", "")
        return Chem.MolFromSmiles(smi)
    
    drop_first = False # first line to be dropped?
    returns = []       # might be molecule objects or smiles
    
    with open(file_name) as f:
        lines = f.readlines()
        
        # check if there is a molecule in first line
        if Chem.MolFromSmiles(drop_salt(lines[0].strip().split(",")[0].strip())) == None:  
            drop_first = True
            
        for l in lines:
            if drop_first:              
                drop_first = False
                continue
            # rest of the lines (containing structures in first column)    
            l = l.strip().split(",")[0]
            smi = drop_salt(l.strip())
            okay = True
            if file_type == "MOL":
                okay = check_for_heavy_elements(smi)
            if okay:
                if as_smiles == False:
                    if remove_chiral:
                        try:
                            mol = remove_chiral(Chem.MolFromSmiles(smi))
                            if mol != None:
                                returns.append(mol)
                            else:
                                raise Exception("Invalid Molecule")
                        except:
                            print("Broken input molecule:", smi)
                    else:
                        try:
                            returns.append(Chem.MolFromSmiles(smi))
                        except:
                            print("Broken input molecule:", smi)
                else:
                    returns.append(smi)
                
    return returns


# read lead fragments from fragment file
def read_fragments(filename):
    with open(filename) as inp:
        lines = inp.readlines()
    
    lead_sets = []    
    for i, line in enumerate(lines):
        if i == 0:
            continue
        linelist = line.strip("\n").split(",")
        smiles = linelist[2].split(".")
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        lead_sets.append(mols)
    return lead_sets


# write fragment file
# fragment_sets = list where every element corresponds to fragments that can build up one molecule
# molecules: list of molecules formed by the fragment sets (in the same order)
# filename: name of the output file
def write_fragments(fragment_sets, molecules, filename):
    with open(filename, "w") as out:
        out.write("SMILES,Name,Fragments\n")
        for i, frags in enumerate(fragment_sets):
            out.write("{},Mol{},".format(Chem.MolToSmiles(molecules[i]), i) +
                      ".".join([Chem.MolToSmiles(f) for f in frags]) + "\n")


# read configuration from configuration file
def read_config(config_file):
    
    if os.path.exists(config_file) == False:
        raise Exception("Error! Configuration file does not exist: {}".format(config_file))
    
    def string_to_bool(string):
        if string == "True":
            return True
        elif string == "False":
            return False
        else:
            print("Unvalid literal for bool: {}".format(string))
    
    with open(config_file) as inp:
        lines = inp.readlines()
    
    for line in lines:
        info = line.split()                                         # first item is parameter name, third is value
        if line.startswith('#') == False and len(info) > 0:         # comments start with #
            if info[0] not in gl.PARAMS.keys():
                print("Unknown parameter {} in configuration file".format(info[0]))
            elif (info[0] == "MORE_OUTPUT" or info[0] == "WRITE_DECODINGS" or info[0] == "CLEAN_GOOD" or 
                  info[0] == "ONLY_FINAL_SETS" or info[0] == "REACTION_JUMP" or info[0] == "USE_CLASSES"):
                gl.PARAMS[info[0]] = string_to_bool(info[2])   # bool
            elif info[0] == "ETA" or info[0] == "GAMMA" or info[0] == "MAX_SCORE":
                gl.PARAMS[info[0]] = float(info[2])            # float
            elif (info[0] == "SPLIT_SMARTS" or info[0] == "FREEZE_FRAGS" or info[0] == "FRAGMENT_FILES" or
                  info[0] == "ATTACHMENTS" or  info[0] == "LOAD_MODEL" or info[0] == "LEAD_MOLECULES"):
                gl.PARAMS[info[0]] = info[2].split(",")        # list of strings
            elif (info[0] == "DECODINGS_FILE" or info[0] == "LEAD_FILE" or info[0] == "MODEL_CONF" or
                  info[0] == "RXN_FOLDER" or info[0] == "SCORING_FUNCTION" or info[0] == "REACTIONFILE"or
                  info[0] == "BBLOCKS" or info[0] == "MODIFY_PROBS" or info[0] == "LEAD_FRAGMENTS"): 
                gl.PARAMS[info[0]] = info[2]                   # string
            elif info[0] == "SF_KWARGS":
                gl.PARAMS[info[0]] = ast.literal_eval(info[2])             # dictionary
            else:
                gl.PARAMS[info[0]] = int(info[2])              # int    


# writes a matrix of smiles into a csv file
def smilesMatrix_to_csv(smiles_matrix, filename):
    with open(filename, "w") as out:
        for line in smiles_matrix:
            out.write(",".join(line)+"\n")
            
# writes list of smiles to csv            
def smiles_to_csv(smiles, filename):
    with open(filename, "w") as out:
        out.write("SMILES,NAME\n")
        for i, s in enumerate(smiles):
            out.write("{},mol{}\n".format(s, i))

# writes list of smiles to smi file (can be viewed with vida)               
def smiles_to_smi(smiles, filename, namelist=[]):
    with open(filename, "w") as out:
        for i, s in enumerate(smiles):
            if len(namelist) != len(smiles):
                out.write("{} mol{}\n".format(s, i))
            else:
                out.write("{} {}\n".format(s, namelist[i]))


# write smiles codes of a list of fragments to csv    
# frags = list of fragments (as rdkit.Mol)
# names = list of names, each corresponding to one molecule 
# filename = name of the file that is created
# titles = titles of the columns in the csv file ('None' if you don't want titles)
#          if more columns than smiles and names are desired, 
#          their content has to be stored in the names, separated by comma
# mode = should stuff be written into a new file or appended to an already existing one?
def fragments_to_smilesCsv(frags, names, filename, titles = ["SMILES", "NAME"], mode="w"):
    with open(filename, mode) as out:
        if titles:
            out.write(",".join(titles)+"\n")
        for i, frag in enumerate(frags):
            try:
                smi = Chem.MolToSmiles(frag)
                out.write(smi+","+names[i]+"\n")
            except:
                pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
