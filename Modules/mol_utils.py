import re
import glob
import itertools
import datetime
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import Modules.global_parameters as gl
import Modules.file_io as file_io
from submodules.virtuallibrary.source.RouteFinder import RouteFinderAttachments, create_libgen_interfaces


# Main module for handling the interactions with molecules

# periodic table
PSE = ["XX",
   "H",                                                                                                                                                                                       "He",
   "Li", "Be",                                                                                                                                                  "B",  "C",  "N",  "O",  "F",  "Ne",
   "Na", "Mg",                                                                                                                                                  "Al", "Si", "P",  "S",  "Cl", "Ar",
   "K",  "Ca",                                                                                      "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
   "Rb", "Sr",                                                                                      "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
   "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",  "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]


# sorts fragments so they can be joined together later
# smi_list: list of fragments as SMILES
#           attachment points are marked with numbers where same number means there was a split between them  
def sort_fragments(smi_list):
    
    # find the first fragment in smi_list which has attachment point with number n
    def find_fragment_with_attachment(n):
        for i, smi in enumerate(smi_list):
            if "[{}*]".format(n) in smi:
                return smi, i
    
    # replaces all attachment points except the first with rare element symbols
    # smi = SMILES string of fragment with attachment points as numbers
    # fa = numbers that mark attachment points to be replaced
    def replace_further_attachments(smi, fa):
        for i, a, in enumerate(fa):
            smi = smi.replace("[{}*]".format(a), "[{}]".format(PSE[gl.PARAMS["MOL_SPLIT_START"]+i+1]))
        return smi
    
    fragments = []              # list of fragments that will be returned
    a = re.compile("\[\d+\*]")  # search pattern for "[<number>*]" with regex
    next_attach = [1,1]         # keep track of which attachment point will be treated next (at first both fragments with "1" have to be found)
    counter = 0
    
    while counter < len(next_attach):  # go through the attachment points
        smi, index = find_fragment_with_attachment(next_attach[counter])  # find the next fragment
        smi = smi.replace("[{}*]".format(next_attach[counter]), "[{}]".format(PSE[gl.PARAMS["MOL_SPLIT_START"]])) # replace first attachment point
        further_attachments = match_to_int(a.findall(smi))           # find further attachment points...
        smi = replace_further_attachments(smi, further_attachments)  # ...and replace them
        smi_list[index] = smi  # replace fragment in smi_list so this fragment can't be found again
        fragments.append(Chem.MolFromSmiles(smi))  # append fragment to results
        next_attach += further_attachments         # append next attachment points   
        counter += 1                               # go to next attachment point in list
    
    return fragments


# Split a molecule into fragments   
# result is a list of molecules where each field is a fragment where attachment points are rare elements 
def split_molecule(mol):
    
    # check if both atoms of atom pair are in one of the freezed fragments
    def check_freeze(atom_pair, freeze_indices):
        for fr in freeze_indices:
            if atom_pair[0] in fr and atom_pair[1] in fr:
                return True
        return False
    
    # find atom indices of freezed fragments
    freeze_indices = []  # one tuple of atom indices for every fragment in molecule that is freezed
    for ff_smi in gl.PARAMS["FREEZE_FRAGS"]:
        ff = Chem.MolFromSmiles(ff_smi)
        idxs = mol.GetSubstructMatch(ff)
        if len(idxs) != 0:
            freeze_indices.append(idxs)
      
    # find bonds according to pattern
    bis = []   # list of tuples with atom pairs corresponding to bonds to be splitted
    for pattern in gl.PARAMS["SPLIT_SMARTS"]:
        bi = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))  # find pattern
        for b in bi:
            if check_freeze(b, freeze_indices) == False:
                bis.append(tuple(sorted(b)))  # bond (1,2) is the same as bond (2,1)
    bis = list(set(bis))  # remove bonds that are counted twice
    bond_indices = []  # bond indices of these bonds
    bond_labels = []   # labels are defined so that attachment points belonging to same bond are marked with same label
    for i,b in enumerate(bis):
        bond_indices.append(mol.GetBondBetweenAtoms(b[0],b[1]).GetIdx())
        bond_labels.append((i+1,i+1))
        
    # fragment at these bonds   
    frag_mol = Chem.FragmentOnBonds(mol, bond_indices, dummyLabels=bond_labels)  
    smi_list = Chem.MolToSmiles(frag_mol).split(".")
    
    # sort the fragments, so they can be joined together later
    return sort_fragments(smi_list)


# Split a molecule into fragments according to rxn files (given as Library_Generation_Interface objects)
# result is a list of fragment sets where each field is a fragment where attachment points are rare elements 
def split_molecule_rxn(mol, lib_gens):
    
    # count how often each freezed fragment appears in a molecule or in a list of molecules
    def count_substructure_matches(mollist):
        result = [0] * len(gl.PARAMS["FREEZE_FRAGS"])
        for i, freeze in enumerate(gl.PARAMS["FREEZE_FRAGS"]):
            freeze_mol = Chem.MolFromSmiles(freeze)
            for mol in mollist:
                idxs = mol.GetSubstructMatches(freeze_mol)
                result[i] += len(idxs)
        return result
    
    # determine which of the given sets leaves the freezed fragments untouched
    def determine_valid_freeze_sets(sets):
        if len(gl.PARAMS["FREEZE_FRAGS"]) != 0:  # if freezed fragments are given:
            valid_sets = []                      # only those sets may be used that contain the same number of those fragments
            freeze_frags_count = count_substructure_matches([mol])
            for uset in sets:
                uset_as_mol = [Chem.MolFromSmiles(mol) for mol in uset]
                if count_substructure_matches(uset_as_mol) == freeze_frags_count:
                    valid_sets.append(uset)
        else:                                    # if no fragments are freezed:
            valid_sets = sets                    # all sets are valid
        return valid_sets
    
    fragment_sets = []
    
    # main work is done by class RouteFinderAttachments from virtual library submodule
    rfa = RouteFinderAttachments(Chem.MolToSmiles(mol), lib_gens, gl.PARAMS["ATTACHMENTS"])
    rfa.run_wrapper(100)
    if gl.PARAMS["MORE_OUTPUT"]:
        print("{}: Route building finished.".format(datetime.datetime.now()))
    if not gl.PARAMS["ONLY_FINAL_SETS"]:  # use all unique sets
        rfa.create_unique_sets(100)
        sets = rfa.unique_sets
    else:                                 # only use final sets
        rfa.create_final_sets(100)
        sets = rfa.final_sets
    valid_sets = determine_valid_freeze_sets(sets)
    for rs in valid_sets:
        if len(rs) > 1:  # only sets containing more than 1 fragment
            fragment_sets.append(sort_fragments(rs))
    return fragment_sets


# Join a list of fragments together into a molecule
# Throws an exception if it is not possible to join all fragments.
def join_fragments(fragments):
    
    # replaces attachment points marked by rare elements with "[<number>*]"
    # number is defined by attach_counter so order of attachment points remains clear
    def replace_rare_elements_by_numbers(smi):
        nonlocal attach_counter
        for counter in range(gl.PARAMS["MAX_FREE"]):
            smi = smi.replace("[{}]".format(PSE[gl.PARAMS["MOL_SPLIT_START"]+counter]), 
                              "[{}*]".format(attach_counter))
            attach_counter += 1
        return smi
    
    # replace the first attachment point (marked with numbers) by first rare element symbol
    def replace_lowest_number_by_symbol(smi):
        a = re.compile("\[\d+\*]")  # search pattern for "[<number>*]" with regex
        numbers = match_to_int(a.findall(smi))  # find all attachment points
        if len(numbers) == 0:
            raise RuntimeError("No attachment point in fragment.")
        smi = smi.replace("[{}*]".format(min(numbers)), "[{}]".format(PSE[gl.PARAMS["MOL_SPLIT_START"]]))
        return smi
    
    # replace attachment points in growing molecule
    def replace_attachs(frag):
        smi = Chem.MolToSmiles(frag)
        smi = replace_rare_elements_by_numbers(smi)
        smi = replace_lowest_number_by_symbol(smi)
        return Chem.MolFromSmiles(smi)
    
    # removes the first attachment point / dummy atom from an editable mol (after connecting the fragments)
    # removing all attachment points at once is not possible as the indices of the atoms change when removing the first
    # during this the stereo atoms and the atoms connected before are updated according to the new indices
    def remove_first_attach(emol, stereo_atoms, connect_atoms):
        mol = emol.GetMol()
        for i, a in enumerate(mol.GetAtoms()):  # look for attachment points
            if a.GetSymbol() == PSE[gl.PARAMS["MOL_SPLIT_START"]]:
                # remove attachment point (this is the easy part)
                emol.RemoveAtom(a.GetIdx())
                # update stereo atoms
                stereo_new = []
                for sa_pair in stereo_atoms:
                    neighbor_indices = [n.GetIdx() for n in a.GetNeighbors()]
                    # if stereo atom is removed: set it to the newly attached atom
                    if sa_pair[0] == i:
                        if connect_atoms[0] in neighbor_indices:
                            sa_pair[0] = connect_atoms[1]
                        if connect_atoms[1] in neighbor_indices:
                            sa_pair[0] = connect_atoms[0]
                    if sa_pair[1] == i:
                        if connect_atoms[0] in neighbor_indices:
                            sa_pair[1] = connect_atoms[1]
                        if connect_atoms[1] in neighbor_indices:
                            sa_pair[1] = connect_atoms[0]
                    # all indices > i are decreased by 1
                    if sa_pair[0] > i:
                        sa_pair[0] = sa_pair[0]-1
                    if sa_pair[1] > i:
                        sa_pair[1] = sa_pair[1]-1
                    stereo_new.append(sa_pair)
                # update connected atoms (all indices > i are decreased by 1)
                if connect_atoms[0] > i:
                    connect_atoms[0] = connect_atoms[0]-1
                if connect_atoms[1] > i:
                    connect_atoms[1] = connect_atoms[1]-1
                # after having removed first attachment point: exit function
                break
        return emol, stereo_new, connect_atoms
    
    # collects the stereo atoms for each cis or trans double bond
    # this means the atoms neighboring to the bond, to which the E or Z marker refers
    def collect_stereo_atoms(mol):
        stereo_atoms = []
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE and (bond.GetStereo() == Chem.BondStereo.STEREOE 
                                                               or bond.GetStereo() == Chem.BondStereo.STEREOZ):
                stereo_atoms.append([bond.GetStereoAtoms()[0], bond.GetStereoAtoms()[1]])
        return stereo_atoms
    
    # adds the stereo atoms collected above to the molecule
    def create_mol_with_stereo(emol, stereo_atoms):
        mol = emol.GetMol()
        i = 0
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE  and (bond.GetStereo() == Chem.BondStereo.STEREOE 
                                                                or bond.GetStereo() == Chem.BondStereo.STEREOZ):
                bond.SetStereoAtoms(stereo_atoms[i][0], stereo_atoms[i][1])
                i += 1
        AllChem.Compute2DCoords(mol) # this is necessary to get correct stereoinformation in picture and SMILES
        return mol
    
    # connect two fragments at attachment points, corresponding to first rare element symbol (usually Yb)
    def connect_fragments(f1, f2):
        mol = Chem.CombineMols(f1, f2)
        connect_atoms = []  # indices of atoms to be connected
        bond_type = []      # bond types of dangling bonds to be connected (should be the same for both atoms)
        for a in mol.GetAtoms():       # find which atoms to connect (neighbors of dummy atoms)
            if a.GetSymbol() == PSE[gl.PARAMS["MOL_SPLIT_START"]]:
                neighbor = a.GetNeighbors()[0].GetIdx()
                connect_atoms.append(neighbor)
                bond_type.append(mol.GetBondBetweenAtoms(a.GetIdx(),neighbor).GetBondType())
        if len(connect_atoms) == 2:    # connect them and remove the dummy atoms
            if bond_type[0] != bond_type[1]:
                raise RuntimeError("Cannot connect bonds because they are of different bond type.")
            emol = Chem.EditableMol(mol)
            emol.AddBond(connect_atoms[0], connect_atoms[1], bond_type[0])
            stereo_atoms = collect_stereo_atoms(mol)
            emol, stereo_atoms, connect_atoms = remove_first_attach(emol, stereo_atoms, connect_atoms)
            emol, stereo_atoms, connect_atoms = remove_first_attach(emol, stereo_atoms, connect_atoms)
            return create_mol_with_stereo(emol, stereo_atoms)  # return molecule
        else:
            raise RuntimeError("Something went wrong: {} atoms to connect.".format(len(connect_atoms)))
           
    growing_mol = fragments[0]  # molecule starts with first fragment
    counter = 1                 # fragment counter
    attach_counter = 1          # counter for the numbering of attachment points
    
    while counter < len(fragments):  # go through all fragments
        f1 = replace_attachs(growing_mol)       # mark correct attachment point with first rare element symbol and others with numbers
        f2 = fragments[counter]                 # get next fragment
        growing_mol = connect_fragments(f1, f2) # connect next fragment to growing molecule
        counter += 1
    
    # check that there are no more attachment points in molecule
    for atom in growing_mol.GetAtoms():
        if atom.GetSymbol() == "*" or atom.GetAtomicNum() >= gl.PARAMS["MOL_SPLIT_START"]:
            raise RuntimeError("Still attachment points in molecule after joining all fragments.")
    
    return growing_mol

        
# extracts the numbers from a list of attachment points (pattern "[<number>*]")
# and returns them as a list of ints
def match_to_int(matches):
    ret = []
    for m in matches:
        ret.append(int(m[1:-2]))
    return ret     


# replaces all attachment points in fragments by those defined by MOL_SPLIT_START
# if there are several attachment points, all permutations are created
def replace_attachment_points(fragments):
    
    ret = []
    
    for mol in fragments:
        number_of_attachs = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in gl.PARAMS["ATTACHMENTS"]:
                number_of_attachs += 1
        permutations = list(itertools.permutations(gl.PARAMS["ATTACHMENTS"][:number_of_attachs]))
        
        smi = Chem.MolToSmiles(mol)
        for p in permutations:
            new_smi = smi
            for i, a in enumerate(p):
                new_smi = new_smi.replace(a, PSE[gl.PARAMS["MOL_SPLIT_START"]+i])
            ret.append(Chem.MolFromSmiles(new_smi))
    
    return ret


# Decide the class of a fragment (fragment is an rdkit molecule)
# Class corresponds to number of attachment points
def get_class(fragment):
    n = 0   # number of attachment points
    for a in fragment.GetAtoms():  # for every atom in fragment
        if a.GetAtomicNum() >= gl.PARAMS["MOL_SPLIT_START"]: # if atom is pseudo-atom corresponding to attachment point
            n += 1
    return n


# Enforce conditions on fragments (fragment is an rdkit molecule)
def should_use(fragment):
    
    n = 0  # number of attachment points
    m = 0  # number of heavy atoms
    for a in fragment.GetAtoms(): # for every atom in fragment
        m += 1
        if a.GetAtomicNum() >= gl.PARAMS["MOL_SPLIT_START"]:
            n += 1
        if n > gl.PARAMS["MAX_FREE"]:
            return False
        if m > gl.PARAMS["MAX_ATOMS"]:
            return False

    return True


# Split a list of molecules into fragments.
# Fragments are also written into file 'filename'
# returns (fragments, used_mols, fragment_sets)
# fragments = dict{smiles: tuple} where tuple is (molecule, class)
# used_mols = np-array of bools where those molecules that are fragmented are marked as True
# fragment_sets = list where every element corresponds to fragments that can build up one molecule
def get_fragments(mols, filename=None):

    used_mols = np.zeros(len(mols)) != 0

    fragments = dict()
    fragment_sets = []
    
    # create LibararyGeneration objects (if splitting is done by RXN files)
    if gl.PARAMS["SPLIT_OPTION"] == 2:
        rxn_files = glob.glob("{}/*.rxn".format(gl.PARAMS["RXN_FOLDER"]))
        lib_gens = create_libgen_interfaces(rxn_files)
    
    # split molecules
    for i, mol in enumerate(mols):
        
        if gl.PARAMS["MORE_OUTPUT"] and i>0:
            print("{}: {} of {} molecules fragmented".format(datetime.datetime.now(), i, len(mols)))
             
        try:
            if gl.PARAMS["SPLIT_OPTION"] == 1:
                fsets = [split_molecule(mol)]      # fragments for one molecule (as list)
            elif gl.PARAMS["SPLIT_OPTION"] == 2:
                fsets = split_molecule_rxn(mol, lib_gens)    # list of possible fragment sets
            else:
                raise Exception("Invalid option given for SPLIT_OPTION: {}".format(gl.PARAMS["SPLIT_OPTION"]))
        except:
            continue
        
        # determine which fragment sets are used
        for fs in fsets:
            if all(map(should_use, fs)):
                used_mols[i] = True
                fragment_sets.append(fs)
        
    # add used fragments to dictionary
    for fs in fragment_sets:
        for f in fs:
            cl = get_class(f)
            fragments[Chem.MolToSmiles(f)] = (f, cl)
    
    # write fragment output file
    if filename is not None:     
        file_io.write_fragments(fragment_sets, [join_fragments(f) for f in fragment_sets], filename)
    
    return fragments, used_mols, fragment_sets


# convert molecules to fragments
# returns fragments = dict{smiles: tuple} where tuple is (molecule, class)
def mols_to_frags(mols):
    fragments = dict()
    for m in mols:
        cl = get_class(m)
        fragments[Chem.MolToSmiles(m)] = (m, cl)
    return fragments


# get reagent name from fragment (as rdkit.Mol)   
# frag_type = reagent type (e.g. "amineprim_sc__anilineprim_sc")
# reagents_dict = dictionary containing {SMILES: name} dictionaries for all reagent types
def get_name_from_frag(frag, frag_type, reagents_dict):
    
    # search one of the dictionaries for current fragment and return name
    def search_dict_for_mol(mol, reagents):
        try:
            return reagents[Chem.MolToSmiles(mol)]
        except KeyError:
            return None
    
    # frag_type might contain several reagent types -> convert to list
    frag_type = frag_type.split("__") if "__" in frag_type else [frag_type]
    
    # search for name    
    for ft in frag_type:  # loop over reagent types
        for reagent_type in reagents_dict:
            if reagent_type == ft:  # find correct reagent type
                name = search_dict_for_mol(frag, reagents_dict[ft])  # seach dictionary
                if name: return name
    
