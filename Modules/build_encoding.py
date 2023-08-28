import os
import bisect
import glob
import time
import math
import datetime
import ast
import numpy as np
import rdkit.Chem as Chem

from joblib import Parallel, delayed

import Modules.mol_utils as mol_utils
import Modules.global_parameters as gl
from Modules.file_io import read_molfile
from Modules.file_io import create_folder
from Modules.file_io import read_fragments
from Modules.visualization import mols_to_svg
from Modules.similarity import similarity
from Modules.tree import build_tree_from_list


# create decodings
# can either be done from
#     * molecules (to be fragmented)
#     * fragments (to be decoded)
#     * decodings (just to be read in)
# lead_frags: lead fragments as dict
def create_decodings(lead_frags):
    
    frag_files = []
    for pattern in gl.PARAMS["FRAGMENT_FILES"]:
        frag_files += glob.glob(pattern)
    
    if gl.PARAMS["GET_DECODINGS"] == 1:    # from fragmenting molecules
        fragment_mols = []
        for fragment_file in frag_files:
            fragment_mols += read_molfile(fragment_file)
        fragments = mol_utils.get_fragments(fragment_mols, "mol_frags.csv")[0]
        fragments.update(lead_frags)
        return get_encodings(fragments)
    
    elif gl.PARAMS["GET_DECODINGS"] == 2:  # from fragments
        fragment_mols = []
        for fragment_file in frag_files:
            fragment_mols += read_molfile(fragment_file, "FRAG")
        fragment_mols = mol_utils.replace_attachment_points(fragment_mols)
        fragments = mol_utils.mols_to_frags(fragment_mols)
        fragments.update(lead_frags)
        return get_encodings(fragments)
    
    elif gl.PARAMS["GET_DECODINGS"] == 3:  # directly from decodings file
        decodings = read_decodings(gl.PARAMS["DECODINGS_FILE"])
        if not check_leadfrags(lead_frags, decodings):
            print("Unknown fragment in lead molecules. Rebuild decodings.")
            fragments = mol_utils.mols_to_frags(list(decodings.values()))
            fragments.update(lead_frags)
            encodings, decodings = get_encodings(fragments)
        encodings = decodings_to_encodings(decodings)
        return encodings, decodings
    
    elif gl.PARAMS["GET_DECODINGS"] == 4:  # from a fragments file
        for fragment_file in frag_files:
            mols_frag_sets = read_fragments(fragment_file)
            fragments = {}
            for frag_set in mols_frag_sets:
                fragments.update(mol_utils.mols_to_frags(frag_set))
        fragments.update(lead_frags)
        return get_encodings(fragments)    
                
    else:
        raise RuntimeError("Unknown option for GET_DECODINGS: {}".format(gl.PARAMS["GET_DECODINGS"]))
    
# check if all lead fragments are found in decodings    
def check_leadfrags(lead_frags, decodings):
    decodings_smiles = []
    for d in decodings.values():
        decodings_smiles.append(Chem.MolToSmiles(d))
    for lf in lead_frags:
        if lf not in decodings_smiles:
            return False
    return True

# create decodings for one type of reagents (may contain several building block types)
def make_decodings(fragment_type):
    fragments = dict()
    for name in fragment_type:
        filename = gl.PARAMS["BBLOCKS"] + "/" + name + "/" + name + "_final_R1.smi"
        frag_mols = read_molfile(filename, file_type="FRAG", remove_chiral=False)
        frags = mol_utils.mols_to_frags(frag_mols)
        fragments.update(frags)
    return get_encodings(fragments)

######################### ENCODING AND DECODING OF MOLECULES ##############################################


# Molecule representations:
#    - rdkit.Mol
#    - string encoding: binary codes for fragments, separated by -
#    - array encoding:  matrix has a fixed number of lines and one more column than there are bits in binary code
#                       each line represents one fragment where the first bit only shows if there's a fragment or not
#                       the rest of the bits are the binary code


# Attention: encode_molecule() and encode_list() are only there for historical reasons and debugging purpose
#            please use function encode_frags() instead

# rdkit.Mol -> string encoding
def encode_molecule(m, encodings):
    if gl.PARAMS["SPLIT_OPTION"] == 1:
        fs = [Chem.MolToSmiles(f) for f in mol_utils.split_molecule(m)]
    elif gl.PARAMS["SPLIT_OPTION"] == 2:
        raise Exception("Splitting molecules with according to rxn files is not supported by function encode_molecule().")
    else:
        raise Exception("Invalid option given for SPLIT_OPTION: {}".format(gl.PARAMS["SPLIT_OPTION"]))
    encoded = "-".join([encodings[f] for f in fs])
    return encoded

# rdkit.Mol -> array encoding (for list of molecules)
def encode_list(mols, encodings, max_frag):
    enc_size = len(list(encodings.values())[0])
    encoded_mols = [encode_molecule(m, encodings) for m in mols]
    X_mat = np.zeros((len(encoded_mols), max_frag, enc_size + 1))

    for i in range(X_mat.shape[0]):
        es = encoded_mols[i].split("-")
        for j in range(X_mat.shape[1]):
            if j < len(es):
                e = np.asarray([int(c) for c in es[j]])
                X_mat[i,j,0] = 1
                X_mat[i,j,1:] = e
    return X_mat

# fragment list -> array encoding (for list of molecules)
# frag_sets: contains a list of rdkit.Mols that correspond to fragments
def encode_frags(frag_sets, encodings, max_frag):
    enc_size = len(list(encodings.values())[0])   # length of bitcode
    X_mat = np.zeros((len(frag_sets), max_frag, enc_size + 1))
    
    for i in range(X_mat.shape[0]):  # for every fragment set
        for j, f in enumerate(frag_sets[i]):  # for every fragment in set 
            enc = encodings[Chem.MolToSmiles(f)]
            e = np.asarray([int(c) for c in enc])
            X_mat[i,j,0] = 1
            X_mat[i,j,1:] = e
    return X_mat

# string encoding -> rdkit.Mol
def decode_molecule(enc, decodings):
    fs = [Chem.Mol(decodings[x]) for x in enc.split("-")]
    return mol_utils.join_fragments(fs)

# array encoding -> rdkit.Mol
def decode(x, translation):
    enc = ["".join([str(int(y)) for y in e[1:]]) for e in x if e[0] == 1]
    fs = [Chem.Mol(translation[e]) for e in enc]
    try:
        return mol_utils.join_fragments(fs)
    except:
        raise RuntimeError("Something went wrong when joining fragments.")

# array encoding -> rdkit.Mol (for list of molecules)
# ignores those that throw an error
def decode_list(arraylist, decodings):
    mollist = []
    for a in arraylist:
        try:
            mollist.append(decode(a, decodings))
        except (KeyError, RuntimeError):
            mollist.append(None)
    return mollist

# determine which reaction has to be used to decode array encoding
def determine_reaction(array_enc, num_bits_frag, bblock_encodings, reactions_dict):    
    frag_list = []  # fragment types from array_enc
    for frag in array_enc:
        frag_enc = np.array2string(frag[:num_bits_frag], separator="")[1:-1]
        frag_list.append(bblock_encodings[frag_enc].split("__"))
    for reaction in reactions_dict:  # search for reaction where fragment types fit (including scaffold but not "NoneXX")
        reagent_list = [x for x in [reactions_dict[reaction][0].split(",")] + reactions_dict[reaction][1:] if x != ["NoneXX"]]
        if reagent_list == frag_list:
            return reaction
    raise Exception("No reaction found for given reagent types")

# converts array for one molecule to a list of fragments (as rdkit.Mol)
def create_fragmol(a, fragment_types, decodings_dict):
    fragmol = []
    for i, f in enumerate(fragment_types):  # for each reagent type in reaction
        current_decodings = decodings_dict[f][1]  # choose corresponding decodings tree
        num_bits = len(list(current_decodings.keys())[0])  # determine number of bits in decodings string
        decoding_str = "".join([str(bit) for bit in a[i][-num_bits:]])  # build decodings string
        if decoding_str in current_decodings:  # use dictionary to convert decoding to fragment
            frag = current_decodings[decoding_str]
            fragmol.append(frag)
        else:
            raise Exception("Decoding does not exist.")
    return fragmol
    
# convert array encoding to molecule name (as <reaction>__<reagent1>__<reagent2>...)
def get_name(org_mol, reactions_dict, inverse_reagents_dict, decodings_dict):
    num_bits_frag = len(list(decodings_dict["bblock_encodings"].keys())[0])
    try:
        reaction = determine_reaction(org_mol, num_bits_frag, decodings_dict["bblock_encodings"], reactions_dict)
        fragment_types = ["__".join(x) for x in [reactions_dict[reaction][0].split(",")] + reactions_dict[reaction][1:] if x != ["NoneXX"]]
        fragmol = create_fragmol(org_mol, fragment_types, decodings_dict)
    except:
        return "invalid"
    name = reaction
    for i, frag in enumerate(fragmol):
        fragname = mol_utils.get_name_from_frag(frag, fragment_types[i], inverse_reagents_dict)
        name += "___{}".format(fragname)
    return name

# array encoding -> rdkit.Mol (for list of molecules)
# fragments are connected by xenonized reaction
# ignores those that throw an error
# decodings_dict = decodings for each fragment type saved as {"fragment_type": (encodings, decodings)}
# vl_xe = Virtual_Library_Xe object, used to connect fragments
def decode_reaction_codes(arraylist, decodings_dict, vl_xe):
    
    num_bits_frag = len(list(decodings_dict["bblock_encodings"].keys())[0])
    
    mollist = []
    for a in arraylist:  # for every molecule
        try:
            reaction = determine_reaction(a, num_bits_frag, decodings_dict["bblock_encodings"], vl_xe.reactions_dict)
            fragment_types = ["__".join(x) for x in [vl_xe.reactions_dict[reaction][0].split(",")] + vl_xe.reactions_dict[reaction][1:] if x != ["NoneXX"]]
            fragmol = create_fragmol(a, fragment_types, decodings_dict)
            frag_smiles = [Chem.MolToSmiles(frag) for frag in fragmol]
            smiles = frag_smiles[0]  # scaffold
            for i, f in enumerate(frag_smiles[1:]):  # use Virtual_Library_Xe object to connect fragments
                smiles = vl_xe.MergeXeSmiles(smiles, f, gl.PARAMS["ATTACHMENTS"][i], False)
            mollist.append(Chem.MolFromSmiles(smiles))
        except:
            mollist.append(None)
    return mollist

# checks if two array encodings correspond to the same molecule (in DeepVL)
def same_molecule(array1, array2, decodings_dict, vl_xe):
    mol1 = decode_reaction_codes([array1], decodings_dict, vl_xe)
    mol2 = decode_reaction_codes([array2], decodings_dict, vl_xe)
    return mol1 == mol2

# convert array encoding of single fragment to string
def array_to_string(fragment):
    array = fragment[1:]  # remove first bit
    return "".join([str(int(a)) for a in array])

# convert freezed fragments to string encoding
def encode_freeze(decodings):
    enc_freeze = []
    for d in decodings:
        for ff_smi in gl.PARAMS["FREEZE_FRAGS"]:
            ff = Chem.MolFromSmiles(ff_smi)
            if len(decodings[d].GetSubstructMatch(ff)) != 0:
                enc_freeze.append(d)
    return enc_freeze     

# Save all decodings as a file (fragments are stored as SMILES)
def save_decodings(decodings, filename="History/decodings"):
    decodings_smi = dict([(x,Chem.MolToSmiles(m)) for x,m in decodings.items()])
    create_folder("History")
    with open("{}.txt".format(filename), "w+") as f:
        f.write(str(decodings_smi))
    if gl.PARAMS["MORE_OUTPUT"]:
        mols_to_svg(list(decodings.values()), list(decodings.keys()), "{}.svg".format(filename))

# Read encoding list from file
def read_decodings(filename):
    if filename == None:
        raise RuntimeError("No decodings file given!")
    if os.path.exists(filename) == False:
        raise RuntimeError("Decodings file does not exist: {}".format(filename))
    
    with open(filename,"r") as f:
        x = f.readline()                # take first line
        if x.startswith("{") == False:  # check format
            raise Exception("Are you sure your input decodings file is formatted correctly?")
        d = ast.literal_eval(x)                     # get dictionary
        return dict([(x,Chem.MolFromSmiles(m)) for x,m in d.items()])
    
# convert decodings to encodings    
def decodings_to_encodings(decodings):
    tuples = []
    for key in decodings:
        tuples.append((Chem.MolToSmiles(decodings[key]), key))
    return dict(tuples)

# convert encodings to decodings
def encodings_to_decodings(encodings):
    tuples = []
    for key in encodings:
        tuples.append((encodings[key], Chem.MolFromSmiles(key)))
    return dict(tuples)


######################### SIMILARITY STUFF ##############################################


## Parallelization Helperfunction ##
# function to create task lists (diagonal from upper-right to lower-left)
# returns several list of index pairs, each list corresponds to one of the parallel processes
# implemented in a way that it can start and end any anywhere in the matrix (for doing it in batches)
def create_tasks(i_start, j_start, j_max, i_max, matrix_size):
    tasks = [None] * gl.PARAMS["NUM_CORES"]
    for i, _ in enumerate(tasks):
        tasks[i] = []
    counter = 0
    
    def add_to_tasks(i, j):
        nonlocal counter
        tasks[counter%gl.PARAMS["NUM_CORES"]].append([i,j])
        counter += 1
    
    i, j = i_start, j_start
    while True:
        add_to_tasks(i, j)
        if i+1 < j-1:  # "normal" step
            i = i+1
            j = j-1
        elif i+2 == matrix_size and j+1 == matrix_size:  # reached the end of the matrix
            return "stop", "stop", "stop", "stop", tasks
        elif j_max + 1 < matrix_size:  # jump to next column
            j = j_max + 1
            j_max = j
            i = 0
        elif i_max + 1 < matrix_size:  # jump to next line
            i = i_max + 1
            i_max = i
            j = matrix_size-1
        else:  # every case should be covered by one of the cases above
            raise Exception("This should not happen!")
        if counter == gl.PARAMS["BATCHSIZE_D"]:
            return i, j, j_max, i_max, tasks

## Parallelization Helperfunction ##        
# function to compute any function for a list of arbitrary index pairs
# returns input indices as well as results (necessary for putting results to matrix)
def task_function(pairs, func):
    return [(x, y, func(x, y)) for x,y in pairs]


# Get a matrix containing the similarity of different fragments
def get_dist_matrix(fragments):
    
    # function to compute similarity between two molecules (i,j = indices)
    def calc_similarity(i, j):
        return similarity(id_dict[i], id_dict[j], ms[i], ms[j])

    id_dict = {}  # dictionary {index: smi}
    ms = []       # molecules

    i = 0
    for smi, (m, _) in fragments.items():
        ms.append(m)
        id_dict[i] = smi
        i += 1
        
    distance_matrix = np.zeros([len(ms)] * 2)  # distance matrix (to be filled)
    
    print("Building tree from fragments ({})...".format(len(ms)))
    start = time.time()
    
    if gl.PARAMS["NUM_CORES"] == 0 or len(ms) < 2:
        for i in range(len(ms)):
            for j in range(i+1,len(ms)):
                distance_matrix[i,j] = calc_similarity(i, j)
                distance_matrix[j,i] = distance_matrix[i,j]        
    else:
        i, j, j_max, i_max = 0, 1, 1, 0
        batch_count = 0
        num_batches = math.ceil(len(ms)*(len(ms)-1) / (2*gl.PARAMS["BATCHSIZE_D"]))
        while i != "stop" and j != "stop":  # go through batches
            # compute the values
            i, j, j_max, i_max, tasks = create_tasks(i, j, j_max, i_max, len(ms))
            tasks_results = Parallel(n_jobs=gl.PARAMS["NUM_CORES"])(delayed(task_function)(t, calc_similarity) for t in tasks)
            # fill the values into matrix
            for res_one_task in tasks_results:
                for r in res_one_task:
                    distance_matrix[r[0], r[1]] = r[2]
                    distance_matrix[r[1], r[0]] = r[2]
            batch_count += 1
            if gl.PARAMS["MORE_OUTPUT"]:
                print("{}: Finished batch {} of {}.".format(str(datetime.datetime.now())[:-7], batch_count, num_batches))
                
    if gl.PARAMS["MORE_OUTPUT"]:            
        print("Building distance matrix finished after {} seconds".format(time.time()-start))
        
    return distance_matrix, id_dict


# Create pairs of fragments in a greedy way based on a similarity matrix
def find_pairs(distance_matrix):

    left = np.ones(distance_matrix.shape[0])
    pairs = []

    candidates = sorted(zip(distance_matrix.max(1),zip(range(distance_matrix.shape[0]),
                                                       distance_matrix.argmax(1))))

    while len(candidates) > 0:
        (c1,c2) = candidates.pop()[1]

        if left[c1] + left[c2] == 2:
            left[c1] = 0
            left[c2] = 0
            pairs.append([c1,c2])

        elif np.sum(left) == 1: # Just one sample left
            sampl = np.argmax(left)
            pairs.append([sampl])
            left[sampl] = 0


        elif left[c1] == 1:
            row = distance_matrix[c1,:] * left
            c2_new = row.argmax()
            v_new = row[c2_new]
            new =  (v_new, (c1, c2_new))
            bisect.insort(candidates, new)

    return pairs


# Create a new similarity matrix from a given set of pairs
# The new similarity is the maximal similarity of any fragment in the sets that are combined.
def build_matrix(pairs, old_matrix):
    
    # function to compute maximum similarity (i, j = indices)
    def calc_max(i, j):
        return np.max((old_matrix[pairs[i]])[:,[pairs[j]]])

    new_mat = np.zeros([len(pairs)] * 2) - 0.1
    
    if gl.PARAMS["NUM_CORES"] == 0 or len(pairs) < 2:
        for i in range(len(pairs)):
            for j in range(i+1, len(pairs)):
                new_mat[i,j] = calc_max(i, j)
                new_mat[j,i] = new_mat[i,j]
    else:
        i, j, j_max, i_max = 0, 1, 1, 0
        batch_count = 0
        while i != "stop" and j != "stop":  # go through batches
            # compute the values
            i, j, j_max, i_max, tasks = create_tasks(i, j, j_max, i_max, len(pairs))
            tasks_results = Parallel(n_jobs=gl.PARAMS["NUM_CORES"])(delayed(task_function)(t, calc_max) for t in tasks)
            # fill the values into matrix
            for res_one_task in tasks_results:
                for r in res_one_task:
                    new_mat[r[0], r[1]] = r[2]
                    new_mat[r[1], r[0]] = r[2]
            batch_count += 1
    return new_mat


# Get a containing pairs of nested lists where the similarity between fragments in a list is higher than between
#   fragments which are not in the same list.
def get_hierarchy(fragments):

    distance_matrix,  id_dict = get_dist_matrix(fragments)
    working_mat = (distance_matrix + 0.001) * (1- np.eye(distance_matrix.shape[0]))

    start = time.time()
    pairings = []

    while working_mat.shape[0] > 1:
        pairings.append(find_pairs(working_mat))
        working_mat = build_matrix(pairings[-1], working_mat)
    
    if gl.PARAMS["MORE_OUTPUT"]:    
        print("Converting matrix finished after {} seconds".format(time.time()-start))

    return pairings, id_dict


# takes list of fragments and sorts them by the number of attachment points
# returns a list where each element contains the fragments with i+1 attachment points
def sort_into_classes(fragments):
    x = ["placeholder"] * gl.PARAMS["MAX_FREE"]
    for i, _ in enumerate(x): x[i] = {}
    for f in fragments:
        x[fragments[f][1]-1][f] = fragments[f]
    return x


# Build a binary tree from a list of fragments where the most similar fragments are neighbouring in the tree.
# This paths from the root in the tree to the fragments in the leafs is then used to build encode fragments.
def get_encodings_one_class(fragments):
    pairings, id_dict = get_hierarchy(fragments)
    t = build_tree_from_list(pairings, lookup=id_dict)
    encodings = dict(t.encode_leafs())
    decodings = dict([(v, fragments[k][0]) for k,v in encodings.items()])
    if gl.PARAMS["MORE_OUTPUT"]:
        print("{}: Tree finished.".format(str(datetime.datetime.now())[:-7]))
    return encodings, decodings


# determine number of bits for fragment class
def determine_class_bitnum(frags_classes):
    x = np.log(len(frags_classes)) / np.log(2)
    num_bits_class = int(np.ceil(x))
    gl.PARAMS["FIX_BITS"] += num_bits_class
    return num_bits_class


# encodes fragments for each class, compute bits needed for fragment
def encode_mols(frags_classes):
    num_bits = []
    encodings_all = []
    for frags in frags_classes:
        if len(frags) < 2:
            print("Warning! It doesn't make sense to create a distance matrix from less than 2 fragments. Ignoring...")
            continue
        encodings, decodings = get_encodings_one_class(frags)
        num_bits.append(len(list(decodings.keys())[0]))
        encodings_all.append(encodings)
    max_bits = max(num_bits)
    return max_bits, encodings_all


# fills encodings because different fragment classes might need a different number of bits
# adds class encodings and puts all encodings into one dictionary
def fill_and_connect(max_bits, encodings_all, num_bits_class):
    encodings_new = {}
    for i, encodings in enumerate(encodings_all):  # go through fragment classes
        num_fill = max_bits - len(list(encodings.values())[0])  # bits to fill because of different number of bits
        class_encoding = bin(i)[2:].zfill(num_bits_class)       # bits for fragment class
        for enc in encodings:
            encodings[enc] = class_encoding + num_fill*"0" + encodings[enc]
        encodings_new.update(encodings)
    return encodings_new
    

# Build a binary tree from a list of fragments
def get_encodings(fragments):
    
    if gl.PARAMS["USE_CLASSES"]:  # own tree for each fragment class (determined by number of attachment points)
        frags_classes = sort_into_classes(fragments)
        num_bits_class = determine_class_bitnum(frags_classes)
        max_bits, encodings_all = encode_mols(frags_classes)
        encodings = fill_and_connect(max_bits, encodings_all, num_bits_class)
        decodings = encodings_to_decodings(encodings)
        return encodings, decodings
            
    else:     # all decodings in one tree
        return get_encodings_one_class(fragments)
