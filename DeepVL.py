import numpy as np
import ast
import os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2', '3'}

import Modules.global_parameters as gl
from Modules import file_io, build_encoding, models, training

from submodules.scoring_functions import scoring_functions
from submodules.virtuallibrary.source import Virtual_library_Xe

from rdkit import rdBase, Chem
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

def init_tensorflow():    # only for versions >= 2
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)


def parse_args():
    """Parses input arguments."""

    parser = argparse.ArgumentParser(description="Options for DeepVL")
    parser.add_argument("--configuration", "-c",
                        help="Configuration file for everything except scoring function, DEFAULT: configuration.txt",
                        type=str, default="configuration.txt")
    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}    


# convert lead molecule to decodings array
# lead: lead molecule as <reaction>___<scaffold>___<reagent1>__<reagent2>...
# VL_xe: VirtualLibrary object
# decodings dict: decodings dictionaries for all BBlock types + BBlocks themselves
# num_bits: number of bits for fragment (without BBlock type)
def encode_lead(lead, VL_xe, decodings_dict, num_bits):
    
    # search dictionary for value, return key
    def get_key_from_value(v, d):
        for k in d:
            if d[k] == v:
                return k

    # {name: SMILES} for all reagents
    reagents_name_to_smiles = {k:v for x in VL_xe.reagents_dict.values() for k,v in x.items()}  
    decoding = []
    leadlist = lead.split("___")
    # which building block types are used for reaction? (including scaffold)
    fragment_types = ["__".join(ft) for ft in [VL_xe.reactions_dict[leadlist[0]][0].split(",")] + VL_xe.reactions_dict[leadlist[0]][1:]] 
    fragments = leadlist[1:]
    for i, f in enumerate(fragments):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(reagents_name_to_smiles[f].replace("@", "")))
        decoding_str = decodings_dict[fragment_types[i]][0][smiles]
        # create decodings from building block encodings and fragment encodings
        # if decodings have a different number of bits: fill with zeros
        decoding.append([int(d) for d in list(get_key_from_value(fragment_types[i], decodings_dict["bblock_encodings"]))]
                            + [0]*(num_bits - len(decoding_str)) + [int(d) for d in list(decoding_str)])
    return decoding
    
# invert the reagents dictionaries in order to get name from SMILES faster
def invert_reagent_dict(reagent_dict):
    inverted_dict = {}
    for bblock in reagent_dict:  # go through all reagent_types
        current_dict = reagent_dict[bblock]
        new_dict = {}
        for reagent_name in current_dict:  # go through all reagents
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(current_dict[reagent_name].replace("@", "")))
            new_dict[smiles] = reagent_name
        inverted_dict[bblock] = new_dict
    return inverted_dict


# main function
def main():
    
    # create the scoring function with kwargs
    sf = scoring_functions.get_scoring_function(gl.PARAMS["SCORING_FUNCTION"], **gl.PARAMS["SF_KWARGS"])
    
    # use VirtualLibrary submodule to read reactions and reagents
    VL_xe = Virtual_library_Xe.Virtual_Library_Xe()
    VL_xe.init_reactions(gl.PARAMS["REACTIONFILE"])
    VL_xe.init_reagents(gl.PARAMS["BBLOCKS"])
    # {reagent_type: {SMILES: name, SMILES: name}, reagent_type: ...}, SMILES are canonical
    inverse_reagent_dict = invert_reagent_dict(VL_xe.reagents_dict)
    print()
    
    # build decodings tree for each reagent type
    # decodings are saved as {"fragment_type": (encodings, decodings)}
    # fragment_type includes scaffold but not "NoneXX"
    decodings_dict = dict()
    all_fragtypes = []
    for r in VL_xe.reactions_dict:
        fragment_types = [VL_xe.reactions_dict[r][0].split(",")] + VL_xe.reactions_dict[r][1:]
        for frag_type in fragment_types:
            ft = "__".join(frag_type)
            if ft == "NoneXX":
                break
            if ft not in all_fragtypes:
                all_fragtypes.append(ft)
            else:
                continue
            print("Looking at " + ft)
            if gl.PARAMS["GET_DECODINGS"] == 3:  # try to read decodings from decodings file
                filename = "{}/decodings_{}.txt".format(gl.PARAMS["BBLOCKS"], ft)
                if os.path.exists(filename):
                    decodings = build_encoding.read_decodings(filename)
                    encodings = build_encoding.decodings_to_encodings(decodings)
                    decodings_dict[ft] = encodings, decodings
            if ft not in decodings_dict:  # build decodings from building blocks
                decodings_dict[ft] = build_encoding.make_decodings(frag_type)
            build_encoding.save_decodings(decodings_dict[ft][1], "decodings_"+ft)
            
    # if only decodings are written: exit program here
    if (gl.PARAMS["WRITE_DECODINGS"]): exit()   

    # determine number of bits for molecule
    enc_dict = [dd[1] for dd in [decodings_dict[ft] for ft in decodings_dict.keys()]]
    encs = sum([list(d.keys()) for d in enc_dict], [])
    num_bits = max([len(enc) for enc in encs])
    
    # no reaction jump possible
    if not gl.PARAMS["REACTION_JUMP"]:
        # determine number of bits for fragment type
        x = np.log(len(all_fragtypes)) / np.log(2)
        num_bits_frag = int(np.ceil(x))
        gl.PARAMS["FIX_BITS"] = num_bits_frag
    
        # create bblock encodings and also save them into decodings_dict
        bblock_encodings = {}
        for i, ft in enumerate(all_fragtypes):
            bblock_encodings[bin(i)[2:].zfill(num_bits_frag)] = ft
        decodings_dict["bblock_encodings"] = bblock_encodings
    
    # reaction jump possible    
    else:
        with open("decodings_bblocks.txt") as f:              # read bblock encodings
            bblock_encodings = ast.literal_eval(f.read())
        for ft in all_fragtypes:
            if ft not in bblock_encodings.values():
                raise Exception("No decoding given for {} in file decodings_bblocks.txt".format(ft))
        num_bits_frag = len(list(bblock_encodings.keys())[0]) # number of bits
        decodings_dict["bblock_encodings"] = bblock_encodings
        
    print("\nDecodings consist of {} bits: {} for fragment and {} for building block type".format(num_bits+num_bits_frag, num_bits, num_bits_frag))
    
    # encode lead molecules   
    lead_codes = []     
    for lead in gl.PARAMS["LEAD_MOLECULES"]:
        decoding = encode_lead(lead, VL_xe, decodings_dict, num_bits)
        lead_codes.append(decoding)
    lead_codes = np.asarray(lead_codes)
    
    # run network
    actor, critic = models.build_models(lead_codes.shape[1:], False)
    results = training.run_deepVL(lead_codes, sf, actor, critic, decodings_dict, VL_xe, inverse_reagent_dict)
    
    # write outputfile    
    with open("sampled_unique.csv", "w") as out:
        out.write("SMILES,NAME,SCORE\n")
        for r in results:
            out.write(r.get_smiline())
            

if __name__ == "__main__":
    init_tensorflow()
    args = parse_args()                         # parse command line arguments
    file_io.read_config(args["configuration"])  # read configuration file   
    main()
