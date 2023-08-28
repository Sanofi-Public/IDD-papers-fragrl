import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2', '3'}

import Modules.file_io as file_io
import Modules.visualization as visualization
import Modules.build_encoding as build_encoding
import Modules.mol_utils as mol_utils
import Modules.global_parameters as gl
from Modules.models import build_models, read_model
from Modules.training import run

from submodules.scoring_functions import scoring_functions

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


def main():
    
    # create the scoring function with kwargs
    sf = scoring_functions.get_scoring_function(gl.PARAMS["SCORING_FUNCTION"], **gl.PARAMS["SF_KWARGS"])
    
    # get lead molecules and fragment them
    if gl.PARAMS["LEAD_FRAGMENTS"] is not None:  # read leads as fragments
        lead_sets = file_io.read_fragments(gl.PARAMS["LEAD_FRAGMENTS"])
        lead_frags = {} 
        lead_mols = []
        lead_smiles = []
        for lead_set in lead_sets:
            lead_frags.update(mol_utils.mols_to_frags(lead_set))
            lead_mol = mol_utils.join_fragments(lead_set)
            smi = Chem.MolToSmiles(lead_mol)
            if smi not in lead_smiles:
                lead_mols.append(lead_mol)
                lead_smiles.append(smi)
        print("We have {} lead fragment sets, forming {} lead molecules".format(len(lead_sets), len(lead_mols)))
    elif gl.PARAMS["LEAD_FILE"] is not None:  # read leads as molecules
        lead_mols_total = file_io.read_molfile(gl.PARAMS["LEAD_FILE"])
        lead_frags, used, lead_sets = mol_utils.get_fragments(lead_mols_total, "lead_frags.csv")
        lead_mols = []   # use only lead molecules that can be fragmented
        for i, lead in enumerate(lead_mols_total):
            if used[i] == True:
                lead_mols.append(lead)
        print("{} out of {} lead molecules have been fragmented, resulting in {} lead fragment sets".format(
              len(lead_mols), len(lead_mols_total), len(lead_sets)))
    else:
        raise Exception("No files with lead molecules or fragments given!")
    
    # create decodings and encode freezed fragments
    encodings, decodings = build_encoding.create_decodings(lead_frags)
    build_encoding.save_decodings(decodings)
    freeze_encodings = build_encoding.encode_freeze(decodings)  
    
    # optional: write lead molecules and fragments (+ their encodings) into file
    if gl.PARAMS["MORE_OUTPUT"]:
        file_io.create_folder("History")
        lfs_mols = [l[0] for l in lead_frags.values()]
        lead_smiles = [Chem.MolToSmiles(lf) for lf in lfs_mols] 
        lead_keys = [encodings[smi] for smi in lead_smiles]
        file_io.smiles_to_smi(lead_smiles, "History/lead_frags.smi", lead_keys)
        visualization.mols_to_svg(lead_mols, filename="History/leads.svg")
        visualization.mols_to_svg(lfs_mols, filename="History/lead_frags.svg", namelist=lead_keys)
    
    # if only decodings are written: exit program here
    if (gl.PARAMS["WRITE_DECODINGS"]): exit()                                       
        
    # encode lead molecules
    max_frag = max([len(lead_set) for lead_set in lead_sets])
    num_bits = len(list(decodings.keys())[0])  # this is only encoding for fragment (not this first bit that will be added later)
    print("Size of encodings: {} fragments with {} bits per fragment".format(max_frag, num_bits)) 
    lead_codes = build_encoding.encode_frags(lead_sets, encodings, max_frag)
    
    # if there are no more lead molecules left nothing can be done
    if len(lead_codes) == 0:
        raise Exception("Sorry, there are no lead molecules left. Leaving program...")
    
    # generate molecules
    if gl.PARAMS["LOAD_MODEL"] == None:
        actor, critic = build_models(lead_codes.shape[1:])                # build model
        results = run(lead_codes, sf, freeze_encodings, actor, decodings, critic)   # train and run model   
    elif len(gl.PARAMS["LOAD_MODEL"]) == 1:  # if only actor given: just sample molecules
        actor = read_model(gl.PARAMS["LOAD_MODEL"][0])         # load model
        results = run(lead_codes, sf, freeze_encodings, actor, decodings)  # run model
    else:                                      # if actor and critic given: retrain model
        actor = read_model(gl.PARAMS["LOAD_MODEL"][0])
        critic = read_model(gl.PARAMS["LOAD_MODEL"][1])
        results = run(lead_codes, sf, freeze_encodings, actor, decodings, critic)
    
    # write outputfile    
    with open("sampled_unique.csv", "w") as out:
        out.write("SMILES,NAME,SCORE\n")
        for r in results:
            out.write(r.get_smiline())


def parse_args():
    """Parses input arguments."""

    parser = argparse.ArgumentParser(description="Options for DeepFMPO")
    parser.add_argument("--configuration", "-c",
                        help="Configuration file for everything except scoring function, DEFAULT: configuration.txt",
                        type=str, default="configuration.txt")
    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


if __name__ == "__main__":
    init_tensorflow()
    args = parse_args()                         # parse command line arguments
    file_io.read_config(args["configuration"])  # read configuration file   
    main()                                      # run program


