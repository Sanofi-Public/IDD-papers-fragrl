import argparse
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem 

import Modules.file_io as file_io
import Modules.global_parameters as gl
import Modules.mol_utils as mol_utils
import Modules.build_encoding as build_encoding


TRANSLATOR = {}  # store most similar fragments as {smiles_old: molecule_new}, significant performance improvement


# compute fingerprint similarity 
def calc_similarity(mol1, mol2):
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    return DataStructs.DiceSimilarity(fp1, fp2)


# find most similar fragment to 'lead' fragment that exists in decodings
def find_similar(lead, decodings):
    lead_smi = Chem.MolToSmiles(lead)
    # if this fragment is already known: take modified fragment from saved translator
    if lead_smi in TRANSLATOR:  
        return TRANSLATOR[lead_smi]
    # if it is not known we have to search
    num_attachments = mol_utils.get_class(lead)
    max_sim = 0
    current_mol = None
    for mol in decodings:
        if mol_utils.get_class(mol) == num_attachments:  # only fragments with same number of attachment points 
            sim = calc_similarity(mol, lead)
            if sim > max_sim:
                max_sim = sim
                current_mol = mol
    TRANSLATOR[lead_smi] = current_mol  # save to translator
    return current_mol


# convert lead fragment set to new set
def create_new_set(lead_set, decodings, decodings_smiles):
    new_set = []
    changed = False
    for frag in lead_set:
        if Chem.MolToSmiles(frag) in decodings_smiles:
            new_set.append(frag)
        else:  # if fragment is not in decodings: replace it
            changed = True
            new_frag = find_similar(frag, decodings)
            if new_frag is None:  # no similar fragment found
                print("Warning! Fragment {} cannot be replaced.".format(Chem.MolToSmiles(frag)))
                return None, None
            new_set.append(new_frag)
    return new_set, changed


# write titelline of output file
def write_output_titelline(max_frags):
    with open("lead_preparation.csv", "w") as out:
        out.write("Original,")
        for i in range(max_frags):
            out.write("Frag {},".format(i+1))
            out.write("Frag_new {},".format(i+1))   
        out.write("Molecule,changed\n")


# write outputline for each fragment set        
def write_output_line(lead_set, new_set, max_frags, changed):
    with open("lead_preparation.csv", "a") as out:
        out.write(Chem.MolToSmiles(mol_utils.join_fragments(lead_set)) + ",")
        for i in range(max_frags):
            if i < len(lead_set):
                out.write(Chem.MolToSmiles(lead_set[i]) + ",")
                if new_set is not None:
                    out.write(Chem.MolToSmiles(new_set[i]) + ",")
                else:  # in case no new_set was created
                    out.write(",")
            else:
                out.write(",,")
        if new_set is not None:
            out.write(Chem.MolToSmiles(mol_utils.join_fragments(new_set)) + ",{}\n".format(changed))
        else:  # in case no new_set was created
            out.write(",Error\n")
             

def main():
    # read decodings and convert them to SMILES
    decodings = build_encoding.read_decodings(gl.PARAMS["DECODINGS_FILE"])
    decodings_smiles = [Chem.MolToSmiles(d) for d in decodings.values()]
    
    # read lead molecules and split them
    lead_mols_total = file_io.read_molfile(gl.PARAMS["LEAD_FILE"])
    _, _, lead_sets = mol_utils.get_fragments(lead_mols_total)
    max_frags = max([len(lead_set) for lead_set in lead_sets])
    
    # write titelline of outputfile
    write_output_titelline(max_frags)
    
    # convert every lead fragment set to similar molecule where all fragments are in decodings
    new_sets = []
    for i, lead_set in enumerate(lead_sets):
        print("{} of {} lead fragment sets".format(i+1, len(lead_sets)))
        new_set, changed = create_new_set(lead_set, decodings.values(), decodings_smiles)
        write_output_line(lead_set, new_set, max_frags, changed)
        if new_set is not None:
            new_sets.append(new_set)
    
    # write new fragment sets to fragments file        
    file_io.write_fragments(new_sets, [mol_utils.join_fragments(f) for f in new_sets], "lead_frags.csv")


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument("--configuration", "-c",
                        help="Configuration file for everything except scoring function, DEFAULT: configuration.txt",
                        type=str, default="configuration.txt")
    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


if __name__ == "__main__":
    args = parse_args()                         # parse command line arguments
    file_io.read_config(args["configuration"])  # read configuration file   
    if file_io.check_files([gl.PARAMS["LEAD_FILE"]]):                                                         
        main()                                  # run program
    else:
        raise RuntimeError("Necessary input files do not exist!")