import argparse, glob
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem 

import Modules.file_io as file_io
import Modules.global_parameters as gl
import Modules.mol_utils as mol_utils


def main(outputfile):
    # get filenames
    frag_files = []
    for pattern in gl.PARAMS["FRAGMENT_FILES"]:
        frag_files += glob.glob(pattern)
    
    # read files
    fragment_mols = []    
    for fragment_file in frag_files:
        fragment_mols += file_io.read_molfile(fragment_file)
        
    # write fragments to file    
    mol_utils.get_fragments(fragment_mols, outputfile)


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument("--configuration", "-c", type=str, default="configuration.txt",
                        help="Configuration file for everything except scoring function, DEFAULT: configuration.txt")
    parser.add_argument("--output", "-o", type=str, default="mol_frags.csv", help="Name of output file")
    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


if __name__ == "__main__":
    args = parse_args()                         # parse command line arguments
    file_io.read_config(args["configuration"])  # read configuration file   
    main(args["output"])                        # run program