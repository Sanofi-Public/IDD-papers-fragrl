import math
from rdkit import Chem
from rdkit.Chem import Draw

# save a list of molecules to a png image
# namelist = list of molecule names (optional)
# filename = name of the file where it is written to
def mols_to_svg(mollist, namelist=[], filename="molfile.svg"):
    one_row = int(math.sqrt(len(mollist)))+1
    if len(namelist) == len(mollist):
        svg = Draw.MolsToGridImage(mollist,molsPerRow=one_row,subImgSize=(555,555),legends=namelist,useSVG=True)
    else:
        svg = Draw.MolsToGridImage(mollist,molsPerRow=one_row,subImgSize=(555,555),useSVG=True)
    with open(filename, "w") as out:
        out.write(svg)

# save a matrix of smiles to png
# row of the matrix is row in grid    
def smilesMatrix_to_svg(smiles_matrix, filename, namelist=[]):
    mollist = []
    for line in smiles_matrix:
        for col in line:
            mollist.append(Chem.MolFromSmiles(col))
    if len(namelist) != len(mollist):
        svg = Draw.MolsToGridImage(mollist,molsPerRow=len(smiles_matrix[0]),subImgSize=(555,555),useSVG=True)
    else:
        svg = Draw.MolsToGridImage(mollist,molsPerRow=len(smiles_matrix[0]),subImgSize=(555,555),legends=namelist,useSVG=True)
    with open(filename, "w") as out:
        out.write(svg)
