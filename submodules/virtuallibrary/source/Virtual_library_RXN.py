#!/usr/bin/env python
"""
Library generation based on oelibgen
starting from a smirks pattern and reactant file

Created on 12/01/2019
@author: Christoph Grebner
"""
from __future__ import print_function
import sys
import argparse
import random
import time
from openeye.oechem import *

def parse_Arguments():
    '''
    parsing of command line arguments
    '''
    parser = argparse.ArgumentParser(
        description="""Enumerator for RXN and SMIRKS.
        """,
        epilog="", formatter_class=SmartFormatter)
    parser.add_argument("--smirks", help="SMIRKS of a reaction.")
    parser.add_argument("--smirks_file", help="SMIRKS file: <name> <smirks> [<example reactants>].")
    parser.add_argument("--rxn", help="MDL reaction file.")
    parser.add_argument("-r", "--reactants", help="Comma-separated list of input reactant files.")
    parser.add_argument("-o", "--output", help="Product output file name. If neglected, output is given to stdout. Only SMI files are supported.")
    parser.add_argument("-m", "--max_products", help="Maximum number of products", default=100)
    #OELibraryGen options
    parser.add_argument("--relax", help="unmapped atoms on reactant side are not deleted during reaction.", type=str2bool, choices=[False,"no","f","n",True,"yes","t","y"], nargs='?', default=False)
    parser.add_argument("--implicitH", help="reaction will be perfomed using implicit hydrogens.", type=str2bool, choices=[False,"no","f","n",True,"yes","t","y"], nargs='?', default=False)
    parser.add_argument("--valence", help="automatic valence correction will be applied.", type=str2bool, choices=[False,"no","f","n",True,"yes","t","y"], nargs='?', default=True)
    parser.add_argument("--randomize", help="randomize reactant-list before enumeration.", type=str2bool, choices=[False,"no","f","n",True,"yes","t","y"], nargs='?', default=True)
    #product smiles generation
    parser.add_argument("--unique", help="use unique reagents.", type=str2bool, choices=[False,"no","f","n",True,"yes","t","y"], nargs='?', default=True)
    parser.add_argument("--isomeric", help="include atom and bond stereochemistry in product smiles string.", type=str2bool, choices=[False,"no","f","n",True,"yes","t","y"], nargs='?', default=True)    
    
    return parser.parse_args()

def printatoms(label, begin):
    print(label)
    for atom in begin:
        print(" atom:", atom.GetIdx(), OEGetAtomicSymbol(atom.GetAtomicNum()), end=" ")
        print("in component", atom.GetData(OEProperty_Component), end=" ")
        print("with map index", atom.GetMapIdx())

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', 1):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', 0):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class SmartFormatter(argparse.HelpFormatter):
    '''
    Helper class for argparse
            help beginning with R| will be parsed with new lines
    '''
    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('R|'):
            return text[2:].splitlines()  
        return argparse.HelpFormatter._split_lines(self, text, width)





class Library_Generation():
    '''
    @summary: class for generating libraries starting from a smirks and reactants
    @author: Christoph Grebner
    @version: 1.0
    @since: Jan 2018
    '''
    def __init__(self, args):
        #OEThrow.SetLevel(OEErrorLevel_Quiet)
        #OEThrow.SetLevel(OEErrorLevel_Info)
        #OEThrow.SetLevel(OEErrorLevel_Warning)
        OEThrow.SetLevel(OEErrorLevel_Error)
     
        self.smirks_string = args.smirks
        self.rxn_file = args.rxn
        self.reaction_title = "RXN"
        self.relax = args.relax
        self.unique = args.unique
        self.implicitH = args.implicitH
        self.valcorrect = args.valence
        self.isomeric = args.isomeric
        self.max_products = int(args.max_products)
        self.randomize = args.randomize
        self.output = args.output
        self.reactants = args.reactants
        self.use_filter = True

    def Init_reaction(self, smirks_string=None, rxn_file=None):   
        '''
        define SMIRKS
        setup reaction 
        smirks_string: <NAME> <SMIRKS> [<optional reactant examples>]
        
        '''
        if smirks_string == None:
            smirks_string = self.smirks_string
        if rxn_file == None:
            rxn_file = self.rxn_file
        
        #move smirks to init and argument for Init_reaction
        if not smirks_string and not rxn_file:
            OEThrow.Fatal("Please provide SMIRKS string or MDL reaction file")
    
        if smirks_string and rxn_file:
            OEThrow.Fatal("Please provide only SMIRKS string or MDL reaction file")
        
        self.reaction = OEQMol()
        if smirks_string:
            smirks = smirks_string
            if not OEParseSmirks(self.reaction, smirks):
                OEThrow.Fatal("Unable to parse SMIRKS: %s" % smirks)
        else:
            rxn = rxn_file           
            rfile = oemolistream(rxn)
            self.reaction_title = rfile.GetFileName().split(".")[0]
            self.reaction_title = self.reaction_title.split("/")[-1]
            opt = OEMDLQueryOpts_ReactionQuery | OEMDLQueryOpts_SuppressExplicitH

            # reading reaction
            opt = OEMDLQueryOpts_ReactionQuery | OEMDLQueryOpts_SuppressExplicitH
            if not OEReadMDLReactionQueryFile(rfile, self.reaction, opt):
                OEThrow.Fatal("Unable to read reaction file: %s" % rxn) 


        self.reaction.SetTitle(self.reaction_title)        
        self.libgen = OELibraryGen()
        if not self.libgen.Init(self.reaction, not self.relax):
            OEThrow.Fatal("failed to initialize library generator")
        self.libgen.SetValenceCorrection(self.valcorrect)
        self.libgen.SetExplicitHydrogens(not self.implicitH)
        self.libgen.SetRemoveUnmappedFragments(True)
        self.libgen.SetValidateKekule(True)
        #libgen.SetClearCoordinates(True)
        self.libgen.SetTitleSeparator("___")



    def Init_reactants(self, reactants):
        '''
        define the reactants:
        - reactants: comma-separated list of files containing the input reactants.
            either one file containing all reactants
            or one file for each required reactant
        
        '''
        nrReacts = self.libgen.NumReactants()
        reactant_files = reactants.split(',')
        
        if len(reactant_files) != nrReacts:
            if len(reactant_files) == 1:
                reactant_files = []
                count = 0
                while count < nrReacts:
                    reactant_files.append(reactants)
                    count += 1
            else:
                print("Reactions requires exactly ONE reactant file containg all merged reactants or %d reactant files for each single reactant!" % self.libgen.NumReactants())
                exit()
        #initialize counters for each reactant match
        nrMatches=[0]*nrReacts
        
        self.nrPosProducts = 1
        self.reactants_back = [[] for i in range(nrReacts)]
        for i in range(0, nrReacts):
            fileName = reactant_files[i]
            if i >= self.libgen.NumReactants():
                OEThrow.Fatal("Number of reactant files exceeds number of reactants specified in reaction")
            ifs = oemolistream()
            if not ifs.open(fileName):
                OEThrow.Fatal("Unable to read %s reactant file" % fileName)
            for mol in ifs.GetOEGraphMols():
                match = self.libgen.AddStartingMaterial(mol, i, self.unique)
                if match != 0:
                    self.reactants_back[i].append(mol.CreateCopy())
                nrMatches[i] += match
   
        for nrMatch in nrMatches:
            self.nrPosProducts *= nrMatch
        print("Matching reactants:", nrMatches) 
                 
    def Generate_molecules(self):       
        products = self.LibGen(self.libgen, self.unique, self.isomeric, self.max_products)

        print("Created", len(products),"molecules out of", self.nrPosProducts, "possible products." )
        return products
                
    def LibGen(self, libgen, unique, isomeric, max_products):
        counter = 0
        smiflag = OESMILESFlag_DEFAULT  # Canonical|AtomMaps|Rgroup
        if isomeric:
            smiflag |= OESMILESFlag_ISOMERIC
        #access products
        products = set()
        smiles_uniq = []
        #can be mol or smiles
        outputflag = "smiles" 
        
        ####################
        #randomize reactant lists
        if self.randomize:
            print("Randomizing reactants")
            nrReacts = libgen.NumReactants()            
            nrMatches=[0]*nrReacts        
            for i in range(0, nrReacts):
                random.shuffle(self.reactants_back[i])
                  
            for i in range(0, nrReacts):
                libgen.ClearStartingMaterial(i)
                reac_count = 0
                for reactant in self.reactants_back[i]:
                    reac_count += 1
                    nrMatches[i] += libgen.AddStartingMaterial(reactant, i, self.unique)
            print(nrMatches)
            self.nrPosProducts = 1
            for nrMatch in nrMatches:
                self.nrPosProducts *= nrMatch
        #####################
               
        ##### percentage bar        
        percentage = False
        total = max_products
        if total < 2:
            percentage =  False 
        point = total / 100
        increment = total / 20
        #compound_iter = 0  
        if point == 0:
            percentage =  False
        if increment == 0:
            percentage = False   
        #percentage = False
        ##### percentage bar end    
        ref_weight = 600.0
        ref_N_rot = 15
        ref_N_heavy = 55        
        for mol in libgen.GetProducts():
            ##### percentage bar
            if percentage:
                if(counter % (5 * point) == 0):
                    sys.stdout.write("\r[" + "=" * (counter / increment) +  " " * ((total - counter)/ increment) + "]" +  str(counter / point) + "%")
                    sys.stdout.flush()
                #compound_iter += 1
                if counter == total:
                    print
            ##### percentage bar end
            
            #only generate max_products
            if counter >= max_products:
                break
            
            #filtering
            if self.use_filter:
                mol_weight = float(OECalculateMolecularWeight(mol))
                N_rot = int(OECount(mol, OEIsRotor()))
                N_heavy = int(OECount(mol, OEIsHeavy()))
                if mol_weight > ref_weight:
                    continue
                if N_rot > ref_N_rot:
                    continue
                if N_heavy > ref_N_heavy:
                    continue
                
            if outputflag == "mol":
                products.add(mol.CreateCopy())
                counter = len(products)
                
            elif outputflag == "smiles":
                smiles = OECreateSmiString(mol, smiflag)
                products.add(smiles+" "+self.reaction_title+"___"+mol.GetTitle())
                counter = len(products)
                    
        return products
    
    def CanSmi(self, mol, isomeric, kekule):
        OEFindRingAtomsAndBonds(mol)
        OEAssignAromaticFlags(mol, OEAroModel_OpenEye)
        smiflag = OESMILESFlag_Canonical
        if isomeric:
            smiflag |= OESMILESFlag_ISOMERIC
    
        if kekule:
            for bond in mol.GetBonds(OEIsAromaticBond()):
                bond.SetIntType(5)
            OECanonicalOrderAtoms(mol)
            OECanonicalOrderBonds(mol)
            OEClearAromaticFlags(mol)
            OEKekulize(mol)
    
        smi = OECreateSmiString(mol, smiflag)
        return smi
        
if __name__ == "__main__":
    args = parse_Arguments()  
    
    print("ARGUMEWNTS:", args)
    print("bla")
    
    if args.smirks_file and not (args.smirks and args.rxn):
        if args.output:
            with open(args.output, "w") as outfile:
                outfile.write("")
        with open(args.smirks_file) as smirks_file:
            #may contain several reaction smirks
            for line in smirks_file:
                library_gen = Library_Generation(args)
                line = line.split(" ")
                reaction_title = line[0]
                reaction_smirks = line[1]
                
                print("Initializing reaction", reaction_title)
                library_gen.reaction_title = reaction_title
                library_gen.Init_reaction(reaction_smirks, args.rxn)
                
                library_gen.Init_reactants(library_gen.reactants)
                
                products = library_gen.Generate_molecules()
                
                
                ofs = oemolostream()
                if not args.output:
                    ofs.open(reaction_title+".smi")
                else:
                    ofs.open(args.output)
                for product in products:    
                    if type(product) == str:
                        ofs.write(product+"\n", len(product)+1)
                    else:
                        OEWriteMolecule(ofs, product)

                del(products)
                del(library_gen)
    else:
        library_gen = Library_Generation(args)
        print("Initializing reaction", args.smirks, args.rxn)
        library_gen.Init_reaction(args.smirks, args.rxn)
        
        print("Initializing reactants")
        library_gen.Init_reactants(library_gen.reactants)
        
        print("Generate products")
        products = library_gen.Generate_molecules()
        
        
        ofs = oemolostream()
        if args.output:
            ofs.open(args.output)
            for product in products:
                if type(product) == str:
                    ofs.write(product+"\n", len(product)+1)
                else:
                    OEWriteMolecule(ofs, product)
        else:
            for product in products:
                if type(product) == str:
                    ofs.write(product+"\n", len(product)+1)
                else:
                    print(OECreateIsoSmiString(product), product.GetTitle())
    
    
    
    
    
    
    
    
    
