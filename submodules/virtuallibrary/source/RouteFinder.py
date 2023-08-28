import os, sys, re, io
import glob, copy
import json
import argparse
import signal

from rdkit import rdBase, Chem
from rdkit.Chem import Draw, AllChem
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

from openeye import oechem
from graphviz import Digraph

if __name__ == "__main__":
    from Virtual_library_RXN import Library_Generation
else:
    from .Virtual_library_RXN import Library_Generation

# read molecules from file
# first column = SMILES, second column = name, no titels 
# result is a dictionary of {name: SMILES}        
def read_molfile(file_name):
    
    # Drop salt from SMILES string
    def drop_salt(s):
        s = s.split(".")
        return [x for _, x in sorted(zip(map(len,s), s), reverse=True)][0]
    
    # removes empty strings from a list of strings
    def remove_empty(stringlist):
        return [s for s in stringlist if s != ""]
    
    drop_first = False  # first line to be dropped?
    returns = dict()    # name: SMILES
    separator = " "     # separator for columns
    
    with open(file_name) as f:
        lines = f.readlines()
        
        # check if there is a molecule in first line
        if Chem.MolFromSmiles(drop_salt(lines[0].strip().split(",")[0].strip())) == None:  
            drop_first = True
        
        # check for separator    
        if "," in lines[0]:
            separator = ","
        
        # read in molecules    
        for l in lines:
            if drop_first:              
                drop_first = False
                continue
            # rest of the lines (containing structures in first column and name in the second) 
            if not l.startswith("#"):   
                ll = remove_empty(l.strip().split(separator))
                name = ll[1]
                smi = drop_salt(ll[0].strip())
                returns[name] = smi
                
    return returns


# create library generation interfaces for RXN files
def create_libgen_interfaces(rxn_files):
    lib_gens = []
    for rxn_file in rxn_files:
        library_gen = Library_Generation_Interface()
        library_gen.set_reaction(rxn_file)
        library_gen.Init_reaction(rxn_file = rxn_file)
        lib_gens.append(library_gen)
    return lib_gens


# modified version of Library_Generation
# used for finding synthetic route for possible products
class Library_Generation_Interface(Library_Generation): 
    
    # initialization with default values
    def __init__(self):
        self.smirks_string = None
        self.rxn_file = None
        self.reaction_title = "RXN"
        self.relax = False
        self.unique = True
        self.implicitH = False
        self.valcorrect = True
        self.isomeric = True
        self.max_products = 100
        self.randomize = True
        self.output = None
        self.reactants = None
        # if set to True, reagents are filtered by molecular weight, rotable bonds and number of heavy atoms
        self.use_filter = False  
        # do not give too many warnings from openeye
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
    
    # rxn = name of rxn-file for backwards reaction   
    def set_reaction(self, rxn):
        self.rxn = rxn
    
    # smi = SMILES of one molecule (desired product) 
    def Init_reactants(self, smi):
        nrReacts = self.libgen.NumReactants()
        if nrReacts != 1:
            raise RuntimeError("Invalid number of reactants:", nrReacts)
        nrMatches=[0]*nrReacts
        
        self.nrPosProducts = 1
        self.reactants_back = [[] for i in range(nrReacts)]
        for i in range(0, nrReacts):
            ifs = oechem.oemolistream()
            ifs.SetFormat(oechem.OEFormat_CAN)
            ifs.openstring(smi)
            for mol in ifs.GetOEGraphMols():
                match = self.libgen.AddStartingMaterial(mol, i, self.unique)
                if match != 0:
                    self.reactants_back[i].append(mol.CreateCopy())
                nrMatches[i] += match
   
        for nrMatch in nrMatches:
            self.nrPosProducts *= nrMatch
    
    # runs reaction and generates products        
    def Generate_molecules(self):
        sys.stdout = open(os.devnull, "w")
        products = Library_Generation.Generate_molecules(self)
        sys.stdout = sys.__stdout__
        return products
    

# class to find reaction routes for a given product        
class RouteFinder:
    
    # init with product smiles and list of Library_Generation_Interface corresponding to backwards reactions
    def __init__(self, smi, lib_gens):
        # input
        self.product = smi             # smiles string of the desired product
        self.lib_gens = lib_gens       # library generation interfaces for the backwards reactions
        # output
        self.routes = dict()           # dictionary of possible reaction routes
        self.flattend_dict = dict()    # flattend dictionary of all molecules from the route tree
        self.reagent_sets = []         # sets of reagents that can form product (for RouteFinder identical to unique_sets, for RouteFinderAttachments numbering of attachment points might be different)
        self.unique_sets = []          # unique sets of reagents
        self.final_sets = []           # (unique) reagent sets consisting only of reagents that cannot be split into further reagents
        # helpervariables for graph generation
        self.mol_counter = 0           # counts molecule (for unique labels)
        self.reaction_counter = 0      # counts reactions (for unique labels)
        # also use ring reactions?
        self.ring_reactions = False
    
    # finds reagents for a given product and a given reaction
    # result is a set where every item is a route
    # the route is stored in form of a string "SMILES reaction_name"
    # where several reagents are separated by dot in SMILES        
    def find_reagents(self, library_gen):
        library_gen.Init_reactants(self.product)
        reagents = library_gen.Generate_molecules()
        if self.ring_reactions == False:
            reagents = self.remove_ring_reactions(reagents)
        return reagents
    
    # remove reagent sets with only one reagent as this is normally a ring reaction
    def remove_ring_reactions(self, reagents):
        reagents_new = copy.copy(reagents)
        for r in reagents:
            if len(r.split(".")) == 1:
                reagents_new.remove(r)
        return reagents_new
    
    # convert a route string to a dictionary
    # during this the routes for the reagents are evaluated recursively
    def convert_route_to_dict(self, route):
        info = dict()
        info["type"] = "reaction"
        info["name"] = route.split()[1]
        info["children"] = []     # molecules
        for reagent in route.split()[0].split("."):
            mol = {"type": "mol", "smiles": reagent}
            rf = RouteFinder(reagent, self.lib_gens)
            rf.run()
            mol["children"] = rf.routes["children"]
            info["children"].append(mol)
        return info
    
    # converts a list of route strings to a dictionary
    def create_dict(self, route_strings):
        route_dicts = []
        for route in route_strings:
            route_dicts.append(self.convert_route_to_dict(route))
        self.routes = dict()
        self.routes["type"] = "mol"
        self.routes["smiles"] = self.product
        self.routes["children"] = route_dicts
    
    # run the RouteFinder (fills routes)
    def run(self):
        route_strings = []
        for lg in self.lib_gens:
            route_strings += self.find_reagents(lg)
        self.create_dict(route_strings)

    # create a flatted version of route dictionary
    # result is self.flattened_dict 
    # which is a dictionary where every element corresponds to a molecule in the form {SMILES: tree}   
    def flatten_dict(self, root):
        self.flattend_dict[root["smiles"]] = root
        for r in root["children"]:
            for m in r["children"]:
                self.flatten_dict(m)        
            
    # creates reagent sets, i.e. a list where every item consists of SMILES strings
    # the SMILES string of one list can build up the product                                        
    def create_reagent_sets(self):
        
        # replace product in current_reags by its reagents according to reaction
        # product = SMILES of product
        # reaction = reaction dictionary from self.routes
        # current_reags = list of SMILES
        # returns modified SMILES list    
        def replace_reaction(product, reaction, current_reags):
            current_reagents = copy.copy(current_reags)
            reagents = [mol["smiles"] for mol in reaction["children"]]
            current_reagents.remove(product)
            current_reagents += reagents
            return current_reagents   
        
        # create flattened version of route tree
        if self.flattend_dict == dict():
            self.flatten_dict(self.routes)
            
        self.reagent_sets.clear()
        sets_before = [[self.routes["smiles"]]]
        
        # this loop goes through reaction depth until no molecule in this depth has any reactions
        while len(sets_before) != 0:  
            sets_current = []
            for s in sets_before:  # reagent sets from loop before
                for m in s:        # molecule in reagent set
                    tree = self.flattend_dict[m]  # get reaction tree for this molecule
                    for r in tree["children"]:    # reactions
                        current_set = replace_reaction(m, r, s)  # create new set from this reaction
                        current_set_std = self.standardize_set(current_set)  # standardize set 
                        if current_set_std not in self.reagent_sets:  # add standardize set to reagent_sets
                            self.reagent_sets.append(current_set_std)
                        sets_current.append(current_set)  # to continue searching tree use non-standardized set
            sets_before = sets_current
            
    @staticmethod   
    # doesn't do anything, is overridden by RouteFinderAttachments     
    def standardize_smi(s):
        return s
    
    @staticmethod
    # standardize reagent set (in this case: sort fragments)
    def standardize_set(rs):
        return sorted(rs)
    
    # creates unique reagent sets
    # here they are identical to reagent sets but function is overridden for RouteFinderAttachments
    def create_unique_sets(self):
        self.create_reagent_sets()
        self.unique_sets = self.reagent_sets
    
    # creates final sets            
    def create_final_sets(self):
        self.final_sets.clear()
        self.create_unique_sets()
        for s in self.unique_sets:
            final = True
            for m in s:  # go through all molecules in set
                tree = self.flattend_dict[m]
                if len(tree["children"]) != 0:  # if one of the molecules has children:
                    final = False               # set is not a final set
                    break
            if final:
                self.final_sets.append(s)
    
    # writes routes into a json file
    def write_json(self, filename):
        with open(filename, "w") as out:
            json.dump(self.routes, out, indent=4)
    
    @staticmethod    
    # creates an image from a SMILES string and returns a HTML string containing this image    
    def create_image_label(smiles, scratch):
        filename = "{}/{}.png".format(scratch, smiles.replace("/",""))
        if os.path.exists(filename) == False:
            mol = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(mol)
            Draw.MolToFile(mol, filename)
        return "<<table><tr><td><img src='{}'/></td></tr><tr><td>{}</td></tr></table>>".format(filename, smiles)
    
    @staticmethod    
    # returns the label for a node, consisting of several molecules, with images and SMILES
    # smiles = list of SMILES strings
    # labels = list of labels (used to mark the node in order to add edges) 
    # scratch = name of scratch folder for images
    def create_several_image_label(smiles, labels, scratch):
        if len(smiles) != len(labels):
            raise RuntimeError("Wrong number of labels for given SMILES.")
        content = "<<table><tr>"
        for smi in smiles:  # create images and add row containing them
            filename = "{}/{}.png".format(scratch, smi.replace("/",""))
            if os.path.exists(filename) == False:
                mol = Chem.MolFromSmiles(smi)
                AllChem.Compute2DCoords(mol)
                Draw.MolToFile(mol, filename)
            content += "<td><img src='{}'/></td>".format(filename)
        content += "</tr><tr>"
        for i, smi in enumerate(smiles):  # add row with SMILES, labels are added here
            content += "<td port='f{}'>{}</td>".format(labels[i], smi)
        content += "</tr></table>>"
        return content
    
    # helperfunction for graph generation that is called recursively
    # molecule = molecule as dictionary
    # mol_name = name of the graph (sub)node corresponding to molecule
    # dot = graph where nodes and edges are added  
    # scratch = name of scratch folder for images      
    def create_graph_helper(self, molecule, mol_name, dot, scratch):
        for r in molecule["children"]:  # reactions
            reaction_name = "{}{}".format(r["name"], self.reaction_counter)
            molnames, smiles, labels = [], [], []
            for m in r["children"]:     # molecules
                smiles.append(m["smiles"])
                labels.append(self.mol_counter)
                molnames.append("{}:f{}".format(reaction_name, self.mol_counter))
                self.mol_counter += 1
            content = RouteFinder.create_several_image_label(smiles, labels, scratch)
            dot.node(reaction_name, content)                    # node with subnodes for reagents
            dot.edge(mol_name, reaction_name, label=r["name"])  # edge for reaction
            self.reaction_counter += 1
            for i, m in enumerate(r["children"]):               # extend graph from reagents
                dot = self.create_graph_helper(m, molnames[i], dot, scratch)
        return dot
    
    # function to create graph visualization
    # filename = name of output file without ending .pdf    
    # scratch_folder = name of scratch folder for images (has to exist)    
    def create_graph(self, filename, scratch_folder):
        self.counter = 0
        self.reaction_counter = 0
        dot = Digraph(node_attr={'shape': 'none'})
        dot.node("root", label=RouteFinder.create_image_label(self.routes["smiles"], scratch_folder))
        dot = self.create_graph_helper(self.routes, "root", dot, scratch_folder)        
        dot.render(filename, view=False)   

            

# class to find reaction routes for a given product from reactions 
# where attachment points are marked by heavy atoms
# those attachment points are replaced by [<number>*] where same number means that those atoms were connected
class RouteFinderAttachments(RouteFinder):
    
    PATTERN = re.compile("\[\d+\*]")  # regex pattern for "[<number>*]"
    
    # attachments = list of element symbols that mark attachment points
    # attach_counter = counts for attachment points while splitting molecule recursively
    #                  should normally not be set from the outside
    def __init__(self, smi, lib_gens, attachments, attach_counter=0):
        RouteFinder.__init__(self, smi, lib_gens)
        self.attachments = attachments
        self.attach_counter = attach_counter
    
    # wrapper for run function to include time alarm
    def run_wrapper(self, max_time):

        # handler for time alarm
        def handler(signum, frame):
            print("Time Error in RouteFinder.run()")
            signal.alarm(max_time)  # start new alarm counter (in case first is ignored because of __del__)
            raise Exception("Time Error")
            
        # create signal for time alarm
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(max_time)
        # run RouteFinder (until time is over)
        try:
            self.run()
        except Exception:         # time is over
            self.routes = dict()  # delete routes that might be created
            return                # leave function
        signal.alarm(0)           # deactivate alarm
        
    ### some static methods that help in dealing with attachment points
    
    @staticmethod
    # standardizes a smiles string with attachment points (done during create_reagent_sets())
    def standardize_smi(s):
        return RouteFinderAttachments.PATTERN.sub("*", s)  # replace all attachment points by *
        
    @staticmethod   
    # standardizes a list of SMILES (done during create_unique_sets()) 
    def create_standardized_smiles(smiles): 
        return [Chem.MolToSmiles(Chem.MolFromSmiles(RouteFinderAttachments.standardize_smi(s))) for s in smiles]
    
    @staticmethod          
    # create a list of standardized sets from given sets where attachment points are marked with numbers
    # this means SMILES are canonical, all attachment points are identical and list is sorted
    def create_standardized_sets(sets):
        standardized_sets = []
        for rs in sets:
            rs = RouteFinderAttachments.create_standardized_smiles(rs)
            standardized_sets.append(sorted(rs))  # sorted list
        return standardized_sets
    
    @staticmethod
    # happens during create_reagent_sets()  
    # set is not completely standard because numbers of attachment points can still be different 
    def standardize_set(rs):
        
        # extracts the numbers from a list of attachment points (pattern "[<number>*]")
        # and returns them as a list of unique ints
        def match_to_unique_int(matches):
            ret = []
            for m in matches:
                ret.append(int(m[1:-2]))
            return sorted(list(set(ret)))  
        
        # numbers of attachment points start with 1
        rs_string = ".".join(rs)
        aps = match_to_unique_int(RouteFinderAttachments.PATTERN.findall(rs_string))
        for j, ap in enumerate(aps):
            rs_string = rs_string.replace("[{}*]".format(ap), "[{}*]".format(j+1))
        rs = rs_string.split(".")
        return sorted(rs)
    
    ### functions from baseclass that are overridden
    
    # function is overridden to replace rare element symbols by numbers    
    def find_reagents(self, rxn_file):
        reagents = list(RouteFinder.find_reagents(self, rxn_file))
        for i, rs in enumerate(reagents):
            for a in self.attachments:
                if "[{}]".format(a) in rs:
                    self.attach_counter += 1
                    rs = rs.replace("[{}]".format(a), "[{}*]".format(self.attach_counter))
                    reagents[i] = rs
        return set(reagents)
    
    # function is overridden because here the recursive call of another object of the class happens
    # during this couting of attachment points has to be handled
    def convert_route_to_dict(self, route):
        info = dict()
        info["type"] = "reaction"
        info["name"] = route.split()[1]
        info["children"] = []     # molecules
        for reagent in route.split()[0].split("."):
            mol = {"type": "mol", "smiles": reagent}
            rf = RouteFinderAttachments(reagent, self.lib_gens, self.attachments, self.attach_counter)
            rf.run()
            self.attach_counter = rf.attach_counter
            mol["children"] = rf.routes["children"]
            info["children"].append(mol)
        return info
            
    # function is overridden to ensure that reagents with different markers for attachment points are treated equally
    # if creating reagent sets needs more than 'max_time' seconds, it is stopped
    def create_unique_sets(self, max_time):
        
        # handler for time alarm
        def handler_r(signum, frame):
            print("Time Error in create_reagent_sets()")
            raise Exception("Time Error")
        
        # handler for time alarm
        def handler_u(signum, frame):
            print("Time Error in create_unique_sets()")
            raise Exception("Time Error")
        
        # create signal for time alarm (reagent_sets)
        signal.signal(signal.SIGALRM, handler_r)
        signal.alarm(max_time)
        # create reagent sets (until time is over)
        try:
            self.create_reagent_sets()
        except Exception:  # if time is over, leave function
            return
        signal.alarm(0)
        
        # create signal for time alarm (unique_sets)
        signal.signal(signal.SIGALRM, handler_u)
        signal.alarm(max_time)
        # standardize sets (until time is over)
        try:
            reagent_sets_standard = RouteFinderAttachments.create_standardized_sets(self.reagent_sets)
        except Exception:  # if time is over, leave function
            return
        signal.alarm(0)
        
        # create unique sets
        self.unique_sets.clear()
        helper_sets = []
        for i, s in enumerate(reagent_sets_standard):
            if s not in helper_sets:
                helper_sets.append(s)
                self.unique_sets.append(self.reagent_sets[i])
        
    # function is overriden because unique_sets and flattened_dict cannot be directly compared 
    # if there are more than max_sets reagent sets for a molecule, it is discarded
    def create_final_sets(self, max_time):
        self.final_sets.clear()
        self.create_unique_sets(max_time)
        unique_sets_standard = RouteFinderAttachments.create_standardized_sets(self.unique_sets)
        number_of_child_reactions = dict()  # {standardized_smiles : number_of_child_reactions}
        for key in self.flattend_dict:      # get number of child reactions for every molecule
            number_of_child_reactions[RouteFinderAttachments.create_standardized_smiles([key])[0]] = len(self.flattend_dict[key]["children"])
        # fill final sets
        for i, s in enumerate(unique_sets_standard):
            final = True
            for m in s:
                if number_of_child_reactions[m] != 0:
                    final = False
                    break
            if final:
                self.final_sets.append(self.unique_sets[i])
            
            
def parse_args():
    """Parses input arguments."""

    parser = argparse.ArgumentParser(description="Finds possible reaction routes for molecules.")

    parser.add_argument("--molecules", "-m",
                        help="Input file for molecules", type=str, required=True)
    parser.add_argument("--reactions", "-r",
                        help="Folder with rxn files for backwards reactions", type=str, required=True)
    parser.add_argument("--scratch", "-s",
                        help="Scratch folder (has to exist)", type=str, required=True)
    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}     

    
if __name__ == "__main__":
    args = parse_args()
    rxn_filenames = glob.glob("{}/*.rxn".format(args["reactions"]))
    mols = read_molfile(args["molecules"])
    
    lib_gens = create_libgen_interfaces(rxn_filenames)
    
    for m in mols:
        rf = RouteFinder(mols[m], lib_gens)
        rf.run()
        rf.write_json("{}.json".format(m))
        rf.create_graph("{}".format(m), args["scratch"])
        rf.create_final_sets()
        print("\n{}".format(m))
        print("unique sets")
        for rs in rf.unique_sets:
            print(".".join(rs), ", Number of reagents:", len(rs))
        print("final sets")
        for rs in rf.final_sets:
            print(".".join(rs), ", Number of reagents:", len(rs))
    
    
##### This is an example how RouteFinderAttachments might be used #####  

    # ATTACHMENTS = ["Xe", "Kr"]  # this is how attachment points are marked in rxn-files
    # for m in mols:
        # rfa = RouteFinderAttachments(mols[m], lib_gens, ATTACHMENTS)
        # rfa.run()
        # print("\n{}".format(m))
        # rfa.create_unique_sets(sys.maxsize)
        # rfa.create_final_sets(sys.maxsize)
        # rfa.write_json("{}.json".format(m))
        # rfa.create_graph("{}".format(m), args["scratch"])
        # print("unique sets")
        # for rs in rfa.unique_sets:
            # print(".".join(rs), ", Number of reagents:", len(rs))
        # print("final sets")
        # for rs in rfa.final_sets:
            # print(".".join(rs), ", Number of reagents:", len(rs))
        
