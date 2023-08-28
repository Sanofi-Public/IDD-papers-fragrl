# GLOBAL PARAMETERS / USER INPUT

# default parameters
PARAMS = {# Input Files for Fragmentation
          "LEAD_FILE": None,         # name of files containing lead molecules
          "LEAD_FRAGMENTS": None,  # lead fragments file
          "FRAGMENT_FILES": ["/path/*_R1.smi",     # name of file(s) with molecules or fragments for creating decodings 
                             "/path/*_R1.smi"],  # (only for GET_DECODINGS = 1 or 2)
          "ATTACHMENTS": ["Xe", "Kr"],      # attachment points in file (only for GET_DECODINGS = 2 or SPLIT_OPTION = 2)
          "DECODINGS_FILE": None,           # name of decodings file (only for GET_DECODINGS = 3)
          "RXN_FOLDER": "./rxn", # folder with rxn files (only for SPLIT_OPTION = 2)
          
          # Scoring Function
          "SCORING_FUNCTION": "ScoringFunctionCalculator",  # scoring function to be used
          "SF_KWARGS": {"conf": "model_conf.conf"},         # kwargs for scoring function
          
          # General stuff
          "MORE_OUTPUT": False,     # Do you want more output?
          "NUM_CORES": 0,           # cores used for parallelization of distance matrix (0 = non-parallel)
          "BATCHSIZE_D": 1000000,   # batch size for parallelized distance matrix 
          "GET_DECODINGS": 2,       # 1 = from splitting molecules, 2 = from fragments, 3 = from decodings file, 4 = from a fragments file
          "WRITE_DECODINGS": False, # only write decodings (no molecule generation)
          "LOAD_MODEL": None,       # name of model to be run (if None, a new model is trained)
          
          # Fragmenting and building the encoding
          "MOL_SPLIT_START": 70,         # atoms with (atomic number >= this) are interpreted as attachment points
          "MAX_ATOMS": 12,               # maximum number of heavy atoms in fragment
          "MAX_FREE": 3,                 # maximum number of attachment points in fragment
          "SPLIT_OPTION": 1,             # 1 = SMARTS, 2 = RXN-Files
          "SPLIT_SMARTS": ["[R]-&!@*"],  # splitting pattern (only for SPLIT_OPTION = 1)
          "ONLY_FINAL_SETS": True,       # only use fragment sets where no fragment can further be split up (only for SPLIT_OPTION = 2)
          "USE_CLASSES": False,          # build separate decodings trees for fragments with different number of attachment points
          
          # Similarity parameters
          "ETA": 0.1,
          
          # Generation parameters
          "FIX_BITS": 1,      # how many bits are fixed?
          "FREEZE_FRAGS": [], # encodings of fragments that should not be replaced 
          
          # Model parameters
          "N_DENSE": 256,
          "N_DENSE2": 64,
          "N_LSTM": 64,       # Times 2 neurons, since there are both a forward and a backward pass in the bidirectional LSTM
          "MODIFY_PROBS": "unity",
          
          # RL training
          "GAMMA": 0.95,
          "BATCH_SIZE": 512,  # how many molecules are chosen each epoch?
          "EPOCHS": 1000,     # how many epochs?
          "MAX_SCORE": 1,     # break training if this score is reached
          "TIMES": 8,         # how many modifications are performed on each molecule during an epoch?
          
          # Sampling
          "NUM_SAMPLES": 200,       # number of output molecules
          "SAMPLE_EPOCHS": None,    # samples are only taken from last ... epochs
          
          # for DeepVL
          "REACTIONFILE": "reactions_ugi.txt",                                           # reaction file
          "BBLOCKS": "./BBlocks",                              # building block folder
          "LEAD_MOLECULES": ["ugi_R1___R2___R3___R4",  # lead molecules
                             "ugi_R11___R22___R33___R44"],
          "REACTION_JUMP": False}  # make it possible to jump from one reaction to another
