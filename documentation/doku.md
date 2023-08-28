# How to reproduce the computations

### Clone Repository

```bash
git clone https://github.com/Sanofi-Pulic/IDD-papers-fragrl.git
cd IDD-papers-fragrl
git submodule update --init  ## to get MolScore submodule
```

### Create and activate conda environment

```bash
conda env create -f environment.yml
conda activate env_fragrl
```

### Download Files from original DeepFMPO repo

```bash
wget https://raw.githubusercontent.com/stan-his/DeepFMPO/master/Data/molecules.smi
wget https://raw.githubusercontent.com/stan-his/DeepFMPO/master/Data/dopamineD4props.csv
```

### Create Configuration Files 

#### MolScore

* The configuration file "model_config.json" for your scoring function can be created via streamlit:
```bash
python submodules/scoring_functions/create_config.py 
```

* In our example, it looks like this (sum of scores from MolWt, ClogP, and TPSA):

```json
{
  "task": "molscore",
  "output_dir": "./molscore_results",
  "load_from_previous": false,
  "dash_monitor": {
    "run": false,
    "pdb_path": null
  },
  "diversity_filter": {
    "run": false
  },
  "scoring_functions": [
    {
      "name": "RDKitDescriptors",
      "run": true,
      "parameters": {
        "prefix": "desc",
        "n_jobs": 1
      }
    }
  ],
  "scoring": {
    "method": "wsum",
    "metrics": [
      {
        "name": "desc_MolWt",
        "weight": 0.33,
        "modifier": "gauss",
        "parameters": {
          "objective": "range",
          "mu": 370,
          "sigma": 50
        }
      },
      {
        "name": "desc_CLogP",
        "weight": 0.33,
        "modifier": "gauss",
        "parameters": {
          "objective": "range",
          "mu": 3,
          "sigma": 1
        }
      },
      {
        "name": "desc_TPSA",
        "weight": 0.33,
        "modifier": "gauss",
        "parameters": {
          "objective": "range",
          "mu": 50,
          "sigma": 10
        }
      }
    ]
  }
}
```

#### FragRL: 

This is the configuration file ("configuration.txt") that is used for the computations. The only options that were changed are "SPLIT_OPTION", 
"USE_CLASSES", and "MODIFY_PROBS".

```
LEAD_FILE = dopamineD4props.csv       # name of files containing lead molecules
#LEAD_FRAGMENTS = lead_frags.csv      # leads as fragments (instead of LEAD_FILE)
FRAGMENT_FILES = molecules.smi        # files with molecules or fragments
ATTACHMENTS = Xe,Kr                   # attachment points in rxn files (only for splitting by rxn), or in fragments
DECODINGS_FILE = decodings.txt        # name of decodings file (only if decodings are read in directly)
RXN_FOLDER = /path/to/deepfmpo/rxn/   # folder with rxn files for splitting (only for splitting by rxn)

SCORING_FUNCTION = MolScoreCalculator        # scoring function class to be used
SF_KWARGS = {"config":"model_config.json"}   # kwargs for scoring function class. Attention! No spaces in dictionary!!!

MORE_OUTPUT = False              # Do you want more output?
GET_DECODINGS = 1                # 1 = from splitting molecules, 2 = from fragments, 3 = from decodings file
WRITE_DECODINGS = False          # only write decodings (no molecule generation)
#LOAD_MODEL = final_actor.h5     # name of model(s) to be run (if not given, training is done from scratch)

# Fragmenting and building the encoding
MOL_SPLIT_START = 70     # atoms with (atomic number >= this) are interpreted as attachment points
MAX_ATOMS = 12           # maximum number of heavy atoms in fragment
MAX_FREE = 3             # maximum number of attachment points in fragment
SPLIT_OPTION = 1         # 1 = SMARTS, 2 = RXN-Files (!!set to 2 for splitting by rxn!!)
SPLIT_SMARTS = [R]-&!@*  # splitting pattern, several patterns are separated by komma without space between them
ONLY_FINAL_SETS = True   # only use fragment sets where no fragment can further be split up (only for splitting by rxn)
USE_CLASSES = True       # !!set to 1 if only one tree should be built regardless of the number of attachment points

# Similarity parameter
ETA = 0.1
          
# Generation parameters
FIX_BITS = 1                          # how many bits are fixed?
#FREEZE_FRAGS = CN1CCOCCC1,OC1CCNCC1  # fragments that should be freezed (comment out if nothing should be fixed)
          
# Model parameters 
N_DENSE = 256
N_DENSE2 = 64
N_LSTM = 64      
MODIFY_PROBS = unity   # !!change to 'linear' or 'square' if you want to modify the output probabilities!! 
          
# RL training
GAMMA = 0.95
BATCH_SIZE = 512  # how many molecules are chosen each epoch?
EPOCHS = 1000     # how many epochs?
#MAX_SCORE = 0.9  # break training if this score is reached (alternative to number of epochs)
TIMES = 8         # how many modifications are performed on each molecule during an epoch?

# Sampling
NUM_SAMPLES = 2000   # number of output molecules
#SAMPLE_EPOCHS = 50  # samples are only taken from last ... epochs (comment out if samples should be used from all epochs)
```

### Run

```
python /path/to/deepfmpo/Main.py -c configuration.txt
```
