try:
    from submodules.scoring_functions.MolScoreCalculator import MolScoreCalculator
    from submodules.scoring_functions.QEDCalculator import QEDCalculator
except:
    from MolScoreCalculator import MolScoreCalculator
    from QEDCalculator import QEDCalculator
    

# this is a list of available classes of scoring functions
SCORING_FUNCTION_CLASSES = [MolScoreCalculator, QEDCalculator]

"""
Scoring function should be a class where some tasks that are shared for every call
can be reallocated to the __init__, and has a __call__ method which computed the scores
from the SMILES strings of given molecules. 
"""

def get_scoring_function(scoring_function, **kwargs):
    """Function that initializes and returns a scoring function by name"""
    
    scoring_functions = [f.__name__ for f in SCORING_FUNCTION_CLASSES]
    scoring_function_class = [f for f in SCORING_FUNCTION_CLASSES if f.__name__ == scoring_function][0]

    if scoring_function not in scoring_functions:
        raise ValueError("Scoring function must be one of {}".format([f for f in scoring_functions]))

    return scoring_function_class(**kwargs)

