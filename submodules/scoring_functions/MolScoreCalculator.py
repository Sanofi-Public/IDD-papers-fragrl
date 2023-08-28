import logging
import sys, os
sys.path.append("{}/submodules/MolScore".format(os.path.dirname(os.path.abspath(__file__))))

from molscore.manager import MolScore

class MolScoreCalculator():
    """
    wrapper for MolScore submodule
    """
    
    # expected kwargs and their default values
    DEFAULT_KWARGS = {"config": None}
    
    # sets attribute <name> either to the value given in kwargs or to default value
    def set_value_from_kwargs(self, name, default, **kwargs):
        if name in kwargs.keys():
            setattr(self, name, kwargs[name])
        else:
            setattr(self, name, default)
    
    # initialization function
    def __init__(self, **kwargs): 
        
        # get attributes from kwargs
        for dk in MolScoreCalculator.DEFAULT_KWARGS:
            self.set_value_from_kwargs(dk, MolScoreCalculator.DEFAULT_KWARGS[dk], **kwargs)
        
        # remove output
        logger = logging.getLogger('molscore')
        logger.setLevel(logging.ERROR)
        
        # create MolScore object
        self.ms = MolScore(model_name="molscore_model", task_config=self.config)
        
        # take care of dangling files and processes
        self.ms.fh.close()                # close filehandler 
        if self.ms.dash_monitor == True:  # set dash monitor to False
            print("Dash monitor will not be activated as it must be killed manually")
            self.ms.dash_monitor = False
        
    # compute scores           
    def __call__(self, smiles): 
        
        # initialize some variables
        all_scores = []
        single_score_names = []
        
        # compute scores and fill main_df (this is done by score_only=False)
        scores = self.ms(smiles, step=0, flt=True, score_only=False)
        self.ms.fh.close()  # close filehandler (otherwise it will not be possible to delete files)
        
        # get some important information
        num_old_smiles = len(self.ms.main_df)-len(smiles)  # number of entries before current calculation
        for sm in self.ms.configs["scoring"]["metrics"]:   # get names of single scores
            single_score_names.append("{}_{}".format(sm["modifier"], sm["name"]))
            
        # fill 'all_scores' by single and total scores and return it
        for i, s in enumerate(scores):
            all_scores.append({"score_total": s})
            for name in single_score_names:  # keep in mind that main_df contains information about all molecules evaluates until now
                all_scores[i]["score_"+name] = self.ms.main_df[name][i+num_old_smiles]
        return all_scores