from rdkit import Chem


# class to compute QED scores without any configuration necessary
# can be used as a dummy scoring function class
class QEDCalculator():
    
    # initialization function
    def __init__(self, **kwargs): 
        pass
        
    # compute scores           
    def __call__(self, smiles): 
        all_scores = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            score = Chem.QED.qed(mol) if mol else 0
            all_scores.append({"score_total": score})
        return all_scores