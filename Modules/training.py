import datetime
import numpy as np
from rdkit import Chem, rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

import Modules.global_parameters as gl
import Modules.visualization as visualization
import Modules.file_io as file_io
from Modules.models import ModifyProbs, MODEL_PARAMS
from Modules.build_encoding import decode, decode_list, array_to_string            # DeepRL
from Modules.build_encoding import decode_reaction_codes, get_name, same_molecule  # DeepVL

# container for result molecule
class ResultMol:
    def __init__(self, mol, score, name):
        self.smiles = Chem.MolToSmiles(mol)
        self.score = score
        self.name = name
    def get_smiline(self):
        return "{},{},{}\n".format(self.smiles, self.name, self.score)
    def __eq__(self, other):  # overload == operator for use of "if ... is in ..." on list of ResultMols
        return self.smiles == other.smiles
    
    
### Some functions to avoid double code for DeepRL and DeepVL ###


# choose start molecules from leads    
def init_start_mols(X):
    rand_n = np.random.randint(0, X.shape[0], gl.PARAMS["BATCH_SIZE"]) # array of length BATCH_SIZE which contains integers corresponding to lead molecules from X
    batch_mol = X[rand_n].copy() # will be output molecules, but at first fetch lead molecules from X
    org_mols = batch_mol.copy()  # input molecules
    r_tot = np.zeros(gl.PARAMS["BATCH_SIZE"])
    return org_mols, batch_mol, r_tot

# initialize some parameters for model
def init_action(t, batch_mol, actor):
    tm = (np.ones((gl.PARAMS["BATCH_SIZE"],1)) * t) / gl.PARAMS["TIMES"]
    probs = actor.predict([batch_mol, tm])
    probs = ModifyProbs.modify_probs(probs)  # modify probabilities
    actions = np.zeros((gl.PARAMS["BATCH_SIZE"]))
    old_batch = batch_mol.copy()  # molecules at the beginning of this TIME (is put into actor and critic)
    rewards = np.zeros((gl.PARAMS["BATCH_SIZE"],1)) # rewards for every molecule in batch
    return probs, actions, old_batch, rewards, tm

# select action (which bit of which fragment should be modified?)
def select_action(i, actions, probs, n_actions, mutable_bits):
    # Select modification actions according to probabilities from actor
    a = np.random.choice(range(n_actions), 1, p=probs[i])
    actions[i] = a[0]     # output of np.random.choice() is an array with one element      
    # determine which fragment is modified at which bit
    action = int(actions[i])               
    a = action // mutable_bits  # which fragment is modified?
    s = action % mutable_bits  # which bit is modified?
    return a, s

# function for fragment modification
def modify_fragment(f, swap):
    f[-(1 + swap)] = (f[-(1 + swap)] + 1) % 2
    return f

# evaluate scores for molecules
def evaluate_scores(batch_mol, sf, org_mols, rewards, xe=False, **kwargs):
    
    # check if two decodings correspond to the same molecule
    def identical(om, bm):
        if xe:
            return same_molecule(om, bm, kwargs["decodings_dict"], kwargs["vl_xe"])
        else:
            return np.all(org_mols[i] == batch_mol[i])
    
    # convert to molecules
    if xe:
        mols = decode_reaction_codes(batch_mol, kwargs["decodings_dict"], kwargs["vl_xe"])  
    else:
        mols = decode_list(batch_mol, kwargs["decodings"])                         
    # convert to smiles and get scores
    smiles = [Chem.MolToSmiles(mol) if mol else "X" for mol in mols]  
    all_scores = sf(smiles)                                          
    for i in range(batch_mol.shape[0]): # for every molecule
        if not identical(org_mols[i], batch_mol[i]):  # if molecule was modified                        
            rewards[i] += all_scores[i]["score_total"]  
    return all_scores, rewards         

# train model (actor and critic)
def train(actor, critic, old_batch, batch_mol, actions, probs, tm, rewards, Vs):
    # Calculate TD-error = estimator of advantage function (critic predicts the value function)
    # see https://www.freecodecamp.org/news/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d/
    target = rewards + gl.PARAMS["GAMMA"] * critic.predict([batch_mol, tm+1.0/gl.PARAMS["TIMES"]])  # target = current rewards + future prediction
    td_error = target - Vs  # td_error = improvement in comparison to before

    # Improve critic model
    critic.fit([old_batch,tm], target, verbose=0)

    # Determine for each action how good it is (improvement over average action)
    target_actor = np.zeros_like(probs)
    for i in range(gl.PARAMS["BATCH_SIZE"]):
        a = int(actions[i])
        #loss = -np.log(probs[i,a]) * td_error[i]
        target_actor[i,a] = td_error[i]

    # Maximize expected reward (improve actor model)
    actor.fit([old_batch,tm], target_actor, verbose=0)
    return actor, critic

# write input and output molecules to files
def write_output(org_mols, batch_mol, epoch, all_scores, xe=False, **kwargs):
    
    # create molecules' names
    def get_names(mols):
        if not xe:  # mol0, mol1, ...
            return ["mol{}".format(i) for i,_ in enumerate(mols)]
        else:       # contains information about reaction and reagents
            return [get_name(mol, kwargs["vl_xe"].reactions_dict, kwargs["inverse_reagent_dict"], 
                             kwargs["decodings_dict"]) for mol in mols]
    
    # array encoding -> rdkit.Mol
    def decode_mols(mols):
        if not xe:
            return decode_list(mols, kwargs["decodings"])
        else:
            return decode_reaction_codes(mols, kwargs["decodings_dict"], kwargs["vl_xe"])
    
    # write out input molecules as smiles
    names = get_names(org_mols)
    org_mols = decode_mols(org_mols)
    file_io.fragments_to_smilesCsv(org_mols, names, "History/in-{}.csv".format(epoch))
        
    # get single scores for each output molecule
    names = get_names(batch_mol)
    for counter, scores in enumerate(all_scores):
        for score_name in scores.keys():
            score = scores[score_name]
            names[counter] += ",{}".format(score)
            
    # write out output molecules as smiles    
    batch_mol = decode_mols(batch_mol)
    file_io.fragments_to_smilesCsv(batch_mol, names, "History/out-{}.csv".format(epoch),
                                    ["SMILES", "NAME"] + list(scores.keys()))
    return batch_mol, names

# save best NUM_SAMPLES molecules
def save_result_mols(result_mols, batch_mol, all_scores, names):
    for i, mol in enumerate(batch_mol):                        # add all new molecules to results
        if mol is not None:
            r = ResultMol(mol, all_scores[i]["score_total"], names[i].split(",")[0])
            if r not in result_mols:                           # only unique molecules
                result_mols.append(r)                         
    result_mols.sort(key=lambda mol: mol.score, reverse=True)  # sort results by score
    result_mols = result_mols[:gl.PARAMS["NUM_SAMPLES"]]       # only keep best
    return result_mols

# writes mean score for each epoch into file
def write_mean_scores(scores):
    with open("average_score.csv", "w") as out:
        out.write("Epoch,Score\n")
        for i, s in enumerate(scores):
            out.write("{},{}\n".format(i, s))


### Training functions ###


# Run Network (DeepRL)
# if critic is given: train actor and critic
# if not: just use actor to create molecules
def run(X, sf, freeze_codes, actor, decodings, critic=None):
    
    # X = list of lead molecules
    # X.shape[0] = number of lead molecules
    # X.shape[1] = maximum number of fragments
    # X.shape[2] = length of binary code (including first bit that only tells if fragment present or not)
    
    # number of possible actions (number of fragments x number of bits that can be modified)
    mutable_bits = MODEL_PARAMS["MUTABLE_BITS"]
    n_actions = MODEL_PARAMS["N_ACTIONS"]
    
    print()
    
    # best NUM_SAMLPES molecules found
    result_mols = [] 
    # current mean score
    current_score = 0.0
    # mean score for each epoch
    mean_scores = []
    
    # if SAMPLE_EPOCHS not defined: samples are taken from all epochs
    if gl.PARAMS["SAMPLE_EPOCHS"] == None:
        gl.PARAMS["SAMPLE_EPOCHS"] = gl.PARAMS["EPOCHS"]

    # For every epoch
    for e in range(gl.PARAMS["EPOCHS"]):
        
        # if maximum score is reached: save actor and critic and stop training
        if current_score > gl.PARAMS["MAX_SCORE"]:
            print("Maximum score reached. Finishing program.")
            if critic:    
                actor.save("./History/final_actor.h5")   
                critic.save("./History/final_critic.h5") 
            return result_mols

        # Select random starting lead molecules
        org_mols, batch_mol, r_tot = init_start_mols(X)
        # saves vector of smiles for every batch (only used for advanced visualization in runs with 1 epoch)
        batch_over_time = [ [Chem.MolToSmiles(decode(org_mol, decodings))] for org_mol in org_mols ]

        # For all modification steps
        for t in range(gl.PARAMS["TIMES"]):
            
            # initialize some stuff
            probs, actions, old_batch, rewards, tm = init_action(t, batch_mol, actor)
            
            # Initial critic value (value function before modifications)
            if critic:  
                Vs = critic.predict([batch_mol,tm])

            # Modification of Molecules
            for i in range(gl.PARAMS["BATCH_SIZE"]):
                # select action (a = fragment, s = which bit)
                a, s = select_action(i, actions, probs, n_actions, mutable_bits)
                
                # apply action to molecule
                if batch_mol[i,a,0] == 0: # if chosen fragment does not exist:
                    rewards[i] -= 0.1     # the reward is decreased and nothing is done
                elif array_to_string(batch_mol[i,a]) in freeze_codes:    # if the chosen fragment is freezed:
                    rewards[i] -= 0.1                                    # the reward is decreased and nothing is done
                else:                                                    # only if both is not the case:
                    batch_mol[i,a] = modify_fragment(batch_mol[i,a], s)  # modify molecule                   
                
                # save stuff for advanced output    
                if gl.PARAMS["MORE_OUTPUT"] and gl.PARAMS["EPOCHS"] == 1:
                    try:
                        mol = decode(batch_mol[i], decodings)
                        smi = Chem.MolToSmiles(mol)
                        batch_over_time[i].append(smi)
                    except KeyError:      # Fragment codes that don't belong to any fragments
                        batch_over_time[i].append("[Yb]")
                    except RuntimeError:  # Error when joining Fragments
                        batch_over_time[i].append("[Hf]")

            # If final round of epoch: compute rewards from scoring function
            if t + 1 == gl.PARAMS["TIMES"]:
                all_scores, rewards = evaluate_scores(batch_mol, sf, org_mols, rewards, decodings=decodings)                                      

            ### The following is done after every modification (TIMES per Epoch)
            
            if critic:
                actor, critic = train(actor, critic, old_batch, batch_mol, actions, probs, tm, rewards, Vs)
            
            # reward for every molecule in batch
            r_tot += rewards[:,0]  # add up rewards from all TIMES
            
        ### AT THE END OF EACH EPOCH
        current_score = np.mean(r_tot)
        mean_scores.append(current_score)
            
        # console output: summary
        print("{}: \t Epoch {} \t Mean score: {:.3}".format(str(datetime.datetime.now())[:-7], e, current_score))
        
        # advanced output for runs with one epoch    
        if gl.PARAMS["MORE_OUTPUT"] and gl.PARAMS["EPOCHS"] == 1:
            try:         
                visualization.smilesMatrix_to_svg(batch_over_time, "History/modification.svg")
            except (OSError, RuntimeError):
                print("I'm sorry it is not possible to view the modifications as image. Printing them as SMILES instead.")
                file_io.smilesMatrix_to_csv(batch_over_time, "History/modification.csv")
        
        # write output to file
        mols, names = write_output(org_mols, batch_mol, e, all_scores, decodings=decodings)
        
        # save result mols (i.e. highest score)
        if e >= gl.PARAMS["EPOCHS"] - gl.PARAMS["SAMPLE_EPOCHS"]:
            result_mols = save_result_mols(result_mols, mols, all_scores, names)
    
    # if model was trained: save final actor and critic
    if critic:    
        actor.save("./History/final_actor.h5")   
        critic.save("./History/final_critic.h5") 
    
    # write scores for each epoch into file    
    write_mean_scores(mean_scores)
    
    # return best molecules    
    return result_mols


# Run Network (DeepVL)
def run_deepVL(X, sf, actor, critic, decodings_dict, vl_xe, inverse_reagent_dict):
    
    # X = list of lead molecules
    # X.shape[0] = number of lead molecules
    # X.shape[1] = maximum number of fragments
    # X.shape[2] = length of binary code
    
    # number of possible actions (number of fragments x number of bits)
    mutable_bits = MODEL_PARAMS["MUTABLE_BITS"]
    n_actions = MODEL_PARAMS["N_ACTIONS"]
    
    print()
    
    # best NUM_SAMLPES molecules found
    result_mols = [] 
    # mean scores for each epoch
    mean_scores = []
    
    # if SAMPLE_EPOCHS not defined: samples are taken from all epochs
    if gl.PARAMS["SAMPLE_EPOCHS"] == None:
        gl.PARAMS["SAMPLE_EPOCHS"] = gl.PARAMS["EPOCHS"]

    # For every epoch
    for e in range(gl.PARAMS["EPOCHS"]):
        
        # Select random starting lead molecules
        org_mols, batch_mol, r_tot = init_start_mols(X)

        # For all modification steps
        for t in range(gl.PARAMS["TIMES"]):
            
            # initialize some stuff
            probs, actions, old_batch, rewards, tm = init_action(t, batch_mol, actor)
            
            # Initial critic value (value function before modifications)
            Vs = critic.predict([batch_mol,tm])

            # Modification of Molecules
            for i in range(gl.PARAMS["BATCH_SIZE"]):
                # select action (a = fragment, s = which bit)
                a, s = select_action(i, actions, probs, n_actions, mutable_bits)
                # apply action to molecule
                batch_mol[i,a] = modify_fragment(batch_mol[i,a], s)  # modify molecule                   

            # If final round of epoch: compute rewards from scoring function
            if t + 1 == gl.PARAMS["TIMES"]:
                all_scores, rewards = evaluate_scores(batch_mol, sf, org_mols, rewards, True, vl_xe=vl_xe, decodings_dict=decodings_dict)                                       

            ### The following is done after every modification (TIMES per Epoch)
            actor, critic = train(actor, critic, old_batch, batch_mol, actions, probs, tm, rewards, Vs)

            # reward for every molecule in batch
            r_tot += rewards[:,0]  # add up rewards from all TIMES
            
        ### AT THE END OF EACH EPOCH
        current_score = np.mean(r_tot)
        mean_scores.append(current_score)
            
        # console output: summary
        print("{}: \t Epoch {} \t Mean score: {:.3}".format(str(datetime.datetime.now())[:-7], e, current_score))
        
        # write output to file
        mols, names = write_output(org_mols, batch_mol, e, all_scores, True, vl_xe=vl_xe, decodings_dict=decodings_dict,
                                   inverse_reagent_dict=inverse_reagent_dict)
        
        # save result mols (i.e. highest score)
        if e >= gl.PARAMS["EPOCHS"] - gl.PARAMS["SAMPLE_EPOCHS"]:
            result_mols = save_result_mols(result_mols, mols, all_scores, names)
    
    # save final actor and critic
    actor.save("./History/final_actor.h5")   
    critic.save("./History/final_critic.h5") 
    
    # write scores for each epoch into file
    write_mean_scores(mean_scores)
    
    # return best molecules    
    return result_mols
