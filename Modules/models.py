import math
import numpy as np
import tensorflow as tf

from keras import Model, models, backend
from keras.layers import LSTM, TimeDistributed, Dense, Input, LeakyReLU, Bidirectional, Concatenate, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2

import Modules.global_parameters as gl

# These are some parameters needed for the model but outside 'build_models()'
MODEL_PARAMS = {"N_ACTIONS": None, "MUTABLE_BITS": None}

# Create models
# inp_shape: shape of a matrix representation of one molecule
#            inp_shape[0]: number of fragments
#            inp_shape[1]: number of bits in fragment (including first bit that only tells if fragment present or not)
# exists_switch: if True first bit of each fragment is ignored because it only tells if this fragment exists
def build_models(inp_shape, exist_switch=True):
    
    # number of possible actions = number of fragments * number of bits that are not fixed
    if exist_switch:
        MODEL_PARAMS["N_ACTIONS"] = inp_shape[0] * (inp_shape[1] - gl.PARAMS["FIX_BITS"] - 1)
        MODEL_PARAMS["MUTABLE_BITS"] = inp_shape[1] - gl.PARAMS["FIX_BITS"] - 1
    else:
        MODEL_PARAMS["N_ACTIONS"] = inp_shape[0] * (inp_shape[1] - gl.PARAMS["FIX_BITS"])
        MODEL_PARAMS["MUTABLE_BITS"] = inp_shape[1] - gl.PARAMS["FIX_BITS"]
        
    # Build the actor
    inp = Input(inp_shape)
    hidden_inp = LeakyReLU(0.1)(TimeDistributed(Dense(gl.PARAMS["N_DENSE"], activation="linear"))(inp))
    hidden = LSTM(gl.PARAMS["N_LSTM"], return_sequences=True)(hidden_inp)
    hidden = Flatten()(hidden)

    hidden2 = LSTM(gl.PARAMS["N_LSTM"], return_sequences=True, go_backwards=True)(hidden_inp)
    hidden2 = Flatten()(hidden2)

    inp2 = Input((1,))
    hidden = Concatenate()([hidden, hidden2, inp2])

    hidden = LeakyReLU(0.1)(Dense(gl.PARAMS["N_DENSE2"], activation="linear")(hidden))
    out = Dense(MODEL_PARAMS["N_ACTIONS"], activation="softmax", activity_regularizer=l2(0.001))(hidden)

    actor = Model([inp,inp2], out)
    #modify_hyper(actor)  # modify hyperparameters
    actor.compile(loss=maximization, optimizer=Adam(0.0005))

    # Build the critic
    inp = Input(inp_shape)
    hidden = LeakyReLU(0.1)(TimeDistributed(Dense(gl.PARAMS["N_DENSE"], activation="linear"))(inp))
    hidden = Bidirectional(LSTM(2*gl.PARAMS["N_LSTM"]))(hidden)

    inp2 = Input((1,))
    hidden = Concatenate()([hidden, inp2])
    hidden = LeakyReLU(0.1)(Dense(gl.PARAMS["N_DENSE2"], activation="linear")(hidden))
    out = Dense(1, activation="linear")(hidden)

    critic = Model([inp,inp2], out)
    critic.compile(loss="MSE", optimizer=Adam(0.0001))

    return actor, critic


### This is some code to modify weights of the model

# def create_vector():
    # x = []
    # for i in range(90):
        # x.append((10 - i%10)**2)
    # return x
    #
# def create_weights():
    # l = []
    # for _ in range(64):
        # x = create_vector()
        # l.append(x)
    # return np.array(l)
    #
# def create_biases():
    # x = create_vector()
    # return np.array(x)
    #
# def modify_hyper(model):
    # for layer in model.layers:
        # if layer.name == "dense_3":
            # layer.set_weights([create_weights(), create_biases()])
            
            
### Static class to modify probabilities
class ModifyProbs():
    
    # The modification matrices have different shape because for just getting probability outputs the whole batch of 
    # molecules is evaluated at once while in loss function batches of size 32 are used.
    MatrixOutput = None     # 'normal' output
    MatrixObjective = None  # used in loss function
    
    # different functions to modify probabilities
    
    @staticmethod
    def unity(x):
        return 1
    
    @staticmethod
    def square(x):
        return x**2
    
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def exponential(x):
        return math.exp(x)
    
    # choose one of the above functions to modify probabilities
    @staticmethod
    def choose_function(s):
        return getattr(ModifyProbs, s)

    # create matrix by which the probabilities should be modified              
    @staticmethod   
    def create_matrix(num_rows):
        x = []
        for _ in range(num_rows):
            row = []
            for j in range(MODEL_PARAMS["N_ACTIONS"]):
                row.append(ModifyProbs.choose_function(gl.PARAMS["MODIFY_PROBS"])(MODEL_PARAMS["MUTABLE_BITS"] - j % MODEL_PARAMS["MUTABLE_BITS"]))
            x.append(row)
        return np.array(x)
    
    # last step of model: modify probability output  
    @staticmethod        
    def modify_probs(probs, shape=None, tensor=False):
        if tensor:  # in loss function
            if ModifyProbs.MatrixObjective is None:
                ModifyProbs.MatrixObjective = ModifyProbs.create_matrix(shape)
            p = tf.multiply(probs, ModifyProbs.MatrixObjective)
            d = tf.math.reduce_sum(p, axis=1, keepdims=True)
            return p / d
        else:  # for getting outputs
            if ModifyProbs.MatrixOutput is None:
                ModifyProbs.MatrixOutput = ModifyProbs.create_matrix(probs.shape[0])
            p = np.multiply(probs, ModifyProbs.MatrixOutput)
            d = p.sum(axis=1, keepdims=1)
            return p / d


# Objective to optimize
def maximization(y_true, y_pred):
    #return backend.mean(-backend.log(y_pred) * y_true)
    return backend.mean(-backend.log(ModifyProbs.modify_probs(y_pred, 32, True)) * y_true)  # 32 is default batch-size of keras.fit

# read model from file
def read_model(filename):
    return models.load_model(filename, custom_objects={"maximization": maximization})
