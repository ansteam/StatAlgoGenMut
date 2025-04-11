#%%
import sys
from time import time
import numpy as np
from numpy.random import rand, randint
import random as rn
import tensorflow as tf  # just to use one thread (no parallelism to be reproducile)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.models import load_model
from keras.losses import mean_absolute_percentage_error as mape
from keras.callbacks import Callback
from keras import backend as K

class PrintDot(Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0 and epoch > 0: print()
        print('.', end='')
        sys.stdout.flush()

#%% Training data
# dataset = np.loadtxt('datasets/gendata_0mut_train_pred_aucune_rep=500.txt', delimiter=" ")
dataset = np.loadtxt('datasets/gendata_0mut_train_all_rep=500.txt', delimiter=" ")

X = dataset[:, 2:27]        # take 25 stats (the last one, ie. the difficulty, is discarded)
Y = dataset[:, 0:2]         # 2 params for this model
# Normalization by column: not better,
# achieves map error about 1.9 % (and easyli below 1.5 without any scaling).
# norm_factor = X.max(axis=0) - X.min(axis=0)
# There are two columns filled with zeros (effectively unused)
# Need to either correct the normalizing factor (to non nul)
# or to entirely drop the whole columns (seems better but let's try easy correction first)
# norm_mask = norm_factor == 0
# norm_factor[norm_mask] = 1
# print("norm factor", (X.max(axis=0) - X.min(axis=0)) > 0)
# X = (X - X.min(axis=0)) / norm_factor

# Normalization by a scalar max (global)
# X = (X) / (X).max()
# Affine normalization is a general disaster
# norm_factor = X.max() - X.min()
# X = (X - X.min()) / norm_factor
# first attempt bizarre
# norm_factor = np.abs(X).max()
# X = (X - X.min(axis=0)) / norm_factor

#%% Test / validation data

# dataset_test = np.loadtxt('datasets/gendata_0mut_test_pred_aucune_rep=500.txt', delimiter=" ")
dataset_test = np.loadtxt('datasets/gendata_0mut_test_all_rep=500.txt', delimiter=" ")


X_test = dataset_test[:, 2:27]
y_test = dataset_test[:, 0:2]
# Normalization by column
# norm_factor = X_test.max(axis=0) - X_test.min(axis=0)
# norm_mask = norm_factor == 0
# norm_factor[norm_mask] = 1
# X_test = (X_test - X_test.min(axis=0)) / norm_factor

# Normalization by a scalar max (global)
# X_test = (X_test) / (X_test).max()
# Affine normalization is a disaster
# norm_factor = X_test.max() - X_test.min()
# X_test = (X_test - X_test.min()) / norm_factor
# first attemt bizarre
# norm_factor = np.abs(X_test).max()
# X_test = (X_test - X_test.min()) / norm_factor

# Reproducibility
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                               inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

#%% Learning 
n_models = 50
best_score = 1000
for i in range(n_models):
    # Reproducibility
    # np.random.seed(31*i)
    # rn.seed(31*i)
    # tf.set_random_seed(31*i)

    model = Sequential()
    model.add(Dense(128, input_dim=25, kernel_initializer='uniform', activation='relu'))
    # model.add(Dense(64, input_dim=28, kernel_initializer='uniform', activation='relu'))
    # model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(2, kernel_initializer='uniform', activation='linear'))
    model.compile(loss='mape', optimizer='adam', metrics=['mape'])
    
    print("-" * 80,"\nIteration", i," fitting, please wait...")
    tic = time()
    # Versions more verbose for terminal
    # model.fit(X, Y, epochs=300, batch_size=500, verbose=2)
    # model.fit(X, Y, epochs=300, batch_size=10, verbose=1, callbacks=[PrintDot()])
    # Version serveur
    model.fit(X, Y, epochs=300, batch_size=10, verbose=0)
    toc = time()    
    print("\nTime elapsed =", (toc-tic)//60, "min", np.round((toc-tic)%60), "sec.", flush=True)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Metric", model.metrics_names[1], score[1])
    if score[1] < best_score:
        print("*** Best model so far, saving")
        best_score = score[1]
        model.save('model0_est_param.h5')
print("Best score:", best_score)


