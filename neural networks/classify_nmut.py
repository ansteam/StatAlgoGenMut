#%%
import sys
from time import time
import numpy as np
from numpy.random import rand, randint

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.models import load_model
from keras.utils import np_utils
from keras.losses import mean_absolute_percentage_error as mape
from keras.callbacks import Callback, EarlyStopping

class PrintDot(Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0 and epoch > 0: print()
        print('.', end='')
        sys.stdout.flush()

# Training data
# Données générées avec les arbres du fichier "Predisposition Aucune"
# dataset0 = np.loadtxt('datasets/gendata_0mut_train_pred_aucune_rep=500.txt', delimiter=" ")
# dataset1 = np.loadtxt('datasets/gendata_1mut_train_pred_aucune_rep=500.txt', delimiter=" ")
# dataset2 = np.loadtxt('datasets/gendata_2mut_train_pred_aucune_rep=500.txt', delimiter=" ")
# Données générées avec tous les arbres (All_SO)
dataset0 = np.loadtxt('datasets/gendata_0mut_train_all_rep=500.txt', delimiter=" ")
dataset1 = np.loadtxt('datasets/gendata_1mut_train_all_rep=500.txt', delimiter=" ")
dataset2 = np.loadtxt('datasets/gendata_2mut_train_all_rep=500.txt', delimiter=" ")

# Extracting 25 stats from each dataset
X = np.zeros((75000, 25), dtype=np.float32)
X[0:25000, :] = dataset0[:, 2:27]
X[25000:50000, :] = dataset1[:, 3:28]
X[50000:75000, :] = dataset2[:, 4:29]

# Normalization by column does not work for this task,
# it gives random-level accuracy of 0.34!!
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
#X = X / np.abs(X).max()
# Affine normalization is a disaster!
# norm_factor = X.max() - X.min()
# X = (X - X.min()) / norm_factor
# First attempt bizarre but 100% accurate
# norm_factor = np.abs(X).max()
# X = (X - X.min(axis=0)) / norm_factor


# Categorical (one-hot) encoding
Y = np.zeros((75000, 3), dtype=np.float32)
Y[0:25000, 0] = 1
Y[25000:50000, 1] = 1
Y[50000:75000, 2] = 1


#%% Test / validation data
# Test correspondant aux arbres "Aucune prédisposition"
# dataset0_test = np.loadtxt('datasets/gendata_0mut_test_pred_aucune_rep=500.txt', delimiter=" ")
# dataset1_test = np.loadtxt('datasets/gendata_1mut_test_pred_aucune_rep=500.txt', delimiter=" ")
# dataset2_test = np.loadtxt('datasets/gendata_2mut_test_pred_aucune_rep=500.txt', delimiter=" ")
# Test pour tous les arbres
dataset0_test = np.loadtxt('datasets/gendata_0mut_test_all_rep=500.txt', delimiter=" ")
dataset1_test = np.loadtxt('datasets/gendata_1mut_test_all_rep=500.txt', delimiter=" ")
dataset2_test = np.loadtxt('datasets/gendata_2mut_test_all_rep=500.txt', delimiter=" ")


X_test = np.vstack((dataset0_test[:, 2:27], dataset1_test[:, 3:28], dataset2_test[:, 4:29]))

# Normalization by a scalar max (global)
# X_test = (X_test) / (X_test).max()
# Affine normalization is a disaster!
# norm_factor = np.abs(X).max()
# X = X / norm_factor
# First attempt bizarre but 100% accurate (!?)
# norm_factor = np.abs(X_test).max()
# X_test = (X_test - X_test.min(axis=0)) / norm_factor
# Affine normalisation is a disaster !
# norm_factor = X_test.max() - X_test.min()
# X_test = (X_test - X_test.min()) / norm_factor



n_per_model = X_test.shape[0] // 3
y_test = np.zeros((n_per_model*3, 3), dtype=np.float32)
y_test[n_per_model:2*n_per_model, 1] = 1
y_test[2*n_per_model:, 2] = 1

#%% Learning
best_score = 0
for i in range(25):
    model = Sequential()
    model.add(Dense(32, input_dim=25, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(3, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("-" * 80,"\nIteration", i," fitting, please wait...")
    tic = time()
    # model.fit(X, Y, epochs=200, batch_size=10, verbose=0, callbacks=[PrintDot()])
    model.fit(X, Y, epochs=200, batch_size=10, verbose=0, 
            callbacks=[PrintDot(), EarlyStopping(monitor='acc', patience=15, verbose=1)])
    # model.fit(X, Y, epochs=150, batch_size=1000, verbose=1, callbacks=[EarlyStopping(monitor='loss', patience=10, verbose=1)])
    toc = time()    
    print("\nIteration ", i, "Time elapsed =", (toc-tic)//60, "min", np.round((toc-tic)%60), "sec.")

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Metric", model.metrics_names[1], score[1])
    if score[1] > best_score:
        print("*** Best model so far, saving")
        best_score = score[1]
        model.save('classifier_nn.h5')
print("Best score:", best_score)


#%%
