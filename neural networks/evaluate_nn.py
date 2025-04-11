import sys
import os.path

# Before a slow keras import, check the the input
if len(sys.argv) < 2:
    print("Give me a NN, any kind.")
    sys.exit(0)
model_filename = sys.argv[1]
if not os.path.isfile(model_filename):
    print("404 File not found.")
    sys.exit(1)


import numpy as np
from numpy.random import rand, randint
from keras.models import load_model
from sklearn.metrics import confusion_matrix

model = load_model(model_filename)
model.summary()


# The size of the output and the activation function
# determine the type of the model  
shape = model.layers[-1].output_shape
activ = model.layers[-1].get_config()['activation']

if shape[1] == 2:
    # no mutations
    nmut = 0 
elif shape[1] == 3:
    if activ == 'linear':
        # one mutation
        nmut = 1
    elif activ == 'softmax':
        # direct classifier (finds number of mutations)
        nmut = -1
    else:
        print("Unexpected model output layer, terminating.")
        sys.exit(1)    
elif shape[1] == 4:
    # two mutations
    nmut = 2
else:
    print("Unexpected model output layer, quitting.")
    sys.exit(1)
if nmut >=0:
    print("\nRecognized NN for parameter estimation in a model with", nmut, "mutations.")
else:
    print("\nRecognized a NN for direct classification of the number of mutations.")


if nmut >= 0:
    # Evaluate estimation of parameters
    for replic in [1, 5, 10, 500]:
        rep = str(replic)
        par_size = shape[1]
        dataset_test = np.loadtxt(
            'datasets/gendata_'+ str(nmut) +'mut_test_pred_aucune_rep='+ rep +'.txt', delimiter=" ")
        # 25 stats, the last parameter (difficulty) is discarded                              
        X_test = dataset_test[:, par_size:par_size+25] 
        y_test = dataset_test[:, 0:par_size]
        # Normalisation if ever needed
        # X_test = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0))
        score = model.evaluate(X_test, y_test, verbose=0)
        print('\nEvaluation on the set created with', rep, 'replications' )
        print("Metric", model.metrics_names[1], score[1])
else:
    # Evaluate model classification
    # rep is the number of replications in the generator
    # (controls the level of noise in the test datasets)
    for replic in [1, 5, 10, 500]:
        rep = str(replic)
        dataset0_test = np.loadtxt('datasets/gendata_0mut_test_pred_aucune_rep='+rep+'.txt',
                                delimiter=" ")
        dataset1_test = np.loadtxt('datasets/gendata_1mut_test_pred_aucune_rep='+rep+'.txt',
                                delimiter=" ")
        dataset2_test = np.loadtxt('datasets/gendata_2mut_test_pred_aucune_rep='+rep+'.txt',
                                delimiter=" ")
        X_test = np.vstack((dataset0_test[:, 2:27], dataset1_test[:, 3:28], dataset2_test[:, 4:29]))
        n_per_model = X_test.shape[0] // 3
        y_test = np.zeros((n_per_model*3, 3), dtype=np.float32)
        y_test[n_per_model:2*n_per_model, 1] = 1
        y_test[2*n_per_model:, 2] = 1
        print("\nEvaluation of the classifier on the test set with", rep, "replics")
        score = model.evaluate(X_test, y_test, verbose=0)
        print("\n---> Metric", model.metrics_names[1], score[1])

        # For confusion matrix
        pred_onehot = model.predict(X_test)
        pred_int = pred_onehot.argmax(axis=1)
        y_test_int = y_test.argmax(axis=1)

        # print(pred_int.shape)
        # print(y_test_int.shape)
        cmat = confusion_matrix(y_test_int, pred_int)
        np.set_printoptions(suppress=True, precision=5, floatmode='unique')
        print(cmat)
        print("Pourcentages:")
        print(np.round(cmat / 5000*100, 3))