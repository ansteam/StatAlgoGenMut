# Neural networks

Scripts and data for the neural network part of the article. The scripts allow to


- perform estimation of parameters for our algorithms for three models (0, 1, and 2 mutations)
  `estimat_param_model_0.py`
  `estimat_param_model_1.py`
  `estimat_param_model_2.py`

- perform direct detection of the most likely model given the pedigree:
  `classify_nmut.py`

- evaluate obtained models:
  `evaluate_nn.py`

The subfolder `datasets` contain some simulated data used to fit the neural networks
and evaluate obtained models.

All scripts were written using Keras API version 2.


Usage

1. Parameter estimation. 
In a terminal type

python estimat_param_model_0.py 
or
python estimat_param_model_1.py 
or
python estimat_param_model_2.py 
according to the model you wish to use.

The script saves the best neural model that estimates the parameters in a file
`model0_est_param.h5`
`model1_est_param.h5`
`model2_est_param.h5`
respectively.


2. Direct detection of the type of model:

python classify_nmut.py

The resulting model is saved as `classifier_nn.h5`

3. Model evaluation.
To evaluate saved models, use

python evaluate_nn.py filename

where filename is on of the saved neural network files.
The script automatically infers the type model from it architecture and performs 
evaluation on a corresponding test set.





