## Representation Transfer

This repo contains the codebase that was used to train, compare and replace representations, as described in the paper. There consist multiple main files that can be run and all
have a parser containing the different arguments one can pass.

Later main functions depend on previous ones beeing run before (as one can't compare representations when no model is trained).

Initially one has to run train_model.py, ideally 10 times with different splits. After this one has to make sure the activations are extracted. This can take a lot of disk space,
so an iterative approach might be suitable.

Following this one can calculate the best prediction weights (on the test set) via linear regression by caling compare_models.py

Once the weights are calculated we use them to predict the representations and replace them with fuse_models.py

Moreover internal_noise_sensitivity calcualtes the noise, currently it needs the activation stats to be calculated
(Which happens in compare_models.py -- i believe).

-----

#### Training new models.

train_model.py --- Train one model of an arbitrary architecture.

All other functions depend on the already trained models!

#### Extract activations (if not already done in train_model.py

eval_trained_models.py --- Extracts activations [Should the training error after finishing, or not automatically save activations (arg: -na 1 passed to it)

#### Calculate similarity and Linear regression for prediction weights

compare_models.py --- Calculate the prediction weights of the linear regression between model of the same architecture

#### Replacement of representations through predictions from other model

fuse_models.py --- Replaces all quartiles of representations in the target model one after another

#### Noise evaluation

internal_noise_sensitivity_evaluation.py --- Adds the noise to the different trained models

-----

There are other functions that were investigated, which are not part of the submission present in the repo. We kindly ask you to ignore these.

#### Full Ensemble (all 10 models) calculation

eval_ensembles
