## Similarity Bench (Vision)

This repository contains code to train and compare representations of vision models.

Currently this is separated into a multi-stage process:
1. Training of the models (on some datasets)
2. Re-Loading the models with subsequent comparison of 
   1. Representations
   2. Output Behavior

-----
### Code structure
The codebase is split into 
1. `data` containing all datamodules that are used to train and compare models
2. `arch` architectures (including some locations where to extract representations) -- This is probably not necessary to do as is with some better logic where to extract representations
3. `comp` Some comparison metrics
4. `metrics` Some output similarity metrics (cohens kappa, jensen shannon divergence, etc.)
5. `similarity_benchmark` Logic to load models, extract representations and compare them
6. `util` Utilities to load and read stuff


