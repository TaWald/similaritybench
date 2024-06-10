# ReSi: A Comprehensive Benchmark for Representational Similarity Measures
In the following, we describe how the experiments from our Benchmark Paper can be reprduced, and how additional measures could be added.


## 1. Setting up the Repository

### 1.1 Virtual environment
Skip the conda rows if you already have the correct Python version installed.
```shell
conda create -n ENVNAME python=3.10
conda active ENVNAME
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt .
```


### 1.2 Loading Result Files

The results from all our experiments are stored in a `results.parquet` file, which is located in the èxperiments/results/` directory. Based on this, the plots can be reproduced, and if you want to implement a text a new measure, you can simply add the corresponding results to this file.


### 1.3 Loading Datasets and Models

If you want to rerun our experiments from scratch, or test a measure that you have implemented, it is required that the necessary models and datasets have been downloaded.
Regarding the dataset, for the 
* NLP domain, you need to load x into y
* Vision domain, you need to...
* Graph domain, we chose datasts that are already included in either the `pythorch geometric`or `ogb` packages. Upon extracting representations for the first time, these datasts will be downloaded automatically into the `experiments/datasets/graphs/`subdirectory.

Regarding the models, you need to download the model files from --ZENODO-LINK(S)-- and unpack the zipped files into corresponding subdirectories of ´experiments/models`.


## 2. Running the Benchmark

The main way to (re)run experiments from your benchmark is to set up a `config.yaml` file, and then simply run
```bash
    python3 -m repsim.run -c path/to/config.yaml
```
In the `configs/` subdirectory, you can find all the config files necessary to reproduce our experiments. 

If you want only want to run experiments on specific tests or domains, we provide a detailed descriptions on the config files below.

### 2.1 Instructions on Config files

TODO

### 2.2 About Parallelization

If you want to run multiple experiments in parallel, it is crucial, that these never write/work on the same results parquet file, as specified by `raw_results_filename` in the configs. Otherwise, the results files will be corrupted due to overwriting each other in a non-complementing way, and later on, the result files may be incomplete, or the evaluation may even crash.

### 2.3 Running Tests in Graph Domain without Configs

For the graph domain, another option to (re)run individual tests on a given dataset is to run

```bash
    python3 -m repsim.run -t {test_name} -d {dataset} -m {measures}
```
where the latter flag for measures is optional, and by default, all measures will be used.


### 2.4 Merging Result Files and Plotting Results

After all computations are done, plots can be produced via the `xyz.ipynb` notebook


## 3 Adding a New Measure

If you want to use our benchmark on a measure that has not been implemented yet, you can easily add your measure to the benchmark with the following steps:
