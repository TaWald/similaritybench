# ReSi: A Comprehensive Benchmark for Representatinal Similarity Measures
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

To rerun experiments using the consig, run
```bash
    python3 -m repsim.run -c path/to/config.yaml
```

For the graph domain, another option to re-run individual tests on a given dataset is to run

```bash
    python3 -m repsim.run -t {test_name} -d {dataset} -m {measures}
```
where the latter flag for measures is optional, and by default, all measures will be used.

### 2.2 Instructions on Config files


### 2.3 Plotting Results

After all computations are done, plots can be produced via the `xyz.ipynb` notebook


## 3 Adding a new Measure


