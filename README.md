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

parquets are stored in dir xyz, bla

### 1.3 Loading Datasets

For NLP and Vision, you need to load x into y.

For the graph domain, we chose datasts that are already included in either the `pythorch geometric`or `ogb` packages. Upon extracting representations for the first time, these datasts will be downloaded automatically.


### 1.3 Loading Model Files

download from zenodo here and store in subdirectory xyz


## 2. Running the Benchmark

To rerun experiments using the consifg, run
```bash
    python3 -m repsim.run -c path/to/config.yaml
```

For the graph domain, another option to re-run individual tests on a given dataset is to run

```bash
    python3 -m repsim.run -t {test_name} -d {dataset} -m {measures}
```
where the latter flag for measures is optional, and by default, all measures will be used.

### 2.2 Instructions on config files


### Plotting Results

After all computations are done, plots can be produced via the `xyz.ipynb` notebook
