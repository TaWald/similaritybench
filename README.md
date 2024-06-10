# `ReSi`: A Comprehensive Benchmark for Representational Similarity Measures
In the following, we describe how the experiments from our Benchmark Paper can be reproduced, and how additional measures could be added.


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

### 1.2 Set up Experiment Path


### 1.3 Loading Result Files

The results from all our experiments are stored in a `results.parquet` file, which is located in the èxperiments/results/` directory. Based on this, the plots can be reproduced, and if you want to implement a text a new measure, you can simply add the corresponding results to this file.


### 1.4 Loading Datasets and Models

If you want to rerun our experiments from scratch, or test a measure that you have implemented, it is required that the necessary models and datasets have been downloaded.
Regarding the dataset, for the 
* NLP domain, you need to load x into y
* Vision domain, you need to...
* Graph domain, we chose datasts that are already included in either the `pythorch geometric`or `ogb` packages. Upon extracting representations for the first time, these datasts will be downloaded automatically into the `EXPERIMENT_PATH/datasets/graphs/`subdirectory.

Regarding the models, you need to download the model files from --ZENODO-LINK(S)-- and unpack the zipped files into corresponding subdirectories of ´EXPERIMENT_PATH/models`.


## 2. Running the Benchmark

The main way to (re)run experiments from your benchmark is to set up a `config.yaml` file, and then simply run
```bash
    python3 -m repsim.run -c path/to/config.yaml
```
In the `configs/` subdirectory, you can find all the config files necessary to reproduce our experiments. 

If you want only want to run experiments on specific tests or domains, we provide detailed instructions on how to write these config files below.

### 2.1 Instructions on Config files

TODO

### 2.2 About Parallelization and Overwriting of Result Files

If you want to run multiple experiments in parallel, it is crucial that these **NEVER** write/work on the same results parquet file, as specified by `raw_results_filename` in the configs, at the same time. Otherwise, the results files will be corrupted due to overwriting each other in a non-complementing way, and later on, the result files may be incomplete, or the evaluation may even crash.
It is, however, no issue to write on an already existing parquet file with a single process - this will simply append the new results.

Regarding the CSVs of (aggregated) results, which are specied in configs under `table_creation` -> `filename` and `full_df_filename`, it is crucial to consider that if the name of an existing file is provided, this existing file will always be overwritten.
**NOTE:** The given config files in the `configs` directory were designed such that no such overwriting can occur, and thus these can safely be run in parallel.

### 2.3 Running Tests in Graph Domain without Specifying Configs

For the graph domain, another option to (re)run individual tests for all the graph models (GCN, GraphSAGE, GAT) on a given dataset is to run

```bash
    python3 -m repsim.run_graphs -t {test_name} -d {dataset} [-m {measures}]
```
Implicitly, this scripts creates a config file as described above, which is then used to run a test. The config files stored in the configs directory were also generated from this script.
Valid dataset names are `cora`, `flickr`, and `ogbn-arxiv`, valid test names are `label_test`, `shortcut_test`, `augmentation_test`, `layel_test`, and `output_correlation_test`, where the latter runs Tests 1 and 2 from our benchmark simultaneously. 
The argument for measures is optional, and by default, all measures that are registered under `ALL_MEASURES` in the `repsim.measures` module will be used. 
In this case, results will be saved into files called `graphs_{test_name}_{dataset}.parquet`, `graphs_{test_name}_{dataset}.csv` (`filename`), and `graphs_{test_name}_{dataset}_full.csv` (`full_df_filename`).
When specific measures that should be used are specified, the corresponding measure names will be appended to the result file names to avoid problems with files overwriting each other (cf. Section 2.3 above).
The name of the generated config file will follow the same pattern.


### 2.4 Merging Result Files

To merge all the parquet files you have produced into a single file, you can apply the script XYZ.


### 2.5  Plotting Results
To plot the results, we however utilite the csv file


## 3 Adding a New Measure

If you want to use our benchmark on a measure that has not been implemented yet, you can easily add your measure to the benchmark with the following steps:

#### 1. Set up a Script
Add a python script `your_measure.py` to the `repsim.measures` module, in which your similarity measure will be implemented.

#### 2. Implement the similarity function:
In your script, you need to implement a your similariy measure in a function of the following signature
```
def your_measure(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
```
where the shape parameter of type `SHAPE_TYPE = Literal["nd", "ntd", "nchw"]` defines input format of the given representations: `"nd"` represents input matrices in the $n \times d$ format, the other types corresponding higher-dimensional input formats, as common in the vision domain. Your measure should be able to process shapes of all these types. If higher-dimension inputs should simply be flattened to the `"nd"` format, you can use the `flatten` function that we provide in `repsim.measures.utils`. We further provide additional functions for preprocessing/normalizing inputs in this module.

#### 3. Wrap your Function into a class that inherits from `RepresentationalSimilarityMeasure`:

To properly fit into our framework, it is crucial that you implement such a class for your measure, such that, for instance, the semantics of your measure, i.e., whether a higher value indicates more similarity, can be handled properly in our benchmark.
The `RepresentationalSimilarityMeasure` class, as well as its `BaseSimilarityMeasure` parent class, are implemented in and can be imported from `repsim.benchmark.utils`. To wrap your function into such a class, using the following template should be sufficient:
```
class YourMeasure(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=your_measure,
            larger_is_more_similar=False             # Fill in True iff for your measure, higher values indicate more similarity  
            is_metric=True,                          # Fill in True iff your measure satisfies the properties of a distance metric. 
            is_symmetric=True,                       # Fill in True iff your measure is symmetric, i.e., m(R, Rp) = m(Rp,R)
            invariant_to_affine=True,                # Fill in True iff your measure is invariant to affine transformations
            invariant_to_invertible_linear=True,     # Fill in True iff your measure is invariant to invertible linear transformations
            invariant_to_ortho=True,                 # Fill in True iff your measure is invariant to orthogonal transformations
            invariant_to_permutation=True,           # Fill in True iff your measure is invariant to permutations
            invariant_to_isotropic_scaling=True,     # Fill in True iff your measure is invariant to isotropic scaling
            invariant_to_translation=True,           # Fill in True iff your measure is invariant to translations
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:

        # here you can, in priciple, conduct some preprocessing already, such as aligning spatial dimensions for vision inputs
        
        return self.sim_func(R, Rp, shape)
```

#### 4. Register your measure in the module

Open `repsim.benchmark.__init__.py`, import `YourMeasure` class, and append it to the `CLASSES` list that is defined in this script - this will also automatically append it to `ALL_MEASURES`, which is the list of measures considered in our benchmark. Thus, your measure is now registered in our benchmark, and can, for instance, be explicitly included in the `config.yaml` files via its class name.
