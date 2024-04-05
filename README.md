# similaritybench
Representational Similarity Benchmark


### ToDo's
- [ ] (Tassilo) Investigate why Difference metrics do not show proper behaviour.
- [ ] (All) Run the evaluation to get proper results (Shortcuts; Memorization; Monotonicity; Augmentation)
  - [ ] Vision: Add a ViT model for Datasets (all)
- [ ] (Max) Merge TrainedModel with ModelRepresentation

### Overall ToDO's
- [ ] Add functional measures for connection between reps and functional similarity
- [ ] Think about the grouping some more. Does it make more sense to move to "Shortcut Affinity" (Acc with 100 vs 0 SC?) --> Allows for continuous measures

- Proxy for:
  - Stitching Accuracy -- Fine-tuning of linear layer
  - Predictive Similarity (Jensen-Shannon) -- No Finetung (but Linear Probe)
  - Cohens Kappa  - Linear Probe
  - Error Similarity (at that layer) - Linear Probe
  - Pruning (of layers with least change)
  - Rel. ensemble accuracy. (No training)
  - Vision Language Alignment? (How would one test this?)
  - "Foolability" of a metric -- High dissimilarity despite same predictions
    - (Would require differentiability and training)

Linear Probes only necessary if not used at the very last layer
## Structure

- Library (`repsim`)
  - utils
  - Ähnlichkeitsmaße (`measures`)
    - utils für Maße wie padding und preprocessing
- Tests der Maße
  - Invarianzen korrekt
  - Richtige scores für gleiche bzw unterschiedliche inputs
- Benchmark
  - Evaluierungsskripts der einzelnen Experimente
- Vision
- NLP
- Graph
- utils: quasi config, wo wird gespeichert, laden von Ergebnissen/speichern, ...


## Virtual environment
Skip the conda rows if you already have the correct Python version installed.
```shell
conda create -n ENVNAME python=3.10
conda active ENVNAME
python -m venv .venv
source .venv/bin/activate
pip install torch=2.0.1 torchvision torchaudio hydra-core ipykernel flake8 black huggingface-hub ipywidgets matplotlib seaborn numpy scipy pyarrow tokenizers datasets transformers pytest Cython gudhi scikit-learn evaluate accelerate pre-commit
pip install -e .
```

(Max: Die GPU Treiber sind veraltet bei uns, weswegen keine höhere pytorch Version funktioniert.)

Additional dependencies for some measures:
```shell
git clone https://github.com/KhrulkovV/geometry-score.git && cd geometry-score && pip install .
git clone https://github.com/xgfs/imd.git && cd imd && pip install .
```
