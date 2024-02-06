# similaritybench
Representational Similarity Benchmark


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
pip install torch=2.0.1 torchvision torchaudio hydra-core ipykernel flake8 black huggingface-hub ipywidgets matplotlib seaborn numpy scipy pyarrow tokenizers datasets transformers pytest Cython gudhi
```

(Max: Die GPU Treiber sind veraltet bei uns, weswegen keine höhere pytorch Version funktioniert.)

Additional dependencies for some measures:
```shell
git clone https://github.com/KhrulkovV/geometry-score.git && cd geometry-score && pip install .
git clone https://github.com/xgfs/imd.git && cd imd && pip install .
```
