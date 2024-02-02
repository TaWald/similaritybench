# LLM Comparison

## Virtual environment
Had problems installing all the packages with `conda`.
Hence this solution:
```shell
conda create -n ENVNAME python
conda active ENVNAME
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio hydra-core ipykernel flake8 black huggingface-hub ipywidgets matplotlib seaborn numpy scipy pyarrow tokenizers datasets transformers s3fs pytest Cython gudhi
```
Additional dependencies for some measures:
```shell
git clone https://github.com/KhrulkovV/geometry-score.git && cd geometry-score && pip install .
git clone https://github.com/xgfs/imd.git && cd imd && pip install .
```
TODO: can probably run this as `pip install git+https://github.com/KhrulkovV/geometry-score.git`. Try out sometime

## Extract representations

```shell
python extract_representations.py storage.reps_subdir="/llmdata/representations/hf_pipeline" model=falcon-7B,galactica-7B,gptj-7B,llama2-7B,mpt-7B,openllama-7B,opt-7B,pythia-7B,redPajama-7B,stablelmAlpha-7B,bloom-7B dataset=winogrande,arc-easy,mathqa,piqa,webquestions -m
```

## Compare representations
```shell
python compare_representations.py
```


## Run lm-eval-harness
Make sure to login with huggingface cli or restricted models like llama2 cannot be downloaded.

### Compare predictions
```
python compare_predictions.py filter.must_contain_all=[] filter.must_not_contain=["gsm8k_yaml"] storage.results_subdir=results/7b-models_disagreement_v2 hydra.verbose=__main__
```
(gsm8k does not have predictions)
