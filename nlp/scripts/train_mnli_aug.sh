#!/bin/bash

sessionNameTrain="train-mnli-aug"
tmux new -s $sessionNameTrain -d
tmux send-keys -t $sessionNameTrain "source .venv/bin/activate" Enter
tmux send-keys -t $sessionNameTrain "clear" Enter

# Train augmented models
for seed in {0..4}; do
    tmux send-keys -t $sessionNameTrain "python nlp/bert_finetune.py model.seed=$seed dataset=mnli_eda_025,mnli_eda_05,mnli_eda_075,mnli_eda_10 dataset.finetuning.trainer.args.seed=$seed -m" Enter
done
