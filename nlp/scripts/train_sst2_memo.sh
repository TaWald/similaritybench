#!/bin/bash
sessionNameTrain="train-sst2-memo"

tmux kill-session -t $sessionNameTrain

tmux new -s $sessionNameTrain -d
tmux send-keys -t $sessionNameTrain "source .venv/bin/activate" Enter
tmux send-keys -t $sessionNameTrain "clear" Enter

# Train shortcut models
for seed in {0..9}; do
    # Not using the 0.25 steps, because of the class distribution. Instead use 5 steps between naive guessing accuracy and 1.0.
    # tmux send-keys -t $sessionNameTrain "python nlp/bert_finetune.py model.seed=$seed dataset=sst2 dataset.finetuning.trainer.args.seed=$seed memorization_rate=1.0,0.75,0.5,0.25 -m" Enter

    # Leaving out 1.0, because it's the same as before.
    tmux send-keys -t $sessionNameTrain "python nlp/bert_finetune.py model.seed=$seed dataset=sst2 dataset.finetuning.trainer.args.seed=$seed memorization_rate=0.889,0.779,0.668,0.558 -m" Enter
done
