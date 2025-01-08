sessionName="iclr-rtd"
tmux kill-session -t $sessionName
tmux new -s $sessionName -d
tmux send-keys -t $sessionName "source .venv/bin/activate" Enter
tmux send-keys -t $sessionName "clear" Enter

device=6

tmux send-keys -t $sessionName "echo \"Starting aug_mnli_albert_cls at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/aug_mnli_albert_cls.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting aug_mnli_albert_mean at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/aug_mnli_albert_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting aug_mnli_bert_mean at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/aug_mnli_bert_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting aug_mnli_3 at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/aug_mnli_3.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting aug_sst2_2_mean at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/aug_sst2_2_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting aug_sst2_2 at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/aug_sst2_2.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting correlation_nlp_standard_mnli_mean at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/correlation_nlp_standard_mnli_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting correlation_nlp_standard_mnli at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/correlation_nlp_standard_mnli.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting correlation_nlp_standard_sst2_albert at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/correlation_nlp_standard_sst2_albert.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting correlation_nlp_standard_sst2_mean at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/correlation_nlp_standard_sst2_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting correlation_nlp_standard_sst2 at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/correlation_nlp_standard_sst2.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting mem_mnli_3_mean at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/mem_mnli_3_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting mem_mnli_3 at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/mem_mnli_3.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting mem_sst2_3_albert at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/mem_sst2_3_albert.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting mem_sst2_3_mean at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/mem_sst2_3_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting mem_sst2_3 at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/mem_sst2_3.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting monotonicity_nlp_standard_mnli_cls_tok_mean at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/monotonicity_nlp_standard_mnli_cls_tok_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting monotonicity_nlp_standard_mnli_cls_tok at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/monotonicity_nlp_standard_mnli_cls_tok.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting monotonicity_nlp_standard_sst2_cls_tok_albert at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/monotonicity_nlp_standard_sst2_cls_tok_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting monotonicity_nlp_standard_sst2_cls_tok at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/monotonicity_nlp_standard_sst2_cls_tok.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting sc_mnli_3_mean at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/sc_mnli_3_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting sc_mnli_3 at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/sc_mnli_3.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting sc_sst2_3_mean at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/sc_sst2_3_mean.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting sc_sst2_3 at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/sc_sst2_3.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting correlation_nlp_standard_sst2_smollm at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/correlation_nlp_standard_sst2_smollm.yaml" Enter

tmux send-keys -t $sessionName "echo \"Starting sc_sst2_smollm at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/sc_sst2_smollm.yaml" Enter
tmux send-keys -t $sessionName "echo \"Starting mem_sst2_smollm at $(date)\" >> experiments/run_log.txt && CUDA_VISIBLE_DEVICES=$device python repsim/run.py -c configs/language/mem_sst2_smollm.yaml" Enter

# copy result files to iclr results folder
tmux send-keys -t $sessionName "cp experiments/results/aug_mnli_albert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/aug_mnli_albert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/aug_mnli_bert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/aug_mnli_bert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/aug_sst2_bertAndAlbert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/aug_sst2_bertAndAlbert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/correlation_nlp_standard_mnli_bertAndAlbert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/correlation_nlp_standard_mnli_bertAndAlbert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/correlation_nlp_standard_sst2_albert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/correlation_nlp_standard_sst2_bertAndAlbert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/correlation_nlp_standard_sst2_bert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mem_mnli_bertAndAlbert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mem_mnli_bertAndAlbert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mem_sst2_albert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mem_sst2_bertAndAlbert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mem_sst2_bert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mono_nlp_standard_mnli_bertAndAlbert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mono_nlp_standard_mnli_bertAndAlbert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mono_nlp_standard_sst2_albert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mono_nlp_standard_sst2_bertAndAlbert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mono_nlp_standard_sst2_bert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/sc_mnli_bertAndAlbert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/sc_mnli_bertAndAlbert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/sc_sst2_bertAndAlbert_mean.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/sc_sst2_bertAndAlbert_cls.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/sc_sst2_smollm_full.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/correlation_nlp_standard_sst2_smollm.csv experiments/paper_results/nlp_iclr" Enter
tmux send-keys -t $sessionName "cp experiments/results/mem_sst2_smollm_full.csv experiments/paper_results/nlp_iclr" Enter
