#!/bin/bash

python train.py /home/gb/dlc/python/DLMulMix-main/data-bin \
		--arch transformer_iwslt_de_en \
		--share-decoder-input-output-embed \
		--clip-norm 0 \
		--optimizer adam \
		--reset-optimizer \
		--lr 0.0066 \
		--source-lang en \
		--target-lang de \
		--max-tokens 4096 \
		--no-progress-bar \
		--log-interval 100 \
		--min-lr 1e-09 \
		--weight-decay 0.1 \
		--criterion label_smoothed_cross_entropy \
		--label-smoothing 0.20 \
		--lr-scheduler inverse_sqrt \
		--max-update 8000 \
		--warmup-updates 2000 \
		--warmup-init-lr 1e-07 \
		--update-freq 4 \
		--adam-betas 0.9,0.98 \
		--keep-last-epochs 40 \
		--dropout 0.30 \
		--tensorboard-logdir /home/gb/dlc/python/DLMulMix-mainyuanban/results2/en-fr-bpe/bl_log1 \
		--log-format simple \
		--save-dir /home/gb/dlc/python/DLMulMix-mainyuanban/results2/premix10/mmtimg10 \
		--eval-bleu \
		--eval-bleu-remove-bpe \
		--patience 15 \
		--fp16 \
		--Threshold 0.28 \

# 		--warmup-init-lr 1e-07 \  or 5e-03


