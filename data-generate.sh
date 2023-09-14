#!/bin/bash

python3 generate.py /home/gb/dlc/python/DLMulMix-main/data-bin/test2016 \
				--path /home/gb/dlc/python/DLMulMix-mainyuanban/results2/premix10/mmtimg10/model-1.pt \
				--source-lang en --target-lang de \
				--beam 5 \
				--num-workers 12 \
				--batch-size 128 \
				--results-path /home/gb/dlc/python/DLMulMix-mainyuanban/results2/premix10/mmtimg10/test2016 \
				--remove-bpe \
				--Threshold 0.28  
