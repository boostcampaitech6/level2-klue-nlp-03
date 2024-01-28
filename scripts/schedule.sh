#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=rbert logger=wandb

python src/train.py experiment=rbert1 logger=wandb

python src/train.py experiment=rbert2 logger=wandb
