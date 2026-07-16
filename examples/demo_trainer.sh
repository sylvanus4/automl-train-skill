#!/usr/bin/env bash
# Language-agnostic backend proof: a trainer that is NOT Python.
# Reads trial params from env, "trains", prints a metric the loop scrapes.
# eval_loss is a bowl with its optimum at LEARNING_RATE=0.30 (so LR genuinely matters).
lr="${LEARNING_RATE:-0.1}"
steps="${MAX_STEPS:-50}"
# eval_loss = (lr-0.3)^2 + 0.1/ (steps/100)   -- more steps helps a bit
loss=$(awk -v lr="$lr" -v st="$steps" 'BEGIN{ d=lr-0.30; printf "%.6f", d*d + 0.1/(st/100.0+1) }')
echo "{'eval_loss': '$loss', 'lr': '$lr', 'steps': $steps}"
echo "{\"eval_loss\": $loss}"
