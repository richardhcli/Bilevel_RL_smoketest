#!/bin/bash

# --- SETUP ---
module purge
module load anaconda cuda
conda activate mrn

echo "[Smoketest] Purging old results..."
rm -rf exp/LunarLander-v3
rm -rf myresults/lunarlander_smoketest
mkdir -p myresults/lunarlander_smoketest

# --- FAST TRAINING ---
echo "[Smoketest] Running PEBBLE (300 steps total)..."
python train_PEBBLE.py env=LunarLander-v3 num_seed_steps=100 num_unsup_steps=100 num_train_steps=300 eval_frequency=150 num_eval_episodes=1 > myresults/lunarlander_smoketest/pebble.log 2>&1 &
PID_PEBBLE=$!

echo "[Smoketest] Running OURS (300 steps total)..."
python train_V.py env=LunarLander-v3 num_seed_steps=100 num_unsup_steps=100 num_train_steps=300 eval_frequency=150 num_eval_episodes=1 > myresults/lunarlander_smoketest/v_guided.log 2>&1 &
PID_V=$!

# --- WAIT FOR COMPLETION ---
wait $PID_PEBBLE
wait $PID_V

echo "[Smoketest] Training complete. Checking for eval data..."

# --- FAST PLOTTING ---
# We use sed to temporarily point your plotting script to the smoketest directory
sed -i 's|"myresults/lunarlander"|"myresults/lunarlander_smoketest"|g' plot_lunarlander.py
python plot_lunarlander.py
sed -i 's|"myresults/lunarlander_smoketest"|"myresults/lunarlander"|g' plot_lunarlander.py

echo "[Smoketest] Pipeline Complete!"