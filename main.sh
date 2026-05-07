#!/bin/bash

#=======================================================================================================================
# 1. This script is designed to run two training processes (PEBBLE and V) in parallel, monitor their progress, and
#    send notifications about their status using ntfy.sh.
# 2. Automatically terminates training after MAX_TIME to ensure plotting and results aggregation complete.
#========================================================================================================================

# --- PROLOGUE SECTION ---
# WHAT: Initialize timing, environment variables, and the hard timeout limit.
# WHY: We need a reference point to calculate total runtime and ensure the HPC releases resources cleanly.
START_TIME=$(date +%s)
CONDA_ENV="mrn"  
NTFY_TOPIC="cvirl-aortopathy" 
RUN_NAME="LunarLander_Comparison"
MAX_TIME="4h" # The absolute maximum time the training loops are allowed to run.
KILL_GRACE="5m" # If Python freezes, send SIGKILL 5 mins after timeout.

echo "[prologue] Starting $RUN_NAME at $(date)"

# --- ENVIRONMENT SETUP ---
# WHAT: Load Gilbreth's modules and activate the environment.
# WHY: Aligns the execution script with the HPC's architecture.

module purge
module load anaconda cuda
conda activate $CONDA_ENV

# LOCAL MACHINE FALLBACK (Commented out)
# WHAT: Activate the specific Conda environment natively.
# WHY: Prevents the script from using the 'base' environment which lacks dependencies.
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate $CONDA_ENV

# --- DIRECTORY SETUP & CLEANUP ---
# WHAT: Ensure the single source of truth for results exists, and purge old Hydra experiments.
# WHY: Prevents old corrupted data from interfering with the new run and organizes outputs.
RESULTS_DIR="myresults/lunarlander"
mkdir -p $RESULTS_DIR
rm -rf exp/LunarLander-v3

# --- EXECUTION SECTION ---
# WHAT: Launch the training scripts wrapped in 'timeout' using unbuffered output.
# WHY: 'timeout' guarantees the scripts die after 4 hours. '-u' ensures logs hit the disk immediately.
echo "[status] Launching training processes with a $MAX_TIME limit..."
timeout -k $KILL_GRACE $MAX_TIME python -u train_PEBBLE.py env=LunarLander-v3 > $RESULTS_DIR/pebble.log 2>&1 &
PID_PEBBLE=$!

timeout -k $KILL_GRACE $MAX_TIME python -u train_V.py env=LunarLander-v3 > $RESULTS_DIR/v_guided.log 2>&1 &
PID_V=$!

# --- MONITORING ---
# WHAT: A watchdog loop that keeps the bash script alive while Python runs in the background.
while kill -0 $PID_PEBBLE 2>/dev/null || kill -0 $PID_V 2>/dev/null; do
    echo "--- Heartbeat $(date +%H:%M) ---"
    tail -n 1 $RESULTS_DIR/pebble.log
    tail -n 1 $RESULTS_DIR/v_guided.log
    sleep 300 # Check every 5 minutes
done

# --- ERROR & TIMEOUT DETECTION ---
# WHAT: Capture the exit codes. 0 = Natural Finish, 124 = Timeout Reached, Anything else = Crash.
# WHY: We must distinguish between the script being intentionally cut off at 4 hours vs. throwing a Python error.
wait $PID_PEBBLE; EXIT_P=$?
wait $PID_V; EXIT_V=$?

if [ $EXIT_P -eq 124 ] || [ $EXIT_V -eq 124 ]; then
    echo "[event] MAX TIME ($MAX_TIME) reached. Processes cleanly severed."
    STATUS_MSG="Time limit reached ($MAX_TIME). Plotting available data."
elif [ $EXIT_P -ne 0 ] || [ $EXIT_V -ne 0 ]; then
    echo "[event] CRASH DETECTED."
    STATUS_MSG="CRASH DETECTED: PEBBLE ($EXIT_P), V ($EXIT_V). Check logs immediately."
else
    echo "[event] Training completed naturally before the time limit."
    STATUS_MSG="Run completed naturally."
fi

# --- PLOTTING SECTION ---
# WHAT: Trigger the Python plotting script to read the Hydra output and save to myresults.
# WHY: Automates figure generation based on whatever data was collected before the timeout.
echo "[status] Generating figures..."
python plot_lunarlander.py

# --- EPILOGUE SECTION ---
# WHAT: Calculate final runtime and send the completion notification to your phone.
END_TIME=$(date +%s)
RUNTIME_SEC=$((END_TIME - START_TIME))
RUNTIME_FMT=$(printf "%02dh:%02dm:%02ds" $((RUNTIME_SEC / 3600)) $(( (RUNTIME_SEC % 3600) / 60)) $((RUNTIME_SEC % 60)))

MESSAGE="[$RUN_NAME] $STATUS_MSG Runtime: $RUNTIME_FMT."
echo "[epilogue] $MESSAGE"

curl -s -d "$MESSAGE" -H "Title: Run Finished" ntfy.sh/$NTFY_TOPIC > /dev/null