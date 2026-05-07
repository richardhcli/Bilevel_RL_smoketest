import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# WHAT: Automatically find the eval.csv files
# WHY: Hardcoding those massive Hydra directory names leads to broken paths when you change settings.
def find_eval_csv(base_dir, agent_prefix):
    for path in Path(base_dir).rglob('eval.csv'):
        if agent_prefix in str(path):
            return path
    return None

# WHAT: Load the data from the CSV
# WHY: The logger automatically saves every evaluation step here, identical to what is in TensorBoard.
def load_eval_points(csv_path):
    steps, rewards = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            rewards.append(float(row["episode_reward"]))
    return steps, rewards

def main():
    base_dir = "exp/LunarLander-v3"
    
    # WHAT: Explicitly declare the centralized output directory
    # WHY: Keeps the repository clean and aligns with the main.sh file structure.
    output_dir = "myresults/lunarlander"
    os.makedirs(output_dir, exist_ok=True)
    
    pebble_csv = find_eval_csv(base_dir, "PEBBLE")
    ours_csv = find_eval_csv(base_dir, "Value_lambda")
    
    # FAIL-SAFE: Check if the 4-hour timeout killed the script before eval data was generated
    if not pebble_csv or not ours_csv:
        print("\n[WARNING] Plotting Failed!")
        print("Could not find one or both eval.csv files.")
        print("Reason: The 4-hour timeout likely triggered before the agents reached the first Evaluation phase.")
        return

    print(f"Plotting PEBBLE from: {pebble_csv}")
    print(f"Plotting OURS from: {ours_csv}")

    pebble_steps, pebble_rewards = load_eval_points(pebble_csv)
    ours_steps, ours_rewards = load_eval_points(ours_csv)

    # WHAT: Construct the graph using Matplotlib
    # WHY: It handles all the complex vector math for the SVG generation automatically.
    plt.figure(figsize=(8, 5))
    plt.plot(pebble_steps, pebble_rewards, label="PEBBLE (Baseline)", color="#1f77b4", linewidth=2)
    plt.plot(ours_steps, ours_rewards, label="OURS (Value-Guided)", color="#ff7f0e", linewidth=2)

    plt.title("Evaluation Reward on LunarLander-v3", fontsize=14)
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Episode Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save to the targeted results directory
    output_filename = os.path.join(output_dir, "lunar_lander_learning_curve.svg")
    plt.savefig(output_filename, format="svg")
    print(f"Successfully saved plot to {output_filename}")

if __name__ == "__main__":
    main()