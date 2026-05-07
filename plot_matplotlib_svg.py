import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path

def load_eval_points(csv_path):
    # WHAT: Reads the logger's CSV output
    # WHY: The logger saves the exact same data here that it sends to TensorBoard.
    steps, rewards = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            rewards.append(float(row["episode_reward"]))
    return steps, rewards

def find_eval_csv(base_dir, agent_prefix):
    # WHAT: Recursively searches for the CSV files
    # WHY: This prevents the script from breaking when you change hyperparameters and Hydra creates a new folder name.
    for path in Path(base_dir).rglob('eval.csv'):
        if agent_prefix in str(path):
            return path
    return None

def main():
    # Set this to whichever environment you want to plot (e.g., "exp/Pendulum-v1" or "exp/LunarLander-v3")
    target_env = "exp/LunarLander-v3" 
    
    pebble_csv = find_eval_csv(target_env, "PEBBLE")
    ours_csv = find_eval_csv(target_env, "Value_lambda")
    
    if not pebble_csv or not ours_csv:
        print(f"Error: Could not find CSVs in {target_env}. Are you sure the training ran?")
        return

    print(f"Loading PEBBLE from: {pebble_csv}")
    print(f"Loading OURS from: {ours_csv}")

    pebble_steps, pebble_rewards = load_eval_points(pebble_csv)
    ours_steps, ours_rewards = load_eval_points(ours_csv)

    # Initialize the plot
    plt.figure(figsize=(8, 5))
    
    # Plot both lines with distinct colors
    plt.plot(pebble_steps, pebble_rewards, label="PEBBLE (Baseline)", color="#1f77b4", linewidth=2)
    plt.plot(ours_steps, ours_rewards, label="OURS (Value-Guided)", color="#ff7f0e", linewidth=2)

    # Style the graph
    plt.title(f"Evaluation Reward: {target_env.split('/')[-1]}", fontsize=14)
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Episode Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # WHAT: Create the figs directory and save the file
    # WHY: To keep the repository organized, matching the original authors' structure.
    os.makedirs("figs", exist_ok=True)
    output_path = f"figs/comparison_{target_env.split('/')[-1]}.svg"
    plt.savefig(output_path, format="svg")
    print(f"\nSuccess! Graph saved to: {output_path}")

if __name__ == "__main__":
    main()