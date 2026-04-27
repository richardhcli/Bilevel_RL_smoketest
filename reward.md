# Reward / Reproduction Notes

This file gives the minimum setup and run instructions for the corrected codebase used to reproduce the reward plots in this repository.

## 1. Recommended environment

Use WSL Ubuntu, not native Windows Python.

Working Linux-side repo path used for reproduction:

`/root/projects/Bi-level-Reinforcement-Learning-main`

## 2. Install dependencies

### Ubuntu packages

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y unzip gcc g++ make build-essential cmake git wget curl libosmesa6-dev libgl1 libglfw3 patchelf libegl1 libopengl0 libglew-dev
```

### Miniconda

```bash
wget -O /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /root/miniconda.sh -b -p /root/miniconda3
/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Create env

```bash
cd /root/projects/Bi-level-Reinforcement-Learning-main
/root/miniconda3/bin/conda env create -f conda_env.yml
```

### MuJoCo

```bash
mkdir -p /root/.mujoco
wget -O /tmp/mujoco200_linux.zip https://www.roboti.us/download/mujoco200_linux.zip
unzip -o /tmp/mujoco200_linux.zip -d /root/.mujoco
wget -O /root/.mujoco/mjkey.txt https://www.roboti.us/file/mjkey.txt
wget -O /tmp/mujoco210-linux-x86_64.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf /tmp/mujoco210-linux-x86_64.tar.gz -C /root/.mujoco
ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
```

### Python packages and compatibility pins

```bash
/root/miniconda3/bin/conda install -n mrn -c conda-forge -y dm-tree=0.1.5
/root/miniconda3/bin/conda run -n mrn pip install -e /root/projects/Bi-level-Reinforcement-Learning-main
/root/miniconda3/bin/conda run -n mrn pip install dm-env future h5py lxml protobuf==3.19.6 pyopengl scipy tqdm termcolor pybullet scikit-image
/root/miniconda3/bin/conda run -n mrn pip install -e /root/projects/Bi-level-Reinforcement-Learning-main/custom_dmcontrol --no-deps
/root/miniconda3/bin/conda run -n mrn pip install -e /root/projects/Bi-level-Reinforcement-Learning-main/custom_dmc2gym
/root/miniconda3/bin/conda run -n mrn pip install git+https://github.com/rlworkgroup/metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
/root/miniconda3/bin/conda run -n mrn pip install git+https://github.com/facebookresearch/hydra@0.11_branch
/root/miniconda3/bin/conda run -n mrn pip install "Cython<3"
/root/miniconda3/bin/conda run -n mrn pip install --force-reinstall "gym==0.17.3"
```

## 3. Environment variables before training

```bash
export MUJOCO_GL=osmesa
export MJKEY_PATH=/root/.mujoco/mjkey.txt
export MJLIB_PATH=/root/.mujoco/mujoco200_linux/bin/libmujoco200.so
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export LD_LIBRARY_PATH=/root/.mujoco/mujoco200_linux/bin:/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
```

## 4. Training commands used for the reward plots

### Walker: OURS

```bash
python train_V.py env=walker_walk seed=12345 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=50 feed_type=1 experiment=Value_lambda_0.1_fix1
```

### Walker: PEBBLE

For `plot_current_results.py`:

```bash
python train_PEBBLE.py env=walker_walk seed=12345 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=50 feed_type=1 experiment=PEBBLE
```

For `plot_walker_train_200k_compare.py` and `plot_walker_eval_compare.py`:

```bash
python train_PEBBLE.py env=walker_walk seed=12345 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=50 feed_type=1 experiment=PEBBLE_walker_refresh
```

### Door Open: OURS

```bash
python train_V.py env=metaworld_door-open-v2 seed=12349 agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 num_unsup_steps=9000 num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 num_interact=5000 max_feedback=1000 reward_batch=10 reward_update=10 feed_type=1 experiment=Value_lambda_0.1_fix1_dooropen
```

### Door Open: PEBBLE

```bash
python train_PEBBLE.py env=metaworld_door-open-v2 seed=12345 agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 num_unsup_steps=9000 num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 num_interact=5000 max_feedback=1000 reward_batch=10 reward_update=10 feed_type=1 experiment=PEBBLE_dooropen_fixbaseline
```

## 5. Plot generation commands

Generate the combined current training-reward figure:

```bash
python plot_current_results.py
```

Generate Walker 0-200k training comparison:

```bash
python plot_walker_train_200k_compare.py
```

Generate Walker eval comparison:

```bash
python plot_walker_eval_compare.py
```

Generate Door Open eval comparison:

```bash
python plot_door_eval_compare.py
```

## 6. Naming caveat

The Walker PEBBLE experiment name is not fully consistent across plotting scripts:

- `plot_current_results.py` expects `PEBBLE_maxfeed100_Rbatch10_seed12345`
- `plot_walker_train_200k_compare.py` expects `PEBBLE_walker_refresh_maxfeed100_Rbatch10_seed12345`
- `plot_walker_eval_compare.py` expects `PEBBLE_walker_refresh_maxfeed100_Rbatch10_seed12345`

So use the experiment name that matches the figure you want to regenerate.

## 7. Verification note

We reran the Walker experiments from scratch with the same seed and configuration and confirmed that the regenerated logs matched the original stored logs exactly at overlapping checkpoints through nearly the full run. This is strong evidence that the current corrected codebase reproduces the reward plots already present in this repository.
