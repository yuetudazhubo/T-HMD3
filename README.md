## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:
- Conda (for environment management)
- Git LFS (Large File Storage) -- For IsaacLab
- CMake -- For IsaacLab

And the following system packages:
```bash
sudo apt install libglfw3 libgl1-mesa-glx libosmesa6 git-lfs cmake
```

## 📖 Installation

This project requires different Conda environments for different sets of experiments.

### Common Setup
First, ensure the common dependencies are installed as mentioned in the [Prerequisites](#prerequisites) section.

### Environment for HumanoidBench

```bash
conda create -n tdmc_td3_hb -y python=3.10
conda activate tdmc_td3_hb
pip install --editable git+https://github.com/carlosferrazza/humanoid-bench.git#egg=humanoid-bench
pip install -r requirements/requirements.txt
```

### Environment for MuJoCo Playground
```bash
conda create -n tdmc_td3_playground -y python=3.10
conda activate tdmc_td3_playground
pip install -r requirements/requirements_playground.txt
```

**⚠️ Note:** Our `requirements_playground.txt` specifies `Jax==0.4.35`, which we found to be stable for latest GPUs in certain tasks such as `LeapCubeReorient` or `LeapCubeRotateZAxis`


### Environment for IsaacLab
```bash
conda create -n tdmc_td3_isaaclab -y python=3.10
conda activate tdmc_td3_isaaclab

# Install IsaacLab (refer to official documentation for the latest steps)
# Official Quickstart: https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
cd ..

# Install project-specific requirements
pip install -r requirements/requirements.txt
```

### Environment for MTBench
MTBench does not support humanoid experiments, but is a useful multi-task benchmark with massive parallel simulation. This could be useful for users who want to use tdmc_td3 for their multi-task experiments.

```bash
conda create -n tdmc_td3_mtbench -y python=3.8  # Note python version
conda activate tdmc_td3_mtbench

# Install IsaacGym -- recommend to follow instructions in https://github.com/BoosterRobotics/booster_gym
...

# Install MTBench
git clone https://github.com/Viraj-Joshi/MTBench.git
cd MTbench
pip install -e .
pip install skrl
cd ..

# Install project-specific requirements
pip install -r requirements/requirements_isaacgym.txt
```

### (Optional) Accelerate headless GPU rendering in cloud instances

In some cloud VM images the NVIDIA kernel driver is present but the user-space OpenGL/EGL/Vulkan libraries aren't, so MuJoCo falls back to CPU renderer. You can install just the NVIDIA user-space libraries (and skip rebuilding the kernel module) with:

```bash
sudo apt install -y kmod
sudo sh NVIDIA-Linux-x86_64-<your_driver_version>.run -s --no-kernel-module --ui=none --no-questions
```

As a rule-of-thumb, if you're running experiments and rendering is taking longer than 5 seconds, it is very likely that GPU renderer is not used.

## 🚀 Running Experiments

Activate the appropriate Conda environment before running experiments.

Please see `tdmc_td3_td3/hyperparams.py` for information regarding hyperparameters!

### HumanoidBench Experiments
```bash
cd TDMC-TD3
conda activate tdmc_td3_hb
bash run_tdmc_hb.sh

### MuJoCo Playground Experiments
```bash
cd TDMC-TD3
conda activate tdmc_td3_playground
###modyfing the env_name

### IsaacLab Experiments
```bash
cd TDMC-TD3
conda activate tdmc_td3_isaaclab
###modyfing the env_name

### MTBench Experiments
```bash
cd TDMC-TD3
conda activate tdmc_td3_mtbench
###modyfing the env_name

**Quick note:** For boolean-based arguments, you can set them to False by adding `no_` in front each argument, for instance, if you want to disable Clipped Q Learning, you can specify `--no_use_cdq` in your command.

### TD3
```bibtex
@inproceedings{fujimoto2018addressing,
  title={Addressing function approximation error in actor-critic methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International conference on machine learning},
  pages={1587--1596},
  year={2018},
  organization={PMLR}
}
```

### SimbaV2
```bibtex
@article{lee2025hyperspherical,
  title={Hyperspherical normalization for scalable deep reinforcement learning},
  author={Lee, Hojoon and Lee, Youngdo and Seno, Takuma and Kim, Donghu and Stone, Peter and Choo, Jaegul},
  journal={arXiv preprint arXiv:2502.15280},
  year={2025}
}
```

### LeanRL

Following the [LeanRL](https://github.com/pytorch-labs/LeanRL)'s recommendation, we put CleanRL's bibtex here:

```bibtex
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```

### Parallel Q-Learning (PQL)
```bibtex
@inproceedings{li2023parallel,
  title={Parallel $ Q $-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation},
  author={Li, Zechu and Chen, Tao and Hong, Zhang-Wei and Ajay, Anurag and Agrawal, Pulkit},
  booktitle={International Conference on Machine Learning},
  pages={19440--19459},
  year={2023},
  organization={PMLR}
}
```

### HumanoidBench
```bibtex
@inproceedings{sferrazza2024humanoidbench,
  title={Humanoidbench: Simulated humanoid benchmark for whole-body locomotion and manipulation},
  author={Sferrazza, Carmelo and Huang, Dun-Ming and Lin, Xingyu and Lee, Youngwoon and Abbeel, Pieter},
  booktitle={Robotics: Science and Systems},
  year={2024}
}
```

### MuJoCo Playground
```bibtex
@article{zakka2025mujoco,
  title={MuJoCo Playground},
  author={Zakka, Kevin and Tabanpour, Baruch and Liao, Qiayuan and Haiderbhai, Mustafa and Holt, Samuel and Luo, Jing Yuan and Allshire, Arthur and Frey, Erik and Sreenath, Koushil and Kahrs, Lueder A and others},
  journal={arXiv preprint arXiv:2502.08844},
  year={2025}
}
```

### IsaacLab
```bibtex
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

### MTBench
```bibtex
@inproceedings{
joshi2025benchmarking,
title={Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks},
author={Viraj Joshi and Zifan Xu and Bo Liu and Peter Stone and Amy Zhang},
booktitle={Reinforcement Learning Conference},
year={2025},
url={https://openreview.net/forum?id=z0MM0y20I2}
}
```

### Getting SAC to Work on a Massive Parallel Simulator
```bibtex
@article{raffin2025isaacsim,
  title   = "Getting SAC to Work on a Massive Parallel Simulator: An RL Journey With Off-Policy Algorithms",
  author  = "Raffin, Antonin",
  journal = "araffin.github.io",
  year    = "2025",
  month   = "Feb",
  url     = "https://araffin.github.io/post/sac-massive-sim/"
}
```

### Speeding Up SAC with Massively Parallel Simulation
```bibtex
@article{shukla2025fastsac,
  title   = "Speeding Up SAC with Massively Parallel Simulation",
  author  = "Shukla, Arth",
  journal = "https://arthshukla.substack.com",
  year    = "2025",
  month   = "Mar",
  url     = "https://arthshukla.substack.com/p/speeding-up-sac-with-massively-parallel"
}
```
