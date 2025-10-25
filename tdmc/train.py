import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import random
import time
import math

import tqdm
import wandb
import numpy as np

try:
    import isaacgym
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from tensordict import TensorDict

from tdmc.thmd3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    PerTaskRewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    mark_step,
)
from hyperparams import get_args

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass
import os, warnings
warnings.filterwarnings("ignore", message=".*NVML_SUCCESS.*")


os.environ["PYTORCH_NO_DISTRIBUTED_DEBUG"] = "1"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_NVML_BASED_ALLOCATOR_DISABLED"] = "1"

def main():
    args = get_args()
    print(args)


    if hasattr(args, "use_bilstm"):
        if args.use_bilstm:
            args.use_static_encoder = True
            args.static_encoder_type = "gatedmlp"

        delattr(args, "use_bilstm")
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if args.use_wandb:
        try:
            wandb.init(
                project=args.project,
                name=run_name,
                config=vars(args),
                save_code=True,
            )
        except Exception as e:
            print(f"W&B initialization failed: {e}")
            print("Falling back to offline mode")
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=args.project,
                name=run_name,
                config=vars(args),
                save_code=True,
            )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    if args.env_name.startswith("h1hand-") or args.env_name.startswith("h1-"):
        from environments.humanoid_bench_env import HumanoidBenchEnv

        env_type = "humanoid_bench"
        envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
        eval_envs = envs
        render_env = HumanoidBenchEnv(
            args.env_name, 1, render_mode="rgb_array", device=device
        )
    elif args.env_name.startswith("Isaac-"):
        from environments.isaaclab_env import IsaacLabEnv

        env_type = "isaaclab"
        envs = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds,
        )
        eval_envs = envs
        render_env = envs
    elif args.env_name.startswith("MTBench-"):
        from environments.mtbench_env import MTBenchEnv

        env_name = "-".join(args.env_name.split("-")[1:])
        env_type = "mtbench"
        envs = MTBenchEnv(env_name, args.device_rank, args.num_envs, args.seed)
        eval_envs = envs
        render_env = envs
    else:
        from environments.mujoco_playground_env import make_env

        # TODO: Check if re-using same envs for eval could reduce memory usage
        env_type = "mujoco_playground"
        envs, eval_envs, render_env = make_env(
            args.env_name,
            args.seed,
            args.num_envs,
            args.num_eval_envs,
            args.device_rank,
            use_tuned_reward=args.use_tuned_reward,
            use_domain_randomization=args.use_domain_randomization,
            use_push_randomization=args.use_push_randomization,
        )

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if type(envs.num_privileged_obs) == int
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    action_low, action_high = -1.0, 1.0

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    if args.reward_normalization:
        if env_type in ["mtbench"]:
            reward_normalizer = PerTaskRewardNormalizer(
                num_tasks=envs.num_tasks,
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
        else:
            reward_normalizer = RewardNormalizer(
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    if env_type == "mtbench":
        actor_kwargs["n_obs"] = n_obs - envs.num_tasks + args.task_embedding_dim
        critic_kwargs["n_obs"] = n_critic_obs - envs.num_tasks + args.task_embedding_dim
        actor_kwargs["num_tasks"] = envs.num_tasks
        actor_kwargs["task_embedding_dim"] = args.task_embedding_dim
        critic_kwargs["num_tasks"] = envs.num_tasks
        critic_kwargs["task_embedding_dim"] = args.task_embedding_dim

    if args.agent == "td3":
        if env_type in ["mtbench"]:
            from tdmc.td3 import MultiTaskActor, MultiTaskCritic

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from tdmc.td3 import Actor, Critic

            actor_cls = Actor
            critic_cls = Critic

        print("Using TD3")
    elif args.agent == "td3_Hyperspherical":
        if env_type in ["mtbench"]:
            from tdmc.thmd3_hyperspherical import MultiTaskActor, MultiTaskCritic

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from tdmc.thmd3_hyperspherical import Actor, Critic

            actor_cls = Actor
            critic_cls = Critic

        print("Using td3 + Hyperspherical")
        actor_kwargs.pop("init_scale")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
                "alpha_init": 1.0 / (args.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": args.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
                "alpha_init": 1.0 / (args.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
                "num_blocks": args.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )

    elif args.agent == "tdmc":
        from tdmc.thmd3 import HypersphericalActor as Actor, HypersphericalCritic as Critic
        print("Using TD3 + Hyperspherical (multi-critic + lightweight encoder)")
        actor_cls, critic_cls = Actor, Critic
        print(f"n_obs: {n_obs}, n_act: {n_act}") 
        print(f"encoder_feat: {args.encoder_feat}, seq_len: {args.seq_len}") 
        actor_kwargs.pop("init_scale")
        actor_kwargs.update({
            "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
            "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
            "alpha_init": 1.0 / (args.actor_num_blocks + 1),
            "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
            "expansion": 4,
            "c_shift": 3.0,
            "num_blocks": args.actor_num_blocks,
            "use_static_encoder": args.use_static_encoder,
            "static_encoder_type": args.static_encoder_type,
        })

        critic_kwargs.update({
            "encoder_feat": args.encoder_feat, 
            "seq_len": args.seq_len,  
            "encoder_blocks": args.encoder_blocks,  
            "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
            "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
            "alpha_init": 1.0 / (args.critic_num_blocks + 1),
            "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
            "num_blocks": args.critic_num_blocks,
            "expansion": 4,
            "c_shift": 3.0,
            "num_critics": args.num_critics,
            "critic_dropout": args.critic_dropout,
            "critic_layernorm": args.critic_layernorm,
            "use_static_encoder": args.use_static_encoder,
            "static_encoder_type": args.static_encoder_type,
        })
        print("Using TD3 + Hyperspherical ") 
        print(f"\n[Debug] 初始化Hyperspherical Critic参数:")
        print(f"  num_critics: {args.num_critics}")
        print(f"  encoder_feat: {args.encoder_feat}")
        print(f"  seq_len: {args.seq_len}")
        print(f"  critic_kwargs: {critic_kwargs}\n")
    else:
        raise ValueError(f"Agent {args.agent} not supported")


    actor = actor_cls(**actor_kwargs).to(device)

    if env_type in ["mtbench"]:
        # Python 3.8 doesn't support 'from_module' in tensordict
        policy = actor.explore
    else:
        from tensordict import from_module

        actor_detach = actor_cls(**actor_kwargs)
        # Copy params to actor_detach without grad
        from_module(actor).data.to_module(actor_detach)
        policy = actor_detach.explore

    qnet = critic_cls(**critic_kwargs).to(device)
    qnet_target = critic_cls(**critic_kwargs).to(device)
    qnet_target.load_state_dict(qnet.state_dict())

    # ===== Teacher critic initialization (optional) =====
    teacher = None
    teacher_monitor = None
    if getattr(args, "teacher_critic_path", None):
        
        from tdmc.thmd3 import TeacherCriticWrapper, TeacherMonitor

        try:
            crit_kwargs_for_teacher = critic_kwargs.copy()
            # Ensure teacher model on same device
            teacher = TeacherCriticWrapper(args.teacher_critic_path, device, critic_cls, crit_kwargs_for_teacher)
            teacher_monitor = TeacherMonitor(window_size=args.teacher_window, threshold=args.teacher_threshold)
        except Exception as e:
            print(f"[Teacher] Failed to load teacher critic from {args.teacher_critic_path}: {e}")
            teacher = None
            teacher_monitor = None

        print(f"[Teacher] loaded: {bool(teacher)}, mode={args.teacher_mode}, loss_type={args.teacher_loss_type}, weight={args.teacher_weight}")
    # =====================================================



    if args.agent == "tdmc":
        print(f"[Debug] Hyperspherical Critic实例化完成:")
        print(f"  评论家数量(heads): {qnet.ensemble.n_heads}")  
        print(f"  是否使用分布型Q值: {not hasattr(args, 'disable_distributional') or not args.disable_distributional}")
        print(f"  编码器类型: {args.static_encoder_type}\n")
    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    # Add learning rate schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        playground_mode=env_type == "mujoco_playground",
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def evaluate():
        logs_dict.update({
            "action_norm": torch.tensor(0.0, device=device),
        })
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        if env_type == "isaaclab":
            obs = eval_envs.reset(random_start_init=False)
        else:
            obs = eval_envs.reset()

        # Run for a fixed number of steps
        for i in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)

                action_norms = torch.norm(actions, dim=-1).mean().item()
                logs_dict["action_norm"] = action_norms  


            next_obs, rewards, dones, infos = eval_envs.step(actions.float())

            if env_type == "mtbench":
                # We only report success rate in MTBench evaluation
                rewards = (
                    infos["episode"]["success"].float() if "episode" in infos else 0.0
                )
            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            if env_type == "mtbench" and "episode" in infos:
                dones = dones | infos["episode"]["success"]
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        return episode_returns.mean().item(), episode_lengths.mean().item()

    def render_with_rollout():
        # Quick rollout for rendering
        if env_type == "humanoid_bench":
            obs = render_env.reset()
            renders = [render_env.render()]
        elif env_type in ["isaaclab", "mtbench"]:
            raise NotImplementedError(
                "We don't support rendering for IsaacLab and MTBench environments"
            )
        else:
            obs = render_env.reset()
            render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            renders = [render_env.state]
        for i in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)
            next_obs, _, done, _ = render_env.step(actions.float())
            if env_type == "mujoco_playground":
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            if i % 2 == 0:
                if env_type == "humanoid_bench":
                    renders.append(render_env.render())
                else:
                    renders.append(render_env.state)
            if done.any():
                break
            obs = next_obs

        if env_type == "mujoco_playground":
            renders = render_env.render_trajectory(renders)
        return renders

    def update_main(data, logs_dict):
        logs_dict.update({
            "td1_mean": torch.tensor(0.0, device=device),
            "td2_mean": torch.tensor(0.0, device=device),
            "td1_std": torch.tensor(0.0, device=device),
            "td2_std": torch.tensor(0.0, device=device),
            "critic_weight_l2_mean": torch.tensor(0.0, device=device),
            "critic_disagreement": torch.tensor(0.0, device=device),
            "head_q_means": torch.tensor(0.0, device=device),
            "ensemble_min_q": torch.tensor(0.0, device=device),
            "ensemble_mean_q": torch.tensor(0.0, device=device),
            "head_q_maxs": torch.tensor(0.0, device=device),
            "head_q_mins": torch.tensor(0.0, device=device),
        })
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                proj_all, qf_next_target_dist = (
                    qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )       
                qf_next_target_projected_dict = {}
                for i in range(proj_all.size(1)):
                    qf_next_target_projected_dict[f"qf{i}_next_target_projected"] = proj_all[:, i, :]
                qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist

                qf_next_target_value_dict = {}
                for i in range(proj_all.size(1)):
                    key = f"qf{i}_next_target_projected"
                    projected = qf_next_target_projected_dict[key]
                    qf_next_target_value_dict[f"qf{i}_next_target_value"] = qnet_target.get_value(projected)
            
            
            all_logits, qf_next_target_dist2 = qnet(critic_observations, actions)
            qf_dict = {}
            for i in range(all_logits.size(1)):
                qf_dict[f"qf{i}"] = all_logits[:, i, :]


            with torch.no_grad():

                all_logits2 = qnet.ensemble(critic_observations, actions)  
                if not args.disable_distributional:
                    all_probs = F.softmax(all_logits2, dim=-1)  
                    head_values = torch.sum(all_probs * qnet.ensemble.q_support.unsqueeze(0).unsqueeze(0), dim=-1)
                else:
                    head_values = all_logits2.mean(dim=-1)  
                
                head_means = head_values.mean(dim=0)  
                head_maxs = head_values.max(dim=0)[0]  
                head_mins = head_values.min(dim=0)[0] 
                ensemble_min_q = head_values.min(dim=1)[0].mean()  
                ensemble_mean_q = head_values.mean(dim=1).mean()  
                
                logs_dict["head_q_means"] = head_means.cpu().numpy().tolist()
                logs_dict["ensemble_min_q"] = ensemble_min_q
                logs_dict["ensemble_mean_q"] = ensemble_mean_q
                logs_dict["head_q_maxs"] = head_maxs.cpu().numpy().tolist()
                logs_dict["head_q_mins"] = head_mins.cpu().numpy().tolist()

            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf_dict["qf0"], dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf_dict["qf1"], dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss#qf3\qf4...


            # ===== Teacher distillation loss (optional) =====
            distill_loss = torch.tensor(0.0, device=device)
            if teacher is not None and teacher_monitor is not None and teacher_monitor.use_teacher and args.teacher_mode in ["shaping", "bootstrap"]:
                try:

                    with torch.no_grad():
                        teacher_logits = teacher.get_logits(critic_observations, actions)
                    if args.teacher_loss_type == "mse":
                        distill_loss = F.mse_loss(qf_dict["qf0"], teacher_logits)
                    else:
                        teacher_p = F.softmax(teacher_logits.detach(), dim=-1)
                        student_logp = F.log_softmax(qf_dict["qf0"], dim=-1)
                        distill_loss = F.kl_div(student_logp, teacher_p, reduction="batchmean")
                    qf_loss = qf_loss + args.teacher_weight * distill_loss
                    if teacher_monitor is not None:
                        teacher_monitor.update(float(distill_loss.detach()))
                        teacher_monitor.check_disable()
                except Exception as e:
                    print(f"[Teacher] distillation compute failed (step): {e}")
                    distill_loss = torch.tensor(0.0, device=device)
            # =====================================================

##
        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()
        with torch.no_grad():
            try:
                if not hasattr(qnet, "ensemble") or qnet.ensemble is None:
                    q1_val = qnet.get_value(F.softmax(qf_dict["qf0"], dim=1))  
                    q2_val = qnet.get_value(F.softmax(qf_dict["qf1"], dim=1))  
                    head_vals = torch.stack([q1_val, q2_val], dim=1) 
                else:
                    if hasattr(qnet, 'temporal_encoder') and qnet.temporal_encoder is not None:
                        obs_feat = qnet.temporal_encoder(critic_observations) 
                    else:
                        obs_feat = critic_observations  
                    
                    if callable(qnet.ensemble):
                        head_logits = qnet.ensemble(obs_feat, actions)  
                    else:
                        raise ValueError("qnet.ensemble is not callable")
                    
                    if args.disable_distributional:
                        head_vals = head_logits  
                    else:
                        head_probs = F.softmax(head_logits, dim=-1)  
                        if hasattr(qnet.ensemble, "q_support"):
                            q_support = qnet.ensemble.q_support  
                        elif hasattr(qnet, "q_support"):
                            q_support = qnet.q_support  
                        else:
                            q_support = torch.linspace(args.v_min, args.v_max, args.num_atoms, device=device)             
                        head_vals = torch.sum(head_probs * q_support.view(1, 1, -1), dim=-1)  
                

                logs_dict["critic_disagreement"] = head_vals.std(dim=1).mean().detach()

            except Exception as e:
                print(f"Critic disagreement计算错误: {str(e)}")
                logs_dict["critic_disagreement"] = torch.tensor(0.0, device=device)


            try:#对象改变
                sample_td1 = -torch.sum(qf1_next_target_dist * F.log_softmax(qf_dict["qf0"], dim=1), dim=1).detach()
                sample_td2 = -torch.sum(qf2_next_target_dist * F.log_softmax(qf_dict["qf1"], dim=1), dim=1).detach()
                logs_dict["td1_mean"] = sample_td1.mean()
                logs_dict["td2_mean"] = sample_td2.mean()
                logs_dict["td1_std"] = sample_td1.std()
                logs_dict["td2_std"] = sample_td2.std()
            except Exception:
                pass


            try:
                total_norm = torch.tensor(0.0, device=device)
                cnt = 0
                for p in qnet.parameters():
                    total_norm += torch.norm(p.detach())
                    cnt += 1
                logs_dict["critic_weight_l2_mean"] = (total_norm / max(1, cnt)).detach()
            except Exception:
                pass

            if 'distill_loss' in locals() or 'distill_loss' in globals():
                logs_dict["distill_loss"] = (
                    distill_loss.detach()
                    if isinstance(distill_loss, torch.Tensor)
                    else torch.tensor(distill_loss, device=device)
                )
            else:
                logs_dict["distill_loss"] = torch.tensor(0.0, device=device)

            if teacher_monitor is not None:
                logs_dict["teacher_active"] = torch.tensor(
                    1.0 if teacher_monitor.use_teacher else 0.0, device=device
                )
            else:
                logs_dict["teacher_active"] = torch.tensor(0.0, device=device)



        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf_next_target_value_dict["qf0_next_target_value"].max().detach()
        logs_dict["qf_min"] = qf_next_target_value_dict["qf0_next_target_value"].min().detach()
        return logs_dict

    def update_pol(data, logs_dict):

        logs_dict.update({
            "weights_projected": torch.tensor(0.0, device=device),
            "actor_grad_norm_median": torch.tensor(0.0, device=device),
            "actor_grad_norm_mean": torch.tensor(0.0, device=device),
        })
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = (
                data["critic_observations"]
                if envs.asymmetric_obs
                else data["observations"]
            )
            
            all_logits3, qf_next_target_dist2 = qnet(critic_observations, actor(data["observations"]))
            qf_dict = {}
            for i in range(all_logits3.size(1)):
                qf_dict[f"qf{i}"] = all_logits3[:, i, :]
            qf1_value = qnet.get_value(F.softmax(qf_dict["qf0"], dim=1))
            qf2_value = qnet.get_value(F.softmax(qf_dict["qf1"], dim=1))#

            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

            if teacher is not None and teacher_monitor is not None and teacher_monitor.use_teacher and args.teacher_mode == "bootstrap":
                try:
                    with torch.no_grad():
                        actor_actions = actor(data["observations"])
                        teacher_logits_pi = teacher.get_logits(critic_observations, actor_actions)
                        if hasattr(qnet, "ensemble") and hasattr(qnet.ensemble, "q_support"):
                            q_support = qnet.ensemble.q_support.to(teacher_logits_pi.device)
                        elif hasattr(qnet, "q_support"):
                            q_support = qnet.q_support.to(teacher_logits_pi.device)
                        else:
                            q_support = torch.linspace(args.v_min, args.v_max, args.num_atoms, device=teacher_logits_pi.device)
                        teacher_probs_pi = F.softmax(teacher_logits_pi, dim=-1)
                        teacher_value_pi = torch.sum(teacher_probs_pi * q_support.view(1, -1), dim=-1)
                    actor_loss = -teacher_value_pi.mean()
                except Exception as e:
                    print(f"[Teacher] actor bootstrap guidance failed: {e}")

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        with torch.no_grad():
            if getattr(args, "weight_projection", False):
                try:
                    from thmd3 import project_model_weights_to_unit_sphere
                    project_model_weights_to_unit_sphere(actor)
                    project_model_weights_to_unit_sphere(qnet)
                    logs_dict["weights_projected"] = torch.tensor(1.0, device=device)
                except Exception as e:
                    logs_dict["weights_projected"] = torch.tensor(0.0, device=device)
                print("Weight projection failed:", e)

        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]

        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    if args.compile:
        compile_mode = 'default'  #args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode)
        update_pol = torch.compile(update_pol, mode=compile_mode)
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=None)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(
            torch_checkpoint["critic_obs_normalizer_state"]
        )
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    desc = ""

    while global_step < args.total_timesteps:
        mark_step()
        logs_dict = TensorDict()
        if (
            start_time is None
            and global_step >= args.measure_burnin + args.learning_starts
        ):
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]

        if args.reward_normalization:
            if env_type == "mtbench":
                task_ids_one_hot = obs[..., -envs.num_tasks :]
                task_indices = torch.argmax(task_ids_one_hot, dim=1)
                update_stats(rewards, dones.float(), task_ids=task_indices)
            else:
                update_stats(rewards, dones.float())

        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]
        # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs
        rb.extend(transition)

        obs = next_obs
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(
                        data["critic_observations"]
                    )
                    data["next"]["critic_observations"] = normalize_critic_obs(
                        data["next"]["critic_observations"]
                    )
                raw_rewards = data["next"]["rewards"]
                if env_type in ["mtbench"] and args.reward_normalization:
                    # Multi-task reward normalization
                    task_ids_one_hot = data["observations"][..., -envs.num_tasks :]
                    task_indices = torch.argmax(task_ids_one_hot, dim=1)
                    data["next"]["rewards"] = normalize_reward(
                        raw_rewards, task_ids=task_indices
                    )
                else:
                    data["next"]["rewards"] = normalize_reward(raw_rewards)

                logs_dict = update_main(data, logs_dict)
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        logs_dict = update_pol(data, logs_dict)
                else:
                    if global_step % args.policy_frequency == 0:
                        logs_dict = update_pol(data, logs_dict)

                soft_update(qnet, qnet_target, args.tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "env_rewards": rewards.mean(),
                        "buffer_rewards": raw_rewards.mean(),
                        "critic_disagreement" : logs_dict["critic_disagreement"],
                        "td1_mean": logs_dict["td1_mean"],
                        "td2_mean": logs_dict["td2_mean"], 
                        "td1_std": logs_dict["td1_std"],
                        "td2_std": logs_dict["td2_std"],
                        "critic_weight_l2_mean": logs_dict["critic_weight_l2_mean"],
                        "head_q_means":logs_dict["head_q_means"],
                        "ensemble_min_q":logs_dict["ensemble_min_q"],
                        "ensemble_mean_q":logs_dict["ensemble_mean_q"],
                        "head_q_maxs":logs_dict["head_q_maxs"],
                        "head_q_mins":logs_dict["head_q_mins"],
                        "distill_loss":logs_dict["distill_loss"],
                        "teacher_active":logs_dict["teacher_active"],
                    } 

                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        print(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate()
                        if env_type in ["humanoid_bench", "isaaclab", "mtbench"]:
                            obs = envs.reset()
                        logs["eval_avg_return"] = eval_avg_return
                        logs["eval_avg_length"] = eval_avg_length
                        logs["action_norm"] = logs_dict["action_norm"]

                    if (
                        args.render_interval > 0
                        and global_step % args.render_interval == 0
                    ):
                        renders = render_with_rollout()
                        render_video = wandb.Video(
                            np.array(renders).transpose(
                                0, 3, 1, 2
                            ), 
                            fps=30,
                            format="gif",
                        )
                        logs["render_video"] = render_video
                if args.use_wandb:
                    wandb.log(
                        {
                            "speed": speed,
                            "frame": global_step * args.num_envs,
                            "critic_lr": q_scheduler.get_last_lr()[0],
                            "actor_lr": actor_scheduler.get_last_lr()[0],
                            **logs,
                        },
                        step=global_step,
                    )
                if global_step == 0 and args.use_wandb:
                    try:
                        wandb.config.update(vars(args))
                        import subprocess
                        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
                        wandb.run.summary["git_sha"] = git_sha
                    except Exception as e:
                        print("Git hash record failed:", e)

            if (
                args.save_interval > 0
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving model at global step {global_step}")
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    critic_obs_normalizer,
                    args,
                    f"models/{run_name}_{global_step}.pt",
                )

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)

    save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        f"models/{run_name}_final.pt",
    )


if __name__ == "__main__":
    main()
