import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from tdmc.thmd3_hyperspherical import (
        l2normalize,
        HyperEmbedder,
        HyperLERPBlock,
        HyperTanhPolicy,
        DistributionalQNetwork,
    )
except Exception as e:
    raise ImportError(
        "fast_td3_simbav2.py not found or missing expected symbols. "
        "Please ensure fast_td3_simbav2.py is in the same directory and exports "
        "l2normalize, HyperEmbedder, HyperLERPBlock, HyperTanhPolicy, DistributionalQNetwork."
    )

# -------------------------
# Utilities: hypersphere norm & weight projection
# -------------------------
_eps = 1e-8


def hyperspherical_normalize(x: torch.Tensor, eps: float = _eps) -> torch.Tensor:
    """Normalize last dim to unit L2 norm (ℓ2 hypersphere)."""
    return x / (torch.linalg.norm(x, ord=2, dim=-1, keepdim=True) + eps)


def project_model_weights_to_unit_sphere(model: nn.Module, eps: float = 1e-12) -> None:
    """
    Project linear/Dense weight matrices to unit-norm rows (output-feature axis).
    Call AFTER optimizer.step() (and after scaler.update() if using AMP).
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and ("weight" in name):
                row_norm = torch.linalg.norm(p.data, ord=2, dim=1, keepdim=True)
                row_norm = torch.clamp(row_norm, min=eps)
                p.data = p.data / row_norm




# ==== teacher Critic  ====

class TeacherCriticWrapper:
    """
    Wrapper to load a pretrained critic and freeze it.
    Usage: teacher = TeacherCriticWrapper(path, device, critic_class, critic_kwargs)
    - get_logits(obs, actions) returns logits of Q (head 0 logits) [B, num_atoms]
    """
    def __init__(self, path: str, device: torch.device, critic_class, critic_kwargs: dict):
        print(f"[TeacherCritic] Loading teacher critic from: {path}")
        self.device = device
        self.critic = critic_class(**critic_kwargs).to(device)
        ckpt = torch.load(path, map_location=device)
        # Handle both plain and wrapped checkpoint formats
        if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # ---- Safe load (allow mismatched heads) ----
        missing, unexpected = self.critic.load_state_dict(state_dict, strict=False)
        print(f"[Teacher] partially loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"[Teacher] Missing keys example: {missing[:3]}")
        if unexpected:
            print(f"[Teacher] Unexpected keys example: {unexpected[:3]}")

        # ---- Freeze teacher critic ----
        self.critic.eval()
        for p in self.critic.parameters():
            p.requires_grad = False

        print("[TeacherCritic] Loaded and frozen.")


    @torch.no_grad()
    def get_logits(self, obs, actions):
        """
        Return logits for head0 (or first returned head). Compatible with distributional critic.
        Expects obs/actions already on correct device.
        """
        # Some critic implementations accept (obs, actions) and return (logits1, logits2) or [B,n_heads,num_atoms]
        out = self.critic(obs, actions)
        # If critic returns tuple (q1_logits, q2_logits)
        if isinstance(out, tuple) and len(out) >= 1:
            q1 = out[0]
            return q1[:, 0, :]
        # If critic returns stacked logits [B, n_heads, num_atoms], take head 0
        if isinstance(out, torch.Tensor) and out.dim() == 3:
            return out[:, 0, :]
        raise RuntimeError("Unsupported critic.forward return shape for TeacherCriticWrapper")

def compute_teacher_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, loss_type: str = "mse"):
    """
    teacher and student are raw logits (not softmaxed).
    - mse: MSE between logits
    - kl: KL(teacher_prob || student_prob) implemented as KLDiv(student_logprob, teacher_prob)
    """
    if loss_type == "mse":
        return F.mse_loss(student_logits, teacher_logits.detach())
    elif loss_type == "kl":
        teacher_p = F.softmax(teacher_logits.detach(), dim=-1)
        student_logp = F.log_softmax(student_logits, dim=-1)
        return F.kl_div(student_logp, teacher_p, reduction="batchmean")
    else:
        raise ValueError("Unknown loss_type for compute_teacher_loss")

class TeacherMonitor:
    """
    Track recent distillation loss values (sliding window).
    If average of window < threshold -> mark use_teacher = False.
    """
    def __init__(self, window_size: int = 5000, threshold: float = 0.01):
        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self.buf = []
        self.use_teacher = True

    def update(self, loss_value: float):
        self.buf.append(float(loss_value))
        if len(self.buf) > self.window_size:
            # pop oldest
            self.buf.pop(0)

    def check_disable(self):
        if len(self.buf) >= self.window_size:
            avg = sum(self.buf) / len(self.buf)
            if avg < self.threshold:
                self.use_teacher = False
                print(f"[TeacherMonitor] disabling teacher (avg_loss={avg:.6e} < threshold={self.threshold})")
        return self.use_teacher
# === end Teacher Critic Support ===
# -------------------------
# Static encoders (torch.compile friendly)
# -------------------------
class GatedMLPBlock(nn.Module):
    """
    A lightweight gated-MLP block for time mixing.
    Input: x [B, T, D]
    Returns: x [B, T, D]
    """

    def __init__(self, dim: int, seq_len: int, ff_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.ff_hidden = ff_hidden or (dim * 2)
        # time-mixing (linear projection along time dim for each channel)
        # implemented as 1x1 conv across time dimension for efficiency
        self.time_proj = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=True)
        # feedforward projection
        self.fc1 = nn.Linear(dim, self.ff_hidden)
        self.fc2 = nn.Linear(self.ff_hidden, dim)
        # gating unit
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        # time projection: conv1d expects [B, C, T]
        xt = x.transpose(1, 2)  # [B, D, T]
        xt = self.time_proj(xt)  # [B, D, T]
        xt = xt.transpose(1, 2)  # [B, T, D]
        # feedforward + gate
        h = self.act(self.fc1(xt))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(xt))
        out = xt + self.dropout(h * g)
        return out


class GatedMLPEncoder(nn.Module):
    """
    Stack of gated-MLP blocks with time/feature mixing.
    Accepts either [B, D] (treated as T=1) or [B, T, D].
    Outputs pooled [B, feat_dim]
    """

    def __init__(self, input_dim: int, feat_dim: int = 256, seq_len: int = 4, n_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.n_blocks = n_blocks
        self.input_proj = nn.Linear(input_dim, feat_dim)
        self.blocks = nn.ModuleList([GatedMLPBlock(dim=feat_dim, seq_len=seq_len, ff_hidden=feat_dim * 2, dropout=dropout) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)  # will pool across time
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, D] or [B, T, D]
        if obs.ndim == 2:
            x = obs.unsqueeze(1).repeat(1, self.seq_len, 1)  # broadcast single step
        elif obs.ndim == 3:
            x = obs
            # If sequence length mismatches, either pad/trim externally. For simplicity, if T != seq_len, we
            # interpolate/resample by linear interpolation to seq_len (keep static shapes for compile).
            if x.size(1) != self.seq_len:
                # simple: if longer, truncate; if shorter, repeat last
                if x.size(1) > self.seq_len:
                    x = x[:, : self.seq_len, :]
                else:
                    # repeat last to fill
                    last = x[:, -1:, :].expand(-1, self.seq_len - x.size(1), -1)
                    x = torch.cat([x, last], dim=1)
        else:
            raise ValueError("obs must be 2D or 3D")

        x = self.input_proj(x)  # [B, T, feat_dim]
        for blk in self.blocks:
            x = blk(x)
        # pool across time
        x = x.transpose(1, 2)  # [B, feat_dim, T]
        x = self.pool(x).squeeze(-1)  # [B, feat_dim]
        x = self.out_proj(x)
        x = hyperspherical_normalize(x)
        return x


class TemporalConvEncoder(nn.Module):
    """
    TemporalConvEncoder: stack of 1D conv layers across time dimension.
    Input: [B, T, D] (or [B, D] treated as T=1).
    Output: [B, feat_dim]
    """

    def __init__(self, input_dim: int, feat_dim: int = 256, seq_len: int = 4, n_layers: int = 3, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, feat_dim)
        convs = []
        in_ch = feat_dim
        for i in range(n_layers):
            convs.append(nn.Conv1d(in_ch, feat_dim, kernel_size=kernel_size, padding=kernel_size // 2))
            convs.append(nn.ReLU())
            in_ch = feat_dim
        self.conv_net = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 2:
            x = obs.unsqueeze(1).repeat(1, self.seq_len, 1)
        elif obs.ndim == 3:
            x = obs
            if x.size(1) != self.seq_len:
                if x.size(1) > self.seq_len:
                    x = x[:, : self.seq_len, :]
                else:
                    last = x[:, -1:, :].expand(-1, self.seq_len - x.size(1), -1)
                    x = torch.cat([x, last], dim=1)
        else:
            raise ValueError("obs must be 2D or 3D")
        x = self.input_proj(x)  # [B, T, feat_dim]
        x = x.transpose(1, 2)  # [B, feat_dim, T]
        x = self.conv_net(x)
        x = self.pool(x).squeeze(-1)
        x = self.out_proj(x)
        x = self.dropout(x)
        x = hyperspherical_normalize(x)
        return x


# -------------------------
# Ensemble distributional critic
# -------------------------
class EnsembleDistributionalCritic(nn.Module):
    """
    Ensemble wrapper around DistributionalQNetwork heads.
    Returns stacked logits [B, n_heads, num_atoms].
    """

    def __init__(self, use_static_encoder, base_q_cls, n_heads: int, q_kwargs: dict, critic_dropout: float = 0.0, use_layernorm: bool = False):
        super().__init__()
        encoder_feat = q_kwargs.pop("encoder_feat")
        if use_static_encoder:
            q_kwargs["n_obs"] = encoder_feat  #
            print(f"  encoder_feat(编码后维度): {encoder_feat}")
        assert n_heads >= 1
        self.n_heads = n_heads

        print(f"[Debug] EnsembleDistributionalCritic初始化:")
        print(f"  n_heads(评论家数量): {n_heads}")
        print(f"  q_kwargs(n_obs等参数): {q_kwargs}\n")
        print(f"[Debug] EnsembleDistributionalCritic 头初始化:")
        print(f"  传递给每个 head 的 q_kwargs 中 n_obs: {q_kwargs['n_obs']}")  
        self.heads = nn.ModuleList([base_q_cls(**q_kwargs) for _ in range(n_heads)])
        self.dropout = nn.Dropout(critic_dropout) if critic_dropout > 0.0 else None
        self.use_layernorm = use_layernorm
        if use_layernorm:
            hidden_dim = q_kwargs.get("hidden_dim", None)
            if hidden_dim is None:
                raise ValueError("hidden_dim must be provided in q_kwargs when using layernorm")
            self.layernorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_heads)])
        else:
            self.layernorms = None
        # assume first head defines support (tensor) attribute
        self.register_buffer("q_support", self.heads[0].q_support.clone().to(device="cuda:0"))

    def forward(self, obs_feat: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        obs_feat: [B, feat_dim]  (already encoded/pool)
        actions: [B, act_dim]
        returns: logits [B, n_heads, num_atoms]
        """
        logits_list = []
        for idx, head in enumerate(self.heads):
            try:
                # Many DistributionalQNetwork implementations accept (obs, action) and return logits
                logits = head.forward(obs_feat, actions).clone()
            except TypeError:
                if not hasattr(head, "embedder") or head.embedder is None:
                    raise RuntimeError(f"Head {idx} has no valid embedder; cannot use fallback logic")
                x = torch.cat([obs_feat, actions], dim=-1)  
                z = head.embedder(x)
                if self.use_layernorm:
                    z = self.layernorms[idx](z)
                if self.dropout is not None:
                    z = self.dropout(z)
                if not hasattr(head, "encoder") or head.encoder is None:
                    raise RuntimeError(f"Head {idx} has no valid encoder; cannot use fallback logic")
                z = head.encoder(z)
                if not hasattr(head, "predictor") or head.predictor is None:
                    raise RuntimeError(f"Head {idx} has no valid predictor; cannot use fallback logic")
                logits = head.predictor(z)
            logits_list.append(logits)  
        logits_all = torch.stack(logits_list, dim=1)  
        return logits_all 

    def projection(self, obs_feat: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, bootstrap: torch.Tensor, discount: float):
        """
        Compute projected distributions per head; returns tuple for compatibility (head0_proj, head1_proj)
        """
        proj_list = []
        for head in self.heads:
            # use head.projection if it exists
            try:
                proj = head.projection(obs_feat, actions, rewards, bootstrap, discount, head.q_support, head.q_support.device)
            except Exception:
                # best-effort fallback: run forward + categorical projection externally in train loop
                raise RuntimeError("DistributionalQNetwork.projection not available for head; please use base class with projection")
            proj_list.append(proj.unsqueeze(1))
        proj_all = torch.cat(proj_list, dim=1)  
        if self.n_heads >= 2:
            values_all = self.get_value(proj_all) 
            min_head_idx = torch.argmin(values_all, dim=1, keepdim=True) 
            min_proj = torch.gather(proj_all, 1, min_head_idx.unsqueeze(-1).expand(-1, -1, proj_all.size(-1)))[:, 0]  
            mean_proj = torch.mean(proj_all, dim=1) 
            max_head_idx = torch.argmax(values_all, dim=1, keepdim=True)  # [B, 1]
            max_proj = torch.gather(proj_all, 1, max_head_idx.unsqueeze(-1).expand(-1, -1, proj_all.size(-1)))[:, 0]  

            #kalman
            mu = torch.mean(proj_all, dim=1)
            sigma = torch.std(proj_all, dim=1)
            w = 1 / (sigma + 1e-6)
            mu_k = torch.sum(mu * w, dim=1) / torch.sum(w, dim=1)
            #mu_k.unsqueeze(1), mu_k.unsqueeze(1)

            # kalmanv2 
            # probs_all = F.softmax(proj_all, dim=-1)      
            # means = (probs_all * support).sum(dim=-1)       
            # vars = (probs_all * (support - means.unsqueeze(-1))**2).sum(dim=-1) 
            # inv_vars = 1.0 / (vars + 1e-6)                 
            # alphas = inv_vars / inv_vars.sum(dim=1, keepdim=True)  
            # alphas = alphas.unsqueeze(-1)                    
            # probs_weighted = (alphas * probs_all).sum(dim=1) 
            # logits_kf = torch.log(probs_weighted + 1e-12)    
            # mu_kf = (probs_weighted * support).sum(dim=-1)   
            #logits_kf, logits_kf  

            return mu_k.unsqueeze(1), mu_k.unsqueeze(1) 
        else:
            return proj_all[:, 0, :], proj_all[:, 0, :]

    def get_value(self, probs_or_logits: torch.Tensor) -> torch.Tensor:
        while probs_or_logits.ndim > 3 and 1 in probs_or_logits.shape:
            probs_or_logits = probs_or_logits.squeeze()  
        if probs_or_logits.ndim == 3 and probs_or_logits.size(1) == 1:
            probs_or_logits = probs_or_logits.squeeze(1)  
        if probs_or_logits.ndim == 3:
            probs = F.softmax(probs_or_logits, dim=-1) if torch.sum(probs_or_logits**2, dim=-1).mean() > 1.5 else probs_or_logits 
            vals = torch.sum(probs * self.q_support.view(1, 1, -1), dim=-1)
            return vals
        elif probs_or_logits.ndim == 2:
            probs = F.softmax(probs_or_logits, dim=-1) if torch.sum(probs_or_logits**2, dim=-1).mean() > 1.5 else probs_or_logits
            vals = torch.sum(probs * self.q_support.view(1, -1), dim=-1)
            return vals
        elif probs_or_logits.ndim == 1:
            probs = F.softmax(probs_or_logits, dim=-1) if probs_or_logits.var() > 1.0 else probs_or_logits.unsqueeze(0)
            vals = torch.sum(probs * self.q_support.view(1, -1), dim=-1).squeeze(0)
            return vals
        else:
            raise ValueError(f"Unexpected dimension after squeeze: ndim={probs_or_logits.ndim}, shape={probs_or_logits.shape}")
    def projection_raw(self, obs_feat, actions, rewards, bootstrap, discount):
        proj_list = []
        for head in self.heads:
            proj = head.projection(obs_feat, actions, rewards, bootstrap, discount, head.q_support, head.q_support.device).clone().to(head.q_support.device) 
            proj_list.append(proj.unsqueeze(1))
        return torch.cat(proj_list, dim=1)

# -------------------------
# Actor & Critic classes
# -------------------------
class HypersphericalActor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int,
        c_shift: float,
        num_blocks: int,
        device: torch.device = None,
        # encoder options
        use_static_encoder: bool = True,
        static_encoder_type: str = "gatedmlp",  # 'gatedmlp' or 'conv'
        seq_len: int = 4,
        encoder_feat: int = 256,
        encoder_blocks: int = 2,
        encoder_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_act = n_act
        self.device = device
        self.use_static_encoder = use_static_encoder

        if self.use_static_encoder:
            if static_encoder_type == "gatedmlp":
                self.temporal_encoder = GatedMLPEncoder(input_dim=n_obs, feat_dim=encoder_feat, seq_len=seq_len, n_blocks=encoder_blocks, dropout=encoder_dropout)
            else:
                self.temporal_encoder = TemporalConvEncoder(input_dim=n_obs, feat_dim=encoder_feat, seq_len=seq_len, n_layers=encoder_blocks, dropout=encoder_dropout)
            embed_in = encoder_feat
        else:
            embed_in = n_obs

        # Use SimbaV2 HyperEmbedder for downstream blocks
        self.embedder = HyperEmbedder(in_dim=embed_in, hidden_dim=hidden_dim, scaler_init=scaler_init, scaler_scale=scaler_scale, c_shift=c_shift, device=device)
        self.encoder = nn.Sequential(*[HyperLERPBlock(hidden_dim=hidden_dim, scaler_init=scaler_init, scaler_scale=scaler_scale, alpha_init=alpha_init, alpha_scale=alpha_scale, expansion=expansion, device=device) for _ in range(num_blocks)])
        self.predictor = HyperTanhPolicy(hidden_dim=hidden_dim, action_dim=n_act, scaler_init=1.0, scaler_scale=1.0, device=device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs may be [B, obs_dim] or [B, T, obs_dim]
        if self.use_static_encoder:
            enc = self.temporal_encoder(obs)  # [B, feat_dim]
        else:
            enc = obs
        z = self.embedder(enc)
        z = self.encoder(z)
        z = l2normalize(z, axis=-1)
        a = self.predictor(z)
        return a

    def explore(self, obs: torch.Tensor, dones: Optional[torch.Tensor] = None, deterministic: bool = False) -> torch.Tensor:
        a = self.forward(obs)
        if deterministic:
            return a
        # add simple Gaussian noise
        noise = torch.randn_like(a) * 0.1
        return a + noise


class HypersphericalCritic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        num_blocks: int,
        c_shift: float,
        expansion: int,
        device: torch.device = None,
        # ensemble options
        num_critics: int = 2,
        critic_dropout: float = 0.0,
        critic_layernorm: bool = False,
        use_static_encoder: bool = False,
        static_encoder_type: str = "gatedmlp",
        seq_len: int = 4,
        encoder_feat: int = 256,
        encoder_blocks: int = 2,
    ):
        super().__init__()
        self.device = device
        self.num_critics = max(1, int(num_critics))
        self.use_static_encoder = use_static_encoder
        self.static_encoder_type = static_encoder_type
        self.encoder_feat = encoder_feat 
        if self.use_static_encoder:
            if static_encoder_type == "gatedmlp":
                self.temporal_encoder = GatedMLPEncoder(input_dim=n_obs, feat_dim=encoder_feat, seq_len=seq_len, n_blocks=encoder_blocks, dropout=critic_dropout)
            else:
                self.temporal_encoder = TemporalConvEncoder(input_dim=n_obs, feat_dim=encoder_feat, seq_len=seq_len, n_layers=encoder_blocks, dropout=critic_dropout)
            q_in = encoder_feat + n_act
            print(f"Using static encoder, q_in = {q_in}")  
        else:
            self.temporal_encoder = None
            q_in = n_obs + n_act
        print(f"[Debug] HypersphericalCritic 输入维度确认:")
        print(f"  use_static_encoder: {self.use_static_encoder}")
        print(f"  n_obs: {n_obs}, n_act: {n_act}, encoder_feat: {encoder_feat}")
        print(f"  计算得到的 q_in (观测+动作维度): {q_in}")  
        base_q_kwargs = dict(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            num_blocks=num_blocks,
            c_shift=c_shift,
            expansion=expansion,
            device=device,
            encoder_feat=encoder_feat, 
        )
        self.ensemble = EnsembleDistributionalCritic(use_static_encoder=self.use_static_encoder, base_q_cls=DistributionalQNetwork, n_heads=self.num_critics, q_kwargs=base_q_kwargs, critic_dropout=critic_dropout, use_layernorm=critic_layernorm)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.temporal_encoder is not None:
            enc = self.temporal_encoder(obs)
            obs_feat = enc
        else:
            obs_feat = obs
        logits_all = self.ensemble(obs_feat, actions).clone() 
        if self.num_critics == 1:
            return logits_all[:, 0, :], logits_all[:, 0, :]
        else:
            probs_all = F.softmax(logits_all, dim=-1)
            values_all = torch.sum(probs_all * self.ensemble.q_support.view(1, 1, -1), dim=-1)  
            argmin_idx = torch.argmin(values_all, dim=1)  
            min_head_idx = argmin_idx.squeeze(-1) if argmin_idx.dim() > 1 else argmin_idx  
            batch_idx = torch.arange(logits_all.size(0), device=logits_all.device) 
            qf1_min = logits_all[batch_idx, min_head_idx]  
            qf2_mean = torch.mean(logits_all, dim=1)  
            argmax_idx = torch.argmax(values_all, dim=1)  
            max_head_idx = argmax_idx.squeeze(-1) if argmax_idx.dim() > 1 else argmax_idx  
            qf2_max = logits_all[batch_idx, max_head_idx] 

            #kalman
            mu = torch.mean(logits_all, dim=1)
            sigma = torch.std(logits_all, dim=1)
            w = 1 / (sigma + 1e-6)
            mu_k = torch.sum(mu * w, dim=1) / torch.sum(w, dim=1)
            #mu_k.unsqueeze(1), mu_k.unsqueeze(1)      

            # kalmanv2  
            # probs_all = F.softmax(logits_all, dim=-1)        
            # means = (probs_all * support).sum(dim=-1)       
            # vars = (probs_all * (support - means.unsqueeze(-1))**2).sum(dim=-1)  
            # inv_vars = 1.0 / (vars + 1e-6)                   
            # alphas = inv_vars / inv_vars.sum(dim=1, keepdim=True) 
            # alphas = alphas.unsqueeze(-1)                    
            # probs_weighted = (alphas * probs_all).sum(dim=1) 
            # logits_kf = torch.log(probs_weighted + 1e-12)    
            # mu_kf = (probs_weighted * support).sum(dim=-1)  
            #logits_kf, logits_kf  
            #             
            return mu_k.unsqueeze(1), mu_k.unsqueeze(1)

    def projection(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, bootstrap: torch.Tensor, discount: float):
        if self.temporal_encoder is not None:
            enc = self.temporal_encoder(obs)
            return self.ensemble.projection(enc, actions, rewards, bootstrap, discount)
        else:
            return self.ensemble.projection(obs, actions, rewards, bootstrap, discount)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        return self.ensemble.get_value(probs)


# -------------------------
# Factory helpers
# -------------------------
def make_actor_class(**default_kwargs):
    def actor_fn(**override_kwargs):
        kw = dict(default_kwargs)
        kw.update(override_kwargs)
        return HypersphericalActor(**kw)
    return actor_fn


def make_critic_class(**default_kwargs):
    def critic_fn(**override_kwargs):
        kw = dict(default_kwargs)
        kw.update(override_kwargs)
        return HypersphericalCritic(**kw)
    return critic_fn


# -------------------------
# Smoke-test
# -------------------------
if __name__ == "__main__":
    B = 4
    obs_dim = 24
    act_dim = 6
    seq_len = 4
    obs = torch.randn(B, seq_len, obs_dim)
    act = torch.randn(B, act_dim)
    actor = HypersphericalActor(
        n_obs=obs_dim,
        n_act=act_dim,
        num_envs=1,
        hidden_dim=128,
        scaler_init=1.0,
        scaler_scale=1.0,
        alpha_init=0.5,
        alpha_scale=0.1,
        expansion=4,
        c_shift=3.0,
        num_blocks=2,
        device=torch.device("cpu"),
        use_static_encoder=True,
        static_encoder_type="gatedmlp",
        seq_len=seq_len,
        encoder_feat=128,
        encoder_blocks=2,
    )
    critic = HypersphericalCritic(
        n_obs=obs_dim,
        n_act=act_dim,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        hidden_dim=128,
        scaler_init=1.0,
        scaler_scale=1.0,
        alpha_init=0.5,
        alpha_scale=0.1,
        num_blocks=2,
        c_shift=3.0,
        expansion=4,
        device=torch.device("cpu"),
        num_critics=3,
        critic_dropout=0.1,
        critic_layernorm=False,
        use_static_encoder=False,
        static_encoder_type="gatedmlp",
        seq_len=seq_len,
        encoder_feat=128,
        encoder_blocks=2,
    )
    logits1, logits2 = critic(obs, act)
    print("logits1", logits1.shape, "logits2", logits2.shape)
    a = actor(obs)
    print("actor out", a.shape)
