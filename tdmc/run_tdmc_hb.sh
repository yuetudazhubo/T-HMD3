    #!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOSMesa.so.8:/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR=/tmp/xdg_runtime
mkdir -p /tmp/xdg_runtime && chmod 700 /tmp/xdg_runtime
unset DISPLAY
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355
PROJECT="TDMCTD3"
python tdmc/train.py \
    --project ${PROJECT} \
    --env_name h1hand-hurdle-v0 \
    --exp_name TDMCTD3_nums2_shaping_kalman \
    --render_interval 5000 \
    --agent tdmc \
    --batch_size 8192 \
    --critic_learning_rate_end 3e-5 \
    --actor_learning_rate_end 3e-5 \
    --weight_decay 0.0 \
    --critic_hidden_dim 1024 \
    --critic_num_blocks 2 \
    --actor_hidden_dim 256 \
    --actor_num_blocks 1 \
    --seed 1 \
    --policy_frequency 2 \
    --critic_dropout 0.1 \
    --tau 0.1 \
    --num_updates 2 \
    --teacher_critic_path models/teacher_critic_only.pt \
    --teacher_mode shaping \
    --teacher_weight 1.0 \
    --teacher_loss_type mse \
    --teacher_threshold 0.05 \
    --teacher_window 5000 \
