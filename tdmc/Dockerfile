FROM nvcr.io/nvidia/pytorch:24.09-py3
RUN apt-get update && apt-get install -y libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev libegl1 libgles2-mesa-dev mesa-utils libgl1-mesa-dri libglew-dev patchelf libstdc++6 kmod pkg-config libglvnd-dev libvulkan1
RUN wget https://international.download.nvidia.com/tesla/535.104.12/NVIDIA-Linux-x86_64-535.104.12.run && 
sh NVIDIA-Linux-x86_64-535.104.12.run -s --no-kernel-module --ui=none --no-questions --glvnd-egl-config-path=/usr/share/glvnd/egl_vendor.d/ && 
rm NVIDIA-Linux-x86_64-535.104.12.run
RUN cd /usr/lib/x86_64-linux-gnu && ln -sf libglut.so.3.12.0 libglut.so.3  # 替换为实际版本
RUN pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 torchrl==0.7.2 PyOpenGL==3.1.6 PyOpenGL-accelerate==3.1.6
ENV MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/lib/x86_64-linux-gnu/dri NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video MASTER_PORT=12356
WORKDIR /202322801056/FastTD3-main
COPY . .