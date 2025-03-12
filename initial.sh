#!/bin/bash

echo "🔧 Updating package lists..."
sudo apt update && sudo apt upgrade -y

echo "🛠️ Installing essential dependencies..."
sudo apt install -y wget git build-essential openssh-server

echo "🐍 Installing Miniconda for Python..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

echo "📦 Creating Python virtual environment..."
source $HOME/miniconda/bin/activate
conda create -n bart-ai python=3.10 -y
conda activate bart-ai

echo "🔢 Installing CUDA toolkit and dependencies..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda

echo "🔥 Installing PyTorch and torchrun..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate deepspeed bitsandbytes datasets

echo "🖧 Setting up SSH for communication..."
sudo systemctl enable ssh
sudo systemctl start ssh

# Generate SSH key (only if not already generated)
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
    echo "⚡ SSH key generated! Send the public key (~/.ssh/id_rsa.pub) to the master node."
fi

echo "✅ Worker setup complete! Now share your public key with the master node."
