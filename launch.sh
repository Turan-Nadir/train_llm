#!/bin/bash

# Define the IP addresses and ports of all machines
# Replace these with your actual machine IPs
MASTER_ADDR="192.168.1.100"  # IP of the main PC
MASTER_PORT="29500"          # Port for communication
NODE_RANK=$1                 # Will be passed as argument to this script
WORLD_SIZE=10                # Total number of machines (10 PCs)
NUM_GPUS=1                   # Number of GPUs per machine (assuming 1 RTX 4060 per PC)

# Create DeepSpeed config if it doesn't exist (only on master node)
if [ "$NODE_RANK" -eq 0 ]; then
    if [ ! -f "ds_config.json" ]; then
        echo "Creating DeepSpeed config file..."
        cat > ds_config.json << 'EOL'
{
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "steps_per_print": 100,
    "wall_clock_breakdown": false
}
EOL
    fi
fi

# Set environment variables for distributed training
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$NODE_RANK
export WORLD_SIZE=$WORLD_SIZE
export NCCL_SOCKET_IFNAME=eth0  # Change this to your network interface name if needed
export NCCL_DEBUG=INFO          # Set to INFO for debugging, WARN for production

# Launch the training using torchrun
torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NUM_GPUS \
    bart_finetune.py \
    --deepspeed \
    --deepspeed_config=ds_config.json \
    --epochs=3 \
    --batch_size=8 \
    --gradient_accumulation_steps=4 \
    --learning_rate=5e-5 \
    --max_length=512 \
    --split=lat \
    --output_dir=./bart-uzbek-latin-finetuned \
    --save_steps=500 \
    --logging_steps=50 \
    --evaluation_steps=500