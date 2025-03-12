#!/bin/bash

# Define the IP addresses of all machines
# Replace these with your actual machine IPs
MACHINES=(
    "192.168.1.100"  # Main PC (node 0)
    "192.168.1.101"  # Node 1
    "192.168.1.102"  # Node 2
    "192.168.1.103"  # Node 3
    "192.168.1.104"  # Node 4
    "192.168.1.105"  # Node 5
    "192.168.1.106"  # Node 6
    "192.168.1.107"  # Node 7
    "192.168.1.108"  # Node 8
    "192.168.1.109"  # Node 9
)

# Username for SSH
USERNAME="your_username"  # Replace with your actual username

# Create a directory for the project
PROJECT_DIR="~/bart-uzbek-finetuning"

# Local path to the Python script and launch script
PYTHON_SCRIPT="bart_finetune.py"
LAUNCH_SCRIPT="launch_training.sh"

echo "Setting up distributed training environment..."

# Loop through all machines and set up the environment
for i in "${!MACHINES[@]}"; do
    MACHINE=${MACHINES[$i]}
    echo "Setting up machine $i: $MACHINE"
    
    # Create project directory
    ssh $USERNAME@$MACHINE "mkdir -p $PROJECT_DIR"
    
    # Copy scripts to the machine
    scp $PYTHON_SCRIPT $USERNAME@$MACHINE:$PROJECT_DIR/
    scp $LAUNCH_SCRIPT $USERNAME@$MACHINE:$PROJECT_DIR/
    
    # Make launch script executable
    ssh $USERNAME@$MACHINE "chmod +x $PROJECT_DIR/$LAUNCH_SCRIPT"
    
    # Install dependencies if needed (uncomment if you need to install dependencies on each machine)
    # ssh $USERNAME@$MACHINE "cd $PROJECT_DIR && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install transformers accelerate deepspeed bitsandbytes datasets"
    
    echo "Setup completed for machine $i"
done

echo "All machines set up successfully!"
echo "To start training, run the following command on each machine:"
echo "cd $PROJECT_DIR && ./$LAUNCH_SCRIPT <node_rank>"
echo "Where <node_rank> is the rank of the machine (0 for the main PC, 1-9 for the other machines)"