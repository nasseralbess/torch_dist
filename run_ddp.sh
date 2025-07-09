#!/bin/bash

# === Set environment variables for NCCL and DDP ===
export NCCL_SOCKET_IFNAME=eno1              # Or your actual NIC name
export NCCL_PORT_RANGE=45000-46000          # Limit port range to avoid conflicts
export NCCL_IB_DISABLE=1                    # Disable InfiniBand if not used
export NCCL_DEBUG=INFO                      # Enable detailed NCCL logging

# === Set DDP-specific variables ===
export RANK=$1
export WORLD_SIZE=$2
export LOCAL_RANK=$3

# === Run your Python training script ===
python3 multi_node.py 50 10
