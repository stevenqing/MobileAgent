"""
Copy Expert Init: Fix Router Collapse from MoE SFT v1.

MoE SFT v1 (Job 2610726) suffered router collapse:
  - expert_1 weight ≈ 0.9999, others ≈ 1e-6
  - routing_entropy ≈ 0.0002
  - Root cause: balance_weight=0.0, no router constraint

Fix strategy:
  1. Copy expert_1's trained LoRA weights to all other experts (0, 2, 3, 4, 5)
  2. Re-initialize router with small random weights + uniform bias
  3. Save as new checkpoint for v2 training with balance loss

Usage:
    python copy_expert_init.py [--source_dir ...] [--output_dir ...]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def copy_expert_init(source_dir: str, output_dir: str, source_expert: int = 1):
    """
    Copy source expert's LoRA weights to all other experts and reset router.

    Args:
        source_dir: Path to v1 final checkpoint (contains router.pt, experts/, moe_config.json)
        output_dir: Path to save the new copy-init checkpoint
        source_expert: Which expert to copy from (default: 1, the collapsed dominant expert)
    """
    logger.info(f"Source checkpoint: {source_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Source expert: expert_{source_expert}")

    # --- Validate source checkpoint ---
    moe_config_path = os.path.join(source_dir, "moe_config.json")
    router_path = os.path.join(source_dir, "router.pt")
    experts_dir = os.path.join(source_dir, "experts")

    for path, name in [(moe_config_path, "moe_config.json"), (router_path, "router.pt"), (experts_dir, "experts/")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} at {path}")

    # Load MoE config
    with open(moe_config_path, "r") as f:
        moe_config = json.load(f)

    num_experts = moe_config["num_experts"]
    logger.info(f"Number of experts: {num_experts}")

    # --- Load source expert weights ---
    source_expert_dir = os.path.join(experts_dir, f"expert_{source_expert}")
    source_adapter_path = os.path.join(source_expert_dir, "adapter_model.bin")
    if not os.path.exists(source_adapter_path):
        raise FileNotFoundError(f"Source expert not found: {source_adapter_path}")

    source_weights = torch.load(source_adapter_path, map_location="cpu")
    logger.info(f"Loaded source expert_{source_expert}: {len(source_weights)} weight tensors")

    # Log some stats about source weights
    total_params = sum(v.numel() for v in source_weights.values())
    logger.info(f"Source expert parameters: {total_params:,}")

    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)
    out_experts_dir = os.path.join(output_dir, "experts")
    os.makedirs(out_experts_dir, exist_ok=True)

    # --- Copy source expert weights to all experts ---
    for expert_idx in range(num_experts):
        expert_out_dir = os.path.join(out_experts_dir, f"expert_{expert_idx}")
        os.makedirs(expert_out_dir, exist_ok=True)

        # Deep copy weights
        copied_weights = {k: v.clone() for k, v in source_weights.items()}
        torch.save(copied_weights, os.path.join(expert_out_dir, "adapter_model.bin"))

        # Copy adapter config from source
        source_config_path = os.path.join(source_expert_dir, "adapter_config.json")
        if os.path.exists(source_config_path):
            with open(source_config_path, "r") as f:
                adapter_config = json.load(f)
            with open(os.path.join(expert_out_dir, "adapter_config.json"), "w") as f:
                json.dump(adapter_config, f, indent=2)

        if expert_idx == source_expert:
            logger.info(f"  expert_{expert_idx}: copied (source)")
        else:
            logger.info(f"  expert_{expert_idx}: copied from expert_{source_expert}")

    # --- Re-initialize router for uniform routing ---
    logger.info("Re-initializing router...")
    router_state = torch.load(router_path, map_location="cpu")

    # Examine router structure
    logger.info("Original router keys:")
    for k, v in router_state.items():
        logger.info(f"  {k}: shape={v.shape}, mean={v.mean().item():.6f}, std={v.std().item():.6f}")

    # Reset router weights: small random + uniform bias
    new_router_state = {}
    for k, v in router_state.items():
        if "bias" in k and v.shape[0] == num_experts:
            # Output bias → uniform (zeros give equal logits)
            new_router_state[k] = torch.zeros_like(v)
            logger.info(f"  {k}: reset to zeros (uniform bias)")
        elif "weight" in k and v.shape[0] == num_experts:
            # Output weight → small random for symmetry breaking
            new_val = torch.randn_like(v) * 0.01
            new_router_state[k] = new_val
            logger.info(f"  {k}: reset to small random (std=0.01)")
        elif "weight" in k:
            # Hidden layer weights → Xavier init with small gain
            new_val = torch.empty_like(v)
            nn.init.xavier_uniform_(new_val, gain=0.1)
            new_router_state[k] = new_val
            logger.info(f"  {k}: reset to Xavier uniform (gain=0.1)")
        elif "bias" in k:
            # Hidden layer bias → zeros
            new_router_state[k] = torch.zeros_like(v)
            logger.info(f"  {k}: reset to zeros")
        else:
            # Keep as is (unlikely, but safe)
            new_router_state[k] = v.clone()
            logger.info(f"  {k}: kept unchanged")

    torch.save(new_router_state, os.path.join(output_dir, "router.pt"))
    logger.info("Router re-initialized and saved")

    # --- Copy MoE config ---
    with open(os.path.join(output_dir, "moe_config.json"), "w") as f:
        json.dump(moe_config, f, indent=2)
    logger.info("MoE config copied")

    # --- Verification ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Verification")
    logger.info("=" * 60)

    # Verify all experts have identical weights
    expert_0_path = os.path.join(out_experts_dir, "expert_0", "adapter_model.bin")
    ref_weights = torch.load(expert_0_path, map_location="cpu")

    all_match = True
    for expert_idx in range(1, num_experts):
        other_path = os.path.join(out_experts_dir, f"expert_{expert_idx}", "adapter_model.bin")
        other_weights = torch.load(other_path, map_location="cpu")

        for key in ref_weights:
            if not torch.equal(ref_weights[key], other_weights[key]):
                logger.error(f"MISMATCH: expert_0 vs expert_{expert_idx} at {key}")
                all_match = False

    if all_match:
        logger.info(f"All {num_experts} experts have identical weights ✓")
    else:
        logger.error("Expert weight verification FAILED!")
        return False

    # Verify router outputs near-uniform distribution
    new_router = torch.load(os.path.join(output_dir, "router.pt"), map_location="cpu")
    # Simulate: with near-zero weights and zero bias, softmax should be near-uniform
    # Use the output layer to check
    for k, v in new_router.items():
        if v.shape[0] == num_experts and "weight" in k:
            # Simulate routing with random input
            test_input = torch.randn(10, v.shape[1])
            logits = test_input @ v.t()
            # Add bias if available
            bias_key = k.replace("weight", "bias")
            if bias_key in new_router:
                logits = logits + new_router[bias_key]
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            max_entropy = torch.log(torch.tensor(float(num_experts)))
            logger.info(f"Simulated routing entropy: {entropy:.4f} (max: {max_entropy:.4f})")
            logger.info(f"Normalized entropy: {entropy / max_entropy:.4f} (should be close to 1.0)")
            break

    logger.info("")
    logger.info(f"Copy-init checkpoint saved to: {output_dir}")
    logger.info("Ready for v2 training with balance loss.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Copy expert init for MoE v2")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="train_GUI_360/moe_sft/output/moe_sft_v1/final",
        help="Path to v1 final checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train_GUI_360/moe_sft/output/moe_sft_v1_copy_init",
        help="Path to save copy-init checkpoint",
    )
    parser.add_argument(
        "--source_expert",
        type=int,
        default=1,
        help="Expert index to copy from (default: 1, the collapsed dominant expert)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root if not absolute
    project_root = Path(__file__).resolve().parent.parent.parent
    if not os.path.isabs(args.source_dir):
        args.source_dir = str(project_root / args.source_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(project_root / args.output_dir)

    success = copy_expert_init(args.source_dir, args.output_dir, args.source_expert)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
