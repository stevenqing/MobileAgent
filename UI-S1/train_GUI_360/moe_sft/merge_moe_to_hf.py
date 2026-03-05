"""
Merge MoE Expert LoRA into Base Model for HF-format Inference.

Since MoE v1 suffered router collapse (expert_1 weight ≈ 1.0), we can merge
just expert_1's LoRA weights into the base model to get a standard HF model
that vLLM can serve directly.

Merge formula per Linear layer:
    weight_merged = weight_base + (lora_B @ lora_A) * (alpha / r)

Usage:
    python merge_moe_to_hf.py [--base_model ...] [--moe_checkpoint ...] [--output_dir ...]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def merge_expert_lora(
    base_model_path: str,
    moe_checkpoint_dir: str,
    output_dir: str,
    expert_idx: int = 1,
):
    """
    Merge a single expert's LoRA weights into the base model and save as HF format.

    Args:
        base_model_path: Path to base Qwen2.5-VL-7B-Instruct
        moe_checkpoint_dir: Path to MoE checkpoint (containing experts/ and moe_config.json)
        output_dir: Path to save merged HF model
        expert_idx: Which expert to merge (default: 1, the collapsed dominant expert)
    """
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"MoE checkpoint: {moe_checkpoint_dir}")
    logger.info(f"Expert to merge: expert_{expert_idx}")
    logger.info(f"Output: {output_dir}")

    # --- Load MoE config ---
    moe_config_path = os.path.join(moe_checkpoint_dir, "moe_config.json")
    with open(moe_config_path, "r") as f:
        moe_config = json.load(f)

    lora_r = moe_config["expert_lora_r"]
    lora_alpha = moe_config["expert_lora_alpha"]
    scaling = lora_alpha / lora_r
    logger.info(f"LoRA r={lora_r}, alpha={lora_alpha}, scaling={scaling}")

    # --- Load expert LoRA weights ---
    expert_dir = os.path.join(moe_checkpoint_dir, "experts", f"expert_{expert_idx}")
    adapter_path = os.path.join(expert_dir, "adapter_model.bin")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Expert adapter not found: {adapter_path}")

    lora_state_dict = torch.load(adapter_path, map_location="cpu")
    logger.info(f"Loaded expert_{expert_idx}: {len(lora_state_dict)} weight tensors")

    # Group LoRA A/B pairs by module
    lora_pairs = {}
    for key in lora_state_dict:
        if ".lora_A.weight" in key:
            base_key = key.replace(".lora_A.weight", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["A"] = lora_state_dict[key]
        elif ".lora_B.weight" in key:
            base_key = key.replace(".lora_B.weight", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["B"] = lora_state_dict[key]

    logger.info(f"Found {len(lora_pairs)} LoRA module pairs to merge")

    # --- Load base model ---
    # Qwen2.5-VL is a Vision2Seq model, not a pure CausalLM
    logger.info("Loading base model...")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    except (ValueError, KeyError):
        # Fallback for non-VL models
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    # --- Merge LoRA weights ---
    logger.info("Merging LoRA weights into base model...")
    merged_count = 0

    for peft_key, lora_ab in lora_pairs.items():
        if "A" not in lora_ab or "B" not in lora_ab:
            logger.warning(f"Incomplete LoRA pair for {peft_key}, skipping")
            continue

        lora_A = lora_ab["A"]  # [r, in_features]
        lora_B = lora_ab["B"]  # [out_features, r]

        # Convert PEFT key to model key
        # PEFT: "base_model.model.model.layers.0.self_attn.q_proj"
        # Strip prefix: "model.layers.0.self_attn.q_proj"
        model_key = peft_key
        if model_key.startswith("base_model.model."):
            model_key = model_key[len("base_model.model."):]

        # Try multiple key patterns for different model architectures
        # Qwen2.5-VL via AutoModelForVision2Seq: "model.language_model.layers.X..."
        # Qwen2.5-VL via Qwen2_5_VLForConditionalGeneration: "model.layers.X..."
        candidates = [
            model_key + ".weight",
            model_key.replace("model.layers.", "model.language_model.layers.") + ".weight",
        ]

        # Find the parameter in model
        param = None
        matched_key = None
        for name, p in model.named_parameters():
            if name in candidates:
                param = p
                matched_key = name
                break

        if param is None:
            logger.warning(f"Could not find {candidates} in model, skipping")
            continue

        # Merge: weight += (lora_B @ lora_A) * scaling
        delta_W = (lora_B.to(param.dtype) @ lora_A.to(param.dtype)) * scaling
        with torch.no_grad():
            param.add_(delta_W)

        merged_count += 1

    logger.info(f"Merged {merged_count}/{len(lora_pairs)} LoRA modules")

    # --- Save merged model ---
    logger.info(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)

    # Save merge metadata
    merge_info = {
        "base_model": base_model_path,
        "moe_checkpoint": moe_checkpoint_dir,
        "merged_expert": expert_idx,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "scaling": scaling,
        "merged_modules": merged_count,
        "note": "Router collapsed in v1 training, expert_1 weight ≈ 1.0, so merging expert_1 only",
    }
    with open(os.path.join(output_dir, "merge_info.json"), "w") as f:
        json.dump(merge_info, f, indent=2)

    logger.info("Merge complete!")
    logger.info(f"Merged model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge MoE expert LoRA into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="checkpoints/Qwen2.5-VL-7B-Instruct",
        help="Path to base model",
    )
    parser.add_argument(
        "--moe_checkpoint",
        type=str,
        default="train_GUI_360/moe_sft/output/moe_sft_v1/final",
        help="Path to MoE checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train_GUI_360/moe_sft/output/moe_v1_merged",
        help="Path to save merged model",
    )
    parser.add_argument(
        "--expert_idx",
        type=int,
        default=1,
        help="Expert index to merge (default: 1)",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    if not os.path.isabs(args.base_model):
        args.base_model = str(project_root / args.base_model)
    if not os.path.isabs(args.moe_checkpoint):
        args.moe_checkpoint = str(project_root / args.moe_checkpoint)
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(project_root / args.output_dir)

    merge_expert_lora(args.base_model, args.moe_checkpoint, args.output_dir, args.expert_idx)


if __name__ == "__main__":
    main()
