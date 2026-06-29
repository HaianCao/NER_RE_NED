# -*- coding: utf-8 -*-
"""Post-training evaluation script.

Runs inference (generation) on the evaluation dataset and computes
Precision, Recall, and F1 metrics for NER and RE.

Usage::
    python evaluate.py --config configs/default.yaml --checkpoint /kaggle/working/outputs/final_lora_adapter
"""

import argparse
import json
import logging
import sys
from tqdm import tqdm

from unsloth import FastLanguageModel
from transformers import set_seed

from config import PipelineConfig
from data.registry import DatasetRegistry
from prompts.factory import PromptBuilderFactory
from training.metrics import compute_ner_metrics, compute_re_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA adapter folder")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save prediction results (e.g. predictions.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    set_seed(config.model.seed)

    logger.info("Loading test data...")
    if not config.data.test_files:
        logger.warning("No test_files specified in config. Falling back to eval_files.")
        test_files = config.data.eval_files
    else:
        test_files = config.data.test_files
        
    eval_docs = DatasetRegistry(test_files).aggregate()
    logger.info("Test samples loaded: %d", len(eval_docs))

    builder = PromptBuilderFactory.create(config)
    eval_raw = builder.generate_training_data(eval_docs)

    logger.info(f"Loading model from {args.checkpoint}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=config.model.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

    predictions = []
    references = eval_raw["assistant"]
    
    logger.info("Running inference...")
    
    for i in tqdm(range(0, len(eval_raw), args.batch_size)):
        batch = eval_raw[i:i+args.batch_size]
        
        # Prepare inputs based on format
        inputs_text = []
        if config.training.format.value == "instruction":
            for sys_msg, usr_msg in zip(batch["system"], batch["user"]):
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs_text.append(text)
        else:
            # Causal format (Input only)
            for text in batch["text"]:
                # Split at '### Output:\n' to get prompt only
                prompt = text.split("### Output:\n")[0] + "### Output:\n"
                inputs_text.append(prompt)
                
        # Tokenize batch
        inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the newly generated tokens
        input_lengths = [len(inp) for inp in inputs.input_ids]
        for out, in_len in zip(outputs, input_lengths):
            generated_tokens = out[in_len:]
            gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            predictions.append(gen_text)

    # Compute Metrics
    logger.info("Computing metrics...")
    
    if config.training.task_mode.value in ["joint", "ner_only"]:
        ner_res = compute_ner_metrics(predictions, references)
        logger.info(f"NER Metrics: {json.dumps(ner_res, indent=2)}")
        
    if config.training.task_mode.value in ["joint", "re_only"]:
        is_pairwise = (config.training.re_mode.value == "pairwise")
        re_res = compute_re_metrics(predictions, references, pairwise=is_pairwise)
        logger.info(f"RE Metrics: {json.dumps(re_res, indent=2)}")

    if args.output:
        logger.info(f"Saving predictions to {args.output} (JSONL format)...")
        with open(args.output, "w", encoding="utf-8") as f:
            for pred in predictions:
                try:
                    # Parse to ensure it's a valid dict, then dump to a single line
                    parsed = json.loads(pred)
                    f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                except json.JSONDecodeError:
                    # If the LLM generated invalid JSON, wrap it safely to not break JSONL
                    fallback = {"error": "Malformed JSON", "raw_text": pred}
                    f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
        logger.info("Predictions saved successfully!")

if __name__ == "__main__":
    main()
