import argparse
import json
import logging
import sys
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import gather_object
from unsloth import FastLanguageModel
from transformers import set_seed

from config import PipelineConfig
from data.registry import DatasetRegistry
from prompts.factory import PromptBuilderFactory
from training.metrics import compute_ner_metrics, compute_re_metrics

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA adapter folder")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size per GPU")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save prediction results (e.g. predictions.jsonl)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize accelerate for Data Parallelism inference
    accelerator = Accelerator()
    
    # Only print logs from the main process
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    
    config = PipelineConfig.from_yaml(args.config)
    set_seed(config.model.seed)

    logger.info("Loading test data...")
    if not config.data.test_files:
        logger.warning("No test_files specified in config. Falling back to eval_files.")
        test_files = config.data.eval_files
    else:
        test_files = config.data.test_files
        
    eval_docs = DatasetRegistry(test_files).aggregate()
    logger.info("Total Test samples loaded: %d", len(eval_docs))

    builder = PromptBuilderFactory.create(config)
    eval_raw = builder.generate_training_data(eval_docs)
    
    # Shard the dataset across multiple GPUs
    if accelerator.num_processes > 1:
        eval_raw = eval_raw.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)
        logger.info(f"Process {accelerator.process_index} processing {len(eval_raw)} samples")

    logger.info(f"Loading model from {args.checkpoint} on GPU {accelerator.local_process_index}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=config.model.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map={"": accelerator.local_process_index},
    )
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

    predictions = []
    references = eval_raw["assistant"]
    
    logger.info("Running inference...")
    
    # Only show progress bar on main process
    for i in tqdm(range(0, len(eval_raw), args.batch_size), disable=not accelerator.is_main_process):
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
                prompt = text.split("### Output:\n")[0] + "### Output:\n"
                inputs_text.append(prompt)
                
        # Tokenize batch
        inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
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

    # Gather results from all GPUs to the main process
    all_predictions = gather_object(predictions)
    all_references = gather_object(references)

    # Compute Metrics and Save ONLY on the main process
    if accelerator.is_main_process:
        logger.info("Computing metrics...")
        
        if config.training.task_mode.value in ["joint", "ner_only"]:
            ner_res = compute_ner_metrics(all_predictions, all_references)
            logger.info(f"NER Metrics: {json.dumps(ner_res, indent=2)}")
            
        if config.training.task_mode.value in ["joint", "re_only"]:
            is_pairwise = (config.training.re_mode.value == "pairwise")
            re_res = compute_re_metrics(all_predictions, all_references, pairwise=is_pairwise)
            logger.info(f"RE Metrics: {json.dumps(re_res, indent=2)}")

        if args.output:
            logger.info(f"Saving predictions to {args.output} (JSONL format)...")
            with open(args.output, "w", encoding="utf-8") as f:
                for pred in all_predictions:
                    try:
                        parsed = json.loads(pred)
                        f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    except json.JSONDecodeError:
                        fallback = {"error": "Malformed JSON", "raw_text": pred}
                        f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
            logger.info("Predictions saved successfully!")

if __name__ == "__main__":
    main()
