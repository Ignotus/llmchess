import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.training_args import (
    OptimizerNames,
    SchedulerType,
)
from transformers.trainer_utils import SaveStrategy

from trl import (
    SFTConfig,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from peft.utils import TaskType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", default="google/gemma-3-1b-pt", choices=["google/gemma-3-1b-pt"]
    )
    parser.add_argument("--data-file", default="data/train_data.json")
    parser.add_argument("--output-dir", default="output/")
    parser.add_argument("--max-seq-len", default=512, type=int)
    args = parser.parse_args()

    with open(args.data_file, "r") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_len)

    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, num_proc=1, remove_columns=["text"]
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=[
            "down_proj",
            "o_proj",
            "k_proj",
            "q_proj",
            "gate_proj",
            "up_proj",
            "v_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template="Next Chess Move:",
        tokenizer=tokenizer,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        lr_scheduler_type=SchedulerType.COSINE,
        optim=OptimizerNames.PAGED_ADAMW_8BIT,
        logging_steps=10,
        learning_rate=2e-5,
        fp16=False,
        bf16=True,
        save_strategy=SaveStrategy.EPOCH,
        do_train=True,
        report_to=["tensorboard"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
