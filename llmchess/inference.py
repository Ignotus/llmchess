from transformers import (
    AutoTokenizer,
)
from peft import (
    AutoPeftModelForCausalLM,
)
import torch


def main() -> None:
    MODEL_PATH = "./output"

    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    elo = 1500
    text = f"Last Player Elo: {elo}, Next Move: "
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=10,
            num_beams=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(generated_text)


if __name__ == "__main__":
    main()
