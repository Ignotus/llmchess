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
    text = f"Previous Chess Moves: e4 e5 Bc4 Nc6 Nf3 Nf6 c3\nNext Chess Player Elo: {elo}\nNext Chess Move: "
    print(text)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            do_sample=True,
        )

    generated_text = tokenizer.decode(
        output_ids[0, len(input_ids[0]) :], skip_special_tokens=True
    )

    print(generated_text)


if __name__ == "__main__":
    main()
