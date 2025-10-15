import argparse
import time

from transformers import (
    AutoTokenizer,
)
from peft import (
    AutoPeftModelForCausalLM,
)
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elo1", default=1500)
    parser.add_argument("--elo2", default=1400)
    parser.add_argument("--model-path", default="./output")

    args = parser.parse_args()

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")

    model = model.merge_and_unload()

    assert torch.cuda.is_available()

    device = torch.device("cuda")
    model.to(device)
    model.eval()
    print(model)

    elo1 = args.elo1
    elo2 = args.elo2
    moves = ["e4"]

    with torch.no_grad():
        for i in range(1, 50):
            moves_prompt = ""
            if moves:
                moves_prompt = f"Previous Chess Moves: {' '.join(moves)}\n"

            elo = elo1 if i % 2 == 0 else elo2
            prompt = f"{moves_prompt}Next Chess Player Elo: {elo}\nNext Chess Move: "
            t1 = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            output_ids = model.generate(
                **inputs, num_beams=5, do_sample=True, max_new_tokens=64, use_cache=True
            )

            next_move = tokenizer.decode(
                output_ids[0, len(inputs.input_ids[0]) :], skip_special_tokens=True
            ).split("<EOS>", 1)[0]

            t2 = time.time()
            print(f"Generation {t2 - t1:.2f} sec")

            moves.append(next_move)
            print(moves)


if __name__ == "__main__":
    main()
