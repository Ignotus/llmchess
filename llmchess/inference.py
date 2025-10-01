import argparse

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
        args.model_path, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")

    model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

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
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            output_ids = model.generate(
                input_ids,
                max_length=512,
                num_beams=5,
                do_sample=True,
            )

            next_move = tokenizer.decode(
                output_ids[0, len(input_ids[0]) :], skip_special_tokens=True
            ).split("<EOS>", 1)[0]

            moves.append(next_move)
            print(moves)


if __name__ == "__main__":
    main()
