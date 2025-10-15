import argparse
import time
import re

from transformers import (
    AutoTokenizer,
)
from peft import (
    AutoPeftModelForCausalLM,
)
import torch
import chess
import chess.svg
import io
from cairosvg import svg2png
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    generate_enforced,
    build_token_enforcer_tokenizer_data,
)


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
    frames = []

    board = chess.Board()

    def add_move(move: str):
        board.push_san(move)

        last_move = board.peek()
        board_size = 400
        text_height = 25
        svg_string = chess.svg.board(board=board, lastmove=last_move, size=board_size)
        png_bytes = svg2png(bytestring=svg_string)
        board_img = Image.open(io.BytesIO(png_bytes))

        total_height = board_size + text_height
        new_img = Image.new("RGB", (board_size, total_height), color="white")
        new_img.paste(board_img, (0, 0))

        draw = ImageDraw.Draw(new_img)

        white_text = f"White ELO: {elo1}"
        draw.text((10, board_size + 10), white_text, fill=(0, 0, 0))

        black_text = f"Black ELO: {elo2}"
        text_width = draw.textlength(black_text)
        draw.text(
            (board_size - text_width - 10, board_size + 10), black_text, fill=(0, 0, 0)
        )

        frames.append(new_img)

    def list_to_regex(str_list: list[str]) -> str:
        escaped_strings = [re.escape(s) for s in str_list]
        return "(" + "|".join(escaped_strings) + ")"

    add_move(moves[-1])

    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

    with torch.no_grad():
        for i in range(1, 100):
            if board.legal_moves.count() == 0:
                break

            print(f"Move {i}")
            moves_prompt = ""
            if moves:
                moves_prompt = f"Previous Chess Moves: {' '.join(moves)}\n"

            elo = elo1 if i % 2 == 0 else elo2
            prompt = f"{moves_prompt}Next Chess Player Elo: {elo}\nNext Chess Move: "
            t1 = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            generate_kwargs = dict(
                inputs,
                num_beams=5,
                do_sample=True,
                max_new_tokens=64,
                use_cache=True,
            )

            for j in range(2):
                try:
                    if j == 1:
                        move_regex = list_to_regex(
                            [board.san(move) for move in board.legal_moves]
                        )
                        print(move_regex)
                        parser = RegexParser(move_regex)
                        output_ids = generate_enforced(
                            model, tokenizer_data, parser, **generate_kwargs
                        )
                    else:
                        output_ids = model.generate(**generate_kwargs)

                    next_move = (
                        tokenizer.decode(
                            output_ids[0, len(inputs.input_ids[0]) :],
                            skip_special_tokens=True,
                        )
                        .split("<EOS>", 1)[0]
                        .strip()
                    )

                    if j == 1:
                        print(f"Trial {j}: {next_move}")

                    t2 = time.time()
                    print(f"Generation {t2 - t1:.2f} sec")

                    add_move(next_move)
                    break
                except chess.IllegalMoveError as e:
                    print(e)
                    continue

            moves.append(next_move)
            print(moves)

    first_frame = frames[0]
    other_frames = frames[1:]

    first_frame.save(
        "output.gif",
        save_all=True,
        append_images=other_frames,
        duration=1000,
        loop=0,
        optimize=True,  # Optimize GIF size
    )


if __name__ == "__main__":
    main()
