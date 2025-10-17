import argparse
import re
import json
import random
from typing import TypedDict
import chess
import chess.pgn


class Item(TypedDict):
    text: str


def split_pgn_by_move_number(pgn_string: str) -> list[str]:
    game_moves = (
        pgn_string.rsplit("\n", 1)[-1]
        .replace("0-1", "")
        .replace("1-0", "")
        .replace("1/2-1/2", "")
    )

    processed_string = " " + game_moves.strip()

    parts = re.split(r" (\d+\. )", processed_string)

    move_list = []

    for i in range(1, len(parts), 2):
        move_line = parts[i + 1].strip().split(" ", 1)
        move_list += move_line

    return move_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("outputfile")
    parser.add_argument("--time-control", default=3, type=int)
    parser.add_argument("--increment", default=2, type=int)
    parser.add_argument("--retention-rate", default=0.1, type=float)
    parser.add_argument("--num-boards", default=50000, type=int)
    args = parser.parse_args()

    random.seed(42)

    train_data = []

    with open(args.filename) as pgn:
        num_boards = 0
        while (game := chess.pgn.read_game(pgn)) is not None:
            time_control_parts = game.time_control().parts

            if not time_control_parts:
                continue

            time_control_part = time_control_parts[0]

            if (
                time_control_part.time != args.time_control
                and time_control_part.increment != args.increment
            ):
                continue

            game_moves_and_result = str(game)

            if "eval" in game_moves_and_result:
                continue

            headers = game.headers
            white_elo = headers.get("WhiteElo", None)
            black_elo = headers.get("BlackElo", None)

            if not white_elo or not black_elo:
                continue

            white_elo = round(int(white_elo), -1)
            black_elo = round(int(black_elo), -1)

            game_moves = split_pgn_by_move_number(game_moves_and_result)

            num_boards += 1

            board = chess.Board()
            for i, last_move in enumerate(game_moves):
                if random.random() >= args.retention_rate:
                    board.push_san(last_move)
                    continue

                if i % 2 == 0:
                    player = "white"
                else:
                    player = "black"

                elo = white_elo if i % 2 == 0 else black_elo
                if i > 1:
                    text = f"Previous Chess Position:\n{str(board)}\n"
                else:
                    text = ""

                train_data.append(
                    Item(
                        text=text
                        + f"Next Chess Player ({player}) Elo: {elo}\nNext Chess Move: {last_move}<EOS>"
                    )
                )

                board.push_san(last_move)

            print(f"Data collected: {len(train_data)} from {num_boards} boards.")

            if num_boards >= args.num_boards:
                break

    with open(args.outputfile, "w+") as f:
        json.dump(train_data, f, indent=4)


if __name__ == "__main__":
    main()
