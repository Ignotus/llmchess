import argparse
import json
import chess.pgn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("outputfile")
    parser.add_argument("--time-control", default=60, type=int)
    parser.add_argument("--increment", default=0, type=int)
    args = parser.parse_args()

    train_data = []

    with open(args.filename) as pgn:
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

            game_moves = (
                game_moves_and_result.rsplit("\n", 1)[-1]
                .replace("0-1", "")
                .replace("1-0", "")
                .replace("1/2-1/2", "")
            )

            train_data.append(
                {
                    "text": f"White Elo = {white_elo}, Black Elo = {black_elo}, Moves: {game_moves}"
                }
            )

            print(f"Data collected: {len(train_data)}")

    with open(args.output_file, "w+") as f:
    with open(args.outputfile, "w+") as f:
        json.dump(train_data, f, indent=4)


if __name__ == "__main__":
    main()
