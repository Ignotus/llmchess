uv run python3 -m llmchess.preprocess_data data/lichess_db_standard_rated_2014-07.pgn data/train_data.json --retention-rate 0.1 --num-boards 25000

# v1
# uv run python3 -m llmchess.train --max-seq-len 512 --model-id "Qwen/Qwen3-0.6B"
uv run python3 -m llmchess.train --max-seq-len 96 --model-id "Qwen/Qwen3-0.6B"