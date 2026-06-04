python3 scripts/batch_convert.py datasets/RL-11 --output-dir data/documents/RL-11 --use-llm --cleanup
python3 scripts/ingest_voyage.py --name "RL-11" --documents-dir data/documents/RL-11
