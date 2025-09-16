python3 gaussian_allocation/prepare_embedding.py \
    --embedder Qwen/Qwen3-Embedding-0.6B \
    --dataset_file_path /home/hieunt/verl/data/dapo-math-17k.parquet \
    --saved_path /home/hieunt/verl/data/dapo-math-17k-embeddings \
    --num_samples -1 \
    --chunk_size 512
