export node_rank=0
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12345 \
    train_MNodes_torchrun_mfu_preindexparquet.py --data_path=./all_shuffled_data \
    --emb_path=./macrogene_embeddings_1stage \
    --out_path=./out_test \
    --seq_len=862 \
    --batch_size=64 \
    --backend='hccl' \
    --device_type='npu' \
    --compile=false \
    --epoch=1 \
    --num_of_used_data=0 \
    --save_data_interval=5120000 \
    --s3_remote_dir_path='s3://bucket-3028/public/SpeciesLLM/out_test/' \
    &> log/all_shuffled_data.txt

# python -c 'import moxing as mox;mox.file.copy_parallel("/home/ma-user/work/SpeciesLLM", "s3://bucket-3028/public/SpeciesLLM")'
# python -c 'import moxing as mox;mox.file.copy_parallel("s3://bucket-3028/public/SpeciesLLM", "/home/ma-user/work/SpeciesLLM")'
