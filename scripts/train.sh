CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
--dataset_type=shhq_encode \
--exp_dir=exp_shhq_256_128 \
--workers=4 \
--batch_size=4 \
--test_batch_size=4 \
--test_workers=4 \
--val_interval=1000 \
--save_interval=5000 \
--start_from_latent_avg \
--training_stage=1