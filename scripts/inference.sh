CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
--checkpoint_path=/tmp2/r10922033/lab_project/e2style_high_rate/exp_shhq_256_128/checkpoints/best_model.pt \
--data_path=/tmp2/r10922033/lab_project/deepfashion-mulmod/test_images \
--exp_dir=exp_inference/high_rate/resized_output/wild_test_deepfashion_mulmod \
--resize_outputs