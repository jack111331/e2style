python scripts/inference.py \
--exp_dir=exps \
--checkpoint_path=pretrained_models/inversion.pt \
--data_path=/tmp2/r10922033/lab_project/idinvert_pytorch/examples \
--test_batch_size=1 \
--test_workers=4 \
--stage=1 \
--save_inverted_codes \
--couple_outputs \
--resize_outputs