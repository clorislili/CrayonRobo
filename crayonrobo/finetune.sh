#generate training json files first
python ./data/create_dataset.py

#then, start training
OUTPUT_DIR='./exp/train_crayonrobo_0514_twoloss'
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --master_port=1117 --nproc_per_node=1 --use_env main_finetune_w2a.py --batch_size 4 \
   --epochs 12 --warmup_epochs 1 --blr 1e-3 --weight_decay 0.02 \
   --output_dir "$OUTPUT_DIR" \
   --pretrained_path ./ckpts/BIAS_LORA_NORM-336-Chinese-7B.pth \
   --bins True \
   --mlm True\
   --hint 3 \
   --ortho_loss \
   --para_loss \
   --data_config ./data/train_data_json
