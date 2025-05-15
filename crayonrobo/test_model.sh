#-adapter_dir is the dir to the pretrained ckpts
CUDA_VISIBLE_DEVICES=0 python test_model.py \
  --llama_dir ../llama_model_weights \
  --adapter_dir ./exp/crayon_robo_ckpts/checkpoint-11.pth \
  --data_dir ../data_collection/data/test_dataset \
  --out_dir ./prediction_json/crayon_robo_ckpts \
  --action pulling \
  --hint 3
