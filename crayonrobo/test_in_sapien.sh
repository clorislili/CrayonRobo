#out_dir should be the same to the out_dir in test_model.sh
python test_entireprocess_in_sapien.py \
  --llama_dir ../llama_model_weights \
  --data_dir ../data_collection/data/test_dataset\
  --data_fn data_tuple_list.txt \
  --num_processes 5 \
  --num_epochs 0 \
  --out_dir ./prediction_json/crayon_robo_ckpts \
  --out_fn ./result/crayon_robo_test/data_tuple_list_temp.txt \
  --device 0 \
  --no_gui