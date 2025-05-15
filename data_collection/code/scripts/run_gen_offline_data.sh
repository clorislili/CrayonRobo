python gen_offline_data.py \
  --data_dir ../data/train_crayonrobo_data_rep \
  --data_fn ../stats/train_30cats_train_data_list.txt \
  --primact_types pulling \
  --num_processes 5 \
  --num_epochs 100 \
  --ins_cnt_fn ../stats/ins_cnt_30cats.txt

# python gen_offline_data.py \
#   --data_dir ../data/test_crayonrobo_data_rep \
#   --data_fn ../stats/train_30cats_test_data_list.txt \
#   --primact_types pulling \
#   --num_processes 5 \
#   --num_epochs 10 \
#   --ins_cnt_fn ../stats/ins_cnt_30cats.txt



