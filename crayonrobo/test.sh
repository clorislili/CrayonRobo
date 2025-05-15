# step1: load the trained model, adn infer on all test samples first, save into  prediction.json files
bash test_model.sh
# step2: read all prediction json files, and interact with the objects in the Sapien simulator
bash test_in_sapien.sh
# step2: calculate the success rate
bash cal_succ_rate.sh