# CrayonRobo
The official codebase for [CrayonRobo](https://arxiv.org/abs/2505.02166): Object-Centric Prompt-Driven Vision-Language-Action Model for Robotic Manipulation (CVPR 2025)

## Acknowledgement
This repo benefits from [LLama_Adapter](https://github.com/OpenGVLab/LLaMA-Adapter) and [Where2act](https://github.com/daerduoCarey/where2act). Thanks for their wonderful works.

## Setup
1) conda create --name crayonrobo python=3.8

2) conda activate crayonrobo

3) pip install -r crayonrobo

Note that, the installed torch should satisfy your own cuda version

            
## Data Collection
- Download our training and test data: [train data]() and [test data](). The files should be zipped to ./CrayonRobo/data_collection/data.
  ```bash
  ./data/train_dataset
    ├── 148_Faucet_0_pulling_0
    |   └── png/json/...
    ├── 148_Faucet_0_pulling_9
    |   └── png/json/...
    ├── ...
    │   ...
    └── ...

- Or collect data by your own: Download [partnet mobility assets]() and zip to /CrayonRobo/data_collection/assets.
  ```bash
  ./assets
    ├── 148
    |   └── mobility.urdf
    ├── 149
    |   └── mobility.urdf
    ├── ...
    │   ...
    └── ...
  
  cd ./CrayonRobo/data_collection/code
  
  bash scripts/run_gen_offline_data.sh

This command will first generate training dataset and then generate the testing dataset.

## Model Training
- Preparation:

  Download checkpoints for [LLaMa-Adapter](), [LLaMa](https://disk.pku.edu.cn/link/AA682A19DB7FDA4028B112449D24BBC308). The downloaded checkpoints should be placed under ./Crayonrobo/crayonrobo/ckpts. 
    ```plaintext
    ./ckpts/llama_model_weights
    ├── 7B
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   └── params.json
    └── tokenizer.model
    ./ckpts/BIAS_LORA_NORM-336-Chinese-7B.pth
    ./ckpts/ViT-L-14-336px.pt
- Model training: The training requires the server to has a least 40g memory. The command will first generate the training json, then start training

  
  ```bash
  cd ./CrayonRobo/crayonrobo
  
  bash finetune.sh

## Model Testing

-  Download the released [checkpoint]() or use your own trained checkpoint. The link we provide is baiduyun downloading link. If you can not download, feel free to reach out via email to xl3062@columbia.edu, then we will share the ckpts with you directly. Note that, due to the randomness in data collection, the provided testing dataset is different from the ones in paper, so you may result in slightly different but comparable results compared with the results in paper. 

- The testing requires the server to has a least 40g memory. This command will first use the model to infer on all the test samples, and then interact with object in the simulator (SAPIEN).
  
  ```bash
  cd ./CrayonRobo/crayonrobo
  
  bash test.sh

- Remember to change the argument --adapter_dir in test_model.sh to the directory you placed the ckpts. The default dir is at ./CrayonRobo/crayonrobo/exp
