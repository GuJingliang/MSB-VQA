## MSB-VQA: Overcoming Multiple Source Biases for Robust Visual Question Answering

Almost all flags can be set at `utils/config.py`. The dataset paths, the hyperparams can be set accordingly in this 
file.

## GPU used: 
	* One NVIDIA GeForce RTX 2080 Ti
	
## Memory required:
	* 4GB approximately

## Prerequisites
    * python==3.7.11
    * click==8.1.7
    * h5py==3.5.0
    * matplotlib==3.5.3
    * numpy==1.21.6
    * scikit_image==0.19.3
    * scikit_learn==1.0.2
    * scipy==1.7.3
    * tensorboardX==2.4
    * torch==1.10.1+cu111
    * tqdm==4.62.3
    * tsnecuda==3.0.1

## Dataset

* Download questions/answers for VQA v2 ,VQA-CP2 executing bash tools/download.sh
* The image features can be downloaded by following instructions from : https://github.com/hengyuan-hu/bottom-up-attention-vqa.

After downloading the datasets, keep them in the folders set by config.py

## Preprocessing

The preprocessing steps are as follows:

1. process questions and dump dictionary:
    ```
    python tools/create_dictionary.py
    ```
2. process answers and question types, and generate the frequency-based margins:
    ```
    python tools/compute_softscore.py
    ```
3. convert image features to h5:
    ```
    python tools/detection_features_converter.py 
    ```

## Model training instruction
```
    python main.py --output test-run --gpu 0
   ```
## Model evaluation instruction
```
    python main.py --output test-run --gpu 0 --eval-only
   ```

