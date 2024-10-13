# Contrastive-Based Removal of Negative Information in Multimodal Emotion Analysis

This is the open source code for paper: Contrastive-Based Removal of Negative Information in Multimodal Emotion Analysis

## Intro
>Multimodal sentiment analysis bridges the communication gap between humans and machines by accurately recognizing human emotions. However, existing approaches often focus on synchronizing multimodal data to enhance accuracy, overlooking the critical role of negative information. Negative information refers to noise or corrupted data commonly found in real-world scenarios, such as missing text, shuffled word order, frame drops, or blurring in videos. These issues can significantly compromise the effectiveness of sentiment analysis systems. To address this challenge, we propose a novel method based on contrastive learning for the removal of non-relevant features within single modalities, aiming to eliminate negative information in speech, text, and image data. Additionally, we have designed an enhanced multi-head attention mechanism that integrates the cleansed features into a unified representation for emotion analysis. Experimental evaluations on the CMU-MOSI and CMU-MOSEI datasets demonstrate that our method significantly outperforms existing approaches in sentiment analysis tasks. This method not only improves accuracy but also ensures the system’s robustness against diverse noisy data, including corrupted and inconsistent multimodal information often encountered in real-world settings
![](images/1.png)
## Usage

1. As mentioned in our paper, you need to download the CMU-MOSI and CMU-MOSEI dataset. Then place them under the folder `MECAM/datasets`

2. Environment
```
python 3.8
torch 1.7.1
torchvision 0.8.2
tensorboardX 1.9
tensorflow-estimator 2.3.0
tensorflow-gpu 2.3.1
transformers 4.0.0
```
3. Experimental Setup

### Hardware Parameters

| Parameter              | Value                     |
|------------------------|---------------------------|
| CPU                    | I9-10980XE 3.00 GHz       |
| GPU                    | RTX 3090Ti (Single Card)  |
| Cores                  | 18                        |
| Memory                 | 128GB                     |
| Logical Processors      | 36                        |
| Operating System        | Ubuntu 22.04              |

### Partial Model Training Parameters

| Parameter                           | Value                    |
|-------------------------------------|--------------------------|
| Python Version                      | 3.8                      |
| CUDA Version                        | 11.7                     |
| Pre-trained Language Model (PLM)     | bert-base-uncased         |
| Alpha                               | 0.2/0.1                  |
| Beta                                | 0.05/0.05                |
| Visual Size (Hidden size/Output size)| 16/32                    |
| Audio Size (Hidden size/Output size) | 16/16                    |
| Batch Size (Hidden size/Output size) | 32/64                    |
| Optimizer (Optim)                   | adam                     |
| Dropout Audio (dropout a)            | 0.1/0.01                 |
| Dropout Visual (dropout v)           | 0.1/0.1                  |
| Dropout Text (dropout t)             | 0.1/0.1                  |
| Learning Rate (Learn rate)           | 5e-4                     |
| Weight Decay                        | 1e-4                     |
| Early Stopping                      | 20                       |
| Number of Epochs (Num epochs)        | 40                       |
| Seed                                | 1111                     |


4. Start training
```
python main.py
```

## Contact 
```
If you have any question about our work, please feel free to contact us:

- Yaoyang Wang : wangyaoyang@shu.edu.cn
- Xianxu Zhu :  1591694407@qq.com
```

