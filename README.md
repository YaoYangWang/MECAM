# MECAM：Multimodal Emotion Analysis Method Based on Contrastive Learning and Talking-Heads Attention Mechanism

This is the open source code for paper: MECAM：Multimodal Emotion Analysis Method Based on Contrastive Learning and Talking-Heads Attention Mechanism

## Intro
> In the context of rapid digitalization and informatization, multimodal emotion recognition is increasingly becoming a crucial research area in artificial intelligence and human-computer interaction. Accordingly, this study introduces a novel approach to multimodal emotion recognition. Inspired by the concept of impurity removal in chemistry, we have developed a "Filter Model" based on intra-modal contrastive learning, aiming to identify and process emotional data with greater precision. With an improved strategy for integrating attention mechanisms, our model demonstrates significant performance enhancements in merging various emotional signals, effectively capturing and analyzing emotional features from different modalities. Furthermore, extensive experimental analyses were conducted on several authoritative datasets to validate the efficacy of our proposed method. The related code has been made open-source.
>

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

3. Start training
```
python main.py
```

## Citation
Please cite our paper if you find our work useful for your research:
## Contact 
```
If you have any question about our work, please feel free to contact us:

- Yaoyang Wang : wangyaoyang@shu.edu.cn
- Xianxu Zhu :  1591694407@qq.com
```
>>>>>>> cd7ff5f (代码实现)

