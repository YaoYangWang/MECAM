# Filtering Negative Information for Multimodal Emotion Recognition

This is the open source code for paper: Filtering Negative Information for Multimodal Emotion Recognition

## Intro
>Multimodal emotion recognition stands at the forefront of human-computer interaction technology, significantly bridging the communication gap between humans and machines by accurately identifying human emotions. This technology leverages multiple inputs such as facial expressions, speech, text, and physiological signals to gain an in-depth understanding of emotional states. However, existing methods often focus on synchronizing these multimodal data to enhance accuracy while overlooking the critical role of negative information. Negative information typically refers to noise in the data or inconsistencies with the primary emotion labels, such as discrepancies in emotional expressions across different modalities and noisy data. This information can significantly impact the effectiveness of emotion recognition systems. To address this issue, we propose an innovative method inspired by the concept of impurity removal in chemistry. This method is based on contrastive learning and aims to eliminate negative information in speech, text, and image data. Furthermore, it integrates the filtered features for emotion recognition using a multi-head attention mechanism. Specifically, we apply contrastive learning in the feature space of each modality to selectively filter out negative information. Subsequently, we use a multi-head attention mechanism to dynamically focus on relevant features across modalities, ensuring that the integration process is sensitive to subtle differences in emotional expression. Experiments conducted on the CMU-MOSI and CMU-MOSEI datasets validate our approach, demonstrating that our method significantly outperforms existing methods in emotion recognition tasks.

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

3. Start training
```
python main.py
```

## Contact 
```
If you have any question about our work, please feel free to contact us:

- Yaoyang Wang : wangyaoyang@shu.edu.cn
- Xianxu Zhu :  1591694407@qq.com
```

