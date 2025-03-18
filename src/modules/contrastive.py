from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class IntraModalityCL(nn.Module):

    def __init__(self, video_dim, text_dim, temperature=0.03, negative_weight=0.8, positive_negative_ratio=1.0, logger=None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none') 
        self.temperature = temperature 
        self.logger = logger
        self.negative_w = negative_weight  # Weight of negative samples logits.
        self.positive_negative_ratio = positive_negative_ratio  # Ratio of positive to negative samples
        
        # 维度对齐层
        self.align_dim = nn.Linear(video_dim, text_dim)

    def compute_loss(self, logits, mask):
        return -torch.log((F.softmax(logits, dim=1) * mask).sum(1))

    def _get_positive_mask(self, labels, batch_size):
        """
        Create a positive mask based on labels within the same modality.
        """
        # If labels is 2D (e.g., [batch_size, 1]), we squeeze it to 1D
        labels = labels.squeeze()  # This ensures labels is a 1D tensor of shape [batch_size]
        
        mask = torch.zeros(batch_size, batch_size).cuda(non_blocking=True)
        for i in range(batch_size):
            # Positive samples have the same label
            mask[i, labels == labels[i]] = 1
        return mask

    def forward(self, video_features, text_features, video_labels, text_labels):
        if video_labels is None or text_labels is None:
            return torch.tensor(0.0).cuda(non_blocking=True)  # 返回0的损失

        batch_size = video_features.shape[0]

        # 对视频特征进行维度对齐
        video_features = self.align_dim(video_features)

        # Normalize features 
        video_features = nn.functional.normalize(video_features, dim=1)
        text_features = nn.functional.normalize(text_features, dim=1)
        
        # Intra-modality alignment: Video and Video, Text and Text
        # Video Intra-modality logits
        logits_clstr_vid = video_features @ video_features.t()
        # Text Intra-modality logits
        logits_clstr_txt = text_features @ text_features.t()

        # 归一化并加温度缩放
        logits_clstr_vid /= self.temperature 
        logits_clstr_txt /= self.temperature 

        # 为视频和文本创建正样本掩码
        positive_mask_vid = self._get_positive_mask(video_labels, batch_size)
        positive_mask_txt = self._get_positive_mask(text_labels, batch_size)

        # 计算负样本 logits
        negatives_vid = logits_clstr_vid * (1 - positive_mask_vid)
        negatives_txt = logits_clstr_txt * (1 - positive_mask_txt)

        # 控制正负样本的比例
        num_positive_samples = positive_mask_vid.sum()
        num_negative_samples = int(num_positive_samples * self.positive_negative_ratio)
        # print(num_negative_samples)
        # # 确保 num_negative_samples 是整数类型
        # num_negative_samples = int(num_negative_samples)

        # # 或者如果 num_negative_samples 是一个浮动类型的张量，使用 floor 转换为整数
        # num_negative_samples = torch.floor(num_negative_samples).int()

        
        # 如果负样本的数量大于正样本，则从负样本中选择
        if num_negative_samples > negatives_vid.size(0):
            # print("只选择部分")
            # 如果负样本数量大于最大样本数，则只选择部分负样本
            negatives_vid = negatives_vid[:num_negative_samples]
            negatives_txt = negatives_txt[:num_negative_samples]
        else:
            # print("选择全部")
            # 负样本的数量等于正负样本的比例
            negatives_vid = negatives_vid[:num_negative_samples]
            negatives_txt = negatives_txt[:num_negative_samples]

        # 拼接正样本和负样本的 logits
        vid_logits = torch.cat([logits_clstr_vid, self.negative_w * negatives_vid], dim=1)
        txt_logits = torch.cat([logits_clstr_txt, self.negative_w * negatives_txt], dim=1)

        # 为每种模态创建掩码
        mask_vid = torch.cat([positive_mask_vid, torch.zeros_like(negatives_vid)], dim=1)
        mask_txt = torch.cat([positive_mask_txt, torch.zeros_like(negatives_txt)], dim=1)

        # 计算损失
        loss_vid = self.compute_loss(vid_logits, mask_vid)
        loss_txt = self.compute_loss(txt_logits, mask_txt)

        # 返回平均损失
        return (loss_vid.mean() + loss_txt.mean()) / 2

