from torch import nn
import torch
import torch.nn.functional as F
import time 
import numpy as np

class IntraModalityCL(nn.Module):

    def __init__(self, video_dim, text_dim, temperature=0.03, negative_weight=0.8, logger=None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none') 
        self.temperature = temperature 
        self.logger = logger
        self.negative_w = negative_weight  # Weight of negative samples logits.
        
        # 维度对齐层
        self.align_dim = nn.Linear(video_dim, text_dim)

    def compute_loss(self, logits, mask, weights=None):
        loss = -torch.log((F.softmax(logits, dim=1) * mask).sum(1))
        if weights is not None:
            loss = loss * weights  # 加权
        return loss


    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask)
        return mask.cuda(non_blocking=True)

    def forward(self, video_features, text_features):
        batch_size = video_features.shape[0]

        # 对视频特征进行维度对齐
        video_features = self.align_dim(video_features)

        # Normalize features 
        video_features = nn.functional.normalize(video_features, dim=1)
        text_features = nn.functional.normalize(text_features, dim=1)

        # Intra-modality alignment 模态内
        logits_clstr_vid = video_features @ video_features.t()
        logits_clstr_txt = text_features @ text_features.t()

        logits_clstr_vid /= self.temperature 
        logits_clstr_txt /= self.temperature 

        positive_mask = self._get_positive_mask(batch_size)
        negatives_vid = logits_clstr_vid * positive_mask
        negatives_txt = logits_clstr_txt * positive_mask

        vid_logits = torch.cat([self.negative_w * negatives_vid, self.negative_w * negatives_vid], dim=1)
        txt_logits = torch.cat([self.negative_w * negatives_txt, self.negative_w * negatives_txt], dim=1)

        diag = np.eye(batch_size)
        mask_vid = torch.from_numpy((diag)).cuda()
        mask_txt = torch.from_numpy((diag)).cuda()

        mask_neg_v = torch.zeros_like(negatives_vid)
        mask_neg_t = torch.zeros_like(negatives_txt)
        mask_v = torch.cat([mask_vid, mask_neg_v], dim=1)
        mask_t = torch.cat([mask_txt, mask_neg_t], dim=1)

        loss_i = self.compute_loss(vid_logits, mask_v)
        loss_t = self.compute_loss(txt_logits, mask_t)

        return (loss_i.mean() + loss_t.mean()) / 2
