import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet, CustomEncoder

from transformers import BertModel, BertConfig
from modules.contrastive import IntraModalityCL
from modules.bi_attention import BidirectionalCrossAttention
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
class MECAM(nn.Module):
    def __init__(self, hp):
        # Base Encoders
        super().__init__()
        self.hp = hp
        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        self.text_enc = LanguageEmbeddingLayer(hp)
        self.visual_enc = RNNEncoder(
            in_size = hp.d_vin,
            hidden_size = hp.d_vh,
            out_size = hp.d_vout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size = hp.d_ain,
            hidden_size = hp.d_ah,
            out_size = hp.d_aout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )

        # For MI maximization
        self.mi_tv = MMILB(
            x_size = hp.d_tout,
            y_size = hp.d_vout,
            mid_activation = hp.mmilb_mid_activation,
            last_activation = hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size = hp.d_tout,
            y_size = hp.d_aout,
            mid_activation = hp.mmilb_mid_activation,
            last_activation = hp.mmilb_last_activation
        )

        if hp.add_va:
            self.mi_va = MMILB(
                x_size = hp.d_vout,
                y_size = hp.d_aout,
                mid_activation = hp.mmilb_mid_activation,
                last_activation = hp.mmilb_last_activation
            )

        dim_sum = hp.d_aout + hp.d_vout + hp.d_tout


        self.loss_intra_clr_tv = IntraModalityCL(
            text_dim = hp.d_tout,
            video_dim = hp.d_vout
        )
        self.loss_intra_clr_ta = IntraModalityCL(
            text_dim = hp.d_tout,
            video_dim = hp.d_aout
        )
        self.loss_intra_clr_vt = IntraModalityCL(
            text_dim = hp.d_vout,
            video_dim = hp.d_tout
        )

        # Trimodal Settings
        self.fusion_prj = SubNet(
            in_size = dim_sum,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )

    def normalize(self,out):
        # 将 out 归一化到 -1 到 1
        out_min = out.min()
        out_max = out.max()

        # 首先将数据缩放到 0-1
        out_normalized = (out - out_min) / (out_max - out_min)

        # 然后缩放到 -1 到 1
        out_normalized = out_normalized * 2 - 1
        return out_normalized
    def visualize_tensor(self, out1):
        # 创建掩码
        # mask = torch.sign(aa)
        # 可视化的张量
        plt.figure(figsize=(20, 0.2))  # 设置图像的宽度和高度
        plt.imshow(out1.view(1, -1), cmap='Blues', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()
    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None, mem=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        # print(bert_sent_type, bert_sent_mask)
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        text = enc_word[:,0,:] # (batch_size, emb_size)
        acoustic = self.acoustic_enc(acoustic, a_len)
        visual = self.visual_enc(visual, v_len)
        if y is not None:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])
            # for ablation use
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic, labels=y, mem=mem['va'])
        else:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic)


        # Linear proj and pred
        model = BidirectionalCrossAttention(text_dim=768, audio_dim=self.hp.d_aout, heads=8, dim_head=64, dropout=0.1, talking_heads=True, prenorm=True)
        acoustic = model(text, acoustic)
        # print(acoustic)
        model2 = BidirectionalCrossAttention(text_dim=768, audio_dim=self.hp.d_vout, heads=8, dim_head=64, dropout=0.1, talking_heads=True, prenorm=True)
        #print("test shape", output_audio.shape)
        visual = model2(text, visual)
        # text = acoustic + visual
        fusion, preds = self.fusion_prj(torch.cat([text, acoustic, visual], dim=1))
        loss_cc = self.loss_intra_clr_tv(visual,text,labels=y) + self.loss_intra_clr_ta(acoustic,text,labels=y)
        nce = 0.2 * loss_cc

        pn_dic = {'tv':tv_pn, 'ta':ta_pn, 'va': va_pn if self.add_va else None}
        lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)
        H = H_tv + H_ta + (H_va if self.add_va else 0.0)
        return lld, nce, preds, pn_dic, H, text, acoustic, visual
