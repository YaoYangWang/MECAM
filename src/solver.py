import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model import MECAM
import matplotlib.pyplot as plt
class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model

        # Training hyperarams
        self.alpha = hp.alpha
        self.beta = hp.beta

        self.update_batch = hp.update_batch
        self.y_losses = []
        self.nce_losses = []
        self.h_losses = []
        self.epoch_y_losses = []
        self.epoch_nce_losses = []
        self.epoch_h_losses = []
        # initialize the model
        if model is None:
            self.model = model = MECAM(hp)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            model = model.cuda()
        else:
            self.device = torch.device("cpu")

        # criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else: # mosi and mosei are regression datasets
            self.criterion = criterion = nn.L1Loss(reduction="mean")
        
        # optimizer
        self.optimizer={}

        if self.is_train:
            mmilb_param = []
            main_param = []
            bert_param = []

            for name, p in model.named_parameters():
                # print(name)
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    elif 'mi' in name:
                        mmilb_param.append(p)
                    else: 
                        main_param.append(p)
                
                for p in (mmilb_param+main_param):
                    if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                        nn.init.xavier_normal_(p)

        self.optimizer_mmilb = getattr(torch.optim, self.hp.optim)(
            mmilb_param, lr=self.hp.lr_mmilb, weight_decay=hp.weight_decay_club)
        
        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )

        self.scheduler_mmilb = ReduceLROnPlateau(self.optimizer_mmilb, mode='min', patience=hp.when, factor=0.5, verbose=True)
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################
    def plot_losses(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.epoch_y_losses, label='Pred Loss')
        plt.plot(self.epoch_nce_losses, label='Contrastive Loss')
        plt.plot(self.epoch_h_losses, label='Beta Loss')
        plt.title("Losses during Training")
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    def train_and_eval(self):
        model = self.model
        optimizer_mmilb = self.optimizer_mmilb
        optimizer_main = self.optimizer_main

        scheduler_mmilb = self.scheduler_mmilb
        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        # entropy estimate interval
        mem_size = 1

        def train(model, optimizer, criterion, stage=1):
            epoch_loss = 0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            nce_loss = 0.0
            ba_loss = 0.0
            start_time = time.time()

            left_batch = self.update_batch

            mem_pos_tv = []
            mem_neg_tv = []
            mem_pos_ta = []
            mem_neg_ta = []
            if self.hp.add_va:
                mem_pos_va = []
                mem_neg_va = []

                # 用于累计每个 epoch 中的损失值
            epoch_y_loss = 0.0
            epoch_nce_loss = 0.0
            epoch_h_loss = 0.0

            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
                if self.hp.dataset == "mosei":
                    if stage == 0 and i_batch / len(self.train_loader) >= 0.5:
                        break
                model.zero_grad()

                with torch.cuda.device(0):
                    text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                    text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                    bert_sent_type.cuda(), bert_sent_mask.cuda()
                    if self.hp.dataset=="ur_funny":
                        y = y.squeeze()
                
                batch_size = y.size(0)

                if stage == 0:
                    y = None
                    mem = None
                elif stage == 1 and i_batch >= mem_size:
                    mem = {'tv':{'pos':mem_pos_tv, 'neg':mem_neg_tv},
                            'ta':{'pos':mem_pos_ta, 'neg':mem_neg_ta},
                            'va': {'pos':mem_pos_va, 'neg':mem_neg_va} if self.hp.add_va else None}
                else:
                    mem = {'tv': None, 'ta': None, 'va': None}

                lld, nce, preds, pn_dic, H, _, _, _ = model(text, visual, audio, vlens, alens, 
                                                bert_sent, bert_sent_type, bert_sent_mask, y, mem)

                if stage == 1:
                    y_loss = criterion(preds, y)
                    
                    if len(mem_pos_tv) < mem_size:
                        mem_pos_tv.append(pn_dic['tv']['pos'].detach())
                        mem_neg_tv.append(pn_dic['tv']['neg'].detach())
                        mem_pos_ta.append(pn_dic['ta']['pos'].detach())
                        mem_neg_ta.append(pn_dic['ta']['neg'].detach())
                        if self.hp.add_va:
                            mem_pos_va.append(pn_dic['va']['pos'].detach())
                            mem_neg_va.append(pn_dic['va']['neg'].detach())
                   
                    else: # memory is full! replace the oldest with the newest data
                        oldest = i_batch % mem_size
                        mem_pos_tv[oldest] = pn_dic['tv']['pos'].detach()
                        mem_neg_tv[oldest] = pn_dic['tv']['neg'].detach()
                        mem_pos_ta[oldest] = pn_dic['ta']['pos'].detach()
                        mem_neg_ta[oldest] = pn_dic['ta']['neg'].detach()

                        if self.hp.add_va:
                            mem_pos_va[oldest] = pn_dic['va']['pos'].detach()
                            mem_neg_va[oldest] = pn_dic['va']['neg'].detach()

                    if self.hp.contrast:
                        # print("using cpc")
                        loss = y_loss + self.alpha * nce - self.beta * lld
                    else:
                        # print("not using cpc")
                        loss = y_loss + self.alpha * nce
                    if i_batch > mem_size:
                        # print("use H") 用到了
                        loss -= self.beta * H
                    # loss收集部分-----------------------------------------------------------------------------------------------------
                    # 累计每个 batch 的损失
                    epoch_y_loss += y_loss.item() * batch_size
                    epoch_nce_loss += (self.alpha * nce).item() * batch_size
                    epoch_h_loss +=  ((self.beta * H).item() if torch.is_tensor(H) else self.beta * H) *  batch_size * (-1)


                    loss.backward()

                    
                elif stage == 0:
                    # maximize likelihood equals minimize neg-likelihood
                    loss = -lld
                    loss.backward()
                else:
                    raise ValueError('stage index can either be 0 or 1')
                
                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                    optimizer.step()
                
                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                nce_loss += nce.item() * batch_size
                ba_loss += (-H - lld) * batch_size

                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    avg_nce = nce_loss / proc_size
                    avg_ba = ba_loss / proc_size
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss ({}) {:5.4f} | NCE {:.3f} | BA {:.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, 'TASK+BA+CPC' if stage == 1 else 'Neg-lld',
                        avg_loss, avg_nce, avg_ba))
                    proc_loss, proc_size = 0, 0
                    nce_loss = 0.0
                    ba_loss = 0.0
                    start_time = time.time()
                    
            
                # 在每个 epoch 结束时，计算平均损失并添加到损失列表中
            self.epoch_y_losses.append(epoch_y_loss / self.hp.n_train)
            self.epoch_nce_losses.append(epoch_nce_loss / self.hp.n_train)
            self.epoch_h_losses.append(epoch_h_loss / self.hp.n_train)

            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0
            # 1初始化数据存储容器
            all_preds, all_truths, all_texts, all_visuals, all_audios = [], [], [], [], []

            results = []
            truths = []

            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                    with torch.cuda.device(0):
                        text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                        lengths = lengths.cuda()
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                    
                        if self.hp.dataset == 'iemocap':
                            y = y.long()
                    
                        if self.hp.dataset == 'ur_funny':
                            y = y.squeeze()

                    batch_size = lengths.size(0) # bert_sent in size (bs, seq_len, emb_size)

                    # we don't need lld and bound anymore
                    _, _, preds, _, _, processed_t, processed_a, processed_v = model(text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()

                    total_loss += criterion(preds, y).item() * batch_size
                    # print("preds>>>",preds.shape)preds>>> torch.Size([128, 1])
                    # print("y>>>",y.shape) y>>> torch.Size([128, 1])
                    # print("text>>>",text.shape) text>>> torch.Size([0, 128])
                    # print("vision>>>",vision.shape) vision>>> torch.Size([187, 128, 20])
                    # print("audio>>>",audio.shape) audio>>> torch.Size([156, 128, 5])
                    # print(text)
                    # 2收集数据
                    all_preds.append(preds.detach().cpu())
                    all_truths.append(y.detach().cpu())
                    all_texts.append(processed_t.detach().cpu())
                    all_visuals.append(processed_v.detach().cpu())
                    all_audios.append(processed_a.detach().cpu())
                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)
            
            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)


            results = torch.cat(results)
            truths = torch.cat(truths)
            # 3收集数据
            # 保存测试集数据
            # if test:
            save_path = "saved_sne.npz"
            np.savez(save_path,
                    test_preds=torch.cat(all_preds).numpy(),
                    test_truth=torch.cat(all_truths).numpy(),
                    pred_v=torch.cat(all_texts).numpy(),
                    pred_a=torch.cat(all_visuals).numpy(),
                    pred_t=torch.cat(all_audios).numpy())
            print("save npz success!")
            return avg_loss, results, truths

        best_valid = 1e8
        best_mae = 1e8
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()

            self.epoch = epoch

            # maximize likelihood
            if self.hp.contrast:
                train_loss = train(model, optimizer_mmilb, criterion, 0)

            # minimize all losses left
            train_loss = train(model, optimizer_main, criterion, 1)

            val_loss, _, _ = evaluate(model, criterion, test=False)
            test_loss, results, truths = evaluate(model, criterion, test=True)
            
            end = time.time()
            duration = end-start
            scheduler_main.step(val_loss)    # Decay learning rate by validation loss

            # validation F1
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-"*50)
            
            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss
                # for ur_funny we don't care about
                if self.hp.dataset == "ur_funny":
                    eval_humor(results, truths, True)
                elif test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    if self.hp.dataset in ["mosei_senti", "mosei"]:
                        eval_mosei_senti(results, truths, True)

                    elif self.hp.dataset == 'mosi':
                        eval_mosi(results, truths, True)
                    elif self.hp.dataset == 'iemocap':
                        eval_iemocap(results, truths)
                    
                    best_results = results
                    best_truths = truths
                    print(f"Saved model at pre_trained_models/MM.pt!")
                    save_model(self.hp, model)
            else:
                patience -= 1
                if patience == 0:
                    break

        print(f'Best epoch: {best_epoch}')
        if self.hp.dataset in ["mosei_senti", "mosei"]:
            eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)
        elif self.hp.dataset == 'iemocap':
            eval_iemocap(results, truths)       
        sys.stdout.flush()
    def evaluate_single_test_case(self, model, criterion, index):
        model.eval()
        single_test_case = self.test_loader.dataset.get_single_test_case(index)
        # print(single_test_case)
        # 遍历元组中的每个元素
        for element in single_test_case:
            print("遍历元组中的每个元素")
            print(type(element))
            print(element)
        # for element in single_test_case[0]:
        #     print("遍历元组中的每个元素")
        #     print(type(element))
        #     print(element)
        # 展开single_test_case元组
        [_, visual, audio, bert_sent, vlens, alens], y, ids = single_test_case
        text = torch.tensor([])
        # 如果数据需要转移到GPU
        if torch.cuda.is_available():
            text = text.cuda()
            visual = visual.cuda()
            audio = audio.cuda()
            y = y.cuda()
            l = l.cuda()
            bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()

        # 假设您的模型输出一个预测值
        preds = model(text, visual, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)
        loss = criterion(preds, y)

        # 输出预测结果和实际标签，以便进一步分析
        print(f"Prediction: {preds}, Actual: {y}")

        return loss.item()
    def predict(self, text, visual, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask,y,mem=None):
    # """
    # 预测单个数据实例的方法
    # """
        model = self.model  # 将模型设置为评估模式
        model.eval()

        with torch.no_grad():
            # 假设使用了 GPU
            # text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
            #     text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()

            # 调用模型的 forward 方法，传递所有参数
            lld, nce, preds, pn_dic, H, text, acoustic, visual = model(text, visual, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask,None,None)

            # 处理输出
            predicted_result = preds
            # print(text[1])
            # print(text.shape)
            # model.visualize_tensor(text.cpu())

        return predicted_result