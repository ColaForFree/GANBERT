from __future__ import annotations

import gc
import os
import math
import time
import random
import datetime
import copy
from optuna.trial import TrialState

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, classification_report
)
from transformers import get_constant_schedule_with_warmup

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from CA.CAmodel import CategoryAttentionFusion, init_label_embeddings_from_transformer
from G.generators_act import build_generator_with_activation
# -----------------------------
#  组合构建函数
# -----------------------------
from data.dataTrent.dataLoader import build_label_list, stratified_split_dataset, stratified_split_dataset_splite
from model_builder import build_ganbert_components, _map_generator_impl


## 生成器加权采样
def sample_y_indices(batch_size, num_classes, device, prior_probs: torch.Tensor = None):
    """
    返回形如 (B,) 的 LongTensor 类别索引。
    prior_probs: (num_classes,) 概率向量；若为 None 则均匀采样。
    """
    if prior_probs is None:
        return torch.randint(low=0, high=num_classes, size=(batch_size,), device=device)
    else:
        # torch.multinomial 要求 probs 在 device 上，且为 1D
        return torch.multinomial(prior_probs.to(device), num_samples=batch_size, replacement=True)

def cleanup_cuda():
    """尽可能释放 Python 引用 + 触发 GC + 清空 CUDA 缓存/IPC"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
# =========================
# 固定随机种子
# =========================
def set_seed(seed: int = 42):
    random.seed(seed);
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed = 42
MAX_SEQ_LEN = 64
BATCH_SIZE = 64
PRINT_EVERY = 10
WARMUP_RATIO = 0.1
EPS = 1e-8
APPLY_BALANCE = True  # 同原始 GAN-BERT 的“复制少量标注样本”策略
noise_size = 100
# 搜索时为了速度，给每个 trial 设置较小 epoch 与 batch 限制
DEFAULT_EPOCHS_PER_TRIAL = 3
# 动态学习率
lr=2e-5
apply_scheduler = True
warmup_proportion = 0.2
MAX_TRAIN_BATCHES = None  # 例如设为 200 以限速；None 不限
MAX_EVAL_BATCHES = None

# 早停
use_early_stopping = False
PATIENCE = 5  # 连续 5 个 epoch 没有明显提升就停
MIN_DELTA = 1e-4  # 增量阈值

epochs=30
n_splits = 10

# 类别不平衡开关（交叉熵权重）
USE_CLASS_WEIGHT = False

# 使用SA1特征融合开关
USE_SA1 = False

# 1) 注意力打分器的隐藏维度（控制“给token打分的小网络”容量）
# - 越大：表达力更强，但小数据/低标注/类别不平衡时更容易过拟合、学偏（m-F1可能掉）
# - 越小：更稳、更不容易学偏（通常更适合你的场景）
SA1_ATTN_HIDDEN = 64        # 先从64开始，比128更稳

# 2) Dropout（正则强度，主要作用在注意力打分网络上）
# - 值越大：越不容易过拟合，注意力不容易过“尖”（更稳）
# - 值太小：注意力容易过早集中到少数token（对不平衡数据不友好）
SA1_DROPOUT = 0.2           # 低标注/不平衡时建议0.2或0.3

# 3) 注意力温度 tau（控制 softmax 后注意力分布的“尖/平”程度）
# - tau > 1：更平滑（把权重分散一些），对小类更友好，训练更稳
# - tau = 1：原始softmax
# - tau < 1：更尖锐（更像“只盯几个词”），可能更不稳定
SA1_ATTENTION_TAU = 1.5     # 让注意力更平滑，先试1.5

# 4) 注意力池化时是否排除 CLS（第0个token）
# - True：v_att只从“内容token”汇总，避免把CLS重复算进v_att，融合更有意义、更稳
# - False：v_att包含CLS，容易造成v_cls和v_att高度相关，门控学习抖动
SA1_EXCLUDE_CLS = True      # 默认排除CLS参与v_att

# 5) 门控融合层 gate 的 bias 初始化（让模型训练初期更像baseline）
# - bias越大：sigmoid(bias)越接近1，初期更偏向v_cls（稳定）
# - bias=0：初期g≈0.5，v_cls和v_att一上来就混合，可能扰动过大导致效果变差
SA1_GATE_BIAS_INIT = 2.0    # 让g初期更偏向v_cls（sigmoid(2)≈0.88）

# =========================
# Category-Attention Fusion
# =========================
USE_CA = True
CA_DROPOUT = 0.2
# 用作 self-att 的 softmax 温度（tau>1 更平滑更稳）
CA_ROUTER_TAU = 1.5


# =========================
# 基于预测熵的自适应样本筛选机制
# =========================
USE_AS = True

AS_k = 10.0          # sigmoid 陡峭度 k
AS_alpha = 0.05      # EMA 平滑系数 alpha
AS_rho = 0.7         # 分位数 rho（可先用同一标量；后续可扩展为每类向量）
AS_eps = 1e-6        # 数值稳定项 epsilon
AS_warmup_epochs = 5 # unlabeled 伪标签项 ramp-up 的 warmup 轮数
AS_min_count = 8     # 更新某一类阈值所需的最小样本数（避免空/极少导致抖动）

# 预测熵计算
def pred_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    logits: [B, C]
    return: entropy [B]
    """
    p = torch.softmax(logits, dim=-1)
    ent = -(p * (p + eps).log()).sum(dim=-1)
    return ent

def make_detail_log_path(output_file: str, suffix: str = "_detail") -> str:
    """
    由 output_file 生成细粒度日志路径：
    e.g.  results/single_result.log  -> results/single_result_detail.log
    """
    base, ext = os.path.splitext(output_file)
    if ext == "":
        ext = ".log"
    return f"{base}{suffix}{ext}"

def build_loader(tokenizer, input_examples, label_masks, label_map,
                 do_shuffle=False, balance=False, batch_size=BATCH_SIZE):
    examples = []
    num_labeled = int(np.sum(label_masks))
    rate = num_labeled / len(input_examples)

    for idx, ex in enumerate(input_examples):
        if rate == 1 or not balance:
            examples.append((ex, label_masks[idx]))
        else:
            if label_masks[idx]:
                balance_factor = int(math.log(max(1, int(1 / rate)), 2))
                balance_factor = max(balance_factor, 1)
                for _ in range(balance_factor):
                    examples.append((ex, label_masks[idx]))
            else:
                examples.append((ex, label_masks[idx]))

    input_ids, attn_masks, label_ids, label_mask_arr = [], [], [], []
    for (text, label_mask) in examples:
        label_name = text[1]
        if label_name.lower() == "functional":  # 跳过 functional 类
            continue
        enc = tokenizer.encode(text[0], add_special_tokens=True,
                               max_length=MAX_SEQ_LEN, padding="max_length",
                               truncation=True)
        input_ids.append(enc)
        attn_masks.append([int(t > 0) for t in enc])
        label_ids.append(label_map[text[1]])
        label_mask_arr.append(label_mask)

    dataset = TensorDataset(
        torch.tensor(input_ids),
        torch.tensor(attn_masks),
        torch.tensor(label_ids, dtype=torch.long),
        torch.tensor(label_mask_arr)
    )
    sampler = RandomSampler(dataset) if do_shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def generate_data_loader(input_examples, label_masks, tokenizer, label_map, batch_size, do_shuffle=False,
                         balance_label_examples=False):
    '''
    Generate a Dataloader given the input examples, eventually masked if they are
    to be considered NOT labeled.
    '''
    examples = []

    # Count the percentage of labeled examples
    num_labeled_examples = 0
    for label_mask in label_masks:
        if label_mask:
            num_labeled_examples += 1
    label_mask_rate = num_labeled_examples / len(input_examples)

    # if required it applies the balance
    for index, ex in enumerate(input_examples):
        if label_mask_rate == 1 or not balance_label_examples:
            examples.append((ex, label_masks[index]))
        else:
            # IT SIMULATE A LABELED EXAMPLE
            if label_masks[index]:
                balance = int(1 / label_mask_rate)
                balance = int(math.log(balance, 2))
                if balance < 1:
                    balance = 1
                for b in range(0, int(balance)):
                    examples.append((ex, label_masks[index]))
            else:
                examples.append((ex, label_masks[index]))

    # -----------------------------------------------
    # Generate input examples to the Transformer
    # -----------------------------------------------
    input_ids = []
    input_mask_array = []
    label_mask_array = []
    label_id_array = []

    # Tokenization
    for (text, label_mask) in examples:
        enc = tokenizer(
            text[0],
            add_special_tokens=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(enc["input_ids"])
        input_mask_array.append(enc["attention_mask"])
        label_id_array.append(label_map[text[1]])
        label_mask_array.append(label_mask)

    # pad_id = tokenizer.pad_token_id  # roberta 是 1，bert 是 0
    # if pad_id is None:
    #     pad_id = 0  # 兜底（一般不会用到）
    #
    # # Attention to token (to ignore padded input wordpieces)
    # for sent in input_ids:
    #     att_mask = [int(token_id != pad_id) for token_id in sent]
    #     input_mask_array.append(att_mask)
    # Convertion to Tensor
    input_ids = torch.tensor(input_ids)
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.long)
    label_mask_array = torch.tensor(label_mask_array)

    # Building the TensorDataset
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

    if do_shuffle:
        sampler = RandomSampler
    else:
        sampler = SequentialSampler

    # Building the DataLoader
    return DataLoader(
        dataset,  # The training samples.
        sampler=sampler(dataset),
        batch_size=batch_size)  # Trains with this batch size.


def format_time(sec: float) -> str:
    return str(datetime.timedelta(seconds=int(round(sec))))


def allTrain(epochs, transformer, generator, discriminator, train_dataloader, test_dataloader, device, label_list,
             gen_optimizer, dis_optimizer, SA1_fusion, CA_fusion, ce_weights, scheduler_d=None, scheduler_g=None,
             detail_log_path: str = None, fold_id: int = None):
    val_f1s = []
    start_time = time.time()
    log_f = None
    if detail_log_path is not None:
        os.makedirs(os.path.dirname(detail_log_path) or ".", exist_ok=True)
        log_f = open(detail_log_path, "a", encoding="utf-8")

    def log_print(msg: str = ""):
        # 控制台输出
        print(msg)
        # 写文件
        if log_f is not None:
            log_f.write(msg + "\n")
            log_f.flush()

    # For each epoch...
    print("\nTraining...")
    best_f1 = -1.0
    best_epoch = -1
    best_metrics = None
    no_improve_epochs = 0
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        t0 = time.time()
        # Reset the total loss for this epoch.
        tr_g_loss = 0
        tr_d_loss = 0

        # Put the model into training mode.
        transformer.train()
        generator.train()
        discriminator.train()

        if USE_SA1 and SA1_fusion is not None:
            SA1_fusion.train()

        if USE_CA and CA_fusion is not None:
            CA_fusion.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            if step and step % PRINT_EVERY == 0:
                print(
                    f"  [Epoch {epoch_i + 1}] Step {step}/{len(train_dataloader)} Elapsed: {format_time(time.time() - t0)}")

            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_label_mask = batch[3].to(device)

            real_batch_size = b_input_ids.shape[0]

            # 1) transformer编码：拿token序列 + CLS
            token_states, v_cls = get_sentence_embedding(transformer, b_input_ids, b_input_mask)

            # 2) x_base：如果开SA1，用SA1融合；否则就用v_cls
            if USE_SA1 and (SA1_fusion is not None):
                x_base = SA1_fusion(token_states, b_input_mask)  # [B,H]
            else:
                x_base = v_cls  # [B,H]

            # 3) x_real：如果开类别注意力，用label-aware cross-attn增强；否则不增强
            if USE_CA and (CA_fusion is not None):
                x_real = CA_fusion(
                    x_base=x_base,
                    token_states=token_states,
                    attention_mask=b_input_mask
                )  # [B,H]
            else:
                x_real = x_base

            # Generate fake data that should have the same distribution of the ones
            # encoded by the transformer.
            # First noisy input are used in input to the Generator
            noise = torch.zeros(real_batch_size, noise_size, device=device).uniform_(0, 1)

            # 判别是否含condition后，再进行生成器生成
            if getattr(generator, "is_conditional", False):
                num_classes = len(label_list)  # == discriminator 的有标签类别数
                # 用均匀先验。经验先验：在函数外先根据 labeled 统计一个 probs 向量传进来
                y_gen = sample_y_indices(real_batch_size, num_classes, device)
                gen_rep = generator(noise, y_gen)  # 关键：传 y
            else:
                gen_rep = generator(noise)

            # Generate the output of the Discriminator for real and fake data.
            # D(real)：输入x_real（由开关决定是否经过SA1/SA2）
            D_real_features, D_real_logits, D_real_probs = discriminator(x_real)

            # D(fake)：输入生成器向量（始终是向量，无token序列）
            D_fake_features, D_fake_logits, D_fake_probs = discriminator(gen_rep)

            # # Finally, we separate the discriminator's output for the real and fake
            # # data
            # features_list = torch.split(features, real_batch_size)
            # D_real_features = features_list[0]
            # D_fake_features = features_list[1]
            #
            # logits_list = torch.split(logits, real_batch_size)
            # D_real_logits = logits_list[0]
            # D_fake_logits = logits_list[1]
            #
            # probs_list = torch.split(probs, real_batch_size)
            # D_real_probs = probs_list[0]
            # D_fake_probs = probs_list[1]
            # ---------------------------------
            #  LOSS evaluation
            # ---------------------------------
            # Generator's LOSS estimation
            g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:, -1] + EPS))
            g_feat_reg = torch.mean(
                torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
            g_loss = g_loss_d + g_feat_reg

            # Disciminator's LOSS estimation
            # 只取真实样本在真实类别上的 logit（去掉伪类列）
            logits = D_real_logits[:, 0:-1]  # shape: [B, num_labels]

            # =========================================================
            # Unlabeled Softmax Cross-Entropy (Pseudo-label CE)
            # =========================================================
            unl_mask = (~b_label_mask.bool()).to(device)  # [B] unlabeled=True
            unl_idx = unl_mask.nonzero(as_tuple=False).squeeze(-1)

            if USE_AS and unl_idx.numel() > 0:
                logits_u = logits.index_select(0, unl_idx)  # [B_u, C]
                # 伪标签：用当前模型预测的 argmax（注意 detach，避免梯度走入“造标签”路径）
                pseudo_y = torch.softmax(logits_u, dim=-1).detach().argmax(dim=-1)  # [B_u]
                # softmax交叉熵（等价于 CE(logits, pseudo_y)）
                per_ex_u_ce = F.cross_entropy(logits_u, pseudo_y, reduction='none')  # [B_u]
                D_L_UCE = per_ex_u_ce.mean()
            else:
                D_L_UCE = torch.tensor(0.0, device=device)

            # 构造按类权重的 CE
            if ce_weights is not None:
                _ce_weight = ce_weights.to(device)
                per_ex_ce = F.cross_entropy(logits, b_labels, weight=_ce_weight, reduction='none')
            else:
                per_ex_ce = F.cross_entropy(logits, b_labels, reduction='none')

            # 仅统计当前 batch 里 "有标签" 的样本
            mask = b_label_mask.bool().to(device)
            per_ex_ce = torch.masked_select(per_ex_ce, mask)

            if per_ex_ce.numel() == 0:
                D_L_Supervised = torch.tensor(0.0, device=device)
            else:
                D_L_Supervised = per_ex_ce.mean()

            D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + EPS))
            D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + EPS))
            d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U

            # ---------------------------------
            #  OPTIMIZATION
            # ---------------------------------
            # Avoid gradient accumulation
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            # Calculate weigth updates
            # retain_graph=True is required since the underlying graph will be deleted after backward
            g_loss.backward(retain_graph=True)
            d_loss.backward()

            # Apply modifications
            gen_optimizer.step()
            dis_optimizer.step()

            # A detail log of the individual losses
            # print("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".
            #      format(D_L_Supervised, D_L_unsupervised1U, D_L_unsupervised2U,
            #             g_loss_d, g_feat_reg))

            # Save the losses to print them later
            tr_g_loss += g_loss.item()
            tr_d_loss += d_loss.item()

            # Update the learning rate with the scheduler
            if apply_scheduler:
                scheduler_d.step()
                scheduler_g.step()

        dev_metrics = evaluate(transformer, discriminator, SA1_fusion, CA_fusion, test_dataloader, device)
        curr_f1 = dev_metrics["mf1"]
        val_f1s.append(curr_f1)
        # Calculate the average loss over all of the batches.
        avg_train_loss_g = tr_g_loss / len(train_dataloader)
        avg_train_loss_d = tr_d_loss / len(train_dataloader)

        log_print(f"  [Epoch {epoch_i + 1}/{epochs}] Results")
        log_print(f"  Generator Loss : {avg_train_loss_g:.4f}")
        log_print(f"  Discriminator Loss : {avg_train_loss_d:.4f}")
        log_print(f"  Macro-F1 : {dev_metrics['mf1']:.4f}")
        log_print(f"  Weighted-F1 : {dev_metrics['wf1']:.4f}")
        log_print(f"  Precision : {dev_metrics['precision']:.4f}")
        log_print(f"  Recall : {dev_metrics['recall']:.4f}")
        log_print(f"  Accuracy : {dev_metrics['acc']:.4f}")
        log_print("-" * 70)
        log_print("  Per-class Report:")
        log_print(dev_metrics["report"])
        log_print("=" * 70 + "\n")

        # ====== Early Stopping 逻辑 ======
        if best_f1 < 0 or (curr_f1 - best_f1) > MIN_DELTA:
            # 有提升
            best_f1 = curr_f1
            best_epoch = epoch_i
            best_metrics = dev_metrics
            no_improve_epochs = 0

        else:
            # 无明显提升
            no_improve_epochs += 1
            if use_early_stopping and no_improve_epochs >= PATIENCE:
                print(
                    f"===> Early stopping triggered at epoch {epoch_i + 1} "
                    f"(no improvement in last {PATIENCE} epochs)."
                )
                break
    cleanup_cuda()
    # ====== 最优结果 ======
    total_time = format_time(time.time() - start_time)
    log_print("\n===== 单模型实验完成 =====")
    if fold_id is not None:
        log_print(f"[Fold {fold_id}] Best epoch index = {best_epoch}")
    log_print(f"最终 macro-F1: {best_metrics['mf1']:.4f}")
    log_print(f"最终 weighted-F1: {best_metrics['wf1']:.4f}")
    log_print(f"耗时: {total_time}")

    print(f"耗时: {total_time}")
    if log_f is not None:
        log_f.close()

    return best_epoch, best_metrics


@torch.no_grad()
def evaluate(transformer, discriminator, SA1_fusion, CA_fusion, eval_loader, device):
    transformer.eval()
    discriminator.eval()

    if USE_SA1 and SA1_fusion is not None:
        SA1_fusion.eval()

    if USE_CA and CA_fusion is not None:
        CA_fusion.eval()

    total_loss = 0.0
    nll = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []

    for step, batch in enumerate(eval_loader):
        if MAX_EVAL_BATCHES is not None and step >= MAX_EVAL_BATCHES:
            break

        b_input_ids, b_attn_mask, b_labels, _ = [t.to(device) for t in batch]

        # Encode real data in the Transformer
        token_states, v_cls = get_sentence_embedding(transformer, b_input_ids, b_attn_mask)

        if USE_SA1 and SA1_fusion is not None:
            x_base = SA1_fusion(token_states, b_attn_mask)
        else:
            x_base = v_cls

        if USE_CA and CA_fusion is not None:
            x_real = CA_fusion(x_base=x_base, token_states=token_states, attention_mask=b_attn_mask)
        else:
            x_real = x_base

        _, logits, _ = discriminator(x_real)

        filtered_logits = logits[:, :-1]
        total_loss += float(nll(filtered_logits, b_labels).item())

        _, preds = torch.max(filtered_logits, 1)
        all_preds += preds.cpu().tolist()
        all_labels += b_labels.cpu().tolist()

    # 宏平均 (排除 UNK_UNK=0)
    valid_labels = list(range(1, len(LABEL_LIST)))
    precision = precision_score(all_labels, all_preds, average="macro",
                                labels=valid_labels, zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro",
                          labels=valid_labels, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro",
                  labels=valid_labels, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    wf1 = f1_score(all_labels, all_preds, average="weighted",
                   labels=valid_labels, zero_division=0)

    # 每类报告（同样排除 UNK_UNK）
    report_text = classification_report(
        all_labels, all_preds,
        labels=valid_labels,
        target_names=[LABEL_LIST[i] for i in valid_labels],
        digits=4,
        zero_division=0
    )

    avg_loss = total_loss / max(1, (step + 1))
    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "mf1": f1,
        "wf1": wf1,
        "report": report_text
    }

class SA1Fusion(nn.Module):
    def __init__(self, hidden_size: int,
                 attn_hidden: int = SA1_ATTN_HIDDEN,
                 dropout: float = SA1_DROPOUT,
                 tau: float = SA1_ATTENTION_TAU,
                 exclude_cls: bool = SA1_EXCLUDE_CLS,
                 gate_bias_init: float = SA1_GATE_BIAS_INIT):
        super().__init__()
        self.attn_w = nn.Linear(hidden_size, attn_hidden)
        self.attn_v = nn.Linear(attn_hidden, 1, bias=False)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.tau = tau
        self.exclude_cls = exclude_cls

        # 门控偏置初始化：让g初期更偏向v_cls
        nn.init.constant_(self.gate.bias, gate_bias_init)

    def forward(self, last_hidden_state, attention_mask):
        H = last_hidden_state                          # [B,T,H]
        mask = attention_mask.unsqueeze(-1).float()     # [B,T,1]
        v_cls = H[:, 0, :]                              # [B,H]

        H_pool = H
        mask_pool = mask

        e = self.attn_v(torch.tanh(self.attn_w(self.dropout(H_pool))))  # [B,T',1]
        e = e.masked_fill(mask_pool == 0, -1e9)

        # 温度tau：tau>1更平滑
        a = torch.softmax(e / self.tau, dim=1)          # [B,T',1]
        v_att = torch.sum(a * H_pool, dim=1)            # [B,H]

        g = torch.sigmoid(self.gate(torch.cat([v_cls, v_att], dim=-1)))
        v_fused = g * v_cls + (1.0 - g) * v_att
        return v_fused

# Encode real data in the Transformer
def get_sentence_embedding(transformer, input_ids, attention_mask):
    outputs = transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    token_states = outputs.last_hidden_state      # [B,T,H]
    v_cls = token_states[:, 0, :]                 # [B,H]
    return token_states, v_cls

def cuda_cleanup(device=None):
    import gc
    import torch

    # 先断开全局 scheduler 引用（避免 optimizer/param 仍被持有）
    for _n in ["scheduler_d", "scheduler_g"]:
        if _n in globals():
            try:
                globals()[_n] = None
                del globals()[_n]
            except Exception:
                globals()[_n] = None

    # 多轮 GC，尽量清循环引用
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        try:
            if device is None:
                device = torch.cuda.current_device()
        except Exception:
            device = 0

        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass

        # 清缓存 + IPC 回收
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        # 重置统计（不释放显存，但便于你看“是否还在涨”）
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
        try:
            torch.cuda.reset_accumulated_memory_stats(device)
        except Exception:
            pass

    # 再来一轮 GC
    for _ in range(2):
        gc.collect()

def run_single_experiment(dataset_name, data_dir,
                          bert_name, generator_name, discriminator_name,
                          labeled_ratio=0.8, epochs=10,
                          output_file="single_result.log", act=None):
    """
    训练并评估单个 (BERT, Generator, Discriminator) 组合模型
    """
    set_seed(seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    start_time = time.time()
    # ====== 数据准备 ======
    global LABEL_LIST
    LABEL_LIST = build_label_list(data_dir)
    # =========================
    # 1) 准备K折汇总容器
    # =========================
    fold_best_epochs = []  # 每折最优epoch
    fold_best_metrics = []

    # 总日志头
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write(f"===== Dataset: {dataset_name} (KFold NoFunction) =====\n")
        f.write(f"BERT={bert_name}, G={generator_name}, D={discriminator_name}, act={act}\n")
        f.write(f"KFold: k={n_splits}, labeled_ratio={labeled_ratio}, base_seed={seed}\n")
        f.write(f"switch: USE_CLASS_WEIGHT={USE_CLASS_WEIGHT}, USE_SA1={USE_SA1}, USE_CA={USE_CA}, USE_AS={USE_AS}\n")
        f.write("=========================================\n")
    detail_file = make_detail_log_path(output_file)

    with open(detail_file, "a", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write(f"===== DETAIL LOG | Dataset: {dataset_name} =====\n")
        f.write(f"BERT={bert_name}, G={generator_name}, D={discriminator_name}, act={act}\n")
        f.write(f"KFold: k={n_splits}, labeled_ratio={labeled_ratio}, base_seed={seed}\n")
        f.write(f"switch: USE_CLASS_WEIGHT={USE_CLASS_WEIGHT}, USE_SA1={USE_SA1}, USE_CA={USE_CA}, USE_AS={USE_AS}\n")
        f.write(f"start_at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=========================================\n\n")

    # =========================
    # 2) K折主循环：fold_id=0..k-1
    # =========================
    for fold_id in range(n_splits):
        fold_seed = seed + fold_id
        set_seed(fold_seed)  # 每折可复现
        with open(detail_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"===== Fold {fold_id + 1}/{n_splits} | fold_seed={fold_seed} =====\n")
            f.write("=" * 90 + "\n")

        # ---- 2.1 K折划分：k-1折训练，1折测试；训练集再按labeled_ratio分labeled/unlabeled
        labeled, unlabeled, test_examples = stratified_split_dataset_splite(
            data_dir,
            labeled_ratio=labeled_ratio,
            k=n_splits,
            fold_id=fold_id,
            seed=fold_seed
        )
        print(f"\n===== {fold_id + 1}/{n_splits} =====")
        print(f"Total labeled={len(labeled)}, unlabeled={len(unlabeled)}, test={len(test_examples)}")

        # ====== 构建模型组件 ======
        comps = build_ganbert_components(
            model_name=bert_name,
            generator_name=generator_name,
            discriminator_name=discriminator_name,
            num_labels=len(LABEL_LIST),
            noise_size=100,
            multi_gpu=False
        )
        tokenizer = comps["tokenizer"]
        transformer = comps["transformer"]
        generator = comps["generator"]
        discriminator = comps["discriminator"]
        device = comps["device"]
        if USE_SA1:
            SA1_fusion = SA1Fusion(hidden_size=transformer.config.hidden_size).to(device)
        else:
            SA1_fusion = None

        if USE_CA:
            num_classes_for_ca = len(LABEL_LIST) - 1  # 过滤unk类

            CA_fusion = CategoryAttentionFusion(
                hidden_size=transformer.config.hidden_size,
                num_classes=num_classes_for_ca,
                dropout=CA_DROPOUT,
                tau=CA_ROUTER_TAU
            ).to(device)

            init_label_embeddings_from_transformer(
                transformer=transformer,
                tokenizer=tokenizer,
                ca_module=CA_fusion,
                label_list=LABEL_LIST,
                device=device,
                max_len=16
            )
        else:
            CA_fusion = None

        # 若指定了激活函数，则替换为“同构+自选激活”的生成器
        if act is not None:
            try:
                hidden_size = transformer.config.hidden_size
                generator = build_generator_with_activation(
                    base_name=generator_name,
                    act_name=act,
                    noise_size=100,
                    num_classes=len(LABEL_LIST),
                    output_size=hidden_size,
                    use_ln=False
                ).to(device)
                print(f"Use generator activation = {act}")
            except Exception as e:
                print(f"[Warn] build_generator_with_activation failed ({e}). Fallback to original generator.")

        label_map = {}
        for (i, label) in enumerate(LABEL_LIST):
            label_map[label] = i

        # ====== 类权重（基于 labeled）======
        num_labels = len(LABEL_LIST)
        # 默认：各类均匀
        ce_weights = torch.ones(num_labels, dtype=torch.float)
        if USE_CLASS_WEIGHT:
            # ====== 类权重（基于 labeled）======
            label_indices = [label_map[ex[1]] for ex in labeled]
            class_counts = np.bincount(label_indices, minlength=num_labels)
            safe_counts = np.maximum(class_counts, 1)
            weights_np = 1.0 / safe_counts
            weights_np = weights_np * (num_labels / np.sum(weights_np))
            ce_weights = torch.tensor(weights_np, dtype=torch.float)

        # 如果要放到 GPU：
        # ce_weights = ce_weights.to(device)

        # ------------------------------
        #   Load the train dataset
        # ------------------------------
        train_examples = labeled
        # The labeled (train) dataset is assigned with a mask set to True
        train_label_masks = np.ones(len(labeled), dtype=bool)
        # If unlabel examples are available
        if unlabeled:
            train_examples = train_examples + unlabeled
            # The unlabeled (train) dataset is assigned with a mask set to False
            tmp_masks = np.zeros(len(unlabeled), dtype=bool)
            train_label_masks = np.concatenate([train_label_masks, tmp_masks])
        train_dataloader = generate_data_loader(train_examples, train_label_masks, tokenizer, label_map=label_map,
                                                batch_size=BATCH_SIZE, do_shuffle=True,
                                                balance_label_examples=APPLY_BALANCE)

        # ------------------------------
        #   Load the test dataset
        # ------------------------------
        # The labeled (test) dataset is assigned with a mask set to True
        test_label_masks = np.ones(len(test_examples), dtype=bool)

        test_dataloader = generate_data_loader(test_examples, test_label_masks, tokenizer, label_map=label_map,
                                               batch_size=BATCH_SIZE, do_shuffle=False,
                                               balance_label_examples=False)

        # ====== 优化器 ======
        # models parameters
        transformer_vars = [i for i in transformer.parameters()]
        d_vars = transformer_vars + [v for v in discriminator.parameters()]

        if USE_SA1 and SA1_fusion is not None:
            d_vars += [v for v in SA1_fusion.parameters()]

        if USE_CA and CA_fusion is not None:
            d_vars += [v for v in CA_fusion.parameters()]

        g_vars = [v for v in generator.parameters()]

        dis_optimizer = torch.optim.AdamW(d_vars, lr=lr)
        gen_optimizer = torch.optim.AdamW(g_vars, lr=lr)
        scheduler_d = None
        scheduler_g = None
        # scheduler
        if apply_scheduler:
            num_train_examples = len(train_examples)
            num_train_steps = int(num_train_examples / BATCH_SIZE * epochs)
            num_warmup_steps = int(num_train_steps * warmup_proportion)
            scheduler_d = get_constant_schedule_with_warmup(dis_optimizer,
                                                            num_warmup_steps=num_warmup_steps)
            scheduler_g = get_constant_schedule_with_warmup(gen_optimizer,
                                                            num_warmup_steps=num_warmup_steps)
        # ====== 训练 AND TEST======
        best_epoch, best_metrics = allTrain(epochs, transformer, generator, discriminator, train_dataloader,
                                               test_dataloader, device, LABEL_LIST, gen_optimizer, dis_optimizer,
                                            SA1_fusion, CA_fusion, ce_weights, scheduler_d, scheduler_g, detail_file, fold_id + 1)
        fold_best_epochs.append(best_epoch)
        fold_best_metrics.append(best_metrics)
        # 提取各折指标
        mf1_list = [m["mf1"] for m in fold_best_metrics]
        wf1_list = [m["wf1"] for m in fold_best_metrics]
        # epoch 统计
        epoch_mean = np.mean(fold_best_epochs)
        epoch_std = np.std(fold_best_epochs)

        # F1 统计
        mf1_mean = np.mean(mf1_list)
        mf1_std = np.std(mf1_list)

        wf1_mean = np.mean(wf1_list)
        wf1_std = np.std(wf1_list)

        # ====== Fold结束后：强制释放GPU对象 ======
        # 先删大对象
        del transformer, generator, discriminator
        del dis_optimizer, gen_optimizer
        del train_dataloader, test_dataloader, tokenizer, comps
        del SA1_fusion, CA_fusion
        del scheduler_d, scheduler_g

        # 也把本折的数据对象删掉（避免某些闭包/引用链）
        del labeled, unlabeled, test_examples
        del train_examples, train_label_masks, test_label_masks

        cuda_cleanup(device=0)

        print(torch.cuda.memory_summary(device=0, abbreviated=True))


    total_time = format_time(time.time() - start_time)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write(f"===== Dataset: {dataset_name} (K-Fold Summary) =====\n")
        f.write(f"BERT={bert_name}, G={generator_name}, D={discriminator_name}, act={act}\n")
        f.write(f"K={n_splits}, Labeled ratio={labeled_ratio}\n")
        f.write(f"Epochs={epochs}, Batch={BATCH_SIZE}, Base Seed={seed}\n")
        f.write("-----------------------------------------\n")

        # 每折明细
        for i in range(n_splits):
            f.write(
                f"Fold {i + 1}: "
                f"best_epoch={fold_best_epochs[i]}, "
                f"mf1={mf1_list[i]:.4f}, "
                f"wf1={wf1_list[i]:.4f}\n"
            )

        f.write("-----------------------------------------\n")
        f.write(f"Best Epoch (mean ± std): {epoch_mean:.2f} ± {epoch_std:.2f}\n")
        f.write(f"Macro-F1  (mean ± std): {mf1_mean:.4f} ± {mf1_std:.4f}\n")
        f.write(f"Weighted-F1 (mean ± std): {wf1_mean:.4f} ± {wf1_std:.4f}\n")
        f.write(f"Total Time: {total_time}\n")
        f.write("=========================================\n\n")


    return {
        "mf1_mean": mf1_mean,
        "mf1_std": mf1_std,
        "wf1_mean": wf1_mean,
        "wf1_std": wf1_std,
        "epoch_mean": epoch_mean,
        "epoch_std": epoch_std,
        "fold_epochs": fold_best_epochs,
        "fold_metrics": fold_best_metrics
    }
