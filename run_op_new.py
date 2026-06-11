from __future__ import annotations

import gc
import os
import math
import time
import random
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
from datetime import timedelta, datetime
import re
from G.generators_act import build_generator_with_activation
# -----------------------------
#  组合构建函数
# -----------------------------
from data.dataTrent.dataLoader import build_label_list, stratified_split_dataset, stratified_split_dataset_splite
from experimentLogUtil.draw import plot_metrics_curve_dual_axis
from model_builder import build_ganbert_components, _map_generator_impl
from util import cuda_cleanup


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
# 类别不平衡开关（交叉熵权重）
USE_CLASS_WEIGHT = False

epochs=30
n_splits = 10

## s1 object参数设置
s1_epochs = 30
s1_apply_scheduler = True
s1_n_splits = 3

## s2 object参数设置
s2_epochs = 30
s2_apply_scheduler = True
s2_n_splits = 3

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
SA1_EXCLUDE_CLS = False      # 默认排除CLS参与v_att

# 5) 门控融合层 gate 的 bias 初始化（让模型训练初期更像baseline）
# - bias越大：sigmoid(bias)越接近1，初期更偏向v_cls（稳定）
# - bias=0：初期g≈0.5，v_cls和v_att一上来就混合，可能扰动过大导致效果变差
SA1_GATE_BIAS_INIT = 2.0    # 让g初期更偏向v_cls（sigmoid(2)≈0.88）


# 使用SA2（判别器头：CLS->token cross-attn）开关
USE_SA2 = False

# SA2超参
SA2_NUM_HEADS = 8
SA2_DROPOUT = 0.1

USE_SA3 = False
SA3_ENTROPY_THRESHOLD = 0.6   # 归一化熵阈值：[0,1]，越小越严格
SA3_PSEUDO_LOSS_WEIGHT = 0.5  # 伪标签监督项权重


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
    return str(timedelta(seconds=int(round(sec))))


def allTrain(epochs, transformer, generator, discriminator, train_dataloader, test_dataloader, device, label_list,
             gen_optimizer, dis_optimizer, ce_weights, scheduler_d, scheduler_g, report_path=None):
    val_f1s = []
    start_time = time.time()
    best_report_text = None

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

            # Encode real data in the Transformer
            hidden_states = get_sentence_embedding(transformer, b_input_ids, b_input_mask)

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
            # First, we put together the output of the tranformer and the generator
            disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
            # Then, we select the output of the disciminator
            features, logits, probs = discriminator(disciminator_input)

            # Finally, we separate the discriminator's output for the real and fake
            # data
            features_list = torch.split(features, real_batch_size)
            D_real_features = features_list[0]
            D_fake_features = features_list[1]

            logits_list = torch.split(logits, real_batch_size)
            D_real_logits = logits_list[0]
            D_fake_logits = logits_list[1]

            probs_list = torch.split(probs, real_batch_size)
            D_real_probs = probs_list[0]
            D_fake_probs = probs_list[1]

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

            # Update the learning rate with the scheduler
            if s1_apply_scheduler:
                scheduler_d_s1.step()
                scheduler_g_s1.step()

            # Update the learning rate with the scheduler
            if s2_apply_scheduler:
                scheduler_d_s2.step()
                scheduler_g_s2.step()

        dev_metrics = evaluate(transformer, discriminator, test_dataloader, device)
        curr_f1 = dev_metrics["mf1"]
        val_f1s.append(curr_f1)
        # Calculate the average loss over all of the batches.
        avg_train_loss_g = tr_g_loss / len(train_dataloader)
        avg_train_loss_d = tr_d_loss / len(train_dataloader)

        print(f"  [Epoch {epoch_i + 1}/{epochs}] Results")
        print(f"  Generator Loss : {avg_train_loss_g:.4f}")
        print(f"  Discriminator Loss : {avg_train_loss_d:.4f}")
        print(f"  Macro-F1 : {dev_metrics['mf1']:.4f}")
        print(f"  Weighted-F1 : {dev_metrics['wf1']:.4f}")
        print(f"  Macro-Precision : {dev_metrics['mp']:.4f}")
        print(f"  Weighted-Precision : {dev_metrics['wp']:.4f}")
        print(f"  Macro-Recall : {dev_metrics['mr']:.4f}")
        print(f"  Weighted-Recall : {dev_metrics['wr']:.4f}")
        print(f"  Accuracy : {dev_metrics['acc']:.4f}")
        print("-" * 70)
        print("  Per-class Report:")
        print(dev_metrics["report"])
        print("=" * 70 + "\n")

        # ========= 写入该fold的单一report文件：每代一段 =========
        if report_path is not None:
            os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(f"[Epoch {epoch_i + 1:03d}/{epochs}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Generator Loss     : {avg_train_loss_g:.4f}\n")
                f.write(f"Discriminator Loss : {avg_train_loss_d:.4f}\n")
                f.write(
                    f"Macro-F1={dev_metrics.get('mf1', -1):.4f}, "
                    f"Weighted-F1={dev_metrics.get('wf1', -1):.4f}, "
                    f"w-Precision={dev_metrics.get('wp', -1):.4f}, "
                    f"m-Precision={dev_metrics.get('mp', -1):.4f}, "
                    f"w-Recall={dev_metrics.get('wr', -1):.4f}, "
                    f"m-Recall={dev_metrics.get('mr', -1):.4f}, "
                    f"Accuracy={dev_metrics.get('acc', -1):.4f}\n"
                )
                f.write("Per-class Report:\n")
                f.write(str(dev_metrics.get("report", "")))
                if not str(dev_metrics.get("report", "")).endswith("\n"):
                    f.write("\n")
                f.write("-" * 70 + "\n\n")

        # ====== Early Stopping 逻辑 ======
        if best_f1 < 0 or (curr_f1 - best_f1) > MIN_DELTA:
            # 有提升
            best_f1 = curr_f1
            best_epoch = epoch_i
            best_metrics = dev_metrics
            no_improve_epochs = 0
            best_report_text = dev_metrics.get("report", None)

        else:
            # 无明显提升
            no_improve_epochs += 1
            if use_early_stopping and no_improve_epochs >= PATIENCE:
                print(
                    f"===> Early stopping triggered at epoch {epoch_i + 1} "
                    f"(no improvement in last {PATIENCE} epochs)."
                )
                break

    # ====== 最优结果 ======
    total_time = format_time(time.time() - start_time)
    print("\n===== 单模型实验完成 =====")
    print(f"最优 Epoch: {best_epoch + 1}")
    print(f"最终 macro-F1: {best_metrics['mf1']:.4f}")
    print(f"最终 weighted-F1: {best_metrics['wf1']:.4f}")
    print(f"耗时: {total_time}")
    # ========= 训练结束：在文件末尾追加 BEST 汇总 =========
    if report_path is not None:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write("=========================================\n")
            f.write("[BEST EPOCH SUMMARY]\n")
            # best_epoch 你内部存的是 0-based，这里给人看用 1-based 更清晰
            f.write(f"best_epoch={best_epoch + 1}\n")
            f.write(f"best_mf1={best_metrics.get('mf1', -1):.4f}\n")
            f.write(f"best_wf1={best_metrics.get('wf1', -1):.4f}\n")
            f.write(f"best_weight-precision={best_metrics.get('wp', -1):.4f}\n")
            f.write(f"best_weight-recall={best_metrics.get('wr', -1):.4f}\n")
            f.write(f"best_macro-precision={best_metrics.get('mr', -1):.4f}\n")
            f.write(f"best_macro-recall={best_metrics.get('mr', -1):.4f}\n")
            f.write(f"best_acc={best_metrics.get('acc', -1):.4f}\n")
            f.write(f"total_time={total_time}\n\n")
            f.write("[BEST EPOCH PER-CLASS REPORT]\n")
            if best_report_text is not None:
                f.write(str(best_report_text))
                if not str(best_report_text).endswith("\n"):
                    f.write("\n")
            f.write("=========================================\n\n")

    return best_epoch, best_metrics


@torch.no_grad()
def evaluate(transformer, discriminator, eval_loader, device):
    transformer.eval()
    discriminator.eval()

    total_loss = 0.0
    nll = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []

    for step, batch in enumerate(eval_loader):
        if MAX_EVAL_BATCHES is not None and step >= MAX_EVAL_BATCHES:
            break

        b_input_ids, b_attn_mask, b_labels, _ = [t.to(device) for t in batch]

        # Encode real data in the Transformer
        v_cls = get_sentence_embedding(transformer, b_input_ids, b_attn_mask)

        x_base = v_cls

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
    report_dict = classification_report(
        all_labels, all_preds,
        labels=valid_labels,
        target_names=[LABEL_LIST[i] for i in valid_labels],
        zero_division=0,
        output_dict=True
    )

    mp = report_dict["macro avg"]["precision"]
    mr = report_dict["macro avg"]["recall"]
    mf1 = report_dict["macro avg"]["f1-score"]

    wp = report_dict["weighted avg"]["precision"]
    wr = report_dict["weighted avg"]["recall"]
    wf1 = report_dict["weighted avg"]["f1-score"]


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
        "mp": mp, "mr": mr, "mf1": mf1,
        "wp": wp, "wr": wr, "wf1": wf1,
        "report": report_text
    }


# Encode real data in the Transformer
def get_sentence_embedding(transformer, input_ids, attention_mask):
    # 前向计算得到 Transformer 的所有输出，
    # last_hidden_state 维度为 (batch_size, seq_len, hidden_size)
    outputs = transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    # 取 last_hidden_state 的第 0 个位置的向量：
    #   对 BERT 来说，这个位置是 [CLS]
    #   对 RoBERTa 来说，这个位置是 <s>
    # [CLS] or <s> at position 0 (都是特征提取后的句子向量)
    return outputs.last_hidden_state[:, 0, :]

## 清理显存
def _gpu_cleanup():
    try:
        # 等待所有 CUDA kernel 结束，避免释放时仍在使用
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        # 跨进程/worker 的显存句柄回收（有 DataLoader 多进程时更有用）
        torch.cuda.ipc_collect()
    except Exception:
        pass

    # 先做 Python 垃圾回收，再清空 CUDA 缓存
    try:
        gc.collect()
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# =========================
# Optuna 目标函数 s1
# =========================

def make_objective_stage1(
        dataset_name: str
):
    """
    返回一个 closure：objective(trial)
    这个 objective 会在 K-Fold 中调用 run_single_experiment_from_split(...)，
    而 run_single_experiment_from_split 需要 dataset_name，所以这里把 dataset_name 闭包捕获。
    """
    set_seed(seed)
    # 预先做标签到索引的映射（与全局保持一致）
    labels_only = [ex[1] for ex in labeled]

    def objective(trial: optuna.trial.Trial) -> float:
        # ========== 采样 Stage-1 的结构（示例）==========
        bert_name = trial.suggest_categorical(
            "bert_name",
            [
                "bert-base-cased", "bert-base-uncased",
                "bert-large-cased", "bert-large-uncased",
                "distilbert-base-uncased", "distilbert-base-cased",
                "roberta-base", "roberta-large",
                "xlm-roberta-base", "albert-base-v1", "albert-base-v2"
            ]
        )
        generator_base_name = trial.suggest_categorical("generator_name", [
            "mlp_base", "mlp_deep", "res_mlp", "cnn", "transformer_light"
        ])
        ## 是否condition标签
        gen_is_cond = trial.suggest_categorical("gen_is_conditional", [False, True])
        generator_name = _map_generator_impl(generator_base_name, gen_is_cond)

        discriminator_name = trial.suggest_categorical("discriminator_name", [
            "mlp_base", "mlp_deep", "res_mlp", "attention_head"
        ])

        # ========== K-Fold ==========
        skf = StratifiedKFold(n_splits=s1_n_splits, shuffle=True, random_state=42)
        labeled_arr = np.array(labeled, dtype=object)

        fold_best_f1s = []

        for fold_idx, (tr_idx, dv_idx) in enumerate(skf.split(labeled_arr, labels_only), start=1):
            print(f"[Stage1][Trial {trial.number}] Start KFold: "
                  f"BERT={bert_name}, G={generator_name} (cond={gen_is_cond}), D={discriminator_name}, "
                  f"folds={s1_n_splits}")
            labeled_train = labeled_arr[tr_idx].tolist()
            labeled_dev = labeled_arr[dv_idx].tolist()

            try:
                # —— 每折调用“预切分版”的单模型实验（这里就能传 dataset_name）——
                best_f1_fold, dev_metrics = run_single_experiment_from_split(
                    dataset_name=dataset_name,
                    bert_name=bert_name,
                    generator_name=generator_name,
                    discriminator_name=discriminator_name,
                    labeled_train=labeled_train,
                    labeled_dev=labeled_dev,
                    unlabeled=unlabeled,
                    LABEL_LIST=LABEL_LIST,
                    epochs=s1_epochs,
                    output_file=None,
                    log_tag=f"[trial={trial.number}] fold={fold_idx}",
                )

                # 向 Optuna 报告（便于 pruner）
                trial.report(dev_metrics['mf1'], step=(fold_idx - 1) * s1_epochs)
                if trial.should_prune():
                    print(f"[Trial {trial.number}] Pruned after fold {fold_idx}.")
                    _gpu_cleanup()  # prune 时也清理
                    raise optuna.TrialPruned()

                fold_best_f1s.append(dev_metrics['mf1'])
                print(f"[Trial {trial.number}] Fold {fold_idx}/{s1_n_splits} best Dev F1 = {best_f1_fold:.4f}")

            finally:
                # 跨折之前做一次轻清理，避免显存积累
                _gpu_cleanup()

        avg_f1 = float(np.mean(fold_best_f1s))
        print(
            f"[Trial {trial.number}] Avg Dev F1 across {s1_n_splits} folds = {avg_f1:.4f} | per-fold = {fold_best_f1s}")
        trial.set_user_attr("fold_f1s", [float(x) for x in fold_best_f1s])
        trial.set_user_attr("n_splits", s1_n_splits)

        # 整轮 K-Fold 结束再清一次
        _gpu_cleanup()

        return avg_f1

    return objective


def run_single_experiment(dataset_name, data_dir,
                          bert_name, generator_name, discriminator_name,
                          labeled_ratio=0.8, test_ratio=0.2, epochs=10,
                          output_file="single_result.log", act=None):
    """
    训练并评估单个 (BERT, Generator, Discriminator) 组合模型
    """
    set_seed(seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    start_time = time.time()

    # ====== 数据准备 ======
    global labeled, unlabeled, LABEL_LIST
    LABEL_LIST = build_label_list(data_dir)
    labeled, unlabeled, test_examples = stratified_split_dataset(
        data_dir,
        labeled_ratio=labeled_ratio,
        test_ratio=test_ratio
    )

    print(f"\n===== 开始单模型训练 =====")
    print(f"Dataset={dataset_name}")
    print(f"BERT={bert_name}, Generator={generator_name}, Discriminator={discriminator_name}")
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
    g_vars = [v for v in generator.parameters()]

    dis_optimizer = torch.optim.AdamW(d_vars, lr=lr)
    gen_optimizer = torch.optim.AdamW(g_vars, lr=lr)

    # scheduler
    if apply_scheduler:
        num_train_examples = len(train_examples)
        num_train_steps = int(num_train_examples / BATCH_SIZE * epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        global scheduler_d
        global scheduler_g
        scheduler_d = get_constant_schedule_with_warmup(dis_optimizer,
                                                        num_warmup_steps=num_warmup_steps)
        scheduler_g = get_constant_schedule_with_warmup(gen_optimizer,
                                                        num_warmup_steps=num_warmup_steps)

    if s1_apply_scheduler:
        num_train_steps = int(len(train_examples) / BATCH_SIZE * epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        global scheduler_d_s1, scheduler_g_s1
        scheduler_d_s1 = get_constant_schedule_with_warmup(dis_optimizer, num_warmup_steps=num_warmup_steps)
        scheduler_g_s1 = get_constant_schedule_with_warmup(gen_optimizer, num_warmup_steps=num_warmup_steps)

    if s2_apply_scheduler:
        num_train_steps = int(len(train_examples) / BATCH_SIZE * epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        global scheduler_d_s2, scheduler_g_s2
        scheduler_d_s2 = get_constant_schedule_with_warmup(dis_optimizer, num_warmup_steps=num_warmup_steps)
        scheduler_g_s2 = get_constant_schedule_with_warmup(gen_optimizer, num_warmup_steps=num_warmup_steps)
    # ====== 训练 AND TEST======
    best_epoch, best_metrics = allTrain(epochs, transformer, generator, discriminator, train_dataloader,
                                           test_dataloader, device, LABEL_LIST, gen_optimizer, dis_optimizer,
                                           ce_weights)

    total_time = format_time(time.time() - start_time)
    # ====== 写日志 ======
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write(f"===== Dataset: {dataset_name} (Single Model) =====\n")
        f.write(f"BERT={bert_name}, G={generator_name}, D={discriminator_name}, act = {act}\n")
        f.write(f"Labeled ratio: {labeled_ratio}, Test ratio: {test_ratio}\n")
        f.write(f"Epochs: {epochs}, Batch: {BATCH_SIZE}, Seed: {seed}\n")
        f.write("-----------------------------------------\n")
        f.write(f"Best Epoch F1: {best_epoch:.4f}\n")
        f.write(f"Final Macro-F1: {best_metrics['mf1']:.4f}\n")
        f.write(f"Final Weighted-F1: {best_metrics['wf1']:.4f}\n")
        f.write(f"Metrics Report:\n{best_metrics['report']}\n")
        f.write(f"Total Time: {total_time}\n")
        f.write("=========================================\n\n")
    return best_epoch, best_metrics

def _safe_name(s: str) -> str:
    # 只保留字母数字与少量符号，避免路径非法字符
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s))

def _ratio_tag(r: float) -> str:
    # ratio_0p200 这种形式稳定、可排序、不会受浮点误差影响
    x = int(round(r * 100))  # 0.2 -> 20
    return f"ratio_{x:03d}"   # ratio_200 表示 0.200

def run_single_experiment_v2(dataset_name, data_dir,
                          bert_name, generator_name, discriminator_name,
                          labeled_ratio=0.2, epochs=10,
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

    # ---- 4) 输出准备 ----
    root_dir = os.path.dirname(output_file) or "."
    os.makedirs(root_dir, exist_ok=True)

    # 1) 时间文件夹：如果 root_dir 已经是 YYYYMMDD，就复用；否则新建 YYYYMMDD 子目录
    date_tag = datetime.now().strftime("%Y%m%d")
    base_dir = os.path.basename(os.path.normpath(root_dir))
    time_dir = root_dir if (len(base_dir) == 8 and base_dir.isdigit()) else os.path.join(root_dir, date_tag)
    os.makedirs(time_dir, exist_ok=True)

    tag = re.search(r"(\d{6})\.txt$", output_file).group(1)
    # 2) ratio 目录：放在时间文件夹 time_dir 下
    run_tag = _safe_name(f"{dataset_name}_GANBERT_{bert_name}")
    ratio_dir = os.path.join(time_dir, f"{_ratio_tag(labeled_ratio)}_{run_tag}_{tag}")
    os.makedirs(ratio_dir, exist_ok=True)

    # 总日志头
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write(f"===== Dataset: {dataset_name} (KFold NoFunction) =====\n")
        f.write(f"BERT={bert_name}, G={generator_name}, D={discriminator_name}, act={act}\n")
        f.write(f"KFold: k={n_splits}, labeled_ratio={labeled_ratio}, base_seed={seed}\n")
        f.write(f"switch: USE_CLASS_WEIGHT={USE_CLASS_WEIGHT}, USE_SA1={USE_SA1}, USE_SA2={USE_SA2}, USE_SA3={USE_SA3}\n")
        f.write("=========================================\n")
    # =========================
    # 2) K折主循环：fold_id=0..k-1
    # =========================
    for fold_id in range(n_splits):


        detail_report = os.path.join(ratio_dir, f"fold_{fold_id}.report.txt")

        # 每次运行同一折时覆盖写（保证文件内容只对应本次run）
        with open(detail_report, "a", encoding="utf-8") as f:
            f.write("=========================================\n")
            f.write(f"Dataset={dataset_name}\n")
            f.write(f"BERT={bert_name}, G={generator_name}, D={discriminator_name}, act={act}\n")
            f.write(f"KFold: fold_id={fold_id + 1}/{n_splits}, labeled_ratio={labeled_ratio}, fold_seed={seed + fold_id + 1}\n")
            f.write(
                f"switch: USE_CLASS_WEIGHT={USE_CLASS_WEIGHT}, USE_SA1={USE_SA1}, USE_SA2={USE_SA2}, USE_SA3={USE_SA3}\n")
            f.write(f"StartTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=========================================\n\n")

        fold_seed = seed + fold_id
        set_seed(fold_seed)  # 关键：每折可复现

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

        if s1_apply_scheduler:
            num_train_steps = int(len(train_examples) / BATCH_SIZE * epochs)
            num_warmup_steps = int(num_train_steps * warmup_proportion)
            global scheduler_d_s1, scheduler_g_s1
            scheduler_d_s1 = get_constant_schedule_with_warmup(dis_optimizer, num_warmup_steps=num_warmup_steps)
            scheduler_g_s1 = get_constant_schedule_with_warmup(gen_optimizer, num_warmup_steps=num_warmup_steps)

        if s2_apply_scheduler:
            num_train_steps = int(len(train_examples) / BATCH_SIZE * epochs)
            num_warmup_steps = int(num_train_steps * warmup_proportion)
            global scheduler_d_s2, scheduler_g_s2
            scheduler_d_s2 = get_constant_schedule_with_warmup(dis_optimizer, num_warmup_steps=num_warmup_steps)
            scheduler_g_s2 = get_constant_schedule_with_warmup(gen_optimizer, num_warmup_steps=num_warmup_steps)
        # ====== 训练 AND TEST======
        best_epoch, best_metrics = allTrain(epochs, transformer, generator, discriminator, train_dataloader,
                                               test_dataloader, device, LABEL_LIST, gen_optimizer, dis_optimizer, ce_weights, scheduler_d, scheduler_g,
                                            detail_report)
        fold_best_epochs.append(best_epoch)
        fold_best_metrics.append(best_metrics)
        # 提取各折指标
        mf1_list = [m["mf1"] for m in fold_best_metrics]
        wf1_list = [m["wf1"] for m in fold_best_metrics]
        mp_list = [m["mp"] for m in fold_best_metrics]
        mr_list = [m["mr"] for m in fold_best_metrics]
        wp_list = [m["wp"] for m in fold_best_metrics]
        wr_list = [m["wr"] for m in fold_best_metrics]

        # epoch 统计
        epoch_mean = np.mean(fold_best_epochs)
        epoch_std = np.std(fold_best_epochs)

        # F1 统计
        mf1_mean = np.mean(mf1_list)
        mf1_std = np.std(mf1_list)

        wf1_mean = np.mean(wf1_list)
        wf1_std = np.std(wf1_list)

        # Precision/Recall 统计
        mp_mean = float(np.mean(mp_list));
        mp_std = float(np.std(mp_list))
        mr_mean = float(np.mean(mr_list));
        mr_std = float(np.std(mr_list))
        wp_mean = float(np.mean(wp_list));
        wp_std = float(np.std(wp_list))
        wr_mean = float(np.mean(wr_list));
        wr_std = float(np.std(wr_list))

        # ====== Fold结束后：强制释放GPU对象 ======
        # 先删大对象
        del transformer, generator, discriminator
        del dis_optimizer, gen_optimizer
        del train_dataloader, test_dataloader, tokenizer, comps
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
                f"mp={mp_list[i]:.4f}, mr={mr_list[i]:.4f}, mf1={mf1_list[i]:.4f}, "
                f"wp={wp_list[i]:.4f}, wr={wr_list[i]:.4f}, wf1={wf1_list[i]:.4f}\n"
            )

        f.write("-----------------------------------------\n")
        f.write(f"Best Epoch (mean ± std): {epoch_mean:.2f} ± {epoch_std:.2f}\n")
        f.write(f"Macro-F1  (mean ± std): {mf1_mean:.4f} ± {mf1_std:.4f}\n")
        f.write(f"Weighted-F1 (mean ± std): {wf1_mean:.4f} ± {wf1_std:.4f}\n")
        f.write(f"Macro-P   (mean ± std): {mp_mean:.4f} ± {mp_std:.4f}\n")
        f.write(f"Macro-R   (mean ± std): {mr_mean:.4f} ± {mr_std:.4f}\n")
        f.write(f"Macro-F1  (mean ± std): {mf1_mean:.4f} ± {mf1_std:.4f}\n")

        f.write(f"Weight-P  (mean ± std): {wp_mean:.4f} ± {wp_std:.4f}\n")
        f.write(f"Weight-R  (mean ± std): {wr_mean:.4f} ± {wr_std:.4f}\n")
        f.write(f"Weighted-F1 (mean ± std): {wf1_mean:.4f} ± {wf1_std:.4f}\n")
        f.write(f"Total Time: {total_time}\n")
        f.write("=========================================\n\n")

    return {
        "mf1_mean": mf1_mean,
        "mf1_std": mf1_std,
        "wf1_mean": wf1_mean,
        "wf1_std": wf1_std,

        "mp_mean": mp_mean,
        "mp_std": mp_std,
        "mr_mean": mr_mean,
        "mr_std": mr_std,
        "wp_mean": wp_mean,
        "wp_std": wp_std,
        "wr_mean": wr_mean,
        "wr_std": wr_std,

        "epoch_mean": epoch_mean,
        "epoch_std": epoch_std,

        # 方便外层分析/画图
        "fold_epochs": fold_best_epochs,
        "fold_metrics": fold_best_metrics,
        "fold_mp": mp_list,
        "fold_mr": mr_list,
        "fold_wp": wp_list,
        "fold_wr": wr_list,
        "fold_mf1": mf1_list,
        "fold_wf1": wf1_list,
    }



def run_single_experiment_from_split(
        dataset_name: str,
        bert_name: str, generator_name: str, discriminator_name: str,
        labeled_train: list, labeled_dev: list, unlabeled: list,
        LABEL_LIST: list,
        epochs: int = 10,
        output_file: str | None = None,
        log_tag: str = "",
        gen_activation: str | None = None,
):
    """
    使用“预先给定的 train/dev/unlabeled 切分”训练并在 dev 上评估。
    返回：best_epoch_f1 (dev)、dev_metrics（dict）
    """
    set_seed(seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"目前训练结构为：BERT={bert_name}, G={generator_name}, D={discriminator_name}, act={gen_activation}")
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

    # 若指定了激活函数，则替换为“同构+自选激活”的生成器
    if gen_activation is not None:
        try:
            hidden_size = transformer.config.hidden_size
            generator = build_generator_with_activation(
                base_name=generator_name,
                act_name=gen_activation,
                noise_size=100,
                num_classes=len(LABEL_LIST),
                output_size=hidden_size,
                use_ln=False
            ).to(device)
            print(f"[S2] Use generator activation = {gen_activation}")
        except Exception as e:
            print(f"[Warn] build_generator_with_activation failed ({e}). Fallback to original generator.")

    # ====== label_map ======
    label_map = {label: i for i, label in enumerate(LABEL_LIST)}

    # ====== 类权重（基于 labeled_train）======
    num_labels = len(LABEL_LIST)
    # 默认：各类均匀
    ce_weights = torch.ones(num_labels, dtype=torch.float)

    if USE_CLASS_WEIGHT:
        # ====== 类权重（基于 labeled_train）======
        label_indices = [label_map[ex[1]] for ex in labeled_train]
        class_counts = np.bincount(label_indices, minlength=num_labels)
        safe_counts = np.maximum(class_counts, 1)
        weights_np = 1.0 / safe_counts
        weights_np = weights_np * (num_labels / np.sum(weights_np))
        ce_weights = torch.tensor(weights_np, dtype=torch.float)

    # ====== DataLoader ======
    # 训练集 = labeled_train + unlabeled（半监督）
    train_examples = labeled_train + unlabeled
    train_label_masks = np.concatenate([
        np.ones(len(labeled_train), dtype=bool),
        np.zeros(len(unlabeled), dtype=bool)
    ])
    train_loader = generate_data_loader(
        train_examples, train_label_masks, tokenizer, label_map,
        batch_size=BATCH_SIZE, do_shuffle=True, balance_label_examples=APPLY_BALANCE
    )

    # 验证 / Dev：只用 labeled_dev
    dev_label_masks = np.ones(len(labeled_dev), dtype=bool)
    dev_loader = generate_data_loader(
        labeled_dev, dev_label_masks, tokenizer, label_map,
        batch_size=BATCH_SIZE, do_shuffle=False, balance_label_examples=False
    )

    # ====== 优化器 & scheduler ======
    transformer_vars = [p for p in transformer.parameters()]
    d_vars = transformer_vars + [v for v in discriminator.parameters()]
    g_vars = [v for v in generator.parameters()]
    dis_optimizer = torch.optim.AdamW(d_vars, lr=lr)
    gen_optimizer = torch.optim.AdamW(g_vars, lr=lr)

    # scheduler
    if apply_scheduler:
        num_train_examples = len(train_examples)
        num_train_steps = int(num_train_examples / BATCH_SIZE * epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        global scheduler_d
        global scheduler_g
        scheduler_d = get_constant_schedule_with_warmup(dis_optimizer,
                                                        num_warmup_steps=num_warmup_steps)
        scheduler_g = get_constant_schedule_with_warmup(gen_optimizer,
                                                        num_warmup_steps=num_warmup_steps)
    if s1_apply_scheduler:
        num_train_steps = int(len(train_examples) / BATCH_SIZE * epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        global scheduler_d_s1, scheduler_g_s1
        scheduler_d_s1 = get_constant_schedule_with_warmup(dis_optimizer, num_warmup_steps=num_warmup_steps)
        scheduler_g_s1 = get_constant_schedule_with_warmup(gen_optimizer, num_warmup_steps=num_warmup_steps)

    if s2_apply_scheduler:
        num_train_steps = int(len(train_examples) / BATCH_SIZE * epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        global scheduler_d_s2, scheduler_g_s2
        scheduler_d_s2 = get_constant_schedule_with_warmup(dis_optimizer, num_warmup_steps=num_warmup_steps)
        scheduler_g_s2 = get_constant_schedule_with_warmup(gen_optimizer, num_warmup_steps=num_warmup_steps)
    # ====== 训练 & Dev 评估 ======
    best_epoch, best_metrics = allTrain(
        epochs, transformer, generator, discriminator,
        train_loader, dev_loader, device, LABEL_LIST,
        gen_optimizer, dis_optimizer, ce_weights
    )

    # ====== 落盘日志 ======
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("=========================================\n")
            f.write(
                f"===== Dataset: {dataset_name} (Pre-split Single Model){(' ' + log_tag) if log_tag else ''} =====\n")
            f.write(f"BERT={bert_name}, G={generator_name}, D={discriminator_name}\n")
            f.write(f"Epochs: {epochs}, Batch: {BATCH_SIZE}, Seed: {seed}\n")
            f.write("-----------------------------------------\n")
            f.write(f"Best Epoch F1 (Dev): {best_epoch:.4f}\n")
            f.write(f"Dev Macro-F1: {best_metrics['mf1']:.4f}\n")
            f.write(f"Dev Weighted-F1: {best_metrics['wf1']:.4f}\n")
            f.write(f"Dev Report:\n{best_metrics['report']}\n")
            f.write("=========================================\n\n")
    return best_epoch, best_metrics


# =========================
# 暴力实验函数(s1)
# =========================
def run_experiment_bruteforce_s1(
        dataset_name,
        data_dir,
        labeled_ratio,
        test_ratio,
        output_file,
        epochs=None  # 若为 None，则默认使用全局 s1_epochs
):
    """
    Stage-1 暴力遍历（Bruteforce）搜索：
      - 搜索空间 = {BERT} × {Generator 基类} × {是否 conditional} × {Discriminator}
      - 对每个结构组合做 K 折 (s1_n_splits) 评估：
          * 每折调用 run_single_experiment_from_split(...)
          * 以该折 best_metrics['mf1'] 作为 fold 分数
      - 计算每个结构组合的 KFold 平均 Macro-F1，选取最优结构

    返回：
      best_combo_s1 : dict，包含最优结构信息：
          {
            "bert_name": ...,
            "generator_name": ...,
            "discriminator_name": ...,
            "gen_base": ...,
            "gen_is_conditional": ...
          }
      best_avg_f1   : float，最优结构在 Stage-1 KFold 上的平均 Macro-F1
      all_results   : list，记录所有组合的结果，便于后续画表/画图
    """
    set_seed(seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if epochs is None:
        # 如果没传，就用 Stage-1 全局配置
        epochs = s1_epochs

    overall_start = time.time()

    # ====== 日志：记录本次实验参数 ======
    exp_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    param_log_lines = [
        "=" * 60,
        f"[Experiment] run_experiment_bruteforce_s1 @ {exp_time_str}",
        f"[Params] dataset_name={dataset_name}",
        f"[Params] data_dir={data_dir}",
        f"[Params] labeled_ratio={labeled_ratio}, test_ratio={test_ratio}",
        f"[Params] s1_epochs={epochs}, s1_n_splits={s1_n_splits}, batch_size={BATCH_SIZE}, seed={seed}",
        f"[Params] output_file={output_file}",
        "=" * 60,
        ""
    ]
    for line in param_log_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in param_log_lines:
                f.write(line + "\n")

    # ====== 数据准备（与 version2 一致） ======
    global labeled, unlabeled, LABEL_LIST
    LABEL_LIST = build_label_list(data_dir)
    labeled, unlabeled, test_examples = stratified_split_dataset(
        data_dir,
        labeled_ratio=labeled_ratio,
        test_ratio=test_ratio
    )

    num_labeled = len(labeled)
    num_unlabeled = len(unlabeled)
    num_test = len(test_examples)

    data_log_lines = [
        "[Data] 当前标签集合 LABEL_LIST: " + str(LABEL_LIST),
        f"[Data] 数据集划分结果: #labeled={num_labeled}, #unlabeled={num_unlabeled}, #test={num_test}",
        ""
    ]
    for line in data_log_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in data_log_lines:
                f.write(line + "\n")

    # ====== Stage-1 搜索空间定义（与 make_objective_stage1 对齐） ======
    BERT_LIST = [
        "bert-base-cased", "bert-base-uncased",
        "bert-large-cased", "bert-large-uncased",
        "distilbert-base-uncased", "distilbert-base-cased",
        "roberta-base", "roberta-large",
        "xlm-roberta-base", "albert-base-v1", "albert-base-v2"
    ]
    GEN_BASE_LIST = ["mlp_base", "mlp_deep", "res_mlp", "cnn", "transformer_light"]
    GEN_IS_COND_LIST = [False, True]
    DISCRIMINATOR_LIST = ["mlp_base", "mlp_deep", "res_mlp", "attention_head"]

    total_combinations = (
            len(BERT_LIST) *
            len(GEN_BASE_LIST) *
            len(GEN_IS_COND_LIST) *
            len(DISCRIMINATOR_LIST)
    )
    print(f"[Stage-1-Bruteforce] 搜索空间大小: {total_combinations} 组结构")

    # ====== KFold 划分（对 labeled 做一次，所有结构复用同一划分） ======
    labels_only = [ex[1] for ex in labeled]
    labeled_arr = np.array(labeled, dtype=object)
    skf = StratifiedKFold(n_splits=s1_n_splits, shuffle=True, random_state=42)

    # ====== 暴力遍历所有结构组合 ======
    best_avg_f1 = -1.0
    best_combo_s1 = None
    best_metrics = None
    all_results = []  # 用于后续分析/画表

    search_start = time.time()
    combo_idx = 0

    for bert_name in BERT_LIST:
        for gen_base in GEN_BASE_LIST:
            for gen_is_cond in GEN_IS_COND_LIST:
                generator_name = _map_generator_impl(gen_base, gen_is_cond)
                for discriminator_name in DISCRIMINATOR_LIST:
                    combo_idx += 1
                    print(
                        f"\n[Stage-1-Bruteforce][Combo {combo_idx}/{total_combinations}] "
                        f"BERT={bert_name}, G={generator_name} (base={gen_base}, cond={gen_is_cond}), "
                        f"D={discriminator_name}"
                    )

                    per_fold_f1 = []
                    per_fold_wf1 = []

                    # ---- KFold 评估 ----
                    for fold_idx, (tr_idx, dv_idx) in enumerate(
                            skf.split(labeled_arr, labels_only), start=1
                    ):
                        labeled_train = labeled_arr[tr_idx].tolist()
                        labeled_dev = labeled_arr[dv_idx].tolist()

                        try:
                            # allTrain 返回 (best_epoch, best_metrics)，这里我们使用 best_metrics['mf1']
                            _, temp_best_metrics = run_single_experiment_from_split(
                                dataset_name=dataset_name,
                                bert_name=bert_name,
                                generator_name=generator_name,
                                discriminator_name=discriminator_name,
                                labeled_train=labeled_train,
                                labeled_dev=labeled_dev,
                                unlabeled=unlabeled,
                                LABEL_LIST=LABEL_LIST,
                                epochs=s1_epochs,
                                output_file=None,
                                log_tag=(f"[S1-BF] fold={fold_idx}"),
                                gen_activation=None
                            )
                            fold_f1 = float(temp_best_metrics['mf1'])
                            per_fold_f1.append(fold_f1)
                            fold_wf1 = float(temp_best_metrics['wf1'])
                            per_fold_wf1.append(fold_wf1)
                            print(
                                f"[Stage-1-Bruteforce] Fold {fold_idx}/{s1_n_splits} "
                                f"Dev Macro-F1 = {fold_f1:.4f}"
                                f"Dev w-F1 = {fold_wf1:.4f}"
                            )

                        finally:
                            # 每折结束做一次显存清理，避免累积
                            _gpu_cleanup()

                    if per_fold_f1:
                        avg_f1 = float(np.mean(per_fold_f1))
                        avg_wf1 = float(np.mean(per_fold_wf1))
                    else:
                        avg_f1 = -1.0
                        avg_wf1 = -1.0

                    print(
                        f"[Stage-1-Bruteforce] 当前结构 KFold-Avg Dev Macro-F1 = {avg_f1:.4f}"
                        f"[Stage-1-Bruteforce] 当前结构 KFold-Avg Dev w-F1 = {avg_wf1:.4f}"
                    )
                    # ====== 写入日志文件 ======
                    if output_file is not None:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(f"\n[Stage-1-Bruteforce][Combo {combo_idx}/{total_combinations}] ")
                            f.write(f"BERT={bert_name}, G={generator_name} (base={gen_base}, cond={gen_is_cond}), ")
                            f.write(f"D={discriminator_name}")
                            f.write(f"[Stage-1-Bruteforce] 当前结构 KFold-Avg Dev Macro-F1 = {avg_f1:.4f}")
                            f.write(f"[Stage-1-Bruteforce] 当前结构 KFold-Avg Dev Macro-F1 = {avg_wf1:.4f}")
                    # 记录该结构结果
                    result_item = {
                        "bert_name": bert_name,
                        "generator_name": generator_name,
                        "discriminator_name": discriminator_name,
                        "gen_base": gen_base,
                        "gen_is_conditional": gen_is_cond,
                        "fold_f1s": per_fold_f1,
                        "avg_f1": avg_f1
                    }
                    all_results.append(result_item)

                    # 更新最优结构
                    if avg_f1 > best_avg_f1:
                        best_avg_f1 = avg_f1
                        best_metrics = temp_best_metrics
                        best_combo_s1 = {
                            "bert_name": bert_name,
                            "generator_name": generator_name,
                            "discriminator_name": discriminator_name,
                            "gen_is_conditional": gen_is_cond,
                            "act": None
                        }
                        print(
                            f"[Stage-1-Bruteforce] <<< 当前最优更新: "
                            f"Avg Dev Macro-F1 = {best_avg_f1:.4f}, "
                            f"BERT={bert_name}, G={generator_name} "
                            f"(base={gen_base}, cond={gen_is_cond}), "
                            f"D={discriminator_name} >>>"
                        )

    search_time_str = format_time(time.time() - search_start)

    if best_combo_s1 is None:
        raise RuntimeError("[Stage-1-Bruteforce] 暂无有效结构（可能搜索阶段均失败）。")

    print("\n[Stage-1-Bruteforce] 暴力遍历搜索完成")
    print(f"[Stage-1-Bruteforce] Best Avg Dev Macro-F1 = {best_avg_f1:.4f}")
    print(
        f"[Stage-1-Bruteforce] Best combo = "
        f"BERT={best_combo_s1['bert_name']}, "
        f"G={best_combo_s1['generator_name']} "
        f"(cond={best_combo_s1['gen_is_conditional']}), "
        f"D={best_combo_s1['discriminator_name']}"
    )
    print(f"[Stage-1-Bruteforce] Search Cost (time) = {search_time_str}")

    # ====== 写入日志文件 ======
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("========== Stage-1 Bruteforce Search ==========\n")
            f.write(f"Total combinations: {total_combinations}\n")
            f.write(f"KFold Splits     : {s1_n_splits}\n")
            f.write(f"Search Time      : {search_time_str}\n")
            f.write("-----------------------------------------\n")
            for idx, r in enumerate(all_results, 1):
                f.write(
                    f"[Combo {idx}] BERT={r['bert_name']}, "
                    f"G={r['generator_name']} "
                    f"(base={r['gen_base']}, cond={r['gen_is_conditional']}), "
                    f"D={r['discriminator_name']}\n"
                )
                f.write(
                    "  Fold F1s: "
                    + ", ".join(f"{x:.4f}" for x in r["fold_f1s"])
                    + "\n"
                )
                f.write(f"  Avg Dev Macro-F1: {r['avg_f1']:.4f}\n")
            f.write("-----------------------------------------\n")
            f.write(
                "Best combo: "
                f"BERT={best_combo_s1['bert_name']}, "
                f"G={best_combo_s1['generator_name']} "
                f"(base={best_combo_s1['gen_base']}, cond={best_combo_s1['gen_is_conditional']}), "
                f"D={best_combo_s1['discriminator_name']}\n"
            )
            f.write(f"Best Avg Dev Macro-F1: {best_avg_f1:.4f}\n")
            f.write(f"Best Avg Dev W-F1: {best_metrics['wf1']:.4f}\n")
            f.write("=============================================\n\n")
            f.write(f"total Time: {format_time(time.time() - overall_start)}\n")

    overall_time_str = format_time(time.time() - overall_start)
    print(f"[Stage-1-Bruteforce] Overall Time: {overall_time_str}")

    # 收尾清理显存
    try:
        _gpu_cleanup()
    except Exception:
        pass

    return best_combo_s1


# =========================
# 随机搜索实验函数(s1)
# =========================
def run_experiment_random_s1(
        dataset_name,
        data_dir,
        labeled_ratio,
        test_ratio,
        output_file,
        n_trials=20):
    """
    Stage-1 Random Search 版本的架构搜索：
      - 搜索空间：与 Stage-1 Optuna 一致 (BERT × G_base × cond × D)
      - 评估方式：对每个随机采样的结构做 s1_n_splits 折 Stratified K-Fold
                  每折调用 run_single_experiment_from_split(...)
                  使用 Dev Macro-F1 的平均值作为该结构的分数
      - 最终：选择平均分最高的结构，在全量 labeled+unlabeled 上训练，
              用 test_examples 作为最终测试集，输出指标 & 日志。

    返回：
      best_combo  : (best_bert, best_gen, best_dis)
      best_avg_f1 : Stage-1 随机搜索阶段的最佳 KFold 平均 Macro-F1
      test_metrics: 最终在测试集上的评估结果（dict）
    """
    set_seed(seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    overall_start = time.time()

    # ====== 日志：记录本次实验的参数配置 ======
    exp_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    param_log_lines = [
        "=" * 60,
        f"[Experiment] run_experiment_random(Stage-1) @ {exp_time_str}",
        f"[Params] dataset_name={dataset_name}",
        f"[Params] data_dir={data_dir}",
        f"[Params] labeled_ratio={labeled_ratio}, test_ratio={test_ratio}",
        f"[Params] n_trials={n_trials}, "
        f"s1_epochs={s1_epochs}, batch_size={BATCH_SIZE}, seed={seed}",
        f"[Params] output_file={output_file}",
        "=" * 60,
        ""
    ]
    for line in param_log_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in param_log_lines:
                f.write(line + "\n")

    # ====== 数据准备：与 Stage-1 Optuna / version2 一致 ======
    global labeled, unlabeled, LABEL_LIST
    LABEL_LIST = build_label_list(data_dir)
    labeled, unlabeled, test_examples = stratified_split_dataset(
        data_dir,
        labeled_ratio=labeled_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    num_labeled = len(labeled)
    num_unlabeled = len(unlabeled)
    num_test = len(test_examples)

    data_log_lines = [
        "[Data] 当前标签集合 LABEL_LIST: " + str(LABEL_LIST),
        f"[Data] 数据集划分结果: #labeled={num_labeled}, "
        f"#unlabeled={num_unlabeled}, #test={num_test}",
        ""
    ]
    for line in data_log_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in data_log_lines:
                f.write(line + "\n")

    # ====== 定义 Stage-1 Random 的搜索空间 ======
    BERT_LIST = [
        "bert-base-cased", "bert-base-uncased",
        "bert-large-cased", "bert-large-uncased",
        "distilbert-base-uncased", "distilbert-base-cased",
        "roberta-base", "roberta-large",
        "xlm-roberta-base", "albert-base-v1", "albert-base-v2"
    ]
    GEN_BASE_LIST = ["mlp_base", "mlp_deep", "res_mlp", "cnn", "transformer_light"]
    GEN_IS_COND_LIST = [False, True]
    DISCRIMINATOR_LIST = ["mlp_base", "mlp_deep", "res_mlp", "attention_head"]

    # 用于 StratifiedKFold 的标签序列
    labels_only = [ex[1] for ex in labeled]
    labeled_arr = np.array(labeled, dtype=object)

    # 记录
    best_avg_f1 = -1.0
    best_avg_wf1 = -1.0
    best_combo = None  # (bert_name, generator_name, discriminator_name)
    all_trials_info = []

    seen_combos = set()  # 避免重复采样相同 (BERT, G, D)

    print(f"[Stage-1-Random] 开始随机搜索，n_trials={n_trials}, "
          f"KFold={s1_n_splits}")

    search_start = time.time()

    # =========================
    # Stage-1 随机搜索主循环
    # =========================
    for trial_idx in range(1, n_trials + 1):
        # ---- 随机采样一个未出现过的结构组合 ----
        for _ in range(1000):  # 防止极端情况下死循环
            bert_name = random.choice(BERT_LIST)
            gen_base = random.choice(GEN_BASE_LIST)
            gen_is_cond = random.choice(GEN_IS_COND_LIST)
            generator_name = _map_generator_impl(gen_base, gen_is_cond)
            discriminator_name = random.choice(DISCRIMINATOR_LIST)

            combo = (bert_name, generator_name, discriminator_name, None)
            if combo not in seen_combos:
                seen_combos.add(combo)
                break
        else:
            print("[Stage-1-Random] 搜索空间已被用尽，提前终止。")
            break

        print(f"\n[Stage-1-Random][Trial {trial_idx}/{n_trials}] "
              f"BERT={bert_name}, G={generator_name} (base={gen_base}, "
              f"cond={gen_is_cond}), D={discriminator_name}")

        # ---- 对该结构做 KFold 评估 ----
        skf = StratifiedKFold(n_splits=s1_n_splits, shuffle=True, random_state=42)
        per_fold_f1 = []
        per_fold_wf1 = []

        for fold_idx, (tr_idx, dv_idx) in enumerate(
                skf.split(labeled_arr, labels_only), start=1
        ):
            labeled_train = labeled_arr[tr_idx].tolist()
            labeled_dev = labeled_arr[dv_idx].tolist()

            try:
                # 注意：这里不依赖 allTrain 的第一个返回值，而是直接用 best_metrics['mf1']
                _, best_metrics = run_single_experiment_from_split(
                    dataset_name=dataset_name,
                    bert_name=bert_name,
                    generator_name=generator_name,
                    discriminator_name=discriminator_name,
                    labeled_train=labeled_train,
                    labeled_dev=labeled_dev,
                    unlabeled=unlabeled,
                    LABEL_LIST=LABEL_LIST,
                    epochs=s1_epochs,
                    output_file=None,
                    log_tag=f"[Stage1-Random trial={trial_idx}] fold={fold_idx}",
                    gen_activation=None
                )
                fold_f1 = float(best_metrics["mf1"])
                fold_wf1 = float(best_metrics["wf1"])
                per_fold_f1.append(fold_f1)
                per_fold_wf1.append(fold_wf1)
                print(f"[Stage-1-Random][Trial {trial_idx}] "
                      f"Fold {fold_idx}/{s1_n_splits} Dev Macro-F1={fold_f1:.4f}")
            finally:
                # 每折结束做轻量显存清理
                _gpu_cleanup()

        if per_fold_f1:
            avg_f1 = float(np.mean(per_fold_f1))
            avg_wf1 = float(np.mean(per_fold_wf1))
        else:
            avg_f1 = -1.0
            avg_wf1 = -1.0

        print(f"[Stage-1-Random][Trial {trial_idx}] "
              f"KFold-Avg Dev Macro-F1 = {avg_f1:.4f}"
              f"KFold-Avg Dev w-F1 = {avg_wf1:.4f}")
        all_trials_info.append({
            "trial": trial_idx,
            "bert_name": bert_name,
            "generator_name": generator_name,
            "discriminator_name": discriminator_name,
            "gen_base": gen_base,
            "gen_is_cond": gen_is_cond,
            "fold_f1s": per_fold_f1,
            "avg_f1": avg_f1
        })

        # 更新全局最优
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_avg_wf1 = avg_wf1
            best_combo = combo
            print(f"[Stage-1-Random] <<< 当前最优更新: "
                  f"Avg Dev Macro-F1 = {best_avg_f1:.4f}, "
                  f"Avg Dev Macro-F1 = {best_avg_wf1:.4f}, "
                  f"BERT={best_combo[0]}, G={best_combo[1]}, D={best_combo[2]} >>>")

    search_time_str = format_time(time.time() - search_start)

    if best_combo is None:
        raise RuntimeError("[Stage-1-Random] 没有成功评估任何结构组合。")

    (best_bert, best_gen, best_dis) = (best_combo[0], best_combo[1], best_combo[2])
    print("\n[Stage-1-Random] 搜索完成")
    print(f"[Stage-1-Random] Best Avg Dev Macro-F1 = {best_avg_f1:.4f}")
    print(f"[Stage-1-Random] Best combo = "
          f"BERT={best_bert}, G={best_gen}, D={best_dis}")
    print(f"[Stage-1-Random] Search Cost (time) = {search_time_str}")

    # ====== 将 Stage-1 随机搜索结果写入日志文件 ======
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("========== Stage-1 Random Search ==========\n")
            f.write(f"Search Trials: {len(all_trials_info)}\n")
            f.write(f"KFold Splits: {s1_n_splits}\n")
            f.write(f"Search Time : {search_time_str}\n")
            f.write("-----------------------------------------\n")
            for info in all_trials_info:
                f.write(
                    f"[Trial {info['trial']}] "
                    f"BERT={info['bert_name']}, "
                    f"G={info['generator_name']} "
                    f"(base={info['gen_base']}, cond={info['gen_is_cond']}), "
                    f"D={info['discriminator_name']}\n"
                )
                f.write(
                    f"  Fold F1s: {', '.join(f'{x:.4f}' for x in info['fold_f1s'])}\n"
                )
                f.write(f"  Avg Dev Macro-F1: {info['avg_f1']:.4f}\n")
            f.write("-----------------------------------------\n")
            f.write(f"Best combo: BERT={best_bert}, "
                    f"G={best_gen}, D={best_dis}\n")
            f.write(f"Best Avg Dev Macro-F1: {best_avg_f1:.4f}\n")
            f.write(f"Best Avg Dev w-F1: {best_avg_wf1:.4f}\n")
            f.write(f"[Stage-1-Random] Search Cost (time) = {search_time_str}")
            f.write("===========================================\n\n")

    # 结尾清理显存
    try:
        _gpu_cleanup()
    except Exception:
        pass

    return best_combo


# =========================
# stage-1-BO实验函数
# =========================
# =========================
# experiment
# =========================
def run_experiment_s1(dataset_name, data_dir, labeled_ratio, test_ratio, output_file,
                      n_trials=20, epochs=10, seed=42, top_k=5):
    # 数据处理
    set_seed(seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    start1_time = time.time()

    # ====== 日志：记录本次实验的参数配置 ======
    exp_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    param_log_lines = [
        "=" * 60,
        f"[Experiment] run_experiment_version2 @ {exp_time_str}",
        f"[Params] dataset_name={dataset_name}",
        f"[Params] data_dir={data_dir}",
        f"[Params] labeled_ratio={labeled_ratio}, test_ratio={test_ratio}",
        f"[Params] n_trials={n_trials}, epochs={epochs}, s1_epochs={s1_epochs}, s2_epochs={s2_epochs}, batch_size={BATCH_SIZE}, seed={seed}, top_k={top_k}",
        f"[Params] output_file={output_file}",
        "=" * 60,
        ""
    ]
    for line in param_log_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in param_log_lines:
                f.write(line + "\n")

    # ====== 数据准备 ======
    global labeled, unlabeled, LABEL_LIST
    LABEL_LIST = build_label_list(data_dir)
    labeled, unlabeled, test_examples = stratified_split_dataset(
        data_dir,
        labeled_ratio=labeled_ratio,
        test_ratio=test_ratio,
    )

    # ====== 日志：记录当前数据集选择 & 划分规模 ======
    num_labeled = len(labeled)
    num_unlabeled = len(unlabeled)
    num_test = len(test_examples)

    data_log_lines = [
        "[Data] 当前标签集合 LABEL_LIST: " + str(LABEL_LIST),
        f"[Data] 数据集划分结果: #labeled={num_labeled}, #unlabeled={num_unlabeled}, #test={num_test}",
        ""
    ]
    for line in data_log_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in data_log_lines:
                f.write(line + "\n")

    # ====== stage1阶段搜索 ======

    # Optuna配置
    sampler = TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    print(f"[Stage-1] 开始搜索：n_trials={n_trials}, s1_epochs={s1_epochs}, batch_size={BATCH_SIZE}")
    study_s1 = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"ganbert_arch_search_{dataset_name}"
    )

    # —— 构造 Stage-1 的 objective（把 dataset_name 等固化进去）——
    objective_s1 = make_objective_stage1(
        dataset_name=dataset_name
    )
    # 搜索
    study_s1.optimize(objective_s1, n_trials=n_trials, show_progress_bar=True)

    best_trial_s1 = study_s1.best_trial
    best_params_s1 = best_trial_s1.params  # 包含 bert_name / generator_name / discriminator_name
    best_f1_s1 = study_s1.best_value

    bert_name_best = best_params_s1["bert_name"]
    gen_name_best = best_params_s1["generator_name"]
    dis_name_best = best_params_s1["discriminator_name"]
    total1_time = format_time(time.time() - start1_time)
    print("\n[Stage-1] 搜索完成")
    print(f"[Stage-1] Best macro-F1 = {best_f1_s1:.4f}")
    print(f"[Stage-1] Best combo: BERT={bert_name_best}, G={gen_name_best}, D={dis_name_best}")
    print(f"[Stage-1] 搜索时间为: " + total1_time)

    # ====== 从 Stage-1 结果里取 Top-K 组合（去重） ======
    # 仅保留成功完成的 trial
    complete_trials = [t for t in study_s1.trials if t.state == TrialState.COMPLETE and t.value is not None]

    # 按分数从高到低排序
    complete_trials.sort(key=lambda t: t.value, reverse=True)

    # 组装 (value, params) 并做去重（同一三元组只保留一次）
    seen = set()
    topk_list = []
    for t in complete_trials:
        p = t.params
        combo = (p["bert_name"], p["generator_name"], p["discriminator_name"])
        if combo in seen:
            continue
        seen.add(combo)
        topk_list.append({
            "score": float(t.value),
            "bert_name": p["bert_name"],
            "generator_name": p["generator_name"],
            "discriminator_name": p["discriminator_name"],
            "trial_number": t.number
        })
        if len(topk_list) >= top_k:
            break

    if not topk_list:
        raise RuntimeError("[Stage-1] 未找到可用的组合（没有 COMPLETE 的 trial）。")

    # 打印 & 写入文件
    print(f"\n[Stage-1] Top-{len(topk_list)} 组合（用于 Stage-2）：")
    for i, item in enumerate(topk_list, 1):
        print(f"  #{i:02d} F1={item['score']:.4f} "
              f"BERT={item['bert_name']}, G={item['generator_name']}, D={item['discriminator_name']} "
              f"(trial={item['trial_number']})")

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n[Stage-1] 搜索完成")
        f.write(f"[Stage-1] Best macro-F1 = {best_f1_s1:.4f}")
        f.write(f"[Stage-1] Best combo: BERT={bert_name_best}, G={gen_name_best}, D={dis_name_best}")
        f.write(f"[Stage-1] 搜索时间为: " + total1_time)
        f.write(f"\n========== Stage-1 Top-{len(topk_list)} ==========\n")
        for i, item in enumerate(topk_list, 1):
            f.write(f"#{i:02d} F1={item['score']:.4f} "
                    f"BERT={item['bert_name']}, G={item['generator_name']}, D={item['discriminator_name']} "
                    f"(trial={item['trial_number']})\n")
        f.write("=============================================\n\n")

    best_combo = (bert_name_best, gen_name_best, dis_name_best, None)
    try:
        _gpu_cleanup()
    except:
        pass
    return best_combo


# =========================
# 随机搜索实验函数(s2)
# =========================
def run_experiment_random_s2(
        dataset_name,
        data_dir,
        labeled_ratio,
        test_ratio,
        output_file,
        n_trials=20,
        epochs=10,
):
    """
    Stage-1 + Stage-2 联合空间上的随机搜索：
      - 搜索变量同时包含：
          * BERT 结构
          * Generator 基类 + 是否 conditional
          * Discriminator 结构
          * Generator 激活函数（Stage-2 搜索空间）
      - 对每个随机采样的 (BERT, G, D, act) 组合做 s1_n_splits 折 Stratified K-Fold：
          * 每折调用 run_single_experiment_from_split(...)
          * gen_activation=act，使得生成器使用对应激活
          * 以 Dev Macro-F1 的折均值作为该结构的得分
      - 搜索结束后：
          * 选出平均 Dev Macro-F1 最高的 (BERT, G, D, act)
          * 在全量 labeled + unlabeled 上训练，在 test_examples 上评估
    返回：
      best_combo  : (best_bert, best_gen, best_dis, best_act)
      best_avg_f1 : 搜索阶段最优的 KFold 平均 Dev Macro-F1
      test_metrics: 最终在测试集上的评估结果（dict）
    """
    set_seed(seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    overall_start = time.time()

    # ====== 日志：记录本次实验的参数配置 ======
    exp_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    param_log_lines = [
        "=" * 60,
        f"[Experiment] run_experiment_random_s2(Stage-1+2) @ {exp_time_str}",
        f"[Params] dataset_name={dataset_name}",
        f"[Params] data_dir={data_dir}",
        f"[Params] labeled_ratio={labeled_ratio}, test_ratio={test_ratio}",
        f"[Params] n_trials={n_trials}, s2_epochs={s2_epochs}, "
        f"final_epochs={epochs}, batch_size={BATCH_SIZE}, seed={seed}",
        f"[Params] output_file={output_file}",
        "=" * 60,
        ""
    ]
    for line in param_log_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in param_log_lines:
                f.write(line + "\n")

    # ====== 数据准备：与 version2 保持一致 ======
    global labeled, unlabeled, LABEL_LIST
    LABEL_LIST = build_label_list(data_dir)
    labeled, unlabeled, test_examples = stratified_split_dataset(
        data_dir,
        labeled_ratio=labeled_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    num_labeled = len(labeled)
    num_unlabeled = len(unlabeled)
    num_test = len(test_examples)

    data_log_lines = [
        "[Data] 当前标签集合 LABEL_LIST: " + str(LABEL_LIST),
        f"[Data] 数据集划分结果: #labeled={num_labeled}, "
        f"#unlabeled={num_unlabeled}, #test={num_test}",
        ""
    ]
    for line in data_log_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in data_log_lines:
                f.write(line + "\n")

    # ====== 定义联合搜索空间 ======
    BERT_LIST = [
        "bert-base-cased", "bert-base-uncased",
        "bert-large-cased", "bert-large-uncased",
        "distilbert-base-uncased", "distilbert-base-cased",
        "roberta-base", "roberta-large",
        "xlm-roberta-base", "albert-base-v1", "albert-base-v2"
    ]
    GEN_BASE_LIST = ["mlp_base", "mlp_deep", "res_mlp", "cnn", "transformer_light"]
    GEN_IS_COND_LIST = [False, True]
    DISCRIMINATOR_LIST = ["mlp_base", "mlp_deep", "res_mlp", "attention_head"]
    ACT_LIST = ["relu", "leakyrelu", "siren", "gelu", "elu"]

    # StratifiedKFold 需要的标签
    labels_only_all = [ex[1] for ex in labeled]
    labeled_arr_all = np.array(labeled, dtype=object)

    # 记录搜索过程
    best_avg_f1 = -1.0
    best_combo = None  # (bert_name, generator_name, discriminator_name, act)
    all_trials_info = []

    seen_combos = set()  # 避免重复评估同一 (BERT, G, D, act)

    print(f"[Stage-1+2-Random] 开始联合随机搜索，n_trials={n_trials}, "
          f"KFold={s1_n_splits}, act_space={ACT_LIST}")

    search_start = time.time()

    # =========================
    # 联合随机搜索主循环
    # =========================
    for trial_idx in range(1, n_trials + 1):
        # ---- 随机采样一个未出现过的结构+激活组合 ----
        for _ in range(1000):  # 简单防护，避免极端情况下死循环
            bert_name = random.choice(BERT_LIST)
            gen_base = random.choice(GEN_BASE_LIST)
            gen_is_cond = random.choice(GEN_IS_COND_LIST)
            generator_name = _map_generator_impl(gen_base, gen_is_cond)
            discriminator_name = random.choice(DISCRIMINATOR_LIST)
            act_name = random.choice(ACT_LIST)

            combo = (bert_name, generator_name, discriminator_name, act_name)
            if combo not in seen_combos:
                seen_combos.add(combo)
                break
        else:
            print("[Stage-1+2-Random] 搜索空间已被用尽，提前终止。")
            break

        print(f"\n[Stage-1+2-Random][Trial {trial_idx}/{n_trials}] "
              f"BERT={bert_name}, G={generator_name} "
              f"(base={gen_base}, cond={gen_is_cond}), "
              f"D={discriminator_name}, act={act_name}")

        # ---- 对该 (BERT, G, D, act) 做 KFold 评估 ----
        skf = StratifiedKFold(n_splits=s1_n_splits, shuffle=True, random_state=42)
        per_fold_f1 = []

        for fold_idx, (tr_idx, dv_idx) in enumerate(
                skf.split(labeled_arr_all, labels_only_all), start=1
        ):
            labeled_train = labeled_arr_all[tr_idx].tolist()
            labeled_dev = labeled_arr_all[dv_idx].tolist()

            try:
                # 注意：allTrain 返回的是 (best_epoch, best_metrics)，
                # 我们只依赖 best_metrics['mf1'] 作为该折得分。
                _, best_metrics = run_single_experiment_from_split(
                    dataset_name=dataset_name,
                    bert_name=bert_name,
                    generator_name=generator_name,
                    discriminator_name=discriminator_name,
                    labeled_train=labeled_train,
                    labeled_dev=labeled_dev,
                    unlabeled=unlabeled,
                    LABEL_LIST=LABEL_LIST,
                    epochs=s2_epochs,  # 搜索阶段可以用 s2_epochs
                    output_file=None,
                    log_tag=(f"[Rand-S2 trial={trial_idx}] fold={fold_idx}"),
                    gen_activation=act_name  # 激活作为搜索变量
                )
                fold_f1 = float(best_metrics["mf1"])
                per_fold_f1.append(fold_f1)
                print(f"[Stage-1+2-Random][Trial {trial_idx}] "
                      f"Fold {fold_idx}/{s1_n_splits} Dev Macro-F1={fold_f1:.4f}")
            finally:
                # 每折结束显存清理，避免堆积
                _gpu_cleanup()

        if per_fold_f1:
            avg_f1 = float(np.mean(per_fold_f1))
        else:
            avg_f1 = -1.0

        print(f"[Stage-1+2-Random][Trial {trial_idx}] "
              f"KFold-Avg Dev Macro-F1 = {avg_f1:.4f}")

        all_trials_info.append({
            "trial": trial_idx,
            "bert_name": bert_name,
            "generator_name": generator_name,
            "discriminator_name": discriminator_name,
            "gen_base": gen_base,
            "gen_is_cond": gen_is_cond,
            "act_name": act_name,
            "fold_f1s": per_fold_f1,
            "avg_f1": avg_f1
        })

        # 更新全局最优
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_combo = combo
            print(f"[Stage-1+2-Random] <<< 当前最优更新: "
                  f"Avg Dev Macro-F1 = {best_avg_f1:.4f}, "
                  f"BERT={best_combo[0]}, G={best_combo[1]}, "
                  f"D={best_combo[2]}, act={best_combo[3]} >>>")

    search_time_str = format_time(time.time() - search_start)

    if best_combo is None:
        raise RuntimeError("[Stage-1+2-Random] 没有成功评估任何结构组合。")

    best_bert, best_gen, best_dis, best_act = best_combo
    print("\n[Stage-1+2-Random] 联合随机搜索完成")
    print(f"[Stage-1+2-Random] Best Avg Dev Macro-F1 = {best_avg_f1:.4f}")
    print(f"[Stage-1+2-Random] Best combo            = "
          f"BERT={best_bert}, G={best_gen}, D={best_dis}, act={best_act}")
    print(f"[Stage-1+2-Random] Search Cost (time)    = {search_time_str}")

    # ====== 写入搜索阶段日志 ======
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("========== Stage-1+2 Random Search ==========\n")
            f.write(f"Search Trials: {len(all_trials_info)}\n")
            f.write(f"KFold Splits: {s1_n_splits}\n")
            f.write(f"Search Time : {search_time_str}\n")
            f.write("-----------------------------------------\n")
            for info in all_trials_info:
                f.write(
                    f"[Trial {info['trial']}] "
                    f"BERT={info['bert_name']}, "
                    f"G={info['generator_name']} "
                    f"(base={info['gen_base']}, cond={info['gen_is_cond']}), "
                    f"D={info['discriminator_name']}, "
                    f"act={info['act_name']}\n"
                )
                f.write(
                    f"  Fold F1s: "
                    f"{', '.join(f'{x:.4f}' for x in info['fold_f1s'])}\n"
                )
                f.write(f"  Avg Dev Macro-F1: {info['avg_f1']:.4f}\n")
            f.write("-----------------------------------------\n")
            f.write(f"Best combo: BERT={best_bert}, "
                    f"G={best_gen}, D={best_dis}, act={best_act}\n")
            f.write(f"Best Avg Dev Macro-F1: {best_avg_f1:.4f}\n")
            f.write("=============================================\n\n")

    # =========================
    # 最终：用最佳 (BERT, G, D, act) 在全量 labeled 上训练，在 test 上评估
    # =========================
    print("\n[Final-Stage-1+2-Random] 使用最佳结构+激活做最终训练 + 测试集评估 ...")
    print(f"[Final-Stage-1+2-Random] BERT={best_bert}, "
          f"G={best_gen}, D={best_dis}, act={best_act}")

    final_start = time.time()

    # 全量训练：labeled 作为 train，test_examples 作为最终测试集
    _, test_metrics = run_single_experiment_from_split(
        dataset_name=dataset_name,
        bert_name=best_bert,
        generator_name=best_gen,
        discriminator_name=best_dis,
        labeled_train=labeled,  # 全部 labeled 作为训练集合
        labeled_dev=test_examples,  # test_examples 作为最终测试集
        unlabeled=unlabeled,
        LABEL_LIST=LABEL_LIST,
        epochs=epochs,  # 最终训练可以给足 epochs
        output_file=None,
        log_tag="[Stage1+2-Random-Final]",
        gen_activation=best_act
    )

    final_macro_f1 = test_metrics["mf1"]
    final_weighted_f1 = test_metrics["wf1"]
    final_time_str = format_time(time.time() - final_start)
    overall_time_str = format_time(time.time() - overall_start)

    # ====== 最终结果写入日志 ======
    summary_lines = [
        "=========================================",
        f"===== Dataset: {dataset_name} (Stage-1+2 Random) =====",
        f"BERT={best_bert}, G={best_gen}, D={best_dis}, act={best_act}",
        f"Labeled ratio: {labeled_ratio}, Test ratio: {test_ratio}",
        f"Trials: {n_trials}, KFold={s1_n_splits}",
        f"Search Time: {search_time_str}, Final Train+Test Time: {final_time_str}",
        f"Overall Time: {overall_time_str}",
        "-----------------------------------------",
        f"Stage-1+2 Best Avg Dev Macro-F1: {best_avg_f1:.4f}",
        f"Final Test Macro-F1: {final_macro_f1:.4f}",
        f"Final Test Weighted-F1: {final_weighted_f1:.4f}",
        "Final Test Metrics Report:",
        test_metrics["report"].rstrip("\n"),
        "=========================================",
        ""
    ]
    for line in summary_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in summary_lines:
                f.write(line + "\n")

    # 结尾清理显存
    try:
        _gpu_cleanup()
    except Exception:
        pass

    return best_combo


# =========================
# experiment
# =========================
def run_experiment_s2(dataset_name, data_dir, labeled_ratio, test_ratio, output_file,
                      n_trials=20, epochs=10, top_k=5):
    # 数据处理
    set_seed(seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    start1_time = time.time()

    # ====== 日志：记录本次实验的参数配置 ======
    exp_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    param_log_lines = [
        "=" * 60,
        f"[Experiment] run_experiment_version2 @ {exp_time_str}",
        f"[Params] dataset_name={dataset_name}",
        f"[Params] data_dir={data_dir}",
        f"[Params] labeled_ratio={labeled_ratio}, test_ratio={test_ratio}",
        f"[Params] n_trials={n_trials}, epochs={epochs}, s1_epochs={s1_epochs}, s2_epochs={s2_epochs}, batch_size={BATCH_SIZE}, seed={seed}, top_k={top_k}",
        f"[Params] output_file={output_file}",
        "=" * 60,
        ""
    ]
    for line in param_log_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in param_log_lines:
                f.write(line + "\n")

    # ====== 数据准备 ======
    global labeled, unlabeled, LABEL_LIST
    LABEL_LIST = build_label_list(data_dir)
    labeled, unlabeled, test_examples = stratified_split_dataset(
        data_dir,
        labeled_ratio=labeled_ratio,
        test_ratio=test_ratio
    )

    # ====== 日志：记录当前数据集选择 & 划分规模 ======
    num_labeled = len(labeled)
    num_unlabeled = len(unlabeled)
    num_test = len(test_examples)

    data_log_lines = [
        "[Data] 当前标签集合 LABEL_LIST: " + str(LABEL_LIST),
        f"[Data] 数据集划分结果: #labeled={num_labeled}, #unlabeled={num_unlabeled}, #test={num_test}",
        ""
    ]
    for line in data_log_lines:
        print(line)
    if output_file is not None:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in data_log_lines:
                f.write(line + "\n")

    # ====== stage1阶段搜索 ======

    # Optuna配置
    sampler = TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    print(f"[Stage-1] 开始搜索：n_trials={n_trials}, s1_epochs={s1_epochs}, batch_size={BATCH_SIZE}")
    study_s1 = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"ganbert_arch_search_{dataset_name}"
    )

    # —— 构造 Stage-1 的 objective（把 dataset_name 等固化进去）——
    objective_s1 = make_objective_stage1(
        dataset_name=dataset_name
    )
    # 搜索
    study_s1.optimize(objective_s1, n_trials=n_trials, show_progress_bar=True)

    best_trial_s1 = study_s1.best_trial
    best_params_s1 = best_trial_s1.params  # 包含 bert_name / generator_name / discriminator_name
    best_f1_s1 = study_s1.best_value

    bert_name_best = best_params_s1["bert_name"]
    gen_name_best = best_params_s1["generator_name"]
    dis_name_best = best_params_s1["discriminator_name"]
    total1_time = format_time(time.time() - start1_time)
    print("\n[Stage-1] 搜索完成")
    print(f"[Stage-1] Best macro-F1 = {best_f1_s1:.4f}")
    print(f"[Stage-1] Best combo: BERT={bert_name_best}, G={gen_name_best}, D={dis_name_best}")
    print(f"[Stage-1] 搜索时间为: " + total1_time)

    # ====== 从 Stage-1 结果里取 Top-K 组合（去重） ======
    # 仅保留成功完成的 trial
    complete_trials = [t for t in study_s1.trials if t.state == TrialState.COMPLETE and t.value is not None]

    # 按分数从高到低排序
    complete_trials.sort(key=lambda t: t.value, reverse=True)

    # 组装 (value, params) 并做去重（同一三元组只保留一次）
    seen = set()
    topk_list = []
    for t in complete_trials:
        p = t.params
        combo = (p["bert_name"], p["generator_name"], p["discriminator_name"])
        if combo in seen:
            continue
        seen.add(combo)
        topk_list.append({
            "score": float(t.value),
            "bert_name": p["bert_name"],
            "generator_name": p["generator_name"],
            "discriminator_name": p["discriminator_name"],
            "trial_number": t.number
        })
        if len(topk_list) >= top_k:
            break

    if not topk_list:
        raise RuntimeError("[Stage-1] 未找到可用的组合（没有 COMPLETE 的 trial）。")

    # 打印 & 写入文件
    print(f"\n[Stage-1] Top-{len(topk_list)} 组合（用于 Stage-2）：")
    for i, item in enumerate(topk_list, 1):
        print(f"  #{i:02d} F1={item['score']:.4f} "
              f"BERT={item['bert_name']}, G={item['generator_name']}, D={item['discriminator_name']} "
              f"(trial={item['trial_number']})")

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n[Stage-1] 搜索完成")
        f.write(f"[Stage-1] Best macro-F1 = {best_f1_s1:.4f}")
        f.write(f"[Stage-1] Best combo: BERT={bert_name_best}, G={gen_name_best}, D={dis_name_best}")
        f.write(f"[Stage-1] 搜索时间为: " + total1_time)
        f.write(f"\n========== Stage-1 Top-{len(topk_list)} ==========\n")
        for i, item in enumerate(topk_list, 1):
            f.write(f"#{i:02d} F1={item['score']:.4f} "
                    f"BERT={item['bert_name']}, G={item['generator_name']}, D={item['discriminator_name']} "
                    f"(trial={item['trial_number']})\n")
        f.write("=============================================\n\n")

    # 将 Top-K 列表返回（供 Stage-2 使用）
    stage1_topk = topk_list

    # ====== stage2阶段搜索 ======
    # 只在 Top-k 结构上做生成器激活微调
    act_candidates = ["relu", "leakyrelu", "siren", "gelu", "elu"]
    s2_n_splits = s1_n_splits if "s1_n_splits" in globals() else 5  # 与 Stage-1 保持一致，或默认 5

    print(f"\n[Stage-2] 对 Top-{top_k} 组合进行【生成器】激活函数微调：{act_candidates}；KFold={s2_n_splits}")

    # 复用 Stage-1 的分层依据
    labels_only_all = [ex[1] for ex in labeled]
    labeled_arr_all = np.array(labeled, dtype=object)

    stage2_results = []
    start2_time = time.time()
    for i in range(top_k):
        combo = stage1_topk[i]
        bert_name = combo["bert_name"]
        gen_name = combo["generator_name"]
        dis_name = combo["discriminator_name"]

        print(
            f"\n[Stage-2] 基准结构 #{i + 1}: BERT={bert_name}, G={gen_name}, D={dis_name} (from trial={combo['trial_number']})")

        best_act = None
        best_f1 = -1.0
        best_wf1 = -1.0

        for act in act_candidates:
            print(f"[Stage-2] Try activation={act} ...")
            skf = StratifiedKFold(n_splits=s2_n_splits, shuffle=True, random_state=42)
            per_fold_f1 = []
            per_fold_wf1 = []

            for fold_idx, (tr_idx, dv_idx) in enumerate(skf.split(labeled_arr_all, labels_only_all), start=1):
                labeled_train = labeled_arr_all[tr_idx].tolist()
                labeled_dev = labeled_arr_all[dv_idx].tolist()

                try:
                    best_f1_epoch, best_metrics = run_single_experiment_from_split(
                        dataset_name=dataset_name,
                        bert_name=bert_name,
                        generator_name=gen_name,
                        discriminator_name=dis_name,
                        labeled_train=labeled_train,
                        labeled_dev=labeled_dev,
                        unlabeled=unlabeled,
                        LABEL_LIST=LABEL_LIST,
                        epochs=s2_epochs,
                        output_file=None,
                        log_tag=f"[S2 act={act}] fold={fold_idx}",
                        gen_activation=act  # 指定生成器激活
                    )
                    per_fold_f1.append(float(best_metrics['mf1']))
                    per_fold_wf1.append(float(best_metrics['wf1']))
                finally:
                    # 每折结束轻清理，避免显存积累
                    try:
                        _gpu_cleanup()
                    except:
                        pass

            avg_f1 = float(np.mean(per_fold_f1)) if per_fold_f1 else -1.0
            avg_wf1 = float(np.mean(per_fold_wf1)) if per_fold_wf1 else -1.0
            print(f"[Stage-2][{act}] KFold-Avg F1 = {avg_f1:.4f}")

            if avg_f1 > best_f1:
                best_f1, best_act = avg_f1, act
                best_wf1 = avg_wf1

            # 每个激活完成后再清一次
            try:
                _gpu_cleanup()
            except:
                pass

        stage2_results.append({
            "base_combo": (bert_name, gen_name, dis_name, best_act),
            "best_f1": best_f1,
            "best_wf1": best_wf1
        })
        print(f"[Stage-2] 基准结构 #{i + 1} 最优激活：{best_act} -> AvgF1={best_f1:.4f}, AvgWF1={best_wf1:.4f}")

    stage2_results_sorted = sorted(stage2_results, key=lambda r: r["best_f1"], reverse=True)
    best_entry = stage2_results_sorted[0]
    best_combo_s2 = best_entry["base_combo"]
    # —— Stage-2 结果输出与落盘 —— #
    print("\n[Stage-2] 结果汇总：")
    with open(output_file, "a", encoding="utf-8") as f:
        total2_time = format_time(time.time() - start2_time)
        f.write("\n[Stage-2] 搜索完成")
        f.write(f"[Stage-2] Best macro-F1 = {best_entry['best_f1']:.4f}")
        f.write(f"[Stage-2] Best w-F1 = {best_entry['best_wf1']:.4f}")
        f.write(
            f"[Stage-2] Best combo: BERT={best_combo_s2[0]}, G={best_combo_s2[1]}, D={best_combo_s2[2]}, act = {best_combo_s2[3]}")
        f.write(f"[Stage-2] 搜索时间为: " + total2_time)
        f.write(f"========== Stage-2 Activation Tuning (Top-{top_k}) ==========\n")
        for i, r in enumerate(stage2_results_sorted[:top_k], 1):
            (b, g, d, act) = r["base_combo"]
            print(
                f"  #{i:02d} BERT={b}, G={g}, D={d} | best_act={act} | KFold-AvgF1={r['best_f1']:.4f} | KFold-AvgWF1={r['best_wf1']:.4f}")
            f.write(
                f"#{i:02d} BERT={b}, G={g}, D={d} | best_act={act} | KFold-AvgF1={r['best_f1']:.4f} | KFold-AvgWF1={r['best_wf1']:.4f}\n")
        f.write("============================================================\n\n")
    # ====== 结尾清理显存 ======
    try:
        _gpu_cleanup()
    except:
        pass
    return best_combo_s2
