import os
import datetime
import gc
import torch
from attentionGanbert_v2 import run_single_experiment


def gpu_mem(tag=""):
    """打印当前 PyTorch 在 GPU 上的 allocated / reserved（用于判断是否泄漏/累积）"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU MEM]{tag} allocated={alloc:.1f}MiB reserved={reserv:.1f}MiB")

def cleanup_cuda():
    """尽可能释放 Python 引用 + 触发 GC + 清空 CUDA 缓存/IPC"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    # 基线对比实验
    datasets = {
        # "PROMISE": "./data/transData/PROMISE_AF.tsv",
        # "PROMISE_EXP": "./data/transData/PROMISE_exp_AF.tsv",
        "NFR-Review": "./data/transData/NFR-Review-AF.tsv"
        # "NFR-SO": "./data/transData/NFR-SO-AF.tsv"
    }

    # ===============================
    # 动态生成输出文件夹和文件名
    # ===============================
    date_str = datetime.datetime.now().strftime("%Y%m%d")  # 获取当前日期
    time_str = datetime.datetime.now().strftime("%H%M%S")  # 当前时间（防止重复）

    # 输出目录：./results/20251014/
    output_dir = os.path.join(".", "results2", date_str)
    os.makedirs(output_dir, exist_ok=True)

    (bert_name, generator_name, discriminator_name, act) = ("roberta-large", "cnn", "attention_head", "gelu")
    for name, path in datasets.items():
        # 输出文件名：experiment_results_20251014_name_150523.txt
        output_file = os.path.join(output_dir, f"experiment_results_{name}_{date_str}_{time_str}.txt")
        print(f" 实验结果将保存至: {output_file}")
        print(f"\n[{name}] 训练组合: BERT={bert_name}, G={generator_name}, D={discriminator_name}, ACT={act}")
        run_single_experiment(
            dataset_name=name,
            data_dir=path,
            bert_name=bert_name,
            generator_name=generator_name,
            discriminator_name=discriminator_name,
            act=act,
            labeled_ratio=0.2,
            epochs=30,
            output_file=output_file
        )
        # 清理显存
        cleanup_cuda()
        gpu_mem(tag=f" after {name}")

