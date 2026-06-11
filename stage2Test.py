# ============================
# stage2_search_only.py
# ============================

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

# 需要从你的项目中导入
from run_op_new import (
    run_single_experiment_from_split,
    _gpu_cleanup,
    build_label_list,
    stratified_split_dataset,
)

# 激活函数候选
ACT_LIST = ["relu", "leakyrelu", "siren", "gelu", "elu"]


def stage2_activation_search_only(
    dataset_name: str,
    data_dir: str,
    bert_name: str,
    generator_name: str,
    discriminator_name: str,
    k_fold: int = 3,
    epochs: int = 10,
    batch_size: int = 16,
    seed: int = 42,
    output_file: str = "stage2_only.log"
):
    """
    只对指定的 (BERT, Generator, Discriminator) 做 Stage-2 激活函数搜索。
    """
    print("\n========================")
    print("     Stage-2 Only Search")
    print("========================")
    print(f"Dataset={dataset_name}")
    print(f"固定结构：BERT={bert_name}, G={generator_name}, D={discriminator_name}")
    print(f"KFold={k_fold}, Epochs={epochs}, Batch={batch_size}")
    print("========================\n")

    # ========= 加载数据 ==========
    global labeled, unlabeled, LABEL_LIST
    LABEL_LIST = build_label_list(data_dir)

    # stage2 不需要划分 test，只要 labeled/unlabeled
    labeled, unlabeled, _ = stratified_split_dataset(
        data_dir,
        labeled_ratio=1.0,
        test_ratio=0.0,
        seed=seed
    )

    labels_only = [ex[1] for ex in labeled]
    labeled_arr = np.array(labeled, dtype=object)

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)

    # 保存结果
    search_results = {}
    best_act = None
    best_f1 = -1.0

    # ==============================
    #      开始激活函数搜索
    # ==============================
    for act in ACT_LIST:
        print(f"\n[Stage-2] 激活函数 = {act}")
        fold_f1s = []

        for fold_idx, (tr_idx, dv_idx) in enumerate(skf.split(labeled_arr, labels_only), start=1):
            print(f"\n---- Fold {fold_idx}/{k_fold} (act={act}) ----")

            labeled_train = labeled_arr[tr_idx].tolist()
            labeled_dev = labeled_arr[dv_idx].tolist()

            try:
                best_f1_fold, _ = run_single_experiment_from_split(
                    dataset_name=dataset_name,
                    bert_name=bert_name,
                    generator_name=generator_name,
                    discriminator_name=discriminator_name,
                    labeled_train=labeled_train,
                    labeled_dev=labeled_dev,
                    unlabeled=unlabeled,
                    LABEL_LIST=LABEL_LIST,
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=seed,
                    output_file=None,
                    log_tag=f"S2 act={act} fold={fold_idx}",
                    gen_activation=act
                )
                fold_f1s.append(float(best_f1_fold))

            finally:
                _gpu_cleanup()

        avg_f1 = float(np.mean(fold_f1s)) if fold_f1s else -1.0
        print(f"[Stage-2][act={act}] Avg F1 = {avg_f1:.4f}")

        search_results[act] = avg_f1

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_act = act

        _gpu_cleanup()

    # ==============================
    #         输出最终结果
    # ==============================
    print("\n=============================")
    print(" Stage-2 激活函数搜索完成")
    print("=============================")
    print(f"最佳激活函数：{best_act}  (Avg F1={best_f1:.4f})")
    print("=============================\n")

    # 写入日志
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("========== Stage-2 Only ==========\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"BERT={bert_name}, G={generator_name}, D={discriminator_name}\n\n")
        for act, score in search_results.items():
            f.write(f"act={act}: AvgF1={score:.4f}\n")
        f.write(f"\nBest act = {best_act} (AvgF1={best_f1:.4f})\n")
        f.write("=================================\n\n")

    return {
        "best_act": best_act,
        "best_f1": best_f1,
        "results": search_results
    }


# ==============================
#             MAIN
# ==============================
def main():
    stage2_activation_search_only(
        dataset_name="PROMISE",
        data_dir="./data/transData/PROMISE_AF.tsv",
        bert_name="bert-large-uncased",
        generator_name="res_mlp",
        discriminator_name="mlp_base",
        k_fold=3,
        epochs=10,
        batch_size=16,
        seed=42,
        output_file="stage2_test.log"
    )


if __name__ == "__main__":
    main()
