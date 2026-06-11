from model_builder import _map_generator_impl
from run_op_new import run_single_experiment
import os
import datetime

if __name__ == "__main__":
    datasets = {
        "PROMISE_EXP": "./data/transData/PROMISE_exp_AF.tsv"
    }
    # datasets = {
    #     # "PROMISE": "./data/transData/PROMISE_AF.tsv",
    #     # "PROMISE_EXP": "./data/transData/PROMISE_exp_AF.tsv",
    #     # "NFR-Review": "./data/transData/NFR-Review-AF.tsv",
    #     "NFR-SO": "./data/transData/NFR-SO-AF.tsv"
    # }

    # ===============================
    # 动态生成输出文件夹和文件名
    # ===============================
    date_str = datetime.datetime.now().strftime("%Y%m%d")  # 获取当前日期
    time_str = datetime.datetime.now().strftime("%H%M%S")  # 当前时间（防止重复）

    # 输出目录：./results/20251014/
    output_dir = os.path.join(".", "results", "debug", date_str)
    os.makedirs(output_dir, exist_ok=True)

    # 输出文件名：experiment_results_20251014_150523.txt
    output_file = os.path.join(output_dir, f"experiment_results_{date_str}_{time_str}.txt")

    print(f" 实验结果将保存至: {output_file}")

    # # 单个结构组合LIST
    # model_combos = [
    #     # 02 BERT=bert-large-uncased, G=mlp_deep, D=attention_head | best_act=relu | KFold-AvgF1=0.7427
    #     # 03 BERT=roberta-base, G=mlp_base, D=mlp_base | best_act=relu | KFold-AvgF1=0.6875
    #     # BERT = roberta - base, G = mlp_deep, D = attention_head, gen_activation = elu
    #     # ("bert-large-uncased", "mlp_deep", "attention_head"),
    #     # ("bert-large-cased", "transformer_light", "attention_head"),
    #     ("roberta-base", "mlp_deep", "attention_head", "elu")
    #     # ("bert-base-cased", "res_mlp", "mlp_deep"),
    # ]

    model_combos = [
        # ("roberta-base", "mlp_deep", "mlp_base")
        # ("roberta-large", "mlp_deep", "res_mlp"),
        ("roberta-large", "cnn", True, "attention_head", None)
    ]

    for name, path in datasets.items():
        for (bert_name, base_g_name, con, d_name, act) in model_combos:
            g_name = _map_generator_impl(base_g_name, con)
            print(f"\n[{name}] 训练组合: BERT={bert_name}, G={g_name}, D={d_name}, ACT={act}")
            run_single_experiment(
                dataset_name=name,
                data_dir=path,
                bert_name=bert_name,
                generator_name=g_name,
                discriminator_name=d_name,
                labeled_ratio=0.2,
                test_ratio=0.2,
                epochs=30,
                output_file=output_file,
                act=act
            )

