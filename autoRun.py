from run_op_new import run_experiment_random_s1, run_experiment_random_s2, run_experiment_bruteforce_s1, \
    run_experiment_s2, run_single_experiment, run_experiment_s1
import os
import datetime

if __name__ == "__main__":
    ori_datasets = {
        "PROMISE_EXP": "./data/transData/PROMISE_exp_AF.tsv"
    }
    # datasets = {
    #     "PROMISE": "./data/transData/PROMISE_AF.tsv",
    #     "PROMISE_EXP": "./data/transData/PROMISE_exp_AF.tsv",
    #     "NFR-Review": "./data/transData/NFR-Review-AF.tsv",
    #     "NFR-SO": "./data/transData/NFR-SO-AF.tsv"
    # }

    trans_datasets = {
        # "PROMISE": "./data/transData/PROMISE_AF.tsv",
        # "PROMISE_EXP": "./data/transData/PROMISE_exp_AF.tsv",
        # "NFR-Review": "./data/transData/NFR-Review-AF.tsv",
        # "NFR-SO": "./data/transData/NFR-SO-AF.tsv"
    }

    # ===============================
    # 动态生成输出文件夹和文件名
    # ===============================
    date_str = datetime.datetime.now().strftime("%Y%m%d")  # 获取当前日期
    time_str = datetime.datetime.now().strftime("%H%M%S")  # 当前时间（防止重复）

    # 输出目录：./results/20251014/
    output_dir = os.path.join(".", "results", date_str)
    os.makedirs(output_dir, exist_ok=True)

    # 输出文件名：experiment_results_20251014_150523.txt
    output_file = os.path.join(output_dir, f"experiment_results_{date_str}_{time_str}.txt")

    print(f" 实验结果将保存至: {output_file}")

    # ===============================
    # 执行暴力遍历实验(S1)
    # ===============================
    best_combo_s1 = None
    # for name, path in ori_datasets.items():
    #     print(f"\n开始暴力遍历实验: {name} -> {path}")
    #     best_combo_s1 = run_experiment_bruteforce_s1(
    #         dataset_name=name,
    #         data_dir=path,
    #         labeled_ratio=0.2,
    #         test_ratio=0.2,
    #         output_file=output_file,
    #         epochs=30
    #     )

    # # ===============================
    # # 执行random实验(s1)
    # # ===============================
    # best_combo_s1 = None
    # for name, path in ori_datasets.items():
    #     best_combo = run_experiment_random_s1(name, path, labeled_ratio=0.2, test_ratio=0.2,
    #                    output_file=output_file, n_trials=60, epochs=30)




    # # ===============================
    # # 执行OB_s1实验
    # # ===============================
    # best_combo = None
    # for name, path in ori_datasets.items():
    #     best_combo = run_experiment_s1(name, path, labeled_ratio=0.2, test_ratio=0.2,
    #                    output_file=output_file, n_trials=60, epochs=30)


    # # ===============================
    # # 执行random实验(s2)
    # # ===============================
    # for name, path in ori_datasets.items():
    #     best_combo = run_experiment_random_s2(name, path, labeled_ratio=0.2, test_ratio=0.2,
    #                    output_file=output_file, n_trials=60, epochs=30, batch_size=16)

    # ===============================
    # 执行OB(s2)实验 epochs是最终训练迭代次数
    # ===============================
    best_combo = None
    for name, path in ori_datasets.items():
        best_combo = run_experiment_s2(name, path, labeled_ratio=0.2, test_ratio=0.2,
                       output_file=output_file, n_trials=60, epochs=30, top_k=5)


    # 在迁移数据集上进行训练测试

    # # [Stage-1-Bruteforce] Best combo = BERT=roberta-base, G=cnn (cond=False), D=attention_head
    # best_combo_s1 = {
    #     "bert_name": "roberta-base",
    #     "generator_name": "cnn",
    #     "discriminator_name": "attention_head",
    #     "gen_is_conditional": False,
    #     "act": None
    # }
    # 01 BERT=roberta-large, G=mlp_deep, D=res_mlp | best_act=gelu
    # BERT=roberta-large, G=cnn, D=attention_head, act = gelu
    # BERT = roberta - large, G = transformer_light, D = attention_head
    (bert_name, generator_name, discriminator_name, act) = ("roberta-large", "cnn", "attention_head", None)
    for name, path in trans_datasets.items():
        print(f"\n[{name}] 训练组合: BERT={bert_name}, G={generator_name}, D={discriminator_name}, ACT={act}")
        run_single_experiment(
            dataset_name=name,
            data_dir=path,
            bert_name=bert_name,
            generator_name=generator_name,
            discriminator_name=discriminator_name,
            act=act,
            labeled_ratio=0.2,
            test_ratio=0.2,
            epochs=30,
            output_file=output_file
        )
    # (bert_name, generator_name, discriminator_name, act) = ("roberta-large", "transformer_light", "attention_head", "gelu")
    # for name, path in trans_datasets.items():
    #     print(f"\n[{name}] 训练组合: BERT={bert_name}, G={generator_name}, D={discriminator_name}, ACT={act}")
    #     run_single_experiment(
    #         dataset_name=name,
    #         data_dir=path,
    #         bert_name=bert_name,
    #         generator_name=generator_name,
    #         discriminator_name=discriminator_name,
    #         act=act,
    #         labeled_ratio=0.2,
    #         test_ratio=0.2,
    #         epochs=30,
    #         output_file=output_file
    #     )