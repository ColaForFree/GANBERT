import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

def parse_arff_with_strings(path):
    lines = Path(path).read_text(encoding="latin1").splitlines()
    data_section = False
    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if line.lower() == "@data":
            data_section = True
            continue
        if data_section:
            fields = [s.strip().strip("'") for s in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", line)]
            data.append(fields)
    attr_lines = [l for l in lines if l.lower().startswith("@attribute")]
    columns = [re.findall(r"@attribute\s+([^\s]+)", l, re.I)[0] for l in attr_lines]
    return pd.DataFrame(data, columns=columns)

def write_plain_format(df_subset, filepath, with_label=True):
    with open(filepath, "w", encoding="utf-8") as f:
        for _, row in df_subset.iterrows():
            if with_label:
                parts = row['fine_label'].split(":")
                f.write(f"{parts[0]}:{parts[1]} {row['utterance']}\n")
            else:
                f.write(f"UNK:UNK {row['utterance']}\n")


def main():
    input_path = "data/NFR-so/so_nfr.csv"  # .arff 文件路径
    output_dir = Path("./data/NFR-so")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取 so_nfr.csv，并重命名为与原流程一致的字段名
    df = pd.read_csv(input_path, sep="!#!", engine="python", header=0)
    df.columns = ["utterance", "label"]

    # 文本清洗
    df["utterance"] = df["utterance"].astype(str).str.strip()
    df["utterance"] = df["utterance"].str.replace(r'(^"|"$)', '', regex=True)
    df["utterance"] = df["utterance"].str.replace(r'"', '', regex=True)
    df["utterance"] = df["utterance"].str.replace(r"\s+", " ", regex=True).str.strip()

    # 标签映射：0-6 → NFR类别
    label_map = {
        0: "availability",
        1: "performance",
        2: "maintainability",
        3: "portability",
        4: "scalability",
        5: "security",
        6: "fault-tolerance"
    }

    df["label"] = df["label"].astype(int)
    df["fine_label"] = df["label"].map(label_map)
    df["fine_label"] = "NFR:" + df["fine_label"]

    # 删除低频类
    df = df.groupby("fine_label").filter(lambda x: len(x) >= 2)

    # 划分数据集
    if len(df) < 10:
        print(" 样本数量过少，无法进行训练集/测试集划分")
        return

    df_rest, df_test = train_test_split(df, test_size=0.2, stratify=df["fine_label"], random_state=42)
    df_labeled = df_rest.groupby("fine_label", group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))
    df_unlabeled = df_rest.drop(df_labeled.index)

    # 写入TSV文件
    write_plain_format(df_labeled, output_dir / "labeled.tsv", with_label=True)
    write_plain_format(df_unlabeled, output_dir / "unlabeled.tsv", with_label=False)
    write_plain_format(df_test, output_dir / "test.tsv", with_label=True)

    print(" 转换成功，文件输出至 ./data/promiseData/")

if __name__ == "__main__":
    main()
