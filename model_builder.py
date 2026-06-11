# model_builder.py
# 构建组合: tokenizer + transformer(BERT) + generator + discriminator

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
from G.generators import build_generator
from D.discriminators import build_discriminator
from bert_loader.loader import load_bert


def build_ganbert_components(
    model_name="bert-base-cased",          # 只传模型名称
    generator_name="mlp_base",
    discriminator_name="mlp_base",
    num_labels=8,
    noise_size=100,
    device=None,
    multi_gpu=False,
    hf_kwargs=None,
    base_model_dir="/media/aaa/DATA/lcz/models"              # 模型根目录
):
    """
    构建并返回 tokenizer, transformer, generator, discriminator, hidden_size, device
    """
    hf_kwargs = hf_kwargs or {"local_files_only": True}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- BERT (加载 + 投影) ----
    transformer = load_bert(model_name=model_name)

    model_path = os.path.join(base_model_dir, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, **hf_kwargs)

    hidden_size = transformer.config.hidden_size

    # ---- Generator ----
    generator = build_generator(generator_name, noise_size=noise_size, num_classes = num_labels, output_size=hidden_size)

    # ---- Discriminator ----
    discriminator = build_discriminator(discriminator_name, input_size=hidden_size, num_labels=num_labels)

    # ---- 放到设备 ----
    transformer.to(device)
    generator.to(device)
    discriminator.to(device)

    if multi_gpu and torch.cuda.device_count() > 1:
        transformer = nn.DataParallel(transformer)

    return {
        "tokenizer": tokenizer,
        "transformer": transformer,
        "generator": generator,
        "discriminator": discriminator,
        "hidden_size": hidden_size,
        "device": device,
    }

def _map_generator_impl(gen_family: str, gen_is_cond: bool) -> str:
    """
    将生成器候选名（trial 里采样的 gen_family）+ 是否使用条件标签 gen_is_cond
    映射为具体实现名。
    约定：conditional 版本命名为 "<base>_cond"
    - mlp_base      -> mlp_base 或 mlp_base_cond
    - mlp_deep      -> mlp_deep 或 mlp_deep_cond
    - res_mlp       -> res_mlp  或 res_mlp_cond
    - cnn           -> cnn      或 cnn_cond
    - transformer_light -> 仅支持无条件（没有 *_cond 版本）
    """

    # 定义支持的映射；
    registry = {
        "mlp_base":          {"base": "mlp_base",          "cond": "mlp_base_cond"},
        "mlp_deep":          {"base": "mlp_deep",          "cond": "mlp_deep_cond"},
        "res_mlp":           {"base": "res_mlp",           "cond": "res_mlp_cond"},
        "cnn":               {"base": "cnn",               "cond": "cnn_cond"},
        "transformer_light": {"base": "transformer_light", "cond": "transformer_light_cond"},
    }

    if gen_family not in registry:
        raise ValueError(f"Unknown generator class: {gen_family}")

    entry = registry[gen_family]

    # 需要条件版但没有实现时，回退到 base 并给出一次性提示
    if gen_is_cond:
        if entry["cond"] is not None:
            return entry["cond"]
        else:
            print(f"[Warn] Generator '{gen_family}' has no conditional implementation. "
                  f"Fallback to '{entry['base']}'.")
            return entry["base"]
    else:
        return entry["base"]



# --------------------- #
# 测试用例
# --------------------- #
if __name__ == "__main__":
    print(">>> start running model_builder.py")

    comps = build_ganbert_components(
        model_name="roberta-base",           # 会自动去 ./models/roberta-base/
        generator_name="cnn",
        discriminator_name="attention_head",
        num_labels=8,
        target_hidden=768,
        multi_gpu=False,
    )

    tokenizer = comps["tokenizer"]
    transformer = comps["transformer"]
    G = comps["generator"]
    D = comps["discriminator"]
    device = comps["device"]

    # 构造一条假输入
    text = "System shall respond within two seconds under normal load."
    encoded = tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = transformer(input_ids, attention_mask=attn_mask)
        cls_vec = outputs[0][:, 0, :]   # last_hidden_state 的 [CLS]
    print("BERT CLS vec:", cls_vec.shape)

    z = torch.randn(1, 100).to(device)
    fake_vec = G(z)
    print("Generator vec:", fake_vec.shape)

    h, logits, probs = D(cls_vec)
    print("Discriminator logits:", logits.shape)
    print("Discriminator probs:", probs.shape)


