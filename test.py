# from huggingface_hub import snapshot_download
#
# snapshot_download(repo_id="bert-base-cased", local_dir="./models/bert-base-cased")
# from transformers import BertModel, BertTokenizer, BertConfig
# print(BertModel)
# import torch
# print(f"CUDA 是否可用: {torch.cuda.is_available()}")
# print(f"当前 CUDA 版本: {torch.version.cuda}")
# print(f"当前设备 ID: {torch.cuda.current_device()}")
# print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# import torch
# from transformers import BertModel, BertTokenizer
# # 加载预训练模型和分词器
# model = BertModel.from_pretrained('./models/roberta-base')
# tokenizer = BertTokenizer.from_pretrained('mypath/bert-base-chinese')
# # 加载预训练模型的权重
# state_dict = torch.load('mypath/bert-base-chinese.pt')
# model.load_state_dict(state_dict)

# from transformers import AutoModel, AutoTokenizer
# import torch
#
# tok = AutoTokenizer.from_pretrained("./models/roberta-base")
# m = AutoModel.from_pretrained("./models/roberta-base")
#
# x = tok("hello world", return_tensors="pt")
# out = m(**x).last_hidden_state[:,0,:]
#
# print(out.std())   # 看标准差
# import os
# print(os.path.exists("/media/aaa/DATA"))
# print(os.listdir("/media/aaa") if os.path.exists("/media/aaa") else "NO MEDIA")
# import torch
# print("torch:", torch.__version__)
# print("cuda:", torch.version.cuda)
# print("cudnn:", torch.backends.cudnn.version())
# print("cudnn enabled:", torch.backends.cudnn.enabled)
import matplotlib

# 添加下载的字体文件
matplotlib.font_manager.fontManager.addfont('chinese.simhei.ttf')

# 设置 Matplotlib 使用 SimHei 字体
matplotlib.rc('font', family='SimHei')
