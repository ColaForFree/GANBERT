# import matplotlib.pyplot as plt
#
# # x 轴：标注比例（百分数）
# x_percent = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100]
#
# # 三个模型的 F1 结果
# bert = [0.073049, 0.130259, 0.236839, 0.30047, 0.379169, 0.524033, 0.53831, 0.611382, 0.68228, 0.693668]
#
# ganbert = [0.07808534462066688, 0.12991569833675098, 0.23136631773196445, 0.2685846720329479, 0.4704617765959435,
#            0.49845848938394804, 0.5644540644540644, 0.6045318974111544, 0.7551204063374449, 0.7640705529885959]
#
# bo_ganbert = [0.1792810364238936, 0.16486772486772486, 0.3742212009842548, 0.5437264513532262, 0.6461695240838876,
#               0.659577219317479, 0.6994429945386883, 0.7017197168010685, 0.7554498531771259, 0.7779665572522716]
#
#
# plt.figure(figsize=(6, 4))
#
# # 画三条折线
# plt.plot(x_percent, [v * 100 for v in bert],
#          linestyle=':', marker='o', label='BERT')
# plt.plot(x_percent, [v * 100 for v in ganbert],
#          linestyle='-', marker='^', label='GAN-BERT')
# plt.plot(x_percent, [v * 100 for v in bo_ganbert],
#          linestyle='-', marker='s', label='BO-GANBERT')
#
# # 坐标轴与标题
# plt.xlabel('Annotated %')
# plt.ylabel('Macro-F1')
# plt.title('PROMISE_EXP')
#
# # x 轴刻度
# plt.xticks(x_percent, x_percent)
#
# # 网格 & 图例
# plt.grid(True, linestyle='--', alpha=0.4)
# plt.legend()
#
# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt

# x 轴：标注比例（百分数）——只到 50%
x_percent = [1, 2, 5, 10, 20, 30, 40, 50]

# 三个模型的 F1 结果，只取前 8 个值
bert = [0.073049, 0.130259, 0.236839, 0.30047, 0.379169,
        0.524033, 0.53831, 0.611382]

ganbert = [0.07808534462066688, 0.12991569833675098, 0.23136631773196445,
           0.2685846720329479, 0.4704617765959435, 0.49845848938394804,
           0.5644540644540644, 0.6045318974111544]

bo_ganbert = [0.1792810364238936, 0.16486772486772486, 0.3742212009842548,
              0.5437264513532262, 0.6461695240838876, 0.659577219317479,
              0.6994429945386883, 0.7017197168010685]

plt.figure(figsize=(6, 4))

# 画三条折线
plt.plot(x_percent, [v * 100 for v in bert],
         linestyle=':', marker='o', label='BERT')
plt.plot(x_percent, [v * 100 for v in ganbert],
         linestyle='-', marker='^', label='GAN-BERT')
plt.plot(x_percent, [v * 100 for v in bo_ganbert],
         linestyle='-', marker='s', label='BO-GANBERT')

# 坐标轴与标题
plt.xlabel('Annotated %')
plt.ylabel('Macro-F1')
plt.title('PROMISE_EXP')

# x 轴刻度
plt.xticks(x_percent, x_percent)

# 网格 & 图例
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()
