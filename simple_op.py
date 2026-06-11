import optuna
import math

# 黑盒目标函数：我们要最大化它（比如模拟F1得分）
def objective(trial):
    x = trial.suggest_float("x", 0, 6.28)  # 相当于 [0, 2π]
    y = trial.suggest_float("y", 0, 6.28)

    # 模拟的复杂非凸函数
    score = math.sin(x) + math.cos(y)

    return score  # Optuna 默认是最大化时返回这个

# 创建一个优化任务
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
0
# 输出最优结果
print("Best trial:")
print(f"  x = {study.best_params['x']:.4f}")
print(f"  y = {study.best_params['y']:.4f}")
print(f"  max score = {study.best_value:.4f}")
