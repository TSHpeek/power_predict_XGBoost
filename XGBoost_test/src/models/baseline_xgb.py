import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

# 这两行是解决中文方块问题的关键
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 Windows 自带的“黑体”
plt.rcParams['axes.unicode_minus'] = False     # 解决负号 "-" 显示为方块的问题

# ============ 1. 造数据（模拟光伏15min出力） ============
def generate_pv_data(days=30, freq_min=15):
    """
    模拟30天、15min粒度的光伏出力
    白天有功率，晚上为0，加噪声
    """
    periods = days * 24 * (60 // freq_min)  # 30天 * 96步/天 = 2880
    t = pd.date_range(start="2024-01-01", periods=periods, freq=f"{freq_min}min")

    # 时间特征
    hour = t.hour + t.minute / 60  # 0~23.75
    dayofweek = t.dayofweek

    # 模拟辐照度：白天正弦曲线
    irradiance = np.maximum(0, np.sin((hour - 6) / 12 * np.pi))
    irradiance *= np.random.uniform(0.7, 1.0, size=len(t))  # 天气波动

    # 光伏功率 = 辐照度 * 最大功率(50kW) + 噪声
    power = irradiance * 50
    power += np.random.normal(0, 1.5, size=len(t))
    power = np.clip(power, 0, 50)

    df = pd.DataFrame({
        "time": t,
        "power_kw": power,
        "irradiance": irradiance,
        "hour": hour,
        "dayofweek": dayofweek,
    })
    return df


# ============ 2. 滑窗构造特征 ============
def create_sliding_features(df, target_col="power_kw", lookback=16, horizon=4):
    """
    lookback=16 → 用过去4小时(16个15min)预测未来1小时(4个15min)
    horizon=4   → 预测未来4步
    """
    features = []
    targets = []

    values = df[target_col].values
    irradiance = df["irradiance"].values
    hour = df["hour"].values
    dayofweek = df["dayofweek"].values

    for i in range(lookback, len(df) - horizon + 1):
        # --- 特征 ---
        feat = {}

        # 历史功率（过去16步）
        hist_power = values[i - lookback:i]
        for j in range(lookback):
            feat[f"pow_t-{lookback - j}"] = hist_power[j]

        # 历史统计
        feat["hist_mean"] = np.mean(hist_power)
        feat["hist_std"] = np.std(hist_power)
        feat["hist_max"] = np.max(hist_power)
        feat["hist_min"] = np.min(hist_power)

        # 时间特征（用当前时刻）
        feat["hour"] = hour[i]
        feat["dayofweek"] = dayofweek[i]
        feat["is_weekend"] = 1 if dayofweek[i] >= 5 else 0

        # 当前辐照度
        feat["irradiance"] = irradiance[i]

        # 目标：未来4步的功率
        target = values[i:i + horizon]

        features.append(feat)
        targets.append(target)

    feature_df = pd.DataFrame(features)
    target_array = np.array(targets)

    return feature_df, target_array


# ============ 3. 训练 + 评估 ============
def train_and_evaluate():
    print("=" * 50)
    print("光伏超短期预测 - XGBoost 基线")
    print("=" * 50)

    # 1. 造数据
    print("\n[1/5] 生成模拟光伏数据...")
    df = generate_pv_data(days=30)
    print(f"  数据量: {len(df)} 条 ({len(df) // 96} 天)")

    # 2. 滑窗构造
    print("\n[2/5] 滑窗构造特征...")
    lookback, horizon = 16, 4
    X, y = create_sliding_features(df, lookback=lookback, horizon=horizon)
    print(f"  样本数: {len(X)}, 特征数: {X.shape[1]}")
    print(f"  目标: 预测未来 {y.shape[1]} 步 (1小时)")

    # 3. 按原始时间划分训练/测试，避免 horizon 标签跨越边界。
    raw_boundary = int(len(df) * 0.8)
    train_rows = raw_boundary - lookback - horizon + 1
    test_start = raw_boundary - lookback
    X_train = X.iloc[:train_rows]
    X_test = X.iloc[test_start:]
    y_train = y[:train_rows]
    y_test = y[test_start:]

    print(f"\n[3/5] 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 4. 训练 XGBoost（多输出：一步一个模型）
    print("\n[4/5] 训练 XGBoost...")
    models = []
    for step in range(y_train.shape[1]):
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train[:, step])
        models.append(model)
        print(f"  Step {step + 1}/4 训练完成")

    # 5. 预测 + 评估
    print("\n[5/5] 评估...")
    y_pred = np.column_stack([m.predict(X_test) for m in models])

    # 计算指标（只看第1步预测，即15min后）
    mae = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    daylight = y_test[:, 0] > 1e-6
    mape = (
        mean_absolute_percentage_error(y_test[daylight, 0], y_pred[daylight, 0]) * 100
        if np.any(daylight) else np.nan
    )

    print(f"\n{'=' * 50}")
    print(f"结果（预测15min后）:")
    print(f"  MAE  = {mae:.3f} kW")
    print(f"  MAPE = {mape:.2f}%")
    print(f"{'=' * 50}")

    # 特征重要性
    print(f"\n特征重要性 Top 10:")
    importance = models[0].feature_importances_
    feat_names = X.columns
    sorted_idx = np.argsort(importance)[::-1][:10]
    for i, idx in enumerate(sorted_idx):
        print(f"  {i + 1}. {feat_names[idx]:<25s} {importance[idx]:.4f}")

    # 画一条预测对比
    plt.figure(figsize=(12, 4))
    plt.plot(y_test[:96, 0], label="实际", alpha=0.7)
    plt.plot(y_pred[:96, 0], label="预测", alpha=0.7)
    plt.title("测试集前96步(1天) 光伏功率预测对比")
    plt.xlabel("时间步 (15min)")
    plt.ylabel("功率 (kW)")
    plt.legend()
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "prediction_vs_actual.png"
    plt.savefig(result_path, dpi=150)
    print(f"\n预测对比图已保存: {result_path}")

    return mae, mape


if __name__ == "__main__":
    train_and_evaluate()
