"""
端到端流水线：
XGBoost 预测光伏功率 → cvxpy 储能调度 → 输出收益对比
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import cvxpy as cp
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


# ======================== 1. 数据生成 ========================
def generate_pv_data(days=7, freq_min=15):
    """模拟光伏数据（用7天，前5天训练，后2天测试+调度）"""
    periods = days * 24 * (60 // freq_min)
    t = pd.date_range(start="2024-01-01", periods=periods, freq=f"{freq_min}min")

    hour = t.hour + t.minute / 60
    dayofweek = t.dayofweek

    irradiance = np.maximum(0, np.sin((hour - 6) / 12 * np.pi))
    # 加天气波动（模拟阴天）
    cloud_mask = np.random.rand(len(t)) < 0.15  # 15%概率阴天
    irradiance = np.where(cloud_mask, irradiance * np.random.uniform(0.2, 0.5), irradiance)

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


# ======================== 2. 滑窗特征 ========================
def create_features(df, lookback=16, horizon=4):
    """构造滑窗特征"""
    features = []
    targets = []

    power = df["power_kw"].values
    irr = df["irradiance"].values
    hour = df["hour"].values
    dow = df["dayofweek"].values

    for i in range(lookback, len(df) - horizon + 1):
        feat = {}

        # 历史功率
        hist = power[i - lookback:i]
        for j in range(lookback):
            feat[f"p_{j}"] = hist[j]

        feat["hist_mean"] = np.mean(hist)
        feat["hist_std"] = np.std(hist)
        feat["hist_diff"] = hist[-1] - hist[-2]  # 变化趋势

        # 时间
        feat["hour"] = hour[i]
        feat["dayofweek"] = dow[i]
        feat["is_weekend"] = int(dow[i] >= 5)

        # 辐照度
        feat["irradiance"] = irr[i]

        # 目标：未来 horizon 步
        targets.append(power[i:i + horizon])
        features.append(feat)

    return pd.DataFrame(features), np.array(targets)


# ======================== 3. XGBoost 预测 ========================
def train_xgb(X_train, y_train, X_test):
    """训练XGBoost，返回测试集预测"""
    models = []
    preds = np.zeros((len(X_test), y_train.shape[1]))

    for step in range(y_train.shape[1]):
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train[:, step])
        preds[:, step] = model.predict(X_test)
        models.append(model)

    return models, preds


# ======================== 4. 电价生成 ========================
def generate_price(T, base_date="2024-01-01"):
    """生成江苏峰谷电价"""
    t = pd.date_range(start=base_date, periods=T, freq="15min")
    price = np.ones(T) * 0.6

    for i, ti in enumerate(t):
        h = ti.hour + ti.minute / 60
        if 0 <= h < 8:
            price[i] = 0.30
        elif 8 <= h < 11:
            price[i] = 0.60
        elif 11 <= h < 14:
            price[i] = 1.00
        elif 14 <= h < 17:
            price[i] = 0.60
        elif 17 <= h < 22:
            price[i] = 1.00
        else:
            price[i] = 0.30

    return price


# ======================== 5. cvxpy 调度 ========================
def optimize_dispatch(pv_forecast, price, capacity=100, power_max=50,
                      soc_init=0.5, soc_min=0.1, soc_max=0.9,
                      eta_c=0.95, eta_d=0.95, dt=0.25):
    """用线性规划求解光伏+储能的最优调度。

    ``charge`` 的约定保持不变：返回值大于 0 表示充电，小于 0 表示放电。
    光伏优先用于充电或直接上网，电网充电按电价计成本。
    """
    pv_forecast = np.asarray(pv_forecast, dtype=float)
    price = np.asarray(price, dtype=float)
    if pv_forecast.ndim != 1 or price.ndim != 1 or len(pv_forecast) != len(price):
        raise ValueError("pv_forecast 和 price 必须是长度相同的一维数组")
    if len(price) == 0 or np.any(pv_forecast < 0):
        raise ValueError("pv_forecast 必须非负且不能为空")

    T = len(pv_forecast)
    charge_pv = cp.Variable(T, nonneg=True)
    charge_grid = cp.Variable(T, nonneg=True)
    discharge = cp.Variable(T, nonneg=True)
    pv_export = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T)

    charge_in = charge_pv + charge_grid
    constraints = [
        soc >= soc_min * capacity,
        soc <= soc_max * capacity,
        charge_in + discharge <= power_max,
        charge_pv + pv_export <= pv_forecast,
    ]

    for t in range(T):
        previous_soc = soc_init * capacity if t == 0 else soc[t - 1]
        constraints.append(
            soc[t] == previous_soc
            + eta_c * charge_in[t] * dt
            - discharge[t] / eta_d * dt
        )

    # 保持期末电量，避免把初始电量当作本周期收益。
    constraints.append(soc[-1] == soc_init * capacity)

    total_revenue = cp.sum(
        cp.multiply(pv_export * dt, price)
        + cp.multiply(discharge * dt, price)
        - cp.multiply(charge_grid * dt, price)
    )
    prob = cp.Problem(cp.Maximize(total_revenue), constraints)
    try:
        prob.solve(verbose=False)
    except cp.error.SolverError:
        return None, None, 0

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None, None, 0

    # 正值表示充电，负值表示放电。
    charge_values = charge_in.value - discharge.value
    return charge_values, soc.value, float(total_revenue.value)


# ======================== 6. 盲调度基线 ========================
def baseline_dispatch(price, capacity=100, power_max=50, dt=0.25,
                      pv_available=None, soc_init=0.5, soc_min=0.1,
                      soc_max=0.9, eta_c=0.95, eta_d=0.95):
    """按峰谷规则调度；只使用当前时段的光伏功率，不使用未来预测。"""
    price = np.asarray(price, dtype=float)
    T = len(price)
    if T == 0:
        raise ValueError("price 不能为空")
    if pv_available is None:
        pv_available = np.zeros(T)
    else:
        pv_available = np.asarray(pv_available, dtype=float)
        if pv_available.shape != price.shape or np.any(pv_available < 0):
            raise ValueError("pv_available 必须与 price 等长且非负")

    charge = np.zeros(T)
    soc = soc_init * capacity
    soc_min_kwh = soc_min * capacity
    soc_max_kwh = soc_max * capacity
    target_soc = soc_init * capacity
    revenue = 0.0

    for t in range(T):
        pv_to_battery = 0.0
        grid_charge = 0.0
        discharge = 0.0

        if t == T - 1 and abs(soc - target_soc) > 1e-9:
            # 期末回到初始 SOC，保证与优化结果可比较。
            if soc > target_soc:
                discharge = min(power_max, (soc - target_soc) * eta_d / dt)
            else:
                desired_charge = min(power_max, (target_soc - soc) / (eta_c * dt))
                pv_to_battery = min(pv_available[t], desired_charge)
                grid_charge = desired_charge - pv_to_battery
        elif price[t] <= 0.35:
            desired_charge = min(
                power_max, (soc_max_kwh - soc) / (eta_c * dt)
            )
            pv_to_battery = min(pv_available[t], desired_charge)
            grid_charge = desired_charge - pv_to_battery
        elif price[t] >= 0.85:
            discharge = min(power_max, max(0.0, (soc - soc_min_kwh) * eta_d / dt))

        pv_export = max(0.0, pv_available[t] - pv_to_battery)
        charge_in = pv_to_battery + grid_charge
        soc += eta_c * charge_in * dt - discharge / eta_d * dt
        soc = np.clip(soc, soc_min_kwh, soc_max_kwh)
        charge[t] = charge_in - discharge
        revenue += (
            pv_export * dt * price[t]
            + discharge * dt * price[t]
            - grid_charge * dt * price[t]
        )

    return charge, float(revenue)


# ======================== 7. 主流程 ========================
def run_pipeline():
    print("=" * 60)
    print("  光伏预测 + 储能调度 端到端流水线")
    print("=" * 60)

    # ---------- Step 1: 数据 ----------
    print("\n[Step 1/6] 生成数据...")
    df = generate_pv_data(days=7)
    print(f"  总数据: {len(df)} 条 (7天)")

    # ---------- Step 2: 滑窗特征 ----------
    print("[Step 2/6] 构造滑窗特征...")
    lookback, horizon = 16, 4
    X, y = create_features(df, lookback=lookback, horizon=horizon)
    print(f"  样本: {len(X)}, 特征: {X.shape[1]}")

    # 按时间划分：前5天训练，后2天测试
    train_boundary = 5 * 96  # 原始数据中的第6天起点
    train_rows = train_boundary - lookback - horizon + 1
    test_start = train_boundary - lookback
    X_train = X.iloc[:train_rows]
    X_test = X.iloc[test_start:]
    y_train = y[:train_rows]
    y_test = y[test_start:]

    print(f"  训练: {len(X_train)}, 测试: {len(X_test)}")

    # ---------- Step 3: XGBoost 预测 ----------
    print("[Step 3/6] 训练 XGBoost 并预测...")
    models, y_pred = train_xgb(X_train, y_train, X_test)

    mae = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    daylight = y_test[:, 0] > 1e-6
    mape = (
        mean_absolute_percentage_error(y_test[daylight, 0], y_pred[daylight, 0]) * 100
        if np.any(daylight) else np.nan
    )
    print(f"  预测精度 (15min后): MAE={mae:.2f}kW, MAPE={mape:.2f}%")

    # ---------- Step 4: 接电价 + 调度 ----------
    print("[Step 4/6] 生成电价 + 运行储能调度...")
    T_test = len(y_pred)  # 测试集步数
    price = generate_price(T_test, base_date="2024-01-06")

    # 用预测结果作为光伏出力（取第1步预测）
    pv_forecast = np.clip(y_pred[:, 0], 0, 50)

    # 最优调度
    charge_opt, soc_opt, revenue_opt = optimize_dispatch(
        pv_forecast=pv_forecast,
        price=price,
        capacity=100,
        power_max=50,
    )

    if charge_opt is None:
        print("  ⚠️ 优化求解失败！")
        return

    print(f"  最优调度收益: {revenue_opt:.2f} 元")

    # ---------- Step 5: 盲调度对比 ----------
    print("[Step 5/6] 盲调度基线对比...")
    # 基线只使用测试时段已经发生的光伏出力，不使用模型预测结果。
    charge_bl, revenue_bl = baseline_dispatch(price, pv_available=y_test[:, 0])
    print(f"  盲调度收益:   {revenue_bl:.2f} 元")
    print(f"  收益提升:     {((revenue_opt - revenue_bl) / abs(revenue_bl) * 100):.1f}%")

    # ---------- Step 6: 可视化 ----------
    print("[Step 6/6] 生成结果图...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 子图1：预测 vs 实际
    time_ax = np.arange(T_test) * 0.25
    axes[0].plot(time_ax, y_test[:T_test, 0], label="实际出力", alpha=0.7, linewidth=1.5)
    axes[0].plot(time_ax, pv_forecast, label="XGBoost预测", alpha=0.7, linewidth=1.5, linestyle='--')
    axes[0].set_ylabel("功率 (kW)", fontsize=11)
    axes[0].set_title(f"光伏功率预测 (MAPE={mape:.1f}%)", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 子图2：电价
    axes[1].step(time_ax, price, where='mid', color='orange', linewidth=2)
    axes[1].axhline(y=0.30, color='green', linestyle='--', alpha=0.5, label='谷时')
    axes[1].axhline(y=1.00, color='red', linestyle='--', alpha=0.5, label='峰时')
    axes[1].set_ylabel("电价 (元/kWh)", fontsize=11)
    axes[1].set_title("电价曲线", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # 子图3：充放电对比
    axes[2].fill_between(time_ax, charge_opt, where=(charge_opt > 0),
                         color='blue', alpha=0.5, label='最优-充电')
    axes[2].fill_between(time_ax, charge_opt, where=(charge_opt < 0),
                         color='red', alpha=0.5, label='最优-放电')
    axes[2].plot(time_ax, charge_bl, color='gray', linewidth=1, linestyle=':', alpha=0.7, label='盲调度')
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].set_xlabel("时间 (小时)", fontsize=11)
    axes[2].set_ylabel("功率 (kW)", fontsize=11)
    axes[2].set_title("充放电策略对比", fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "pipeline_result.png"
    plt.savefig(result_path, dpi=150)
    print(f"  结果图已保存: {result_path}")

    # ---------- 总结 ----------
    print(f"\n{'=' * 60}")
    print(f"  流水线完成！总结：")
    print(f"  预测模型: XGBoost, MAPE={mape:.2f}%")
    print(f"  最优调度: {revenue_opt:.2f} 元 (测试期间)")
    print(f"  盲调度:   {revenue_bl:.2f} 元")
    print(f"  提升:     {((revenue_opt - revenue_bl) / abs(revenue_bl) * 100):.1f}%")
    print(f"  收益来源: 预测精度→调度决策→套利收益")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_pipeline()
