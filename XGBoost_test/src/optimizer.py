import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def storage_dispatch(
        pv_forecast,  # 光伏预测功率，shape=(T,)，单位 kW
        price,  # 电价，shape=(T,)，单位 元/kWh
        capacity=100,  # 电池容量，kWh
        power_max=50,  # 最大充放电功率，kW（正=充电，负=放电）
        soc_init=0.5,  # 初始 SOC（50%）
        soc_min=0.1,  # 最低 SOC
        soc_max=0.9,  # 最高 SOC
        eta_charge=0.95,  # 充电效率
        eta_discharge=0.95,  # 放电效率
        dt=0.25,  # 时间步长，小时（15min=0.25h）
):
    """
    用 cvxpy 求解储能最优充放电策略

    返回：
    - charge: 充放电功率序列，>0 充电，<0 放电
    - soc: SOC 序列
    - revenue: 总套利收益（元）
    """
    pv_forecast = np.asarray(pv_forecast, dtype=float)
    price = np.asarray(price, dtype=float)
    if pv_forecast.ndim != 1 or price.ndim != 1 or len(pv_forecast) != len(price):
        raise ValueError("pv_forecast 和 price 必须是长度相同的一维数组")
    if len(price) == 0 or np.any(pv_forecast < 0):
        raise ValueError("pv_forecast 必须非负且不能为空")

    T = len(pv_forecast)

    # ============ 变量 ============
    charge_pv = cp.Variable(T, nonneg=True)
    charge_grid = cp.Variable(T, nonneg=True)
    discharge = cp.Variable(T, nonneg=True)
    pv_export = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T)
    charge_in = charge_pv + charge_grid

    # ============ 约束 ============
    constraints = [
        soc >= soc_min * capacity,
        soc <= soc_max * capacity,
        charge_in + discharge <= power_max,
        charge_pv + pv_export <= pv_forecast,
    ]

    # SOC[t] = SOC[t-1] + 充电量*效率*dt - 放电量/效率*dt
    for t in range(T):
        previous_soc = soc_init * capacity if t == 0 else soc[t - 1]
        constraints.append(
            soc[t] == previous_soc
            + charge_in[t] * eta_charge * dt
            - discharge[t] / eta_discharge * dt
        )

    # 不把初始电量直接计入收益。
    constraints.append(soc[-1] == soc_init * capacity)

    # 收益 = 光伏上网收益 + 放电收益 - 电网充电成本。
    total_revenue = cp.sum(
        cp.multiply(pv_export * dt, price)
        + cp.multiply(discharge * dt, price)
        - cp.multiply(charge_grid * dt, price)
    )

    # ============ 求解 ============
    prob = cp.Problem(cp.Maximize(total_revenue), constraints)
    try:
        prob.solve(verbose=False)
    except cp.error.SolverError:
        return None, None, 0

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        print(f"⚠️ 求解状态: {prob.status}")
        return None, None, 0

    # 转成 numpy
    charge_vals = charge_in.value - discharge.value
    soc_vals = soc.value
    revenue_val = float(total_revenue.value)

    return charge_vals, soc_vals, revenue_val


def compare_with_baseline(pv_forecast, price, capacity=100, power_max=50,
                          soc_init=0.5, soc_min=0.1, soc_max=0.9,
                          eta_charge=0.95, eta_discharge=0.95, dt=0.25):
    """
    对比：无预测盲调度 vs 有预测最优调度
    """
    pv_forecast = np.asarray(pv_forecast, dtype=float)
    price = np.asarray(price, dtype=float)
    if pv_forecast.ndim != 1 or price.ndim != 1 or len(pv_forecast) != len(price):
        raise ValueError("pv_forecast 和 price 必须是长度相同的一维数组")
    if len(price) == 0 or np.any(pv_forecast < 0):
        raise ValueError("pv_forecast 必须非负且不能为空")

    T = len(pv_forecast)

    # ===== 盲调度策略（简单规则）=====
    # 谷时充电、峰时放电，不看光伏预测
    charge_baseline = np.zeros(T)
    soc_baseline = np.zeros(T)
    soc = soc_init * capacity
    soc_min_kwh = soc_min * capacity
    soc_max_kwh = soc_max * capacity
    target_soc = soc_init * capacity
    revenue_baseline = 0.0

    for t in range(T):
        pv_to_battery = 0.0
        grid_charge = 0.0
        discharge = 0.0

        if t == T - 1 and abs(soc - target_soc) > 1e-9:
            if soc > target_soc:
                discharge = min(power_max, (soc - target_soc) * eta_discharge / dt)
            else:
                desired_charge = min(
                    power_max, (target_soc - soc) / (eta_charge * dt)
                )
                pv_to_battery = min(pv_forecast[t], desired_charge)
                grid_charge = desired_charge - pv_to_battery
        elif price[t] <= 0.35:  # 谷时：全力充电
            desired_charge = min(
                power_max, (soc_max_kwh - soc) / (eta_charge * dt)
            )
            pv_to_battery = min(pv_forecast[t], desired_charge)
            grid_charge = desired_charge - pv_to_battery
        elif price[t] >= 0.85:  # 峰时：全力放电
            discharge = min(
                power_max, max(0.0, (soc - soc_min_kwh) * eta_discharge / dt)
            )

        # 更新 SOC
        pv_export = max(0.0, pv_forecast[t] - pv_to_battery)
        charge_in = pv_to_battery + grid_charge
        soc += charge_in * eta_charge * dt - discharge / eta_discharge * dt
        soc = np.clip(soc, soc_min_kwh, soc_max_kwh)
        charge_baseline[t] = charge_in - discharge
        soc_baseline[t] = soc
        revenue_baseline += (
            pv_export * dt * price[t]
            + discharge * dt * price[t]
            - grid_charge * dt * price[t]
        )

    return charge_baseline, soc_baseline, revenue_baseline


def run_dispatch_demo():
    """
    演示：生成一天数据 → 跑优化 → 对比 → 画图
    """
    print("=" * 55)
    print("储能峰谷套利调度 - cvxpy 优化器")
    print("=" * 55)

    # ============ 1. 造一天的数据 ============
    T = 96  # 一天96个15min

    # 电价：江苏峰谷（谷0.3, 平0.6, 峰1.0）
    price = np.ones(T) * 0.6  # 默认平段
    for t in range(T):
        hour = t * 0.25  # 当前小时
        if 0 <= hour < 8:
            price[t] = 0.30  # 谷时
        elif 8 <= hour < 11:
            price[t] = 0.60  # 平段
        elif 11 <= hour < 14:
            price[t] = 1.00  # 峰时
        elif 14 <= hour < 17:
            price[t] = 0.60  # 平段
        elif 17 <= hour < 22:
            price[t] = 1.00  # 峰时
        else:
            price[t] = 0.30  # 谷时

    # 光伏预测（模拟：早上6点到18点有出力）
    pv_forecast = np.zeros(T)
    for t in range(T):
        hour = t * 0.25
        if 6 <= hour <= 18:
            # 正弦曲线模拟辐照
            irradiance = np.sin((hour - 6) / 12 * np.pi)
            pv_forecast[t] = irradiance * 45 + np.random.normal(0, 1.5)
    pv_forecast = np.clip(pv_forecast, 0, 50)

    print(f"\n[数据] 一天 {T} 个时段 (15min/步)")
    print(f"  电价范围: {price.min():.2f} ~ {price.max():.2f} 元/kWh")
    print(f"  光伏预测: 峰值 {pv_forecast.max():.1f} kW")

    # ============ 2. 跑 cvxpy 最优调度 ============
    print(f"\n[优化] 求解储能最优充放电策略...")
    charge_opt, soc_opt, revenue_opt = storage_dispatch(
        pv_forecast=pv_forecast,
        price=price,
        capacity=100,
        power_max=50,
    )

    if charge_opt is None:
        print("求解失败！")
        return

    print(f"  最优策略总收益: {revenue_opt:.2f} 元")

    # ============ 3. 跑盲调度对比 ============
    print(f"\n[对比] 盲调度（不看预测，只按峰谷规则）...")
    charge_bl, soc_bl, revenue_bl = compare_with_baseline(
        pv_forecast, price
    )
    print(f"  盲调度总收益: {revenue_bl:.2f} 元")
    print(f"  收益提升: {((revenue_opt - revenue_bl) / abs(revenue_bl) * 100):.1f}%")

    # ============ 4. 画图 ============
    time_axis = np.arange(T) * 0.25  # 小时

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 子图1：电价
    axes[0].step(time_axis, price, where='mid', color='orange', linewidth=2)
    axes[0].axhline(y=0.30, color='green', linestyle='--', alpha=0.5, label='谷时')
    axes[0].axhline(y=1.00, color='red', linestyle='--', alpha=0.5, label='峰时')
    axes[0].set_ylabel("电价 (元/kWh)", fontsize=11)
    axes[0].set_title("电价曲线（江苏峰谷）", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 子图2：光伏预测
    axes[1].fill_between(time_axis, pv_forecast, alpha=0.4, color='gold', label='光伏预测')
    axes[1].set_ylabel("功率 (kW)", fontsize=11)
    axes[1].set_title("光伏超短期预测出力", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # 子图3：充放电功率
    axes[2].fill_between(time_axis, charge_opt, where=(charge_opt > 0),
                         color='blue', alpha=0.5, label='充电')
    axes[2].fill_between(time_axis, charge_opt, where=(charge_opt < 0),
                         color='red', alpha=0.5, label='放电')
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].set_ylabel("功率 (kW)", fontsize=11)
    axes[2].set_title("最优充放电策略", fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # 子图4：SOC
    axes[3].plot(time_axis, soc_opt, color='purple', linewidth=2, label='最优SOC')
    axes[3].plot(time_axis, soc_bl, color='gray', linewidth=1.5, linestyle='--', label='盲调度SOC')
    axes[3].axhline(y=90, color='red', linestyle=':', alpha=0.5, label='SOC上限')
    axes[3].axhline(y=10, color='green', linestyle=':', alpha=0.5, label='SOC下限')
    axes[3].set_xlabel("时间 (小时)", fontsize=11)
    axes[3].set_ylabel("SOC (kWh)", fontsize=11)
    axes[3].set_title("电池 SOC 变化", fontsize=12)
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "dispatch_result.png"
    plt.savefig(result_path, dpi=150)
    print(f"\n调度结果图已保存: {result_path}")

    # ============ 5. 打印时段明细 ============
    print(f"\n{'=' * 55}")
    print(f"{'时段':<8} {'电价':<8} {'光伏':<8} {'充放电':<10} {'SOC':<8}")
    print(f"{'=' * 55}")
    for t in range(0, T, 8):  # 每2小时打印一行
        hour = t * 0.25
        action = "充电" if charge_opt[t] > 1 else ("放电" if charge_opt[t] < -1 else "待机")
        print(f"{hour:>5.1f}h  {price[t]:>6.2f}  {pv_forecast[t]:>6.1f}  "
              f"{charge_opt[t]:>+8.1f}  {soc_opt[t]:>6.1f}  {action}")

    print(f"\n{'=' * 55}")
    print(f"总结:")
    print(f"  最优调度收益: {revenue_opt:.2f} 元/天")
    print(f"  盲调度收益:   {revenue_bl:.2f} 元/天")
    print(f"  提升幅度:     {((revenue_opt - revenue_bl) / abs(revenue_bl) * 100):.1f}%")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    run_dispatch_demo()
