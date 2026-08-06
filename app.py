import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.stats import chi2_contingency,ttest_ind
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import platform

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title='AB测试报告', layout='wide')
st.title('🚀 自动化AB测试报告生成器')
uploaded_file = st.file_uploader('上传你的CSV数据文件', type='csv')

if uploaded_file is not None:
    df = pl.read_csv(io.BytesIO(uploaded_file.read()))
    st.success(f'✅ 加载成功！共 {df.height} 行')

    with st.expander('查看原始数据'):
        st.dataframe(df.head(10))
    with st.expander('🎯 请指定数据列（告诉程序哪一列是什么）', expanded=True):
        cols = df.columns
        group_col = st.selectbox(
            "请选择【分组列】（例如：A组/B组）",
            options=cols,
            index=None,
            placeholder='请选择一列...'
        )
        value_col = st.selectbox(
            "请选择【数值列】（例如：时长、金额）",
            options=cols,
            index=None,
            placeholder='请选择一列...'
        )
        conv_col = st.selectbox(
            "请选择【转化列】）",
            options=cols,
            index=None,
            placeholder='请选择一列'
        )
        st.info(f"✅ 当前配置：分组={group_col}，数值={value_col}，转化={conv_col}")

    if group_col is not None and value_col is not None and conv_col is not None:
        summary = df.group_by(group_col).agg([
            pl.len().alias('样本量'),
            (pl.col(conv_col).mean() * 100).alias('转化率%'),
            pl.col(value_col).mean().alias('平均值')
        ])

        unique_groups = df[group_col].unique().to_list()
        if len(unique_groups) != 2:
            st.error(f"❌ 分组列 '{group_col}' 包含 {len(unique_groups)} 个不同值，需要恰好两组。")
            st.stop()
        group_a_name = unique_groups[0]
        group_b_name = unique_groups[1]

        val_a = df.filter(pl.col(group_col) == group_a_name)[value_col].to_numpy()
        val_b = df.filter(pl.col(group_col) == group_b_name)[value_col].to_numpy()
        t_stat, p_value = ttest_ind(val_a, val_b, equal_var=False)

        # =====================================================
        # 计算效应量：Cohen's d（科恩 d 值）
        # 作用：衡量两组均值差异的"大小"，而非仅仅"有无"
        # 业务解读：d=0.2（小效应）、d=0.5（中效应）、d≥0.8（大效应）
        # =====================================================

        # 1. 计算两组数据的平均值（集中趋势）
        mean_a = np.mean(val_a)  # A组（对照组）的平均值，例如平均收入 50.2 元
        mean_b = np.mean(val_b)  # B组（测试组）的平均值，例如平均收入 85.7 元

        # 2. 获取两组样本量（每组有多少个数据点）
        n1, n2 = len(val_a), len(val_b)  # A组人数、B组人数

        # 3. 计算两组数据的方差（离散程度）
        #    ddof=1 表示使用"样本方差"（分母为 n-1），这是统计学中的标准无偏估计
        var_a = np.var(val_a, ddof=1)  # A组内部数据的波动程度
        var_b = np.var(val_b, ddof=1)  # B组内部数据的波动程度

        # 4. 计算"合并标准差"（pooled standard deviation）
        #    公式：sqrt(((n1-1)*var_a + (n2-1)*var_b) / (n1+n2-2))
        #    含义：将两组数据的波动程度"揉在一起"，计算出一个统一的离散度
        #    通俗理解：平均来看，每个数据点偏离整体平均值多少
        pooled_std = np.sqrt(((n1 - 1) * var_a + (n2 - 1) * var_b) / (n1 + n2 - 2))

        # 5. 计算最终的 Cohen's d
        #    公式：Cohen's d = (均值差) / (合并标准差)
        #    含义：两组均值的差距，相当于"几个标准差"？
        #    如果 d=0.8，表示两组均值差了 0.8 个标准差 → 大效应（肉眼可见的区别）
        #    如果 d=0.2，表示两组均值差了 0.2 个标准差 → 小效应（差异微弱，实际意义不大）
        #    如果 pooled_std == 0，说明数据完全没有波动，此时 d 无意义，直接赋值为 0
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std != 0 else 0
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 分组指标")
            st.dataframe(summary.to_pandas())

        # -------------------------------------------
        # ===== 新增：转化率对比（卡方检验 + 效应量） =====
        st.subheader("📊 转化率深度拆解")

        # 1. 构建交叉表（分组 vs 是否转化）
        # 注意：conv_col 是用户选的转化列（0和1）
        cross_tab = df.group_by(group_col).agg([
            pl.len().alias("总人数"),
            (pl.col(conv_col).sum()).alias("转化人数"),
        ])

        # 计算转化率
        cross_tab = cross_tab.with_columns(
            ((pl.col("转化人数") / pl.col("总人数")) * 100).alias("转化率(%)")
        )

        # 提取两个组的转化人数和未转化人数
        converted = cross_tab["转化人数"].to_list()
        not_converted = (cross_tab["总人数"] - cross_tab["转化人数"]).to_list()
        contingency_table = [converted, not_converted]  # 2x2 表格

        chi2, p_value_conv, dof, expected = chi2_contingency(contingency_table)

        # 3. 计算提升幅度（相对提升）
        rate_a = cross_tab.filter(pl.col(group_col) == group_a_name)["转化率(%)"].to_list()[0]
        rate_b = cross_tab.filter(pl.col(group_col) == group_b_name)["转化率(%)"].to_list()[0]

        # ===== 优化后的提升幅度计算（以对照组为基准） =====
        # 智能识别对照组（支持中英文列名）
        if "对照" in group_a_name or "Control" in group_a_name:
            rate_control = rate_a
            rate_treatment = rate_b
        elif "对照" in group_b_name or "Control" in group_b_name:
            rate_control = rate_b
            rate_treatment = rate_a
        else:
            # 如果列名不含"对照"，默认取转化率较低的作为对照组（保守策略）
            if rate_a < rate_b:
                rate_control, rate_treatment = rate_a, rate_b
            else:
                rate_control, rate_treatment = rate_b, rate_a

        # 计算相对提升（始终为正，表示测试组比对照组提升的百分比）
        lift = ((rate_treatment - rate_control) / rate_control) * 100 if rate_control > 0 else float('inf')
        st.dataframe(cross_tab.to_pandas())
        col_conv1, col_conv2 = st.columns(2)
        with col_conv1:
            st.metric("卡方检验 P值", f"{p_value_conv:.4f}")
            if p_value_conv < 0.05:
                st.success("✅ 转化率差异显著")
            else:
                st.info("ℹ️ 转化率差异不显著")
        with col_conv2:
            st.metric("相对提升幅度", f"{lift:.1f}%", delta=f"{rate_treatment - rate_control:.1f} 个百分点")
            st.caption(f"{group_a_name} 转化率: {rate_a:.1f}% → {group_b_name} 转化率: {rate_b:.1f}%")
        # --------------------------------------------

        with col2:
            st.subheader("📈 检验结论")
            st.metric('P值', f'{p_value:.4f}%')
            if p_value < 0.05:
                st.success(f"✅ 结论：{group_a_name}组 与 {group_b_name}组 差异 **显著**")
            else:
                st.info(f"ℹ️ 结论：{group_a_name}组 与 {group_b_name}组 差异 **不显著**")

        if st.button("刷新页面"):
            st.rerun()

        st.subheader("📉 Distribution Comparison between Groups")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(val_a, bins=20, alpha=0.5, label=f'{group_a_name} (Group A)', color = 'r')
        ax.hist(val_b, bins=20, alpha=0.5, label=f'{group_b_name} (Group B)', color = 'b')
        ax.set_xlabel('Value', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title('Distribution Comparison', fontsize=18)
        ax.legend()
        st.pyplot(fig)

        # =====================================================
        # 🔬 新增：实验设计诊断（统计功效与最小样本量）
        # =====================================================
        with st.expander("🔬 实验设计诊断（统计功效与最小样本量）"):
            st.markdown("""
            **统计功效 (Power)**：若真实差异存在，实验能发现它的概率。一般要求 **> 0.8**。  
            **最小样本量**：在给定预期提升下，为了达到目标功效，每组至少需要的样本数。
            """)

            # ---------- 1. 当前实验的“观测功效”（基于现有数据） ----------
            col_p1, col_p2 = st.columns(2)

            with col_p1:
                st.subheader("📊 均值 T 检验 功效")
                if 'cohens_d' in locals() and len(val_a) > 1 and len(val_b) > 1:
                    # 计算均值检验的当前功效
                    power_analysis = TTestIndPower()
                    # 注意：这里传入的是 Cohen's d 的绝对值，以及两组样本量
                    power_mean = power_analysis.power(
                        effect_size=abs(cohens_d),
                        nobs1=len(val_a),
                        alpha=0.05,
                        ratio=len(val_b) / len(val_a)
                    )
                    st.metric("观测功效", f"{power_mean:.3f}")
                    if power_mean < 0.8:
                        st.warning("⚠️ 功效不足 (< 0.8)，当前样本量可能不足以可靠地检测到该差异。")
                    else:
                        st.success("✅ 功效充足 (>= 0.8)，实验设计较为可靠。")
                else:
                    st.info("暂无均值数据用于功效计算。")

            with col_p2:
                st.subheader("📈 比例差异 (卡方) 功效")
                if 'rate_a' in locals() and 'rate_b' in locals() and len(val_a) > 1:
                    # 计算比例检验（转化率）的当前功效
                    # 先计算 Cohen's h (比例效应量)
                    effect_size_prop = proportion_effectsize(rate_a / 100, rate_b / 100)
                    # 总样本量近似，假设两组样本量相等（取平均）
                    avg_n = (len(val_a) + len(val_b)) / 2
                    power_prop = NormalIndPower().power(
                        effect_size=effect_size_prop,
                        nobs1=avg_n,
                        alpha=0.05,
                        ratio=1
                    )
                    st.metric("观测功效", f"{power_prop:.3f}")
                    if power_prop < 0.8:
                        st.warning("⚠️ 功效不足 (< 0.8)，当前转化率差异可能不够稳健。")
                    else:
                        st.success("✅ 功效充足 (>= 0.8)，转化率结论较为可靠。")
                else:
                    st.info("暂无比例数据用于功效计算。")

        # ---------- 2. 前瞻性规划：计算“最小样本量” ----------
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            # 用户输入：预期的基准转化率和提升幅度
            # 默认值取当前数据的对照组转化率
            default_base = float(rate_a) if 'rate_a' in locals() else 20.0
            target_lift = st.number_input(
                "期望的最小相对提升幅度 (%)",
                min_value=1.0, value=10.0, step=1.0, max_value=100.0,
            ) / 100.0

            base_rate = st.number_input(
                "基准组转化率 (%)",
                min_value=0.1, value=round(default_base, 1), step=0.1, max_value=100.0,
            ) / 100.0

        with col_s2:
            power_target = st.slider(
                "目标统计功效 (Power)",
                min_value=0.5, max_value=0.99, value=0.8, step=0.05
            )
            alpha_target = st.selectbox(
                "显著性水平 (Alpha)",
                options=[0.01, 0.05, 0.10], index=1
            )

        if st.button("🚀 计算所需最小样本量"):
            if base_rate > 0 and target_lift > 0:
                effect_size = proportion_effectsize(
                    base_rate,
                    base_rate * (1 + target_lift),
                )
                if effect_size == 0:
                    st.warning("提升幅度为 0，无法计算样本量。")
                else:
                    sample_size_per_group = NormalIndPower().solve_power(
                        effect_size=effect_size,
                        power=power_target,
                        alpha=alpha_target,
                        ratio=1
                    )
                    required_n = int(np.ceil(sample_size_per_group))
                    total_n = required_n * 2
                    st.success(f"""
                        ✅ 为了有 **{power_target*100:.0f}%** 的把握检测到 **{target_lift*100:.0f}%** 的提升（基准转化率 {base_rate*100:.1f}%）：  
                        - **每组至少需要 {required_n} 个样本**  
                        - **总共需要约 {total_n} 个样本**
                    """)
                    if 'val_a' in locals() and 'val_b' in locals():
                        current_n = min(len(val_a), len(val_b))
                        if current_n < required_n:
                            st.warning(f"📌 当前每组仅有 {current_n} 个样本，建议补充至 {required_n} 个以上。")
                        else:
                            st.info(f"📌 当前每组有 {current_n} 个样本，已满足最低要求。")
            else:
                st.error("请确保基准转化率和提升幅度均大于 0。")
    else:
        #判断是否选择列分析的if
        st.info('请选择列')
else:
    st.info('👈 请现在左侧上传 CSV 文件')
