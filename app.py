import streamlit as st
import polars as pl
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import io
from scipy.stats import chi2_contingency
import platform

# 支持本地（Windows）和云端（Linux）的中文显示
# ===== 字体设置（兼容本地和云端） =====
plt.rcParams['font.sans-serif'] = ['SimHei']
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
            (pl.col(conv_col).mean()*100).alias('转化率%'),
            pl.col(value_col).mean().alias('平均值')
        ])

        unique_groups = df[group_col].unique()

        group_a_name = unique_groups[0]
        group_b_name = unique_groups[1]

        val_a = df.filter(pl.col(group_col) == group_a_name)[value_col].to_numpy()
        val_b = df.filter(pl.col(group_col) == group_b_name)[value_col].to_numpy()

        t_stat, p_value = ttest_ind(val_a, val_b, equal_var=False)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 分组指标")
            st.dataframe(summary.to_pandas())

        #-------------------------------------------
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
        #--------------------------------------------

        with col2:
            st.subheader("📈 检验结论")
            st.metric('P值', f'{p_value:.4f}%')
            if p_value <0.05:
                st.success(f"✅ 结论：{group_a_name}组 与 {group_b_name}组 差异 **显著**")
            else:
                st.info(f"ℹ️ 结论：{group_a_name}组 与 {group_b_name}组 差异 **不显著**")

        st.subheader("📉 两组时长分布对比")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(val_a, bins=20, alpha=0.5, label=f'{group_a_name}组')
        ax.hist(val_b, bins=20, alpha=0.5, label=f'{group_b_name}组')
        ax.set_xlabel('A/B组数量', fontsize=14)
        ax.set_ylabel(value_col, fontsize=14)
        ax.set_title(f"两组在「{value_col}」数值上的分布对比", fontsize=18)
        ax.legend()
        st.pyplot(fig)

    else:
        st.info('请选择列')
else:
    st.info('👈 请现在左侧上传 CSV 文件')
