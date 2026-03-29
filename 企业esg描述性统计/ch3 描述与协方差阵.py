"""
描述性统计与相关系数矩阵
样本：2009-2023 沪深A股上市公司
"""

import pandas as pd
import numpy as np
import pyreadstat
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = r"c:\Users\w1847\Desktop\3月毕业论文设计\毕业论文数据合集"
OUT  = r"c:\Users\w1847\Desktop\毕业论文_自动生成稿_20260310"

# ── 1. 华证 ESG ────────────────────────────────────────────────
print("读取华证ESG...")
esg_raw = pd.read_excel(
    BASE + r"\1.华证【2009-2025】（含细分项+季度)）\华证esg评级2009-2023（细分项）\华证esg评级含细分项（年度）2009-2023.xlsx"
)
print("华证ESG 列名:", list(esg_raw.columns[:10]))

# 列名映射
esg = esg_raw.rename(columns={"股票代码": "id", "年份": "year", "综合得分": "ESG_score",
                               "综合评级": "rating",
                               "是否沪深上市": "is_a", "是否金融": "is_fin",
                               "是否stpt": "is_stpt"})

# 筛选：沪深A股、非金融、非ST/PT、样本期
esg = esg[(esg["is_a"] == 1) & (esg["is_fin"] == 0) & (esg["is_stpt"] == 0)]
esg = esg[(esg["year"] >= 2009) & (esg["year"] <= 2023)]

# 评级 → 1-9 数值
rating_map = {"C":1,"CC":2,"CCC":3,"B":4,"BB":5,"BBB":6,"A":7,"AA":8,"AAA":9}
esg["ESG"] = esg["rating"].map(rating_map)

esg = esg[["id", "year", "ESG", "ESG_score"]].copy()
print(f"华证ESG 筛选后: {len(esg)} 行")

# ── 2. AI 扩展词汇 ─────────────────────────────────────────────
print("读取AI扩展词汇...")
ai_ext, _ = pyreadstat.read_dta(
    BASE + r"\上市公司人工智能MDA-词频总和（2001-2024年）\上市公司人工智能MDA-词频总和（扩展词汇）.dta"
)
print("AI扩展 列名:", list(ai_ext.columns[:8]))

ai = ai_ext.rename(columns={"code": "id", "年份": "year", "人工智能词频和": "ai_freq_ext"})
ai = ai[(ai["year"] >= 2009) & (ai["year"] <= 2023)][["id", "year", "ai_freq_ext"]]
ai["ai_freq_ext"] = pd.to_numeric(ai["ai_freq_ext"], errors="coerce").fillna(0)
ai["AI"] = np.log(1 + ai["ai_freq_ext"].astype(float))
print(f"AI扩展词汇: {len(ai)} 行")

# ── 3. AI 精确词汇 ─────────────────────────────────────────────
print("读取AI精确词汇...")
ai_str_raw, _ = pyreadstat.read_dta(
    BASE + r"\上市公司人工智能MDA-词频总和（2001-2024年）\上市公司人工智能MDA-词频总和（精确词汇）.dta"
)
ai_str = ai_str_raw.rename(columns={"code": "id", "年份": "year", "人工智能词频和": "ai_freq_str"})
ai_str = ai_str[(ai_str["year"] >= 2009) & (ai_str["year"] <= 2023)][["id", "year", "ai_freq_str"]]

# ── 4. 控制变量（已剔除已缩尾）─────────────────────────────────
print("读取控制变量...")
ctrl_raw, meta = pyreadstat.read_dta(
    BASE + r"\上市公司常用控制变量大全2.0 [2006-2024]\常用控制变量-已剔除已缩尾.dta"
)

# 找 SA 指数绝对值的列名（可能含中文）
sa_col = [c for c in ctrl_raw.columns if "SA" in c and "abs" in c.lower()]
if not sa_col:
    sa_col = [c for c in ctrl_raw.columns if "SA" in c]
print("SA列候选:", sa_col)

ctrl = ctrl_raw[["id", "year", "SOE", "Size", "Lev", "ROA", "Growth",
                  "Top1", "Dual", "Board", "ListAge", "是否资不抵债"] + sa_col].copy()
ctrl = ctrl[(ctrl["year"] >= 2009) & (ctrl["year"] <= 2023)]
# 重命名SA列
ctrl = ctrl.rename(columns={sa_col[0]: "SA_abs"})

# ── 5. 合并 ──────────────────────────────────────────────────
print("合并数据...")
df = ctrl.merge(esg,     on=["id","year"], how="inner")
df = df.merge(ai,        on=["id","year"], how="inner")
df = df.merge(ai_str,    on=["id","year"], how="left")

# 剔除资不抵债（双重保险）
df = df[df["是否资不抵债"] != 1]

# 删除核心变量缺失
key_vars = ["ESG","AI","SA_abs","Size","Lev","ROA","Growth",
            "Top1","Dual","Board","SOE","ListAge"]
df = df.dropna(subset=key_vars)

print(f"\n最终样本量: {len(df)} 行")
print(f"公司数:     {df['id'].nunique()} 家")
print(f"年份范围:   {int(df['year'].min())} - {int(df['year'].max())}")
print("\n年份分布:")
print(df.groupby("year").size().to_string())

# ── 6. 描述性统计 ──────────────────────────────────────────────
print("\n\n" + "="*60)
print("描述性统计")
print("="*60)
desc = df[key_vars].describe().T[["count","mean","std","min","max"]]
desc.columns = ["N","均值","标准差","最小值","最大值"]
desc["N"] = desc["N"].astype(int)
print(desc.to_string(float_format=lambda x: f"{x:.4f}"))

# ── 7. 相关系数矩阵 ────────────────────────────────────────────
print("\n\n" + "="*60)
print("Pearson 相关系数矩阵（主要变量）")
print("="*60)

corr_vars  = ["ESG","AI","SA_abs","Size","Lev","ROA"]
corr_vars2 = ["ESG","AI","SA_abs","Growth","Top1","Dual","Board","SOE","ListAge"]

def corr_with_stars(data, variables):
    n = len(variables)
    mat  = pd.DataFrame(index=variables, columns=variables, dtype=object)
    for i, v1 in enumerate(variables):
        for j, v2 in enumerate(variables):
            if i == j:
                mat.loc[v1,v2] = "1"
            elif i < j:
                x = pd.to_numeric(data[v1], errors="coerce")
                y = pd.to_numeric(data[v2], errors="coerce")
                mask = x.notna() & y.notna()
                r, p = stats.pearsonr(x[mask].astype(float), y[mask].astype(float))
                stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
                mat.loc[v1,v2]  = f"{r:.3f}{stars}"
                mat.loc[v2,v1]  = f"{r:.3f}{stars}"
    return mat

corr1 = corr_with_stars(df, corr_vars)
print(corr1.to_string())

print("\n相关系数矩阵（扩展变量）")
corr2 = corr_with_stars(df, corr_vars2)
print(corr2.to_string())

# ── 8. AI原始词频分布 ─────────────────────────────────────────
print("\n\n=== AI 原始词频（扩展口径，用于说明取对数的依据）===")
print(df["ai_freq_ext"].describe())
print(f"  词频=0 的观测数: {(df['ai_freq_ext']==0).sum()}  ({(df['ai_freq_ext']==0).mean()*100:.1f}%)")
print(f"  词频>0 均值:  {df.loc[df['ai_freq_ext']>0,'ai_freq_ext'].mean():.2f}")
print(f"  词频>0 中位数: {df.loc[df['ai_freq_ext']>0,'ai_freq_ext'].median():.1f}")
print(f"  词频>0 最大值: {df['ai_freq_ext'].max():.0f}")

print("\n=== ESG 综合得分（连续值）===")
print(df["ESG_score"].describe())

# ── 9. 保存完整描述统计到 CSV ──────────────────────────────────
desc.to_csv(OUT + r"\desc_stats.csv", encoding="utf-8-sig")
print(f"\n描述统计已保存至: {OUT}\\desc_stats.csv")

# 保存相关矩阵
corr1.to_csv(OUT + r"\corr_matrix.csv",  encoding="utf-8-sig")
corr2.to_csv(OUT + r"\corr_matrix2.csv", encoding="utf-8-sig")
print("相关矩阵已保存")
