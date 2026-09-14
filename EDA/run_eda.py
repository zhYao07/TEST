from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "EDA"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
sns.set_theme(style="whitegrid", font="Microsoft YaHei")

PURPOSE_I = "注汽井"
PURPOSE_P = "生产井"
TARGET_KEY = ["WELL_GROUP_NAME", "PROD_DATE", "INJ_INDICATOR", "PROD_INDICATOR"]
PROD_METRICS = ["LIQ_PROD_DAILY", "OIL_PROD_DAILY", "WATER_CUT", "WH_TEMP", "MAX_TUBING_PRES"]
INJ_METRICS = ["INJ_VOL_DAILY", "INJ_LIQ_DAILY", "WH_TEMP", "STEAM_INJ_PRES"]
COLORS = {"train": "#2F6B9A", "test": "#E07A5F", "short": "#2A9D8F", "long": "#E76F51"}


def read_csv(path: Path, dates=()):
    df = pd.read_csv(path)
    for c in dates:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
    return df


gi = read_csv(DATA / "train" / "well_group_info.csv", ["CONVERT_DATE", "PROD_DATE"])
gi_test = read_csv(DATA / "test" / "well_group_info.csv", ["CONVERT_DATE", "PROD_DATE"])
inj_tr = read_csv(DATA / "train" / "well_inj_data.csv", ["PROD_DATE"])
inj_te = read_csv(DATA / "test" / "well_inj_data.csv", ["PROD_DATE"])
prod_tr = read_csv(DATA / "train" / "well_prod_data.csv", ["PROD_DATE"])
prod_te = read_csv(DATA / "test" / "well_prod_data.csv", ["PROD_DATE"])
y = read_csv(DATA / "train" / "optimal_lag_days.csv", ["PROD_DATE"])
test_template = read_csv(DATA / "test" / "test_optimal_lag_days.csv", ["PROD_DATE"])
y["SHORT"] = y["OPTIMAL_LAG_DAYS"].le(3)


def savefig(name: str, fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def md_table(df: pd.DataFrame, index=False, floatfmt=".3f") -> str:
    show = df.copy()
    if not index:
        show = show.reset_index(drop=True)
    try:
        return show.to_markdown(index=index, floatfmt=floatfmt)
    except Exception:
        cols = ([show.index.name or "index"] if index else []) + list(show.columns)
        rows = []
        if index:
            for idx, row in show.iterrows():
                rows.append([idx] + row.tolist())
        else:
            rows = show.values.tolist()
        head = "| " + " | ".join(map(str, cols)) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(map(str, r)) + " |" for r in rows]
        return "\n".join([head, sep] + body)


def write_md(name: str, text: str):
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")


def pct(v):
    return f"{100 * float(v):.2f}%"


def contest_score(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.rint(np.asarray(y_pred, dtype=float)), 0, 45)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[valid], yp[valid]
    short = yt <= 3
    acc = np.mean(short == (yp <= 3))
    mae_s = np.mean(np.abs(yp[short] - yt[short])) if short.any() else np.nan
    mae_l = np.mean(np.abs(yp[~short] - yt[~short])) if (~short).any() else np.nan
    s_seg = 60 * acc
    s_short = 0.3 * max(0, 100 - 30 * mae_s) if np.isfinite(mae_s) else 0
    s_long = 0.1 * max(0, 100 - 2 * mae_l) if np.isfinite(mae_l) else 0
    return {
        "样本数": len(yt),
        "分段准确率": acc,
        "S_seg": s_seg,
        "MAE_short": mae_s,
        "S_short": s_short,
        "MAE_long": mae_l,
        "S_long": s_long,
        "总分": s_seg + s_short + s_long,
    }


def profile_numeric(df, cols, split):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "split": split,
            "字段": c,
            "行数": len(s),
            "缺失数": int(s.isna().sum()),
            "缺失率": s.isna().mean(),
            "零值数": int(s.eq(0).sum()),
            "零值率_全部行": s.eq(0).mean(),
            "零值率_非空": s.eq(0).sum() / max(1, s.notna().sum()),
            "均值": s.mean(),
            "中位数": s.median(),
            "标准差": s.std(),
            "最小值": s.min(),
            "最大值": s.max(),
        })
    return pd.DataFrame(rows)


def gap_profile(df, value_col=None):
    z = df.sort_values(["WELL_GROUP_NAME", "WELL_NAME", "PROD_DATE"]).copy()
    if value_col is not None:
        z = z[z[value_col].notna()].copy()
    z["gap"] = z.groupby(["WELL_GROUP_NAME", "WELL_NAME"])["PROD_DATE"].diff().dt.days
    return z["gap"].dropna().astype(int)


def successive_value_profile(df, col):
    z = df[df[col].notna()].sort_values(["WELL_GROUP_NAME", "WELL_NAME", "PROD_DATE"]).copy()
    g = z.groupby(["WELL_GROUP_NAME", "WELL_NAME"], sort=False)
    z["prev"] = g[col].shift()
    z["gap"] = g["PROD_DATE"].diff().dt.days
    z = z[z["prev"].notna()].copy()
    z["same"] = np.isclose(z[col], z["prev"], rtol=0, atol=1e-10)
    return z


def month_profile(df, cols, split):
    z = df.copy()
    z["月份"] = z["PROD_DATE"].dt.to_period("M").astype(str)
    rows = []
    for month, g in z.groupby("月份"):
        for c in cols:
            s = g[c]
            rows.append({"split": split, "月份": month, "字段": c, "缺失率": s.isna().mean(), "零值率": s.eq(0).mean(), "均值": s.mean(), "中位数": s.median()})
    return pd.DataFrame(rows)


def well_coverage_profile(df, cols, split):
    rows=[]
    for (group,well),g in df.groupby(["WELL_GROUP_NAME","WELL_NAME"]):
        first,last=g["PROD_DATE"].min(),g["PROD_DATE"].max()
        span=(last-first).days+1
        rec={"split":split,"WELL_GROUP_NAME":group,"WELL_NAME":well,"首日":first,"末日":last,"跨度天数":span,"有记录天数":g["PROD_DATE"].nunique()}
        for c in cols:
            valid=g.loc[g[c].notna(),"PROD_DATE"].nunique()
            rec[f"{c}_有效天数"]=valid
            rec[f"{c}_跨度覆盖率"]=valid/max(1,span)
        rows.append(rec)
    return pd.DataFrame(rows)


def group_day_missing(df, cols, split):
    gd=df.groupby(["WELL_GROUP_NAME","PROD_DATE"])[cols].mean()
    out=gd.isna().groupby(level=0).mean().reset_index()
    out.insert(0,"split",split)
    return out


# -----------------------------------------------------------------------------
# 1. Static topology and relationship audit
# -----------------------------------------------------------------------------
topology = gi.groupby(["WELL_GROUP_NAME", "WELL_PURPOSE"])["WELL_NAME"].nunique().unstack(fill_value=0)
for c in [PURPOSE_I, PURPOSE_P]:
    if c not in topology:
        topology[c] = 0
topology = topology.rename(columns={PURPOSE_I: "注汽井数", PURPOSE_P: "生产井数"})[["注汽井数", "生产井数"]]
topology["拓扑"] = topology["注汽井数"].astype(str) + "I-" + topology["生产井数"].astype(str) + "P"
topology_counts = topology["拓扑"].value_counts().rename_axis("拓扑").reset_index(name="井组数")
topology.reset_index().to_csv(TAB / "well_group_topology.csv", index=False, encoding="utf-8-sig")
topology_counts.to_csv(TAB / "topology_counts.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, col, color in zip(axes, ["注汽井数", "生产井数"], ["#2F6B9A", "#E07A5F"]):
    counts = topology[col].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=color)
    ax.set_title(f"每组{col}分布")
    ax.set_xlabel(col)
    ax.set_ylabel("井组数")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.8, str(v), ha="center", fontsize=9)
savefig("01_topology_hist.png", fig)

fig, ax = plt.subplots(figsize=(10, 5))
tc = topology_counts.head(15).sort_values("井组数")
ax.barh(tc["拓扑"], tc["井组数"], color="#3D5A80")
ax.set_title("井组拓扑组合频数（前15类）")
ax.set_xlabel("井组数")
for i, v in enumerate(tc["井组数"]):
    ax.text(v + 0.3, i, str(v), va="center")
savefig("02_topology_combinations.png", fig)

membership = gi.groupby("WELL_NAME")["WELL_GROUP_NAME"].nunique()
fig, ax = plt.subplots(figsize=(7.5, 4.5))
mc = membership.value_counts().sort_index()
ax.bar(mc.index.astype(str), mc.values, color="#457B9D")
ax.set_title("单井关联井组数量")
ax.set_xlabel("同一物理井关联的井组数")
ax.set_ylabel("井数")
for i, v in enumerate(mc.values):
    ax.text(i, v + 2, str(v), ha="center")
savefig("03_well_membership.png", fig)

static_group = gi.groupby("WELL_GROUP_NAME").agg(
    最早转驱日期=("CONVERT_DATE", "min"),
    最晚转驱日期=("CONVERT_DATE", "max"),
    转驱日期种类=("CONVERT_DATE", "nunique"),
    组织=("ORG_NAME", "first"),
    区块=("BLOCK", "first"),
).join(topology[["注汽井数", "生产井数"]])

label_age = y.merge(gi[["WELL_GROUP_NAME", "WELL_NAME", "CONVERT_DATE"]], on="WELL_GROUP_NAME", how="left")
label_age["转驱年龄年"] = (label_age["PROD_DATE"] - label_age["CONVERT_DATE"]).dt.days / 365.25
age_group = label_age.groupby(TARGET_KEY + ["OPTIMAL_LAG_DAYS", "SHORT"])["转驱年龄年"].median().reset_index()
age_group["年龄分位"] = pd.qcut(age_group["转驱年龄年"], 5, duplicates="drop")
age_stats = age_group.groupby("年龄分位", observed=True).agg(样本数=("SHORT", "size"), 短滞后率=("SHORT", "mean"), 中位lag=("OPTIMAL_LAG_DAYS", "median"), 中位转驱年龄=("转驱年龄年", "median")).reset_index()
fig, ax1 = plt.subplots(figsize=(9, 4.8))
xpos = np.arange(len(age_stats))
ax1.bar(xpos, age_stats["短滞后率"] * 100, color="#2A9D8F", alpha=.8)
ax1.set_ylabel("短滞后率（%）")
ax1.set_xlabel("井组内井的中位转驱年龄分位")
ax1.set_xticks(xpos, [f"Q{i+1}\n{v:.1f}年" for i, v in enumerate(age_stats["中位转驱年龄"])])
ax2 = ax1.twinx()
ax2.plot(xpos, age_stats["中位lag"], color="#E76F51", marker="o", linewidth=2)
ax2.set_ylabel("lag 中位数（天）")
ax1.set_title("静态转驱年龄与标签分布（描述性）")
savefig("04_static_age_label.png", fig)

file_audit = []
for split, objs in [("train", [("well_group_info.csv", gi), ("well_inj_data.csv", inj_tr), ("well_prod_data.csv", prod_tr), ("optimal_lag_days.csv", y.drop(columns=["SHORT"]))]),
                    ("test", [("well_group_info.csv", gi_test), ("well_inj_data.csv", inj_te), ("well_prod_data.csv", prod_te), ("test_optimal_lag_days.csv", test_template)])]:
    for fname, df in objs:
        rec = {"数据集": split, "文件": fname, "行数": len(df), "列数": len(df.columns), "完全重复行": int(df.duplicated().sum())}
        for c in ["WELL_GROUP_NAME", "WELL_NAME", "PROD_DATE"]:
            rec[f"唯一{c}"] = int(df[c].nunique()) if c in df else np.nan
        file_audit.append(rec)
file_audit = pd.DataFrame(file_audit)
file_audit.to_csv(TAB / "file_audit.csv", index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# 2. Injection cadence, missingness, and zero semantics
# -----------------------------------------------------------------------------
inj_num = pd.concat([profile_numeric(inj_tr, INJ_METRICS, "train"), profile_numeric(inj_te, INJ_METRICS, "test")], ignore_index=True)
inj_num.to_csv(TAB / "injection_missing_zero_stats.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, value, title in [(axes[0], "缺失率", "缺失率"), (axes[1], "零值率_非空", "非空记录中的零值率")]:
    pv = inj_num.pivot(index="字段", columns="split", values=value).reindex(INJ_METRICS)
    pv.plot(kind="bar", ax=ax, color=[COLORS["test"], COLORS["train"]] if list(pv.columns)==["test","train"] else [COLORS["train"],COLORS["test"]])
    ax.set_title(f"注汽字段{title}")
    ax.set_ylabel("比例")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="")
savefig("05_injection_missing_zero.png", fig)

inj_gap_tables = []
for split, df in [("train", inj_tr), ("test", inj_te)]:
    for metric in [None] + INJ_METRICS:
        gaps = gap_profile(df, metric)
        name = "任意记录" if metric is None else metric
        total = max(1, len(gaps))
        inj_gap_tables.append({
            "split": split, "序列": name, "间隔数": len(gaps), "中位间隔": gaps.median(), "P90间隔": gaps.quantile(.9),
            "1天占比": (gaps == 1).sum()/total, "2天占比": (gaps == 2).sum()/total,
            "7天占比": (gaps == 7).sum()/total, "30天占比": (gaps == 30).sum()/total, ">1天占比": (gaps > 1).sum()/total,
        })
inj_gap_stats = pd.DataFrame(inj_gap_tables)
inj_gap_stats.to_csv(TAB / "injection_gap_stats.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
for ax, (split, df) in zip(axes, [("train", inj_tr), ("test", inj_te)]):
    gaps = gap_profile(df, "INJ_VOL_DAILY")
    vc = gaps.value_counts().sort_index()
    vc = vc[vc.index <= 45]
    ax.bar(vc.index, vc.values / vc.values.sum() * 100, color=COLORS[split])
    ax.set_title(f"{split}：日注汽量相邻有效记录间隔")
    ax.set_xlabel("间隔天数")
    ax.set_ylabel("占 1～45 天间隔的比例（%）")
    ax.set_xlim(.3, 45.7)
savefig("06_injection_gap_distribution.png", fig)

repeat_rows = []
for split, df in [("train", inj_tr), ("test", inj_te)]:
    z = successive_value_profile(df, "INJ_VOL_DAILY")
    z["间隔组"] = pd.cut(z["gap"], bins=[0,1,2,7,30,np.inf], labels=["1天","2天","3-7天","8-30天",">30天"])
    tmp = z.groupby("间隔组", observed=True).agg(相邻对数=("same","size"), 数值完全相同比例=("same","mean"), 中位绝对变化=("INJ_VOL_DAILY", lambda s: np.nan)).reset_index()
    # Calculate absolute changes directly to avoid an ambiguous aggregation.
    z["abs_change"] = (z["INJ_VOL_DAILY"] - z["prev"]).abs()
    tmp2 = z.groupby("间隔组", observed=True)["abs_change"].median().rename("中位绝对变化").reset_index()
    tmp = tmp.drop(columns="中位绝对变化").merge(tmp2, on="间隔组")
    tmp["split"] = split
    repeat_rows.append(tmp)
inj_repeat = pd.concat(repeat_rows, ignore_index=True)
inj_repeat.to_csv(TAB / "injection_successive_value_stats.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(9.5, 4.8))
sns.barplot(data=inj_repeat, x="间隔组", y="数值完全相同比例", hue="split", palette=COLORS, ax=ax)
ax.set_title("相邻有效日注汽量记录是否重复")
ax.set_ylabel("数值完全相同的比例")
ax.set_xlabel("两条有效记录的时间间隔")
savefig("07_injection_repeat_by_gap.png", fig)

inj_month = pd.concat([month_profile(inj_tr, INJ_METRICS, "train"), month_profile(inj_te, INJ_METRICS, "test")], ignore_index=True)
inj_month.to_csv(TAB / "injection_monthly_profile.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
for ax, metric in zip(axes.flat, INJ_METRICS):
    g = inj_month[inj_month["字段"] == metric]
    ax.plot(g["月份"], g["缺失率"] * 100, marker="o", color="#3D5A80")
    ax.axvline("2026-03", color="#E76F51", linestyle="--", linewidth=1.5)
    ax.set_title(metric)
    ax.set_ylabel("缺失率（%）")
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("注汽字段月度缺失率（虚线为测试期开始）", y=1.02, fontsize=14)
savefig("08_injection_monthly_missing.png", fig)

inj_well_coverage=pd.concat([well_coverage_profile(inj_tr,INJ_METRICS,"train"),well_coverage_profile(inj_te,INJ_METRICS,"test")],ignore_index=True)
inj_well_coverage.to_csv(TAB/"injection_well_effective_days.csv",index=False,encoding="utf-8-sig")
inj_wc_long=[]
for c in INJ_METRICS:
    z=inj_well_coverage[["split","WELL_GROUP_NAME","WELL_NAME",f"{c}_有效天数",f"{c}_跨度覆盖率"]].copy()
    z.columns=["split","WELL_GROUP_NAME","WELL_NAME","有效天数","跨度覆盖率"]; z["字段"]=c; inj_wc_long.append(z)
inj_wc_long=pd.concat(inj_wc_long,ignore_index=True)
fig,axes=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(data=inj_wc_long,x="字段",y="有效天数",hue="split",palette=COLORS,showfliers=False,ax=axes[0]); axes[0].tick_params(axis="x",rotation=25); axes[0].set_title("注汽井组—井：各字段有效天数"); axes[0].set_xlabel("")
sns.boxplot(data=inj_wc_long,x="字段",y="跨度覆盖率",hue="split",palette=COLORS,showfliers=False,ax=axes[1]); axes[1].tick_params(axis="x",rotation=25); axes[1].set_title("注汽井组—井：有效天数/自身日期跨度"); axes[1].set_xlabel("")
savefig("30_injection_well_effective_days.png",fig)

inj_group_missing=pd.concat([group_day_missing(inj_tr,INJ_METRICS,"train"),group_day_missing(inj_te,INJ_METRICS,"test")],ignore_index=True)
inj_group_missing.to_csv(TAB/"injection_group_day_missing.csv",index=False,encoding="utf-8-sig")
fig,axes=plt.subplots(1,2,figsize=(11,13))
for ax,split in zip(axes,["train","test"]):
    z=inj_group_missing[inj_group_missing["split"].eq(split)].set_index("WELL_GROUP_NAME")[INJ_METRICS].sort_values("INJ_VOL_DAILY")
    sns.heatmap(z,cmap="YlOrRd",vmin=0,vmax=1,cbar=split=="test",yticklabels=False,ax=ax)
    ax.set_title(f"{split} 井组日缺失率"); ax.set_xlabel(""); ax.set_ylabel("井组")
savefig("31_injection_group_missing_heatmap.png",fig)


# -----------------------------------------------------------------------------
# 3. Production cadence, missingness, and consistency
# -----------------------------------------------------------------------------
prod_num = pd.concat([profile_numeric(prod_tr, PROD_METRICS, "train"), profile_numeric(prod_te, PROD_METRICS, "test")], ignore_index=True)
prod_num.to_csv(TAB / "production_missing_zero_stats.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, value, title in [(axes[0], "缺失率", "缺失率"), (axes[1], "零值率_非空", "非空记录中的零值率")]:
    pv = prod_num.pivot(index="字段", columns="split", values=value).reindex(PROD_METRICS)
    pv.plot(kind="bar", ax=ax, color=[COLORS.get(c, "#777777") for c in pv.columns])
    ax.set_title(f"采出字段{title}")
    ax.set_ylabel("比例")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="")
savefig("09_production_missing_zero.png", fig)

prod_gap_tables = []
for split, df in [("train", prod_tr), ("test", prod_te)]:
    for metric in [None] + PROD_METRICS:
        gaps = gap_profile(df, metric)
        total = max(1, len(gaps))
        prod_gap_tables.append({
            "split": split, "序列": "任意记录" if metric is None else metric, "间隔数": len(gaps), "中位间隔": gaps.median(), "P90间隔": gaps.quantile(.9),
            "1天占比": (gaps == 1).sum()/total, "2天占比": (gaps == 2).sum()/total,
            "7天占比": (gaps == 7).sum()/total, "30天占比": (gaps == 30).sum()/total, ">1天占比": (gaps > 1).sum()/total,
        })
prod_gap_stats = pd.DataFrame(prod_gap_tables)
prod_gap_stats.to_csv(TAB / "production_gap_stats.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, (split, df) in zip(axes, [("train", prod_tr), ("test", prod_te)]):
    gaps = gap_profile(df, "LIQ_PROD_DAILY")
    vc = gaps.value_counts().sort_index()
    vc = vc[vc.index <= 45]
    ax.bar(vc.index, vc.values / vc.values.sum() * 100, color=COLORS[split])
    ax.set_title(f"{split}：日产液相邻有效记录间隔")
    ax.set_xlabel("间隔天数")
    ax.set_ylabel("占 1～45 天间隔的比例（%）")
    ax.set_xlim(.3, 45.7)
savefig("10_production_gap_distribution.png", fig)

prod_month = pd.concat([month_profile(prod_tr, PROD_METRICS, "train"), month_profile(prod_te, PROD_METRICS, "test")], ignore_index=True)
prod_month.to_csv(TAB / "production_monthly_profile.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
for ax, metric in zip(axes.flat, PROD_METRICS):
    g = prod_month[prod_month["字段"] == metric]
    ax.plot(g["月份"], g["缺失率"] * 100, marker="o", color="#3D5A80")
    ax.axvline("2026-03", color="#E76F51", linestyle="--", linewidth=1.5)
    ax.set_title(metric)
    ax.set_ylabel("缺失率（%）")
    ax.tick_params(axis="x", rotation=45)
axes.flat[-1].axis("off")
fig.suptitle("采出字段月度缺失率（虚线为测试期开始）", y=1.02, fontsize=14)
savefig("11_production_monthly_missing.png", fig)

wc = pd.concat([prod_tr.assign(split="train"), prod_te.assign(split="test")], ignore_index=True)
wc = wc[(wc["LIQ_PROD_DAILY"] > 0) & wc["OIL_PROD_DAILY"].notna() & wc["WATER_CUT"].notna()].copy()
wc["DERIVED_WATER_CUT"] = 100 * (wc["LIQ_PROD_DAILY"] - wc["OIL_PROD_DAILY"]) / wc["LIQ_PROD_DAILY"]
wc["WC_ERROR"] = wc["WATER_CUT"] - wc["DERIVED_WATER_CUT"]
wc_consistency = wc.groupby("split").agg(样本数=("WC_ERROR","size"), 相关系数=("WATER_CUT", lambda s: np.nan), MAE=("WC_ERROR", lambda s: s.abs().mean()), 中位绝对误差=("WC_ERROR", lambda s: s.abs().median()), P95绝对误差=("WC_ERROR", lambda s: s.abs().quantile(.95))).reset_index()
for split in wc_consistency["split"]:
    mask = wc["split"].eq(split)
    wc_consistency.loc[wc_consistency["split"].eq(split), "相关系数"] = wc.loc[mask, ["WATER_CUT","DERIVED_WATER_CUT"]].corr().iloc[0,1]
wc_consistency.to_csv(TAB / "watercut_consistency.csv", index=False, encoding="utf-8-sig")
sample_wc = wc.sample(min(8000, len(wc)), random_state=42)
fig, ax = plt.subplots(figsize=(6.5, 6))
sns.scatterplot(data=sample_wc, x="DERIVED_WATER_CUT", y="WATER_CUT", hue="split", palette=COLORS, alpha=.22, s=12, ax=ax)
ax.plot([0,100],[0,100], color="black", linestyle="--", linewidth=1)
ax.set_xlim(0,105); ax.set_ylim(0,105)
ax.set_xlabel("由液量和油量推导的含水率（%）")
ax.set_ylabel("原始 WATER_CUT（%）")
ax.set_title("含水率口径一致性检查")
savefig("12_watercut_consistency.png", fig)

prod_well_coverage=pd.concat([well_coverage_profile(prod_tr,PROD_METRICS,"train"),well_coverage_profile(prod_te,PROD_METRICS,"test")],ignore_index=True)
prod_well_coverage.to_csv(TAB/"production_well_effective_days.csv",index=False,encoding="utf-8-sig")
prod_wc_long=[]
for c in PROD_METRICS:
    z=prod_well_coverage[["split","WELL_GROUP_NAME","WELL_NAME",f"{c}_有效天数",f"{c}_跨度覆盖率"]].copy()
    z.columns=["split","WELL_GROUP_NAME","WELL_NAME","有效天数","跨度覆盖率"]; z["字段"]=c; prod_wc_long.append(z)
prod_wc_long=pd.concat(prod_wc_long,ignore_index=True)
fig,axes=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(data=prod_wc_long,x="字段",y="有效天数",hue="split",palette=COLORS,showfliers=False,ax=axes[0]); axes[0].tick_params(axis="x",rotation=25); axes[0].set_title("采出井组—井：各字段有效天数"); axes[0].set_xlabel("")
sns.boxplot(data=prod_wc_long,x="字段",y="跨度覆盖率",hue="split",palette=COLORS,showfliers=False,ax=axes[1]); axes[1].tick_params(axis="x",rotation=25); axes[1].set_title("采出井组—井：有效天数/自身日期跨度"); axes[1].set_xlabel("")
savefig("32_production_well_effective_days.png",fig)

prod_group_missing=pd.concat([group_day_missing(prod_tr,PROD_METRICS,"train"),group_day_missing(prod_te,PROD_METRICS,"test")],ignore_index=True)
prod_group_missing.to_csv(TAB/"production_group_day_missing.csv",index=False,encoding="utf-8-sig")
fig,axes=plt.subplots(1,2,figsize=(11,13))
for ax,split in zip(axes,["train","test"]):
    z=prod_group_missing[prod_group_missing["split"].eq(split)].set_index("WELL_GROUP_NAME")[PROD_METRICS].sort_values("LIQ_PROD_DAILY")
    sns.heatmap(z,cmap="YlOrRd",vmin=0,vmax=1,cbar=split=="test",yticklabels=False,ax=ax)
    ax.set_title(f"{split} 井组日缺失率"); ax.set_xlabel(""); ax.set_ylabel("井组")
savefig("33_production_group_missing_heatmap.png",fig)


# -----------------------------------------------------------------------------
# 4. Label, pair, group and time behavior
# -----------------------------------------------------------------------------
freq = y["OPTIMAL_LAG_DAYS"].value_counts().reindex(range(46), fill_value=0).rename_axis("lag").reset_index(name="样本数")
freq["占比"] = freq["样本数"] / len(y)
freq.to_csv(TAB / "label_frequency_0_45.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(15, 5.3))
colors = [COLORS["short"] if k <= 3 else COLORS["long"] for k in freq["lag"]]
ax.bar(freq["lag"], freq["样本数"], color=colors)
ax.axvline(3.5, color="black", linestyle="--", linewidth=1.5, label="评分断点 3/4")
for k in [0,1,2,3,4,44,45]:
    v = int(freq.loc[freq["lag"].eq(k), "样本数"].iloc[0])
    ax.text(k, v + max(freq["样本数"])*.015, f"{v:,}", ha="center", va="bottom", fontsize=8, rotation=45)
ax.set_xticks(range(46))
ax.set_xlabel("OPTIMAL_LAG_DAYS")
ax.set_ylabel("样本数")
ax.set_title("训练标签 0～45 天完整频数")
ax.legend()
savefig("13_label_frequency.png", fig)

pair_stats = y.groupby(["INJ_INDICATOR", "PROD_INDICATOR"]).agg(
    样本数=("OPTIMAL_LAG_DAYS", "size"),
    short比例=("SHORT", "mean"),
    平均lag=("OPTIMAL_LAG_DAYS", "mean"),
    中位lag=("OPTIMAL_LAG_DAYS", "median"),
    lag众数=("OPTIMAL_LAG_DAYS", lambda s: int(s.mode().iloc[0])),
    lag标准差=("OPTIMAL_LAG_DAYS", "std"),
).reset_index()
pair_stats.to_csv(TAB / "indicator_pair_stats.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(3, 1, figsize=(12, 7.5))
for ax, col, fmt, cmap, title in [
    (axes[0], "样本数", ".0f", "Blues", "指标对样本数"),
    (axes[1], "short比例", ".1%", "YlGn", "指标对短滞后比例"),
    (axes[2], "中位lag", ".0f", "YlOrRd", "指标对 lag 中位数"),
]:
    pv = pair_stats.pivot(index="INJ_INDICATOR", columns="PROD_INDICATOR", values=col).reindex(columns=PROD_METRICS)
    sns.heatmap(pv, annot=True, fmt=fmt, cmap=cmap, cbar=False, ax=ax)
    ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("")
savefig("14_indicator_pair_heatmaps.png", fig)

group_stats = y.groupby("WELL_GROUP_NAME").agg(
    样本数=("OPTIMAL_LAG_DAYS","size"), 平均lag=("OPTIMAL_LAG_DAYS","mean"), 中位lag=("OPTIMAL_LAG_DAYS","median"),
    short比例=("SHORT","mean"), label标准差=("OPTIMAL_LAG_DAYS","std"), 最早标签日=("PROD_DATE","min"), 最晚标签日=("PROD_DATE","max")
).reset_index()
gp_stats = y.groupby(["WELL_GROUP_NAME","INJ_INDICATOR","PROD_INDICATOR"]).agg(
    样本数=("OPTIMAL_LAG_DAYS","size"), 平均lag=("OPTIMAL_LAG_DAYS","mean"), 中位lag=("OPTIMAL_LAG_DAYS","median"),
    short比例=("SHORT","mean"), lag标准差=("OPTIMAL_LAG_DAYS","std"), 唯一lag数=("OPTIMAL_LAG_DAYS","nunique")
).reset_index()
group_stats.to_csv(TAB / "group_label_stats.csv", index=False, encoding="utf-8-sig")
gp_stats.to_csv(TAB / "group_indicator_time_stats.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(group_stats["中位lag"], group_stats["short比例"]*100, s=20+group_stats["样本数"]*.08, c=group_stats["label标准差"], cmap="viridis", alpha=.8)
ax.set_xlabel("井组 lag 中位数")
ax.set_ylabel("井组短滞后比例（%）")
ax.set_title("井组条件标签分布（点大小=样本数，颜色=组内标准差）")
plt.colorbar(sc, ax=ax, label="lag 标准差")
savefig("15_group_label_profile.png", fig)

def eta_squared(df, group_cols):
    overall = df["OPTIMAL_LAG_DAYS"].mean()
    total = ((df["OPTIMAL_LAG_DAYS"] - overall)**2).sum()
    st = df.groupby(group_cols)["OPTIMAL_LAG_DAYS"].agg(["size","mean"])
    between = (st["size"] * (st["mean"] - overall)**2).sum()
    return between / total

y["月份"] = y["PROD_DATE"].dt.to_period("M").astype(str)
effect_stats = pd.DataFrame([
    {"分组因素":"井组", "解释方差比例":eta_squared(y,["WELL_GROUP_NAME"])},
    {"分组因素":"采出指标", "解释方差比例":eta_squared(y,["PROD_INDICATOR"])},
    {"分组因素":"月份", "解释方差比例":eta_squared(y,["月份"])},
    {"分组因素":"井组×采出指标", "解释方差比例":eta_squared(y,["WELL_GROUP_NAME","PROD_INDICATOR"])},
]).sort_values("解释方差比例", ascending=False)
effect_stats.to_csv(TAB / "label_effect_eta_squared.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(8, 4.7))
ax.barh(effect_stats["分组因素"], effect_stats["解释方差比例"]*100, color="#577590")
ax.invert_yaxis(); ax.set_xlabel("解释的总方差比例（%）"); ax.set_title("标签的描述性组间效应（η²）")
for i,v in enumerate(effect_stats["解释方差比例"]*100): ax.text(v+.3,i,f"{v:.1f}%",va="center")
savefig("16_label_group_effect.png", fig)

# Time plots: G081 and three stable/three volatile group-pairs with sufficient observations.
fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
g081 = y[y["WELL_GROUP_NAME"].eq("G081")]
for ax, metric in zip(axes, PROD_METRICS):
    z = g081[g081["PROD_INDICATOR"].eq(metric)].sort_values("PROD_DATE")
    ax.plot(z["PROD_DATE"], z["OPTIMAL_LAG_DAYS"], marker="o", markersize=2.5, linewidth=1.2)
    ax.axhline(3.5, color="#E76F51", linestyle="--", linewidth=.8)
    ax.set_ylabel(metric, rotation=0, ha="right", va="center")
    ax.set_ylim(-1,46)
axes[0].set_title("G081：五个采出指标的 lag 随时间变化")
axes[-1].set_xlabel("日期")
savefig("17_g081_label_timeline.png", fig)

eligible_gp = gp_stats[gp_stats["样本数"] >= 30].sort_values("lag标准差")
examples = pd.concat([eligible_gp.head(3), eligible_gp.tail(3)]).drop_duplicates().reset_index(drop=True)
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharey=True)
for ax, (_, r) in zip(axes.flat, examples.iterrows()):
    z = y[(y["WELL_GROUP_NAME"].eq(r["WELL_GROUP_NAME"])) & (y["PROD_INDICATOR"].eq(r["PROD_INDICATOR"]))].sort_values("PROD_DATE")
    ax.plot(z["PROD_DATE"], z["OPTIMAL_LAG_DAYS"], marker="o", markersize=2.5, linewidth=1)
    ax.axhline(3.5, color="#E76F51", linestyle="--", linewidth=.8)
    ax.set_title(f"{r['WELL_GROUP_NAME']} / {r['PROD_INDICATOR']}\nstd={r['lag标准差']:.2f}, n={int(r['样本数'])}")
    ax.set_ylim(-1,46); ax.tick_params(axis="x", rotation=25)
fig.suptitle("稳定与高波动 group × indicator 示例", y=1.01, fontsize=14)
savefig("18_label_timeline_examples.png", fig)

# Consecutive label behavior.
ys = y.sort_values(["WELL_GROUP_NAME","INJ_INDICATOR","PROD_INDICATOR","PROD_DATE"]).copy()
yg = ys.groupby(["WELL_GROUP_NAME","INJ_INDICATOR","PROD_INDICATOR"], sort=False)
ys["prev_lag"] = yg["OPTIMAL_LAG_DAYS"].shift()
ys["label_gap"] = yg["PROD_DATE"].diff().dt.days
trans = ys.dropna(subset=["prev_lag"]).copy()
trans["prev_short"] = trans["prev_lag"].le(3)
trans["abs_change"] = (trans["OPTIMAL_LAG_DAYS"] - trans["prev_lag"]).abs()
trans["gap_bucket"] = pd.cut(trans["label_gap"], [0,1,7,30,np.inf], labels=["1天","2-7天","8-30天",">30天"])
transition_matrix = pd.crosstab(trans["prev_short"].map({True:"前值短",False:"前值长"}), trans["SHORT"].map({True:"当前短",False:"当前长"}), normalize="index")
transition_matrix.to_csv(TAB / "short_long_transition_matrix.csv", encoding="utf-8-sig")
transition_by_gap = trans.groupby("gap_bucket", observed=True).agg(相邻对数=("abs_change","size"), lag完全相同率=("abs_change",lambda s:(s==0).mean()), 同分段率=("SHORT",lambda s: np.nan), 中位绝对变化=("abs_change","median"), 平均绝对变化=("abs_change","mean")).reset_index()
for bucket in transition_by_gap["gap_bucket"]:
    m = trans["gap_bucket"].eq(bucket)
    transition_by_gap.loc[transition_by_gap["gap_bucket"].eq(bucket), "同分段率"] = (trans.loc[m,"SHORT"].to_numpy()==trans.loc[m,"prev_short"].to_numpy()).mean()
transition_by_gap.to_csv(TAB / "label_transition_by_gap.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
sns.heatmap(transition_matrix, annot=True, fmt=".1%", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_title("短/长分段转移概率"); axes[0].set_xlabel(""); axes[0].set_ylabel("")
sns.barplot(data=transition_by_gap, x="gap_bucket", y="同分段率", color="#2A9D8F", ax=axes[1])
axes[1].set_ylim(0,1); axes[1].set_title("不同标签间隔下的同分段率"); axes[1].set_xlabel("标签间隔"); axes[1].set_ylabel("同分段率")
savefig("19_label_transition.png", fig)

# Label date frequency and completeness.
label_gd = y[["WELL_GROUP_NAME","PROD_DATE"]].drop_duplicates().sort_values(["WELL_GROUP_NAME","PROD_DATE"])
label_gd["gap"] = label_gd.groupby("WELL_GROUP_NAME")["PROD_DATE"].diff().dt.days
label_gap = label_gd["gap"].dropna().astype(int)
combo_counts = y.groupby(["WELL_GROUP_NAME","PROD_DATE"]).size()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
vc = label_gap.value_counts().sort_index(); vc = vc[vc.index<=60]
axes[0].bar(vc.index, vc.values, color="#457B9D"); axes[0].set_xlim(.5,60.5); axes[0].set_xlabel("相邻标签日期间隔（天）"); axes[0].set_ylabel("次数"); axes[0].set_title("井组标签日期间隔")
cc = combo_counts.value_counts().sort_index()
axes[1].bar(cc.index.astype(str), cc.values, color="#E9C46A"); axes[1].set_xlabel("同一 group+date 的指标对数量"); axes[1].set_ylabel("井组日期数"); axes[1].set_title("标签组合完整性")
for i,v in enumerate(cc.values): axes[1].text(i,v+20,str(v),ha="center")
savefig("20_label_date_frequency.png", fig)

monthly_label = y.groupby("月份").agg(样本数=("OPTIMAL_LAG_DAYS","size"), 井组数=("WELL_GROUP_NAME","nunique"), short比例=("SHORT","mean"), 平均lag=("OPTIMAL_LAG_DAYS","mean"), 中位lag=("OPTIMAL_LAG_DAYS","median"), lag0比例=("OPTIMAL_LAG_DAYS",lambda s:(s==0).mean()), lag3比例=("OPTIMAL_LAG_DAYS",lambda s:(s==3).mean()), lag44_45比例=("OPTIMAL_LAG_DAYS",lambda s:s.isin([44,45]).mean())).reset_index()
monthly_label.to_csv(TAB / "monthly_label_profile.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
axes[0].plot(monthly_label["月份"], monthly_label["short比例"]*100, marker="o", label="short比例")
axes[0].plot(monthly_label["月份"], monthly_label["lag0比例"]*100, marker="o", label="lag=0")
axes[0].plot(monthly_label["月份"], monthly_label["lag3比例"]*100, marker="o", label="lag=3")
axes[0].plot(monthly_label["月份"], monthly_label["lag44_45比例"]*100, marker="o", label="lag=44/45")
axes[0].set_ylabel("比例（%）"); axes[0].set_title("标签分布的月度漂移"); axes[0].legend(ncol=4)
axes[1].plot(monthly_label["月份"], monthly_label["平均lag"], marker="o", label="均值")
axes[1].plot(monthly_label["月份"], monthly_label["中位lag"], marker="o", label="中位数")
axes[1].set_ylabel("lag（天）"); axes[1].set_xlabel("月份"); axes[1].legend(); axes[1].tick_params(axis="x", rotation=35)
savefig("21_monthly_label_drift.png", fig)


# -----------------------------------------------------------------------------
# 5. Score-aligned descriptive baselines
# -----------------------------------------------------------------------------
baseline_rows = []
for k in range(46):
    rec = contest_score(y["OPTIMAL_LAG_DAYS"], np.full(len(y), k)); rec.update({"实验":"全训练固定值", "预测规则":str(k)})
    baseline_rows.append(rec)
constant_scores = pd.DataFrame(baseline_rows)
best_constant = constant_scores.loc[constant_scores["总分"].idxmax()].copy()

cutoff = pd.Timestamp("2026-02-01")
hist = y[y["PROD_DATE"] < cutoff].copy()
val = y[y["PROD_DATE"] >= cutoff].copy()
global_med = hist["OPTIMAL_LAG_DAYS"].median()
pair_map = hist.groupby(["INJ_INDICATOR","PROD_INDICATOR"])["OPTIMAL_LAG_DAYS"].median()
group_map = hist.groupby("WELL_GROUP_NAME")["OPTIMAL_LAG_DAYS"].median()
gp_map = hist.groupby(["WELL_GROUP_NAME","INJ_INDICATOR","PROD_INDICATOR"])["OPTIMAL_LAG_DAYS"].median()
last_map = hist.sort_values("PROD_DATE").groupby(["WELL_GROUP_NAME","INJ_INDICATOR","PROD_INDICATOR"]).tail(1).set_index(["WELL_GROUP_NAME","INJ_INDICATOR","PROD_INDICATOR"])["OPTIMAL_LAG_DAYS"]

def map_pred(df, mapping, keys, fallback):
    idx = pd.MultiIndex.from_frame(df[keys]) if len(keys)>1 else df[keys[0]]
    p = mapping.reindex(idx).to_numpy(dtype=float)
    return np.where(np.isfinite(p), p, fallback)

temporal_preds = {
    "固定全局中位数": np.full(len(val), global_med),
    "指标对历史中位数": map_pred(val, pair_map, ["INJ_INDICATOR","PROD_INDICATOR"], global_med),
    "井组历史中位数": map_pred(val, group_map, ["WELL_GROUP_NAME"], global_med),
    "井组×指标历史中位数": map_pred(val, gp_map, ["WELL_GROUP_NAME","INJ_INDICATOR","PROD_INDICATOR"], global_med),
    "截止1月底最后标签": map_pred(val, last_map, ["WELL_GROUP_NAME","INJ_INDICATOR","PROD_INDICATOR"], global_med),
}
for name, pred in temporal_preds.items():
    rec = contest_score(val["OPTIMAL_LAG_DAYS"], pred); rec.update({"实验":"2026-02多步时间外推", "预测规则":name}); baseline_rows.append(rec)

# Sequential oracle-like persistence: later validation truths are deliberately used only to quantify local persistence.
seq = pd.concat([hist, val]).sort_values(["WELL_GROUP_NAME","INJ_INDICATOR","PROD_INDICATOR","PROD_DATE"])
seq["prev_true"] = seq.groupby(["WELL_GROUP_NAME","INJ_INDICATOR","PROD_INDICATOR"])["OPTIMAL_LAG_DAYS"].shift()
seqv = seq[seq["PROD_DATE"] >= cutoff].copy()
seqv["prev_true"] = seqv["prev_true"].fillna(global_med)
rec = contest_score(seqv["OPTIMAL_LAG_DAYS"], seqv["prev_true"]); rec.update({"实验":"2026-02诊断上限", "预测规则":"上一时刻真实标签（不可线上使用）"}); baseline_rows.append(rec)
baseline_scores = pd.DataFrame(baseline_rows)
baseline_scores.to_csv(TAB / "official_score_baselines.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(constant_scores["预测规则"].astype(int), constant_scores["总分"], color="#3D5A80")
axes[0].scatter([int(best_constant["预测规则"])],[best_constant["总分"]],color="#E76F51",zorder=3)
axes[0].axhline(60,color="black",linestyle="--",label="60分线")
axes[0].set_xlabel("固定预测值"); axes[0].set_ylabel("官方综合分"); axes[0].set_title("固定整数预测的训练集得分"); axes[0].legend()
show_base = baseline_scores[baseline_scores["实验"].str.contains("2026-02")].sort_values("总分")
axes[1].barh(show_base["预测规则"], show_base["总分"], color=["#E9C46A" if "不可" not in x else "#E76F51" for x in show_base["预测规则"]])
axes[1].axvline(60,color="black",linestyle="--"); axes[1].set_xlabel("官方综合分"); axes[1].set_title("2026-02：简单参照规则评分诊断")
savefig("22_score_baselines.png", fig)

short_mae_grid = np.linspace(0,4,81)
long_mae_grid = np.linspace(0,45,91)
fig, axes = plt.subplots(1, 2, figsize=(11,4.4))
axes[0].plot(short_mae_grid, .3*np.maximum(0,100-30*short_mae_grid), color=COLORS["short"])
axes[0].set_xlabel("MAE_short"); axes[0].set_ylabel("S_short"); axes[0].set_title("短段 MAE 的得分敏感性")
axes[1].plot(long_mae_grid, .1*np.maximum(0,100-2*long_mae_grid), color=COLORS["long"])
axes[1].set_xlabel("MAE_long"); axes[1].set_ylabel("S_long"); axes[1].set_title("长段 MAE 的得分敏感性")
savefig("23_score_sensitivity.png", fig)


# -----------------------------------------------------------------------------
# 6. Aggregation stability and causal cross-correlation probe
# -----------------------------------------------------------------------------
def aggregate_daily(df, cols, funcs):
    chunks = []
    grouped = df.groupby(["WELL_GROUP_NAME","PROD_DATE"])
    for c in cols:
        for f in funcs:
            if f == "sum": s = grouped[c].sum(min_count=1)
            elif f == "mean": s = grouped[c].mean()
            elif f == "max": s = grouped[c].max()
            elif f == "min": s = grouped[c].min()
            elif f == "std": s = grouped[c].std(ddof=0)
            chunks.append(s.rename(f"{c}__{f}"))
    return pd.concat(chunks, axis=1)

inj_aggs = aggregate_daily(inj_tr, ["INJ_VOL_DAILY","WH_TEMP","STEAM_INJ_PRES"], ["sum","mean","max","min","std"])
prod_aggs = aggregate_daily(prod_tr, PROD_METRICS, ["sum","mean","max","min","std"])

stability_rows=[]
for side, aggs in [("注汽",inj_aggs),("采出",prod_aggs)]:
    for col in aggs.columns:
        metric, agg = col.split("__")
        s=aggs[col]
        diffs=s.groupby(level=0).diff().abs()
        by_group=aggs[col].groupby(level=0).agg(["mean","std"])
        stability_rows.append({"侧别":side,"指标":metric,"聚合":agg,"井组日数":len(s),"缺失率":s.isna().mean(),"零值率_非空":s.eq(0).sum()/max(1,s.notna().sum()),"相邻日绝对变化中位数":diffs.median(),"平均组内CV":np.nanmedian((by_group["std"]/by_group["mean"].abs()).replace([np.inf,-np.inf],np.nan))})
aggregation_stability=pd.DataFrame(stability_rows)
aggregation_stability.to_csv(TAB / "aggregation_stability.csv",index=False,encoding="utf-8-sig")
fig, axes=plt.subplots(1,2,figsize=(14,5))
sub=aggregation_stability[aggregation_stability["指标"].isin(["INJ_VOL_DAILY","LIQ_PROD_DAILY","WATER_CUT","WH_TEMP","MAX_TUBING_PRES"])].copy()
sns.barplot(data=sub,x="指标",y="缺失率",hue="聚合",ax=axes[0]); axes[0].tick_params(axis="x",rotation=25); axes[0].set_title("不同聚合后的井组日缺失率"); axes[0].set_xlabel("")
sns.barplot(data=sub,x="指标",y="平均组内CV",hue="聚合",ax=axes[1]); axes[1].tick_params(axis="x",rotation=25); axes[1].set_title("不同聚合后的时间波动（组内CV）"); axes[1].set_xlabel(""); axes[1].set_ylim(0,sub["平均组内CV"].replace([np.inf],np.nan).quantile(.95)*1.15)
savefig("24_aggregation_stability.png",fig)

# Business-default group series.
inj_default = inj_tr.groupby(["WELL_GROUP_NAME","PROD_DATE"])["INJ_VOL_DAILY"].sum(min_count=1).rename("INJ_VOL_DAILY")
prod_grp = prod_tr.groupby(["WELL_GROUP_NAME","PROD_DATE"])
prod_default = pd.DataFrame(index=prod_grp.size().index)
for c in ["LIQ_PROD_DAILY","OIL_PROD_DAILY"]:
    prod_default[c] = prod_grp[c].sum(min_count=1)
prod_default["WH_TEMP"] = prod_grp["WH_TEMP"].mean()
prod_default["MAX_TUBING_PRES"] = prod_grp["MAX_TUBING_PRES"].max()
weighted_num = (prod_tr["WATER_CUT"] * prod_tr["LIQ_PROD_DAILY"]).groupby([prod_tr["WELL_GROUP_NAME"],prod_tr["PROD_DATE"]]).sum(min_count=1)
weighted_den = prod_tr["LIQ_PROD_DAILY"].where(prod_tr["WATER_CUT"].notna()).groupby([prod_tr["WELL_GROUP_NAME"],prod_tr["PROD_DATE"]]).sum(min_count=1)
weighted_wc = weighted_num/weighted_den.replace(0,np.nan)
prod_default["WATER_CUT"] = weighted_wc.reindex(prod_default.index).fillna(prod_grp["WATER_CUT"].mean())

DATE0=pd.Timestamp("2025-03-01"); DATE1=pd.Timestamp("2026-02-28"); DAILY_INDEX=pd.date_range(DATE0,DATE1,freq="D")

def series_for(group, source, col):
    try:
        return source.xs(group,level=0)[col].reindex(DAILY_INDEX).to_numpy(dtype=float)
    except (KeyError, TypeError):
        return np.full(len(DAILY_INDEX),np.nan)


def one_series_for(group, source):
    try:
        return source.xs(group,level=0).reindex(DAILY_INDEX).to_numpy(dtype=float)
    except (KeyError, TypeError):
        return np.full(len(DAILY_INDEX),np.nan)


def transform_arr(a, kind):
    a=np.asarray(a,dtype=float)
    if kind=="raw": return a
    if kind=="diff":
        out=np.full_like(a,np.nan); valid=np.isfinite(a[1:])&np.isfinite(a[:-1]); out[1:][valid]=a[1:][valid]-a[:-1][valid]; return out
    if kind=="rate":
        out=np.full_like(a,np.nan); valid=np.isfinite(a[1:])&np.isfinite(a[:-1]); den=np.abs(a[1:])+np.abs(a[:-1])+1e-8; out[1:][valid]=2*(a[1:][valid]-a[:-1][valid])/den[valid]; return out
    raise ValueError(kind)


def rolling_corr_matrices(x,y,windows,min_frac=.5):
    T=len(y); K=46
    X=np.full((T,K),np.nan)
    for k in range(K):
        X[k:,k]=x[:T-k]
    Y=np.broadcast_to(y[:,None],X.shape)
    mask=np.isfinite(X)&np.isfinite(Y)
    n=mask.astype(float); xv=np.where(mask,X,0); yv=np.where(mask,Y,0)
    arrays=[n,xv,yv,xv*xv,yv*yv,xv*yv]
    cums=[np.vstack([np.zeros((1,K)),np.cumsum(a,axis=0)]) for a in arrays]
    out={}
    for W in windows:
        rolls=[]
        for cs in cums:
            r=np.full((T,K),np.nan); r[W-1:]=cs[W:]-cs[:-W]; rolls.append(r)
        nn,sx,sy,sxx,syy,sxy=rolls
        num=sxy-sx*sy/nn
        den=np.sqrt(np.maximum(0,sxx-sx*sx/nn)*np.maximum(0,syy-sy*sy/nn))
        corr=num/den
        corr[(nn<max(5,math.ceil(W*min_frac)))|(den<=1e-12)]=np.nan
        out[W]=corr
    return out


def xcorr_probe(inj_source, prod_source, configs):
    pred_store={name:np.full(len(y),np.nan) for name,_,_,_,_ in configs}
    peak_store={name:np.full(len(y),np.nan) for name,_,_,_,_ in configs}
    grouped_cfg={}
    for name,W,transform,use_abs,ffill_x in configs:
        grouped_cfg.setdefault((transform,ffill_x),[]).append((name,W,use_abs))
    for group, gy in y.groupby("WELL_GROUP_NAME"):
        x0=one_series_for(group,inj_source)
        for metric, gm in gy.groupby("PROD_INDICATOR"):
            y0=series_for(group,prod_source,metric)
            pos=(gm["PROD_DATE"]-DATE0).dt.days.to_numpy()
            for (transform,ffill_x), items in grouped_cfg.items():
                x=x0.copy()
                if ffill_x:
                    x=pd.Series(x).ffill(limit=45).to_numpy()
                x=transform_arr(x,transform); yy=transform_arr(y0,transform)
                windows=sorted(set(v[1] for v in items))
                cms=rolling_corr_matrices(x,yy,windows)
                for name,W,use_abs in items:
                    vals=cms[W][pos]
                    target=np.abs(vals) if use_abs else vals
                    valid=np.any(np.isfinite(target),axis=1)
                    safe=np.where(np.isfinite(target),target,-np.inf)
                    p=np.argmax(safe,axis=1).astype(float); p[~valid]=np.nan
                    peak=np.max(safe,axis=1); peak[~valid]=np.nan
                    pred_store[name][gm.index.to_numpy()]=p
                    peak_store[name][gm.index.to_numpy()]=peak
    return pred_store,peak_store


configs=[]
for W in [15,30,60,90,180]:
    configs += [
        (f"raw_signed_W{W}",W,"raw",False,False),
        (f"raw_abs_W{W}",W,"raw",True,False),
        (f"diff_abs_W{W}",W,"diff",True,False),
        (f"rate_abs_W{W}",W,"rate",True,False),
        (f"raw_abs_injffill_W{W}",W,"raw",True,True),
    ]
pred_store,peak_store=xcorr_probe(inj_default,prod_default,configs)
pair_fallback=y.groupby("PROD_INDICATOR")["OPTIMAL_LAG_DAYS"].median().to_dict()
xcorr_rows=[]
for name,W,transform,use_abs,ffill_x in configs:
    p=pred_store[name]; valid=np.isfinite(p)
    sc=contest_score(y.loc[valid,"OPTIMAL_LAG_DAYS"],p[valid])
    rho=spearmanr(y.loc[valid,"OPTIMAL_LAG_DAYS"],p[valid],nan_policy="omit").statistic if valid.sum()>2 else np.nan
    fb=y["PROD_INDICATOR"].map(pair_fallback).to_numpy(dtype=float)
    complete=np.where(valid,p,fb)
    sc_full=contest_score(y["OPTIMAL_LAG_DAYS"],complete)
    xcorr_rows.append({"配置":name,"窗口":W,"序列":transform,"取绝对相关":use_abs,"注汽前向填充":ffill_x,"有效覆盖率":valid.mean(),"Spearman":rho,"有效样本MAE":np.mean(np.abs(p[valid]-y.loc[valid,"OPTIMAL_LAG_DAYS"].to_numpy())),"有效样本分段准确率":sc["分段准确率"],"有效样本官方分":sc["总分"],"缺失回退后官方分":sc_full["总分"]})
xcorr_results=pd.DataFrame(xcorr_rows).sort_values("缺失回退后官方分",ascending=False)
xcorr_results.to_csv(TAB / "cross_correlation_window_results.csv",index=False,encoding="utf-8-sig")

fig, axes=plt.subplots(1,2,figsize=(14,5))
for (transform,use_abs,ffill_x),g in xcorr_results.groupby(["序列","取绝对相关","注汽前向填充"]):
    label=f"{transform}-{'abs' if use_abs else 'signed'}"+("-inj_ffill" if ffill_x else "")
    g=g.sort_values("窗口")
    axes[0].plot(g["窗口"],g["有效样本MAE"],marker="o",label=label)
    axes[1].plot(g["窗口"],g["缺失回退后官方分"],marker="o",label=label)
axes[0].set_xlabel("历史窗口（天）"); axes[0].set_ylabel("argmax lag MAE"); axes[0].set_title("互相关候选 lag 与真实标签误差")
axes[1].set_xlabel("历史窗口（天）"); axes[1].set_ylabel("官方综合分"); axes[1].set_title("互相关候选 lag 的评分表现")
axes[1].legend(fontsize=8,ncol=2)
savefig("25_xcorr_window_comparison.png",fig)

best_xcorr=xcorr_results.iloc[0]
best_name=best_xcorr["配置"]
bp=pred_store[best_name]; valid=np.isfinite(bp)
sample_idx=np.random.default_rng(42).choice(np.where(valid)[0],size=min(9000,valid.sum()),replace=False)
fig,axes=plt.subplots(1,2,figsize=(12,5))
axes[0].hexbin(y.loc[sample_idx,"OPTIMAL_LAG_DAYS"],bp[sample_idx],gridsize=30,cmap="Blues",mincnt=1)
axes[0].plot([0,45],[0,45],"--",color="black",linewidth=1); axes[0].set_xlabel("真实 lag"); axes[0].set_ylabel("互相关 argmax lag"); axes[0].set_title(f"最佳配置：{best_name}")
err=np.abs(bp[valid]-y.loc[valid,"OPTIMAL_LAG_DAYS"].to_numpy())
axes[1].hist(err,bins=np.arange(-.5,46.5,1),color="#577590"); axes[1].set_xlabel("绝对误差（天）"); axes[1].set_ylabel("样本数"); axes[1].set_title("互相关候选 lag 绝对误差分布")
savefig("26_xcorr_best_diagnostics.png",fig)

# Compare aggregation choices against labels using the best window/transform family.
agg_cfg=("agg_probe",int(best_xcorr["窗口"]),str(best_xcorr["序列"]),bool(best_xcorr["取绝对相关"]),bool(best_xcorr["注汽前向填充"]))
agg_align_rows=[]
def summarize_agg_alignment(label,p):
    valid=np.isfinite(p); fb=y["PROD_INDICATOR"].map(pair_fallback).to_numpy(dtype=float); complete=np.where(valid,p,fb)
    sc=contest_score(y.loc[valid,"OPTIMAL_LAG_DAYS"],p[valid]); full=contest_score(y["OPTIMAL_LAG_DAYS"],complete)
    return {"方案":label,"有效覆盖率":valid.mean(),"有效样本MAE":np.mean(np.abs(p[valid]-y.loc[valid,"OPTIMAL_LAG_DAYS"].to_numpy())),"有效样本分段准确率":sc["分段准确率"],"有效样本官方分":sc["总分"],"缺失回退后官方分":full["总分"]}

agg_align_rows.append(summarize_agg_alignment("业务默认混合聚合",bp))
for agg in ["sum","mean","max","min"]:
    ps,_=xcorr_probe(inj_aggs[f"INJ_VOL_DAILY__{agg}"],prod_default,[agg_cfg])
    agg_align_rows.append(summarize_agg_alignment(f"仅改变注汽量为{agg}",ps["agg_probe"]))
for agg in ["sum","mean","max","min"]:
    psource=pd.DataFrame(index=prod_aggs.index)
    for metric in PROD_METRICS: psource[metric]=prod_aggs[f"{metric}__{agg}"]
    ps,_=xcorr_probe(inj_default,psource,[agg_cfg])
    agg_align_rows.append(summarize_agg_alignment(f"全部采出指标用{agg}",ps["agg_probe"]))
aggregation_alignment=pd.DataFrame(agg_align_rows).sort_values("缺失回退后官方分",ascending=False)
aggregation_alignment.to_csv(TAB/"aggregation_xcorr_label_alignment.csv",index=False,encoding="utf-8-sig")
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
z=aggregation_alignment.sort_values("有效样本MAE",ascending=False)
axes[0].barh(z["方案"],z["有效样本MAE"],color="#577590"); axes[0].set_xlabel("argmax lag MAE"); axes[0].set_title("聚合方式与标签 lag 的误差")
z2=aggregation_alignment.sort_values("缺失回退后官方分")
axes[1].barh(z2["方案"],z2["缺失回退后官方分"],color="#2A9D8F"); axes[1].set_xlabel("官方综合分"); axes[1].set_title("聚合方式的互相关标签对齐")
savefig("34_aggregation_label_alignment.png",fig)


# -----------------------------------------------------------------------------
# 7. Train/test drift, overlap, and target-date coverage
# -----------------------------------------------------------------------------
test_groups=set(test_template["WELL_GROUP_NAME"]); train_groups=set(y["WELL_GROUP_NAME"]); overlap=sorted(test_groups&train_groups)
overlap_summary=pd.DataFrame([
    {"类别":"训练与测试标签均出现","井组数":len(test_groups&train_groups)},
    {"类别":"仅测试标签出现","井组数":len(test_groups-train_groups)},
    {"类别":"仅训练标签出现","井组数":len(train_groups-test_groups)},
])
test_weight=test_template.groupby("WELL_GROUP_NAME").size().rename("测试行数").reset_index()
test_weight["测试权重"] = test_weight["测试行数"]/len(test_template)
test_weight["是否训练标签已见"] = test_weight["WELL_GROUP_NAME"].isin(train_groups)
test_weight.to_csv(TAB / "test_group_weights.csv",index=False,encoding="utf-8-sig")
fig,axes=plt.subplots(1,2,figsize=(13,5))
axes[0].bar(overlap_summary["类别"],overlap_summary["井组数"],color=["#2A9D8F","#E76F51","#577590"]); axes[0].tick_params(axis="x",rotation=18); axes[0].set_title("训练/测试标签井组关系"); axes[0].set_ylabel("井组数")
tw=test_weight.sort_values("测试行数",ascending=False)
axes[1].bar(np.arange(len(tw)),tw["测试行数"],color=np.where(tw["是否训练标签已见"],"#457B9D","#E76F51")); axes[1].set_xticks(np.arange(len(tw)),tw["WELL_GROUP_NAME"],rotation=90,fontsize=7); axes[1].set_ylabel("测试行数"); axes[1].set_title("测试集井组样本权重")
savefig("27_train_test_group_relation.png",fig)

# Fairer drift comparison: the 32 overlapping target groups only, using group-day default aggregations.
def default_group_daily(inj_df,prod_df):
    ig=inj_df.groupby(["WELL_GROUP_NAME","PROD_DATE"])["INJ_VOL_DAILY"].sum(min_count=1).rename("INJ_VOL_DAILY")
    pg=prod_df.groupby(["WELL_GROUP_NAME","PROD_DATE"])
    out=pd.DataFrame(index=pg.size().index)
    out["LIQ_PROD_DAILY"]=pg["LIQ_PROD_DAILY"].sum(min_count=1)
    out["OIL_PROD_DAILY"]=pg["OIL_PROD_DAILY"].sum(min_count=1)
    out["WATER_CUT"]=pg["WATER_CUT"].mean()
    out["WH_TEMP"]=pg["WH_TEMP"].mean()
    out["MAX_TUBING_PRES"]=pg["MAX_TUBING_PRES"].max()
    return out.join(ig,how="outer").reset_index()

gd_tr=default_group_daily(inj_tr,prod_tr); gd_te=default_group_daily(inj_te,prod_te)
gd_tr=gd_tr[gd_tr["WELL_GROUP_NAME"].isin(overlap)].copy(); gd_te=gd_te[gd_te["WELL_GROUP_NAME"].isin(overlap)].copy()
recent=gd_tr[gd_tr["PROD_DATE"]>=pd.Timestamp("2026-01-01")]
drift_rows=[]
for metric in ["INJ_VOL_DAILY"]+PROD_METRICS:
    a=recent[metric].dropna().to_numpy(); b=gd_te[metric].dropna().to_numpy()
    scale=np.std(a) if len(a) and np.std(a)>0 else 1
    drift_rows.append({"字段":metric,"近期训练非空数":len(a),"测试非空数":len(b),"近期训练均值":np.mean(a) if len(a) else np.nan,"测试均值":np.mean(b) if len(b) else np.nan,"均值变化率":(np.mean(b)-np.mean(a))/(abs(np.mean(a))+1e-9) if len(a) and len(b) else np.nan,"近期训练缺失率":recent[metric].isna().mean(),"测试缺失率":gd_te[metric].isna().mean(),"缺失率变化":gd_te[metric].isna().mean()-recent[metric].isna().mean(),"KS统计量":ks_2samp(a,b).statistic if len(a) and len(b) else np.nan,"标准化Wasserstein":wasserstein_distance(a,b)/scale if len(a) and len(b) else np.nan})
drift_stats=pd.DataFrame(drift_rows)
drift_stats.to_csv(TAB / "recent_train_vs_test_drift.csv",index=False,encoding="utf-8-sig")
fig,axes=plt.subplots(1,2,figsize=(14,5))
sns.barplot(data=drift_stats,x="字段",y="KS统计量",color="#457B9D",ax=axes[0]); axes[0].tick_params(axis="x",rotation=25); axes[0].set_title("近期训练 vs 测试：数值分布 KS 距离"); axes[0].set_xlabel("")
sns.barplot(data=drift_stats,x="字段",y="缺失率变化",color="#E76F51",ax=axes[1]); axes[1].axhline(0,color="black",linewidth=1); axes[1].tick_params(axis="x",rotation=25); axes[1].set_title("测试期缺失率相对近期训练的变化"); axes[1].set_xlabel("")
savefig("28_recent_train_test_drift.png",fig)

# KDE comparison of recent-train and test non-null values.  The display range is
# clipped by pooled quantiles so a few extreme observations do not compress the
# main density shape; missingness remains visible in the separate drift chart.
kde_metrics=["INJ_VOL_DAILY"]+PROD_METRICS
fig,axes=plt.subplots(3,2,figsize=(14,12))
for ax,metric in zip(axes.flat,kde_metrics):
    a=recent[metric].dropna().astype(float)
    b=gd_te[metric].dropna().astype(float)
    pooled=pd.concat([a,b],ignore_index=True)
    if len(pooled)>=2:
        lo,hi=pooled.quantile([0.005,0.995])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo==hi:
            lo,hi=pooled.min(),pooled.max()
        aa=a[(a>=lo)&(a<=hi)]
        bb=b[(b>=lo)&(b<=hi)]
        if aa.nunique()>1:
            sns.kdeplot(x=aa,ax=ax,color=COLORS["train"],linewidth=2,label="近期训练",cut=0)
        if bb.nunique()>1:
            sns.kdeplot(x=bb,ax=ax,color=COLORS["test"],linewidth=2,label="测试",cut=0)
        ax.set_xlim(lo,hi)
    ax.set_title(metric)
    ax.set_xlabel("井组日聚合值（展示 0.5%～99.5% 分位区间）")
    ax.set_ylabel("概率密度")
    ax.legend(frameon=False)
fig.suptitle("近期训练与测试动态指标 KDE 对比（32 个重叠井组）",y=1.01,fontsize=14)
savefig("35_recent_train_test_kde.png",fig)

monthly_raw=[]
for split,df in [("train",gd_tr),("test",gd_te)]:
    z=df.copy(); z["月份"]=z["PROD_DATE"].dt.to_period("M").astype(str)
    for month,g in z.groupby("月份"):
        for metric in ["INJ_VOL_DAILY"]+PROD_METRICS:
            monthly_raw.append({"split":split,"月份":month,"字段":metric,"均值":g[metric].mean(),"中位数":g[metric].median(),"缺失率":g[metric].isna().mean()})
monthly_raw=pd.DataFrame(monthly_raw)
monthly_raw.to_csv(TAB / "overlap_groups_monthly_raw_profile.csv",index=False,encoding="utf-8-sig")
fig,axes=plt.subplots(3,2,figsize=(14,10),sharex=True)
for ax,metric in zip(axes.flat,["INJ_VOL_DAILY"]+PROD_METRICS):
    g=monthly_raw[monthly_raw["字段"].eq(metric)].copy()
    base=g.loc[g["split"].eq("train"),"均值"].median()
    ax.plot(g["月份"],g["均值"]/(abs(base)+1e-9),marker="o",color="#3D5A80")
    ax.axvline("2026-03",color="#E76F51",linestyle="--"); ax.set_title(metric); ax.set_ylabel("相对训练期月均中位数"); ax.tick_params(axis="x",rotation=40)
fig.suptitle("32个重叠测试井组的月度动态漂移",y=1.02,fontsize=14)
savefig("29_monthly_raw_drift.png",fig)

# Target-date raw coverage.
coverage_rows=[]
for split,lab,injdf,proddf in [("train",y,inj_tr,prod_tr),("test",test_template,inj_te,prod_te)]:
    keys=lab[["WELL_GROUP_NAME","PROD_DATE"]].drop_duplicates()
    for source,raw in [("注汽",injdf),("采出",proddf)]:
        avail=raw[["WELL_GROUP_NAME","PROD_DATE"]].drop_duplicates().assign(hit=1)
        m=keys.merge(avail,on=["WELL_GROUP_NAME","PROD_DATE"],how="left")
        coverage_rows.append({"split":split,"来源":source,"目标井组日数":len(keys),"同日有原始记录数":m["hit"].notna().sum(),"同日覆盖率":m["hit"].notna().mean()})
coverage=pd.DataFrame(coverage_rows)
coverage.to_csv(TAB / "target_date_raw_coverage.csv",index=False,encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# 8. Markdown reports
# -----------------------------------------------------------------------------
inj_gap_main=inj_gap_stats[inj_gap_stats["序列"].eq("INJ_VOL_DAILY")].copy()
prod_gap_main=prod_gap_stats[prod_gap_stats["序列"].eq("LIQ_PROD_DAILY")].copy()
inj_repeat_overall=[]
for split,df in [("train",inj_tr),("test",inj_te)]:
    z=successive_value_profile(df,"INJ_VOL_DAILY")
    inj_repeat_overall.append({"split":split,"有效相邻对":len(z),"数值完全相同率":z["same"].mean(),"中位间隔":z["gap"].median(),"大于1天间隔占比":z["gap"].gt(1).mean()})
inj_repeat_overall=pd.DataFrame(inj_repeat_overall)

top10=topology_counts.head(10)
group_extremes=pd.concat([group_stats.nsmallest(8,"short比例"),group_stats.nlargest(8,"short比例")]).drop_duplicates()
gp_summary=pd.DataFrame({
    "统计项":["group×indicator 组合数","lag标准差中位数","lag标准差P25","lag标准差P75","lag完全不变的组合数","short比例≤10%或≥90%的组合数"],
    "数值":[len(gp_stats),gp_stats["lag标准差"].median(),gp_stats["lag标准差"].quantile(.25),gp_stats["lag标准差"].quantile(.75),(gp_stats["lag标准差"].fillna(0)==0).sum(),((gp_stats["short比例"]<=.1)|(gp_stats["short比例"]>=.9)).sum()]
})

write_md("01_井组结构与数据关系.md",f"""
# 井组结构与数据关系 EDA

## 结论

- 静态主表有 **{gi['WELL_GROUP_NAME'].nunique()} 个井组、{gi['WELL_NAME'].nunique()} 口唯一井、{len(gi)} 条井组—井关系**。
- 井组关系中有 **{int((gi.WELL_PURPOSE==PURPOSE_I).sum())} 条注汽井关系**、**{int((gi.WELL_PURPOSE==PURPOSE_P).sum())} 条生产井关系**。平均每组 {topology['注汽井数'].mean():.2f} 口注汽井、{topology['生产井数'].mean():.2f} 口生产井。
- 生产侧拓扑很简单：{int((topology['生产井数']==1).sum())} 个井组为单生产井，只有 {int((topology['生产井数']>1).sum())} 个井组为多生产井；生产井数最大为 {topology['生产井数'].max()}。注汽侧更复杂，注汽井数范围为 {topology['注汽井数'].min()}～{topology['注汽井数'].max()}。
- {int((membership>1).sum())} 口井关联多个井组，最多关联 {membership.max()} 个井组。不能按物理井名全局去重。
- 训练和测试的 `well_group_info.csv` 完全一致：**{gi.equals(gi_test)}**。`CANTON` 是常量；`ORG_NAME` 与 `BLOCK` 只有两个完全对应的组合；静态 `PROD_DATE` 全部为 {gi['PROD_DATE'].dt.strftime('%Y-%m-%d').iloc[0]}，不具备横截面区分度。

## 图表

![每组注汽井与生产井数量](figures/01_topology_hist.png)

![井组拓扑组合](figures/02_topology_combinations.png)

![单井关联井组数](figures/03_well_membership.png)

![转驱年龄与标签](figures/04_static_age_label.png)

## 拓扑组合频数

{md_table(top10)}

完整井组拓扑见 [`tables/well_group_topology.csv`](tables/well_group_topology.csv)，完整组合频数见 [`tables/topology_counts.csv`](tables/topology_counts.csv)。

## 文件级结构审计

{md_table(file_audit)}

## 对后续特征工程的约束

1. 绝大多数井组属于“一口生产井 + 多口注汽井”，生产侧总量与单井量在多数井组接近，但仍要保留多生产井的聚合口径。
2. 注汽侧必须保留总量、最大单井量、井间离散度和有效井数，仅取均值会丢失井组规模。
3. `WELL_PURPOSE` 与动态文件并非完全一致，可能与转驱或历史用途有关，不应仅凭静态用途硬过滤动态记录。
4. 静态年龄与标签图只表示相关关系。年龄与井组、区块、开发阶段共同变化，不能直接解释为因果。
""")

ffill_hint = "支持" if inj_repeat_overall.loc[inj_repeat_overall['split'].eq('train'),'数值完全相同率'].iloc[0] < .15 and inj_gap_main.loc[inj_gap_main['split'].eq('train'),'>1天占比'].iloc[0] > .5 else "不能单独支持"
write_md("02_注汽时序与缺失.md",f"""
# 注汽时序、缺失与零值 EDA

## 结论

- 训练注汽表 {len(inj_tr):,} 行，日期 {inj_tr.PROD_DATE.min():%Y-%m-%d}～{inj_tr.PROD_DATE.max():%Y-%m-%d}；测试注汽表 {len(inj_te):,} 行，日期 {inj_te.PROD_DATE.min():%Y-%m-%d}～{inj_te.PROD_DATE.max():%Y-%m-%d}。
- `INJ_LIQ_DAILY` 在训练和测试中均 **100% 缺失**，当前数据版本不能作为数值特征。
- `INJ_VOL_DAILY` 的行级缺失率分别为 {pct(inj_tr.INJ_VOL_DAILY.isna().mean())} 和 {pct(inj_te.INJ_VOL_DAILY.isna().mean())}；空值与真实 0 同时存在，不能合并。
- 注汽侧 `WH_TEMP` 的缺失率由 {pct(inj_tr.WH_TEMP.isna().mean())} 升至 {pct(inj_te.WH_TEMP.isna().mean())}，缺失机制本身存在明显时间漂移。
- 日注汽量的非空相邻记录中，训练集 {pct(inj_gap_main.loc[inj_gap_main['split'].eq('train'),'1天占比'].iloc[0])} 为连续 1 天记录，且 {pct(inj_repeat_overall.loc[inj_repeat_overall['split'].eq('train'),'数值完全相同率'].iloc[0])} 的相邻数值完全相同。这与“只在数值变化时才记录”的严格状态变更表不一致。
- 因而仅凭“表中有空值” **{ffill_hint}** 全量无限制 forward-fill。建议保留原始缺失版，并把有限期 ffill 作为单独候选特征通过时间验证判断。

## 缺失与零值

![注汽字段缺失率与零值率](figures/05_injection_missing_zero.png)

{md_table(inj_num[['split','字段','缺失数','缺失率','零值数','零值率_非空','均值','中位数']])}

## 时间间隔

![注汽量有效记录间隔](figures/06_injection_gap_distribution.png)

{md_table(inj_gap_main)}

注汽量的相邻有效记录统计：

{md_table(inj_repeat_overall)}

![不同间隔下相邻注汽量是否重复](figures/07_injection_repeat_by_gap.png)

如果这是严格的“只在状态变化时记录”数据，相邻已记录数值应很少完全相同。图中重复率可以检验该假设；实际结果应结合 `gap` 分组看，不能只凭整体非日频就认定 forward-fill 正确。

## 月度缺失模式

![注汽字段月度缺失率](figures/08_injection_monthly_missing.png)

完整表格：[`tables/injection_monthly_profile.csv`](tables/injection_monthly_profile.csv)。

## 按井和井组的有效天数/缺失模式

![注汽井有效天数](figures/30_injection_well_effective_days.png)

![注汽井组日缺失热力图](figures/31_injection_group_missing_heatmap.png)

逐井明细：[`tables/injection_well_effective_days.csv`](tables/injection_well_effective_days.csv)；逐井组日缺失率：[`tables/injection_group_day_missing.csv`](tables/injection_group_day_missing.csv)。

## 建模约束

- 至少同时保留原始缺失标志、距上次有效观测天数、窗口有效率和有限期前向填充值。
- 对 0 单独构造停注/低注状态，不用 `fillna(0)` 覆盖缺失。
- 测试期初的历史窗口应接续训练动态数据；否则 3 月初无法构造 45 天历史。
""")

write_md("03_采出时序与缺失.md",f"""
# 采出时序、缺失与内部一致性 EDA

## 结论

- 训练采出表 {len(prod_tr):,} 行，测试采出表 {len(prod_te):,} 行。
- 日产液、日产油、含水率和采出侧温度常以同一缺失块出现。训练行级缺失率约 41.5%，测试约 31.8%；油管压力更完整。
- 生产记录总体比注汽量有效记录更接近日频，但仍必须在“井组—井”层面检查间隔，不能只看全表每天是否有记录。
- `WATER_CUT` 与由 `LIQ_PROD_DAILY`、`OIL_PROD_DAILY` 推导的含水率高度一致程度见下表。多生产井聚合时，按产液量加权或从汇总液油量推导，比简单平均更符合比例口径。

## 缺失与零值

![采出字段缺失率与零值率](figures/09_production_missing_zero.png)

{md_table(prod_num[['split','字段','缺失数','缺失率','零值数','零值率_非空','均值','中位数']])}

## 时间间隔

![日产液有效记录间隔](figures/10_production_gap_distribution.png)

{md_table(prod_gap_main)}

## 月度缺失模式

![采出字段月度缺失率](figures/11_production_monthly_missing.png)

## 含水率一致性

![含水率一致性](figures/12_watercut_consistency.png)

{md_table(wc_consistency)}

## 按井和井组的有效天数/缺失模式

![采出井有效天数](figures/32_production_well_effective_days.png)

![采出井组日缺失热力图](figures/33_production_group_missing_heatmap.png)

逐井明细：[`tables/production_well_effective_days.csv`](tables/production_well_effective_days.csv)；逐井组日缺失率：[`tables/production_group_day_missing.csv`](tables/production_group_day_missing.csv)。

## 建模约束

- 产液和产油优先保留井组求和及井级分布特征。
- 含水率保留原始加权值、由汇总液油量推导值及二者残差。
- 温度用均值、最大值、升温幅度和井间极差；不使用求和作为物理主口径。
- `WH_TEMP` 在注汽表和采出表重名，合并前必须改名。
""")

write_md("04_标签指标对与井组效应.md",f"""
# 标签、指标对与井组效应 EDA

## 总体标签分布

- 训练样本 {len(y):,} 条，lag 范围 {int(y.OPTIMAL_LAG_DAYS.min())}～{int(y.OPTIMAL_LAG_DAYS.max())}。
- `P(y≤3)` = **{pct(y.SHORT.mean())}**，`P(y≥4)` = **{pct(1-y.SHORT.mean())}**。
- 3 天有 {(y.OPTIMAL_LAG_DAYS==3).sum():,} 条，4 天只有 {(y.OPTIMAL_LAG_DAYS==4).sum():,} 条。3/4 分界两侧极不连续。
- 0 天占 {pct((y.OPTIMAL_LAG_DAYS==0).mean())}；44/45 天合计占 {pct(y.OPTIMAL_LAG_DAYS.isin([44,45]).mean())}。标签存在明显边界聚集。

![0到45完整频数](figures/13_label_frequency.png)

完整频数：[`tables/label_frequency_0_45.csv`](tables/label_frequency_0_45.csv)。

## 指标对

标签中实际只有 **{y.INJ_INDICATOR.nunique()} 个注入指标**、**{y.PROD_INDICATOR.nunique()} 个采出指标**，总计 {y.groupby(['INJ_INDICATOR','PROD_INDICATOR']).ngroups} 个固定指标对，不是理论上的 4×5=20 组合。

![指标对热力图](figures/14_indicator_pair_heatmaps.png)

{md_table(pair_stats)}

## 井组条件标签

![井组标签画像](figures/15_group_label_profile.png)

短滞后比例最低与最高的井组：

{md_table(group_extremes[['WELL_GROUP_NAME','样本数','平均lag','中位lag','short比例','label标准差']])}

## Group effect 与 time-varying effect

![标签组间效应](figures/16_label_group_effect.png)

{md_table(effect_stats)}

同一 `group × indicator` 内部的时间波动：

{md_table(gp_summary)}

这组 η² 是描述性方差分解：数值越高，说明类别均值能解释更多总体差异。即使 `井组×指标` 有明显解释力，组内标准差和后续时间图仍显示 lag 会随时间切换，所以本题既不是纯井组记忆题，也不是完全忽略井组的通用时序题。

完整表格：[`tables/group_label_stats.csv`](tables/group_label_stats.csv) 与 [`tables/group_indicator_time_stats.csv`](tables/group_indicator_time_stats.csv)。
""")

write_md("05_标签时间行为与评分基线.md",f"""
# 标签时间行为、生成频率与简单参照规则 EDA

## G081 与代表性时间序列

![G081标签时间序列](figures/17_g081_label_timeline.png)

![稳定和高波动示例](figures/18_label_timeline_examples.png)

图中横线是 3/4 天评分断点。可见不同生产指标的状态转换并不同步，不能把同一井组日期的五个标签复制为一个值。

## 标签自相关与状态切换

- 相邻可用标签完全相同率：**{pct((trans['abs_change']==0).mean())}**。
- 相邻标签处于同一短/长分段的比例：**{pct((trans['SHORT'].to_numpy()==trans['prev_short'].to_numpy()).mean())}**。
- 相邻标签绝对变化中位数：{trans['abs_change'].median():.1f} 天；均值：{trans['abs_change'].mean():.2f} 天。

![标签转移](figures/19_label_transition.png)

{md_table(transition_by_gap)}

## 标签日期生成频率

![标签日期间隔与组合完整性](figures/20_label_date_frequency.png)

- 每个进入标签表的 `group + date` 固定有 **{int(combo_counts.min())}** 条指标对记录，训练集 {len(combo_counts):,} 个井组日期全部完整。
- 相邻标签日期的中位间隔为 {label_gap.median():.0f} 天，但最大间隔达到 {label_gap.max()} 天，样本并非独立同分布的等间隔观测。

## 月度标签漂移

![月度标签漂移](figures/21_monthly_label_drift.png)

{md_table(monthly_label)}

## 用官方公式评价 EDA 简单参照规则

> **范围说明：** 本节没有训练任何机器学习模型，也没有预测正式测试集。这里计算的是几条最简单的统计规则，用来判断标签仅靠总体分布、井组记忆或时间持续性能达到什么水平。它们属于 EDA 的可预测性诊断，不是正式模型基线，更不是比赛提交结果。“上一时刻真实标签”还使用了验证期真实值，只代表局部自相关的诊断上限，线上无法使用。

![简单参照规则评分诊断](figures/22_score_baselines.png)

2026 年 2 月多步时间外推结果：

{md_table(show_base[['预测规则','样本数','分段准确率','S_seg','MAE_short','S_short','MAE_long','S_long','总分']])}

![评分敏感性](figures/23_score_sensitivity.png)

“上一时刻真实标签”使用了验证期内已经发生的真实标签，只是局部持续性的诊断上限，测试时不可用。正式验证应固定在预测起点，只使用起点以前的标签和动态数据做完整多步外推。
""")

write_md("06_聚合方式与互相关实验.md",f"""
# 井组聚合方式与滞后互相关 EDA

## 聚合方式比较

![聚合稳定性](figures/24_aggregation_stability.png)

完整表格：[`tables/aggregation_stability.csv`](tables/aggregation_stability.csv)。

![聚合方式与标签对齐](figures/34_aggregation_label_alignment.png)

{md_table(aggregation_alignment)}

图表比较 `sum / mean / max / min / std` 在井组日层面的缺失和波动。物理主口径建议为：

- 注汽量：sum 为主，同时保留 max、mean、std 和有效井数。
- 产液、产油：sum 为主。
- 含水率：产液加权或由汇总液油量推导。
- 温度：mean/max/极差，不用 sum 作为主量。
- 压力：max 与 mean 都应验证；当前互相关默认使用井组 max。

## 因果历史窗口互相关实验

对每条训练标签，使用截止该日期的历史窗口，计算 0～45 天候选滞后。比较了：

- 原始值最大正相关。
- 原始值最大绝对相关。
- 一阶差分最大绝对相关。
- 对称变化率最大绝对相关。
- 注汽量最长 45 天有限 forward-fill 后的原始值绝对相关。
- 15、30、60、90、180 天历史窗口。

所有相关都要求窗口内至少 5 个、且不少于窗口一半的有效配对点。无有效候选时，完整评分使用相应采出指标的训练中位 lag 回退。这是标签机制探测实验，不是无泄漏模型验证，因为回退值和配置比较使用了全训练标签。

![互相关窗口比较](figures/25_xcorr_window_comparison.png)

得分最高的描述性配置是 `{best_name}`：有效覆盖率 {pct(best_xcorr['有效覆盖率'])}，有效样本 MAE {best_xcorr['有效样本MAE']:.2f} 天，回退后官方分 {best_xcorr['缺失回退后官方分']:.2f}。

![最佳互相关诊断](figures/26_xcorr_best_diagnostics.png)

前 15 个配置：

{md_table(xcorr_results.head(15))}

## 结论

如果互相关候选 lag 与标签高度一致，应出现较低 MAE、较高分段准确率和接近对角线的诊断图。当前结果显示，单一朴素互相关规则只能解释部分标签；窗口、变换和缺失口径会显著改变结果。互相关更适合作为候选 lag、峰值强度、峰间距和稳定性特征，再交给评分对齐的监督模型校准，而不是直接作为最终预测。
""")

present_test_dates=set(pd.DatetimeIndex(test_template["PROD_DATE"]).normalize().tolist())
missing_dates=sorted(set(pd.date_range(test_template.PROD_DATE.min(),test_template.PROD_DATE.max()).tolist())-present_test_dates)
missing_date_text="、".join(d.strftime("%Y-%m-%d") for d in missing_dates)
write_md("07_时间漂移与训练测试关系.md",f"""
# 时间漂移与训练/测试关系 EDA

## 井组关系

- 训练标签井组：{len(train_groups)} 个。
- 测试标签井组：{len(test_groups)} 个。
- 训练测试重叠：{len(test_groups & train_groups)} 个。
- 测试新井组：{', '.join(sorted(test_groups-train_groups))}。
- 新井组在测试模板中仅 {int(test_weight.loc[~test_weight['是否训练标签已见'],'测试行数'].sum())} 行，占 {pct(test_weight.loc[~test_weight['是否训练标签已见'],'测试权重'].sum())}。

![井组重叠与测试权重](figures/27_train_test_group_relation.png)

完整测试权重：[`tables/test_group_weights.csv`](tables/test_group_weights.csv)。

## 近期训练与测试动态漂移

为了减少井组构成差异，本节只比较 32 个训练测试重叠井组。近期训练定义为 2026-01-01～2026-02-28，测试为 2026-03-01～2026-05-31，并先按井组日做业务默认聚合。

![近期训练测试漂移](figures/28_recent_train_test_drift.png)

{md_table(drift_stats)}

KS 统计量和标准化 Wasserstein 距离越大，数值分布变化越明显；缺失率变化为正表示测试更缺失。这些是漂移筛查指标，不直接说明因果。

### KDE 分布形状对比

![近期训练与测试动态指标KDE对比](figures/35_recent_train_test_kde.png)

KDE 使用各指标的非缺失井组日聚合值，曲线面积分别归一化，因此用于比较分布位置、离散程度和多峰结构，不代表样本量。为防止极端值压缩主体曲线，横轴仅展示近期训练与测试合并样本的 0.5%～99.5% 分位区间。缺失率变化仍应结合上一张图判断。

从曲线形状看，井口温度的多峰结构变化最明显；测试期产液量在 20～30 附近更集中，高产液尾部减弱；最大套管压力更集中在低值区；注汽量主体峰略向低值移动。产油量和含水率整体轮廓较接近，但局部密度仍有变化。以上结论只描述非缺失样本，不能替代缺失机制分析。

## 月度动态轨迹

![重叠井组月度动态漂移](figures/29_monthly_raw_drift.png)

## 标签日期和原始记录覆盖

{md_table(coverage)}

测试模板日期范围为 {test_template.PROD_DATE.min():%Y-%m-%d}～{test_template.PROD_DATE.max():%Y-%m-%d}，但只出现 {test_template.PROD_DATE.nunique()} 个日期。范围内没有任何测试标签的日期共有 {len(missing_dates)} 天：{missing_date_text}。

## 结论

1. 任务主体是已知井组的未来外推，而不是大规模 cold-start。
2. 动态数值和缺失模式存在时间漂移，不能只用全训练均值做归一化。
3. 测试期初特征必须承接训练期末历史。
4. 标签模板决定预测范围。动态表包含的其他井组不能自动扩展为提交样本。
""")

core_findings=pd.DataFrame([
    {"问题":"一个训练样本由哪些信息决定？","EDA答案":"标签键是井组×日期×注入指标×采出指标；当前只有1个注入指标、5个采出指标。应使用该日期以前的井级/井组级动态、静态拓扑、缺失状态和指标类别。"},
    {"问题":"井组时序如何聚合？","EDA答案":"多数为1P+多I。注汽量、产液、产油以sum为主；比例做加权；温压保留mean/max/std；所有指标保留有效井数与缺失特征。"},
    {"问题":"训练与未来测试如何变化？","EDA答案":"33个测试井组中32个已见，但数值分布和缺失率发生漂移；测试初期需要训练末期历史；新井组仅占约0.09%。"},
    {"问题":"这是记忆题还是lag估计题？","EDA答案":"group×indicator存在稳定先验，局部标签自相关很强，但多步外推简单规则的得分明显下降，朴素互相关也不能充分复现标签。因此是井组先验与时变lag信号并存的混合问题。"},
])
write_md("08_EDA结论与建模约束.md",f"""
# EDA 结论与建模约束

## 三个核心问题

{md_table(core_findings)}

## 第一轮必须保留的事实

1. 训练标签共有 {len(y):,} 行，严格对应 {y[['WELL_GROUP_NAME','PROD_DATE']].drop_duplicates().shape[0]:,} 个井组日期 × 5 个采出指标。
2. 标签只有 `INJ_VOL_DAILY` 这一个注入指标；不要根据原始注汽字段自行扩展标签组合。
3. 短段占 {pct(y.SHORT.mean())}，3/4 天边界不连续。评估必须使用官方分数。
4. 井组拓扑以多注汽井、单生产井为主，但同一物理井可属于多个井组。
5. `INJ_LIQ_DAILY` 全空；注汽量空值与零值并存；注汽温度的测试缺失显著增加。
6. 标签局部持续性强，但严格多步时间外推远弱于使用上一真实标签的诊断上限。
7. 指标对、井组和月份都会改变标签分布，不能只训练一个不带类别条件的普通回归器。
8. 朴素互相关只提供部分信号，需要去趋势、差分、峰值稳定性与监督校准。

## 推荐验证约束

- 主验证采用滚动时间切分，最后一个月整段预测，禁止随机行切分。
- 同一 `group + date` 的 5 条记录始终放在同一折。
- 所有历史标签统计、缺失填充参数和归一化参数只用训练折拟合。
- 报告 `S_seg / S_short / S_long / Final`，并按井组、生产指标和月份拆分。
- 额外做井组留出验证，但其权重低于时间外推验证。

## EDA 产物索引

- [01 井组结构与数据关系](01_井组结构与数据关系.md)
- [02 注汽时序与缺失](02_注汽时序与缺失.md)
- [03 采出时序与缺失](03_采出时序与缺失.md)
- [04 标签指标对与井组效应](04_标签指标对与井组效应.md)
- [05 标签时间行为与简单参照规则](05_标签时间行为与评分基线.md)
- [06 聚合方式与互相关实验](06_聚合方式与互相关实验.md)
- [07 时间漂移与训练测试关系](07_时间漂移与训练测试关系.md)

详细数值表位于 [`tables`](tables)，全部图表位于 [`figures`](figures)。分析脚本为 [`run_eda.py`](run_eda.py)，从项目根目录运行 `python EDA/run_eda.py` 可重建全部产物。
""")

write_md("00_EDA总览.md",f"""
# 浅层超稠油注采响应滞后 EDA 总览

## 分析目标

本次 EDA 围绕三件事展开：训练样本的历史信息边界、井组多井时序的聚合方式、训练期到未来测试期的分布变化。分析覆盖四张源表、官方标签结构、简单参照规则的评分诊断和因果历史互相关探测。这里的简单参照规则没有训练模型，也没有生成测试集预测。

## 核心判断

{md_table(core_findings)}

## 最关键的三张结果

### 1. 0～45 天完整标签频数

![标签完整频数](figures/13_label_frequency.png)

### 2. 指标对样本数、short 比例和 median lag

![指标对热力图](figures/14_indicator_pair_heatmaps.png)

### 3. 同一 group × indicator 的标签时间变化

![代表性标签时间序列](figures/18_label_timeline_examples.png)

## 额外关键结果

### 井组拓扑

![井组拓扑](figures/01_topology_hist.png)

### 标签局部转移

![标签转移](figures/19_label_transition.png)

### 互相关窗口实验

![互相关窗口实验](figures/25_xcorr_window_comparison.png)

### 未来分布漂移

![近期训练测试漂移](figures/28_recent_train_test_drift.png)

## 阅读顺序

1. [井组结构与数据关系](01_井组结构与数据关系.md)
2. [注汽时序与缺失](02_注汽时序与缺失.md)
3. [采出时序与缺失](03_采出时序与缺失.md)
4. [标签指标对与井组效应](04_标签指标对与井组效应.md)
5. [标签时间行为与简单参照规则](05_标签时间行为与评分基线.md)
6. [聚合方式与互相关实验](06_聚合方式与互相关实验.md)
7. [时间漂移与训练测试关系](07_时间漂移与训练测试关系.md)
8. [EDA 结论与建模约束](08_EDA结论与建模约束.md)

## 可复现性

- 原始数据未修改。
- 数值明细表保存在 [`tables`](tables)。
- PNG 图表保存在 [`figures`](figures)。
- [`run_eda.py`](run_eda.py) 可一键重建全部分析。
""")

manifest={
    "source_files":[str(p.relative_to(ROOT)) for p in sorted(DATA.rglob("*.csv"))],
    "markdown_files":[p.name for p in sorted(OUT.glob("*.md"))],
    "figure_files":[p.name for p in sorted(FIG.glob("*.png"))],
    "table_files":[p.name for p in sorted(TAB.glob("*.csv"))],
    "summary":{
        "train_labels":len(y),"test_rows":len(test_template),"train_groups":len(train_groups),"test_groups":len(test_groups),
        "short_rate":float(y.SHORT.mean()),"best_constant":int(best_constant["预测规则"]),"best_constant_score":float(best_constant["总分"]),
        "best_xcorr_config":str(best_name),"best_xcorr_score":float(best_xcorr["缺失回退后官方分"]),
    },
}
(OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(manifest["summary"],ensure_ascii=False,indent=2))
