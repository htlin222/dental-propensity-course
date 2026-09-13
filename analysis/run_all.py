"""
牙科自費傾向模型｜五步分析，一次跑完

執行：python analysis/run_all.py
產出：analysis/outputs/ 底下的圖（png）與表（csv）

五個步驟對應教案：
  1. 認識並清理資料
  2. 算出現況（基準線）
  3. 描述性統計
  4. 建立評分模型
  5. 用全新病人驗證
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

TARGET = "selfpay_next90"
NUM = ["age", "missing_teeth", "perio_depth", "no_show_rate", "visits_12m",
       "past_selfpay", "has_line", "days_since_visit"]
CAT = ["income_area", "doctor", "sex", "blood_type"]
FEATURES = NUM + CAT

# 圖表樣式：黑白單色、直角、等寬字、標籤全英文
plt.rcParams.update({
    "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "monospace", "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 14,
    "axes.edgecolor": "black", "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#CCCCCC", "grid.linestyle": ":",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white",
})
BAR = "#1A1A1A"


def bar(labels, values, title, xlabel, name, ymax=None, baseline=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=BAR, width=0.6,
                  edgecolor="black", linewidth=1.0)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.03,
                f"{v:.1f}%", ha="center", fontsize=9.5)
    ax.set_ylim(0, ymax or max(values) * 1.2)
    ax.set_ylabel("Conversion rate (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if baseline is not None:
        ax.axhline(baseline, color="black", ls="--", lw=0.9)
        ax.text(-0.42, baseline + 0.8, f"baseline {baseline:.1f}%", fontsize=8.5,
                ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  [figure] analysis/outputs/{name}.png")


def clean(path):
    d = pd.read_csv(path).drop_duplicates()
    d.loc[d["age"] > 100, "age"] = np.nan
    d["age"] = d["age"].fillna(d["age"].median())
    d["perio_depth"] = d["perio_depth"].fillna(d["perio_depth"].median())
    return d


def rate(d, col):
    t = d.groupby(col, observed=True)[TARGET].agg(conversion_rate="mean", n="size")
    t["conversion_rate"] = (t["conversion_rate"] * 100).round(1)
    return t


def main():
    # ---------- STEP 1 認識並清理 ----------
    print("\nSTEP 1｜認識並清理資料")
    raw = pd.read_csv(DATA / "dental_patients.csv")
    print(f"  原始 {raw.shape[0]} 列 × {raw.shape[1]} 欄")
    print(f"  perio_depth 缺 {raw['perio_depth'].isna().sum()} 筆")
    print(f"  age 最大值 {raw['age'].max()}  <- 不合理")
    df = clean(DATA / "dental_patients.csv")
    print(f"  清理後 {len(df)} 列，age 最大值 {df['age'].max():.0f}")

    # ---------- STEP 2 現況 ----------
    base = df[TARGET].mean() * 100
    print(f"\nSTEP 2｜現況（基準線）= {base:.1f}%")

    # ---------- STEP 3 描述性統計 ----------
    print("\nSTEP 3｜描述性統計")
    df["teeth_group"] = pd.cut(df["missing_teeth"], [-1, 0, 2, 4, 99],
                               labels=["0", "1-2", "3-4", "5+"])
    tables = []
    for c in ["teeth_group", "has_line", "past_selfpay", "doctor",
              "income_area", "sex", "blood_type"]:
        t = rate(df, c).reset_index()
        t.columns = ["group", "conversion_rate", "n"]
        t.insert(0, "variable", c)
        tables.append(t)
        print(f"  {c}: " + ", ".join(
            f"{g}={r}% (n={n})" for g, r, n in
            zip(t["group"].astype(str), t["conversion_rate"], t["n"])))
    pd.concat(tables, ignore_index=True).to_csv(
        OUT / "step3_rates_by_group.csv", index=False, encoding="utf-8-sig")
    print("  [table] analysis/outputs/step3_rates_by_group.csv")

    t = rate(df, "teeth_group")
    bar(t.index.astype(str), t["conversion_rate"].values,
        "More missing teeth, higher self-pay conversion",
        "Missing teeth", "step3_missing_teeth", ymax=50, baseline=base)

    # ---------- STEP 4 建模 ----------
    print("\nSTEP 4｜建立評分模型")
    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])),
        ("clf", LogisticRegression(max_iter=2000)),
    ]).fit(df[FEATURES], df[TARGET])

    names = NUM + list(pipe["pre"].named_transformers_["cat"]
                       .get_feature_names_out(CAT))
    coef = pd.Series(pipe["clf"].coef_[0], index=names).sort_values(
        key=abs, ascending=False)
    print("  模型學到的權重 Top 6")
    print("  " + coef.head(6).round(2).to_string().replace("\n", "\n  "))
    coef.round(4).rename("coefficient").to_csv(
        OUT / "step4_coefficients.csv", encoding="utf-8-sig")

    top = coef.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color=["#1A1A1A" if v > 0 else "white" for v in top],
            edgecolor="black", linewidth=1.0,
            hatch=["" if v > 0 else "///" for v in top])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient (positive = more likely to convert)")
    ax.set_title("What the model learned")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "step4_coefficients.png")
    plt.close(fig)
    print("  [figure] analysis/outputs/step4_coefficients.png")

    demo = pd.DataFrame([
        dict(age=58, missing_teeth=4, perio_depth=4.5, no_show_rate=0.05, visits_12m=3,
             past_selfpay=1, has_line=1, days_since_visit=90, income_area="high",
             doctor="C", sex="F", blood_type="O"),
        dict(age=28, missing_teeth=0, perio_depth=4.5, no_show_rate=0.35, visits_12m=3,
             past_selfpay=0, has_line=0, days_since_visit=90, income_area="low",
             doctor="B", sex="F", blood_type="O")])
    pa, pb = pipe.predict_proba(demo[FEATURES])[:, 1]
    print(f"  病人 A {pa:.1%}   病人 B {pb:.1%}")

    # ---------- STEP 5 驗證 ----------
    print("\nSTEP 5｜用全新病人驗證")
    new = clean(DATA / "dental_patients_new.csv")
    p = pipe.predict_proba(new[FEATURES])[:, 1]
    y = new[TARGET].values
    nb = y.mean() * 100
    auc = roc_auc_score(y, p)
    print(f"  全新 {len(new)} 人，基準線 {nb:.1f}%，AUC {auc:.3f}")

    r = pd.DataFrame({"propensity": p.round(4), "y": y,
                      "patient_id": new["patient_id"].values})
    r = r.sort_values("propensity", ascending=False).reset_index(drop=True)

    rows = []
    for pct in [0.1, 0.2, 0.3, 0.5]:
        k = int(len(r) * pct)
        hit = r["y"][:k].mean() * 100
        rows.append({"top_pct": f"{int(pct*100)}%", "conversion_rate": round(hit, 1),
                     "lift": round(hit / nb, 2), "n": k})
        print(f"    前 {int(pct*100):>2}%  命中 {hit:.1f}%  lift {hit/nb:.2f}")
    pd.DataFrame(rows).to_csv(OUT / "step5_lift.csv", index=False, encoding="utf-8-sig")

    r["quintile"] = pd.qcut(r.index, 5,
                            labels=["Top 20%", "21-40%", "41-60%", "61-80%", "81-100%"])
    q = (r.groupby("quintile", observed=True)["y"].mean() * 100).round(1)
    print("  五等分：" + " -> ".join(f"{v}%" for v in q.values))
    bar(q.index.astype(str), q.values,
        "Validation on 1,200 unseen patients",
        "Ranked by model score", "step5_validation", ymax=60, baseline=nb)

    r.to_csv(OUT / "step5_scored_new_patients.csv", index=False, encoding="utf-8-sig")
    r.head(int(len(r) * 0.2)).to_csv(
        OUT / "step5_action_list_top20.csv", index=False, encoding="utf-8-sig")
    print("  [table] analysis/outputs/step5_action_list_top20.csv")

    print(f"\n結論：100 個諮詢名額，隨機挑約 {nb:.0f} 件，"
          f"照分數挑前 100 人約 {r['y'][:100].mean()*100:.0f} 件。")


if __name__ == "__main__":
    main()
