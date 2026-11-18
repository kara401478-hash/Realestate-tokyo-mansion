"""
価格予測モデルと「相場より安い/高い駅」の抽出 (01_clean.py の出力を読む)

考え方:
  物件の条件(面積・築年数・駅距離・構造・改装・区・取引時期)から、㎡単価の「あるべき水準」を予測し、
  実際の取引価格との差(乖離率)を駅ごとに集計する。
  ・区は説明変数に入れる → 「同じ区の中で、同じ条件なのに安い/高い駅」が残る
  ・駅名・地区名は説明変数に入れない → 駅ごとの差が予測に吸収されず、残差として見える
  ・各取引の予測は5分割の交差検証(その取引を学習に使わない予測)で出す

使い方:
  pip install pandas pyarrow matplotlib scikit-learn
  python 03_model.py                 # 直近3年・駅ごと30件以上
  python 03_model.py --years 2 --min-n 20
  python 03_model.py --keep-special  # 「取引の事情等」付きの取引も含める

出力 (data/output/):
  model_report.txt          精度と除外件数
  station_gap.csv           駅ごとの乖離率・95%区間・判定
  05_pred_vs_actual.png     検証期間の予測と実際
  06_station_gap.png        相場より安い/高い駅
  07_importance.png         価格を左右する要因
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

IN_PATH = Path("data/processed/mansion_tokyo23.parquet")
OUT_DIR = Path("data/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAN = 10_000

_installed = {f.name for f in font_manager.fontManager.ttflist}
for _name in ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "Noto Sans CJK JP", "Noto Sans CJK TC"]:
    if _name in _installed:
        plt.rcParams["font.family"] = _name
        break
plt.rcParams["axes.unicode_minus"] = False

report = []


def say(msg: str = "") -> None:
    print(msg)
    report.append(msg)


CAT_COLS = ["区", "構造", "改装"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "面積": df["area"],
        "築年数": df["age"],
        "駅距離": df["station_min"],
        "取引時期": (df["trade_year"] - 2018) + (df["trade_quarter"] - 1) / 4,
        "区": df["市区町村名"].astype("category").cat.codes,
        "構造": df["structure"].fillna("不明").astype("category").cat.codes,
        "改装": df["renovation"].fillna("不明").astype("category").cat.codes,
    }, index=df.index)


def make_model(columns) -> HistGradientBoostingRegressor:
    mask = [c in CAT_COLS for c in columns]
    # absolute_error(中央値を狙う損失)にして、極端な駅1つに区全体の予測が引きずられるのを防ぐ
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.06, min_samples_leaf=40, l2_regularization=1.0,
        categorical_features=mask, random_state=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0, help="駅ごとの集計に使う直近の年数")
    ap.add_argument("--min-n", type=int, default=30, help="駅ごとの最低件数")
    ap.add_argument("--min-gap", type=float, default=5.0, help="候補とみなす乖離率の最小値(%%)")
    ap.add_argument("--keep-special", action="store_true", help="取引の事情等が付いた取引も含める")
    args = ap.parse_args()

    if not IN_PATH.exists():
        sys.exit(f"{IN_PATH} がありません。先に 01_clean.py を実行してください。")
    df = pd.read_parquet(IN_PATH)
    for c in ["structure", "renovation", "circumstances"]:
        if c not in df.columns:
            df[c] = None
    say(f"読み込み: {len(df):,}件")

    # 特殊な取引(関係者間取引など)の扱い
    special = df["circumstances"].notna()
    say(f"「取引の事情等」が付いた取引: {special.sum():,}件 ({special.mean():.1%})")
    if special.any():
        say("  内訳(上位5): " + " / ".join(f"{k} {v:,}" for k, v in df.loc[special, "circumstances"].value_counts().head(5).items()))
    if not args.keep_special:
        df = df[~special].copy()
        say(f"  → 除外して {len(df):,}件で分析")
    df = df.reset_index(drop=True)

    X = build_features(df)
    y = np.log(df["price_per_m2"])
    t = X["取引時期"]
    tmax = t.max()

    # --- 1) 精度の検証: 直近4四半期を未来として予測 ---------------------------
    test = t >= tmax - 0.75
    train = ~test
    say(f"\n[検証] 学習 {train.sum():,}件 (〜{t[train].max()+2018:.2f}) / 検証 {test.sum():,}件 (直近4四半期)")
    model = make_model(X.columns).fit(X[train], y[train])
    pred = model.predict(X[test])
    actual_m2 = df.loc[test, "price_per_m2"].to_numpy()
    pred_m2 = np.exp(pred)
    mae = np.mean(np.abs(pred_m2 - actual_m2)) / MAN
    mdape = np.median(np.abs(pred_m2 - actual_m2) / actual_m2)
    r2 = r2_score(y[test], pred)

    recent_train = train & (t >= t[train].max() - 0.75)
    base = df[recent_train].groupby("市区町村名")["price_per_m2"].median()
    base_pred = df.loc[test, "市区町村名"].map(base).fillna(base.median()).to_numpy()
    base_mae = np.mean(np.abs(base_pred - actual_m2)) / MAN
    base_mdape = np.median(np.abs(base_pred - actual_m2) / actual_m2)
    say(f"  モデル : 平均誤差 {mae:.1f}万円/㎡ / 誤差率の中央値 {mdape:.1%} / R² {r2:.3f}")
    say(f"  参考   : 区別の直近中央値だけで予測 → 平均誤差 {base_mae:.1f}万円/㎡ / 誤差率の中央値 {base_mdape:.1%}")

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    lim = np.percentile(actual_m2 / MAN, [0.5, 99.5])
    hb = ax.hexbin(actual_m2 / MAN, pred_m2 / MAN, gridsize=50, bins="log", mincnt=1, cmap="Blues",
                   extent=(lim[0], lim[1], lim[0], lim[1]))
    ax.plot(lim, lim, color="crimson", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("実際の㎡単価(万円)"); ax.set_ylabel("予測(万円)")
    ax.set_title("検証期間(直近4四半期)の予測と実際")
    fig.colorbar(hb, label="件数(対数)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "05_pred_vs_actual.png", dpi=150); plt.close(fig)

    # --- 2) 全取引の「あるべき価格」: 5分割の交差検証で、自分を学習に使わない予測 --------
    oof = np.zeros(len(df))
    fold0 = None
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = make_model(X.columns).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
        if fold0 is None:
            fold0 = (m, te)  # 要因分析用(全期間からランダムに抜いた検証データ)
    df["pred_m2"] = np.exp(oof)
    df["resid"] = y - oof

    # --- 3) 駅ごとの乖離率 -------------------------------------------------------
    win = df[(t >= tmax - args.years) & df["最寄駅：名称"].notna()]
    say(f"\n[駅ごとの集計] 直近{args.years:g}年の {len(win):,}件 / 1駅あたり{args.min_n}件以上")
    rng = np.random.default_rng(0)
    rows = []
    for (ward, st), g in win.groupby(["市区町村名", "最寄駅：名称"]):
        if len(g) < args.min_n:
            continue
        r = g["resid"].to_numpy()
        boots = np.median(rng.choice(r, size=(300, len(r)), replace=True), axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        med = np.median(r)
        gap = (np.exp(med) - 1) * 100
        # 統計的に差があり(95%区間が0をまたがない)、かつ実用上の差(既定5%以上)がある駅だけを候補にする
        verdict = ("割安候補" if hi < 0 and gap <= -args.min_gap else
                   "割高候補" if lo > 0 and gap >= args.min_gap else "差は小さい/不明")
        rows.append({
            "区": ward, "駅": st, "件数": len(g),
            "乖離率(%)": round((np.exp(med) - 1) * 100, 1),
            "95%区間下限(%)": round((np.exp(lo) - 1) * 100, 1),
            "95%区間上限(%)": round((np.exp(hi) - 1) * 100, 1),
            "実際㎡単価中央値(万円)": round(g["price_per_m2"].median() / MAN, 1),
            "予測㎡単価中央値(万円)": round(g["pred_m2"].median() / MAN, 1),
            "判定": verdict,
        })
    if not rows:
        sys.exit("条件を満たす駅がありません。--min-n を小さくするか --years を大きくしてください。")
    res = pd.DataFrame(rows).sort_values("乖離率(%)")
    res.to_csv(OUT_DIR / "station_gap.csv", index=False, encoding="utf-8-sig")
    n_cheap = (res["判定"] == "割安候補").sum()
    n_dear = (res["判定"] == "割高候補").sum()
    say(f"  対象 {len(res)}駅 → 割安候補 {n_cheap}駅 / 割高候補 {n_dear}駅 / 差は小さい {len(res)-n_cheap-n_dear}駅")
    say(f"  ※候補の条件: 95%区間が0をまたがず、乖離率が±{args.min_gap:g}%以上。駅数が多いので偶然混じる駅もあります。")
    say("  ※階数・向き・管理状態などはデータにないため、候補は「調べる価値のある駅」であり、割安と断定するものではありません。")
    show = ["区", "駅", "件数", "乖離率(%)", "95%区間下限(%)", "95%区間上限(%)", "実際㎡単価中央値(万円)", "予測㎡単価中央値(万円)", "判定"]
    say("\n[相場より安い駅 上位10]  乖離率がマイナス = 同条件の予測より安く取引されている")
    say(res[show].head(10).to_string(index=False))
    say("\n[相場より高い駅 上位10]")
    say(res[show].tail(10).iloc[::-1].to_string(index=False))

    k = min(12, len(res) // 2)
    pick = pd.concat([res.head(k), res.tail(k)])
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(pick) + 1.5))
    lab = pick["区"].str.replace("区", "") + " " + pick["駅"]
    err = np.array([pick["乖離率(%)"] - pick["95%区間下限(%)"], pick["95%区間上限(%)"] - pick["乖離率(%)"]])
    colors = pick["判定"].map({"割安候補": "#2a7f62", "割高候補": "#b5483a"}).fillna("#b8b8b8")  # 灰色=差は小さい
    ax.barh(lab, pick["乖離率(%)"], xerr=err, color=colors, ecolor="gray", capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("予測との乖離率(%)  マイナス=相場より安い / 緑=割安候補・赤=割高候補・灰=差は小さい")
    ax.set_title(f"相場より安い/高い駅(直近{args.years:g}年・{args.min_n}件以上・95%区間)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "06_station_gap.png", dpi=150); plt.close(fig)

    # --- 4) 何が価格を決めているか ------------------------------------------------
    # 直近4四半期だけだと「取引時期」の幅が狭く影響が過小に出るため、全期間からのランダム検証データで測る
    m0, te0 = fold0
    idx = X.iloc[te0].sample(min(5000, len(te0)), random_state=0).index
    pi = permutation_importance(m0, X.loc[idx], y.loc[idx], n_repeats=3, random_state=0)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    imp.plot.barh(ax=ax, color="#3b6ea5")
    ax.set_xlabel("その要因を入れ替えたときの精度の低下(R²)")
    ax.set_title("価格を左右する要因")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "07_importance.png", dpi=150); plt.close(fig)
    say("\n[価格を左右する要因(大きい順)]")
    say(imp.sort_values(ascending=False).round(3).to_string())

    (OUT_DIR / "model_report.txt").write_text("\n".join(report), encoding="utf-8")
    say(f"\n保存先: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
