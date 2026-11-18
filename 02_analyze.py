"""
東京23区 中古マンション取引の探索的分析 (01_clean.py の出力を読む)

使い方:
  pip install pandas pyarrow matplotlib
  python 02_analyze.py

出力 (data/output/):
  ward_summary.csv      区別の件数・㎡単価(中央値)
  quarterly_trend.csv   四半期別の㎡単価(中央値)・件数
  age_band.csv          築年数帯別の㎡単価(中央値)
  station_band.csv      駅距離帯別の㎡単価(中央値)
  01_ward.png 02_trend.png 03_age.png 04_station.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

IN_PATH = Path("data/processed/mansion_tokyo23.parquet")
OUT_DIR = Path("data/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント(Windows → Mac → Linuxの順で最初に見つかったものを使う)
from matplotlib import font_manager

_installed = {f.name for f in font_manager.fontManager.ttflist}
for _name in ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "Noto Sans CJK JP", "Noto Sans CJK TC"]:
    if _name in _installed:
        plt.rcParams["font.family"] = _name
        break
plt.rcParams["axes.unicode_minus"] = False

MAN = 10_000  # 円 → 万円


def label_quarter(df: pd.DataFrame) -> pd.Series:
    return df["trade_year"].astype(int).astype(str) + "Q" + df["trade_quarter"].astype(int).astype(str)


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = pd.DataFrame({
        "件数": g.size(),
        "㎡単価中央値(万円)": g["price_per_m2"].median() / MAN,
        "価格中央値(万円)": g["price"].median() / MAN,
    }).round(1)
    return out


def save_bar(s: pd.Series, title: str, xlabel: str, path: Path, horizontal: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 6 if horizontal else 4.5))
    if horizontal:
        s.sort_values().plot.barh(ax=ax, color="#3b6ea5")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
    else:
        s.plot.bar(ax=ax, color="#3b6ea5", rot=0)
        ax.set_ylabel(xlabel)
        ax.set_xlabel("")
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not IN_PATH.exists():
        sys.exit(f"{IN_PATH} がありません。先に 01_clean.py を実行してください。")
    df = pd.read_parquet(IN_PATH)
    df["quarter"] = label_quarter(df)
    print(f"読み込み: {len(df):,}件 / 期間 {df['quarter'].min()} 〜 {df['quarter'].max()}")

    # 1) 区別 ------------------------------------------------------------
    ward = summarize(df, "市区町村名").sort_values("㎡単価中央値(万円)", ascending=False)
    ward.to_csv(OUT_DIR / "ward_summary.csv", encoding="utf-8-sig")
    save_bar(ward["㎡単価中央値(万円)"], "区別 ㎡単価(中央値)", "万円/㎡", OUT_DIR / "01_ward.png", horizontal=True)
    print("\n[区別 ㎡単価(万円) 上位5 / 下位5]")
    print(ward.head(5).to_string())
    print(ward.tail(5).to_string())

    # 2) 四半期推移 -------------------------------------------------------
    q = summarize(df, "quarter")
    q.to_csv(OUT_DIR / "quarterly_trend.csv", encoding="utf-8-sig")
    top3 = ward.index[:3].tolist()
    low2 = ward.index[-2:].tolist()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(q.index, q["㎡単価中央値(万円)"], color="black", lw=2.5, label="23区全体")
    for w in top3 + low2:
        s = df[df["市区町村名"] == w].groupby("quarter")["price_per_m2"].median() / MAN
        ax.plot(s.index, s.values, lw=1.4, label=w)
    ticks = list(range(0, len(q.index), 4))
    ax.set_xticks(ticks)
    ax.set_xticklabels([q.index[i] for i in ticks], rotation=45)
    ax.set_ylabel("万円/㎡")
    ax.set_title("四半期別 ㎡単価(中央値)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_trend.png", dpi=150)
    plt.close(fig)

    first, last = q.iloc[0], q.iloc[-1]
    chg = last["㎡単価中央値(万円)"] / first["㎡単価中央値(万円)"] - 1
    print(f"\n[全体の㎡単価] {q.index[0]}: {first['㎡単価中央値(万円)']}万円 → {q.index[-1]}: "
          f"{last['㎡単価中央値(万円)']}万円 ({chg:+.0%})")

    # 3) 築年数帯 ---------------------------------------------------------
    df["築年数帯"] = pd.cut(df["age"], bins=[-1, 5, 10, 20, 30, 40, 200],
                          labels=["〜5年", "6〜10年", "11〜20年", "21〜30年", "31〜40年", "41年〜"])
    a = summarize(df, "築年数帯")
    a.to_csv(OUT_DIR / "age_band.csv", encoding="utf-8-sig")
    save_bar(a["㎡単価中央値(万円)"], "築年数帯別 ㎡単価(中央値)", "万円/㎡", OUT_DIR / "03_age.png")
    print("\n[築年数帯別]")
    print(a.to_string())

    # 4) 駅距離帯 ---------------------------------------------------------
    d = df.dropna(subset=["station_min"]).copy()
    d["駅距離帯"] = pd.cut(d["station_min"], bins=[-1, 1, 3, 5, 10, 15, 1000],
                         labels=["〜1分", "2〜3分", "4〜5分", "6〜10分", "11〜15分", "16分〜"])
    s = summarize(d, "駅距離帯")
    s.to_csv(OUT_DIR / "station_band.csv", encoding="utf-8-sig")
    save_bar(s["㎡単価中央値(万円)"], "駅距離帯別 ㎡単価(中央値)", "万円/㎡", OUT_DIR / "04_station.png")
    print("\n[駅距離帯別]")
    print(s.to_string())

    print(f"\n保存先: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
