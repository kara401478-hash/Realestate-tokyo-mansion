"""
不動産取引価格(中古マンション等・東京23区)の前処理スクリプト  v2

v2の変更点(実CSVの画面を見て修正):
  - 建築年が「2012年」の西暦表記でも読める(和暦・「戦前」にも対応)
  - 構造・用途・都市計画・建ぺい率・容積率・改装・取引の事情等も残す

使い方:
  1. pip install pandas pyarrow
  2. ダウンロードしたCSVのパスを指定して実行する(コピー不要)
       python 01_clean.py "C:\\Users\\名前\\Downloads\\Tokyo_20181_20262.csv"
     引数なしなら data/raw/ のCSVをすべて読む
     (Excelで開いて保存し直さないこと。価格が 1.1E+08 のように丸められます)
  3. 実行した場所の data/processed/ に mansion_tokyo23.parquet と cleaning_log.txt ができる
"""
import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []


def log(msg: str) -> None:
    print(msg)
    log_lines.append(msg)


def read_csv_auto(path: Path) -> pd.DataFrame:
    """文字コードが不明でも読めるようにする(UTF-8 → CP932の順で試す)"""
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"文字コードを判定できません: {path}")


# ---- 列の値を数値に直す関数 ----------------------------------------------
ERA_START = {"明治": 1868, "大正": 1912, "昭和": 1926, "平成": 1989, "令和": 2019}


def parse_build_year(s):
    """'2012年' → 2012 / '平成20年' → 2008 / '戦前' → 1945 / 空欄 → None"""
    if pd.isna(s):
        return None
    s = unicodedata.normalize("NFKC", str(s)).strip()
    if "戦前" in s:
        return 1945
    m = re.match(r"(\d{4})年", s)  # 西暦
    if m:
        return int(m.group(1))
    m = re.match(r"(明治|大正|昭和|平成|令和)(\d+|元)年", s)  # 和暦
    if not m:
        return None
    n = 1 if m.group(2) == "元" else int(m.group(2))
    return ERA_START[m.group(1)] + n - 1


def parse_trade_period(s):
    """'2023年第1四半期' → (2023, 1)"""
    if pd.isna(s):
        return (None, None)
    s = unicodedata.normalize("NFKC", str(s))
    m = re.search(r"(\d{4})年第(\d)四半期", s)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def parse_station_minutes(s):
    """'5' → 5 / '30分?60分' → 45 / '1H?1H30' → 75 などの文字列を分に直す"""
    if pd.isna(s):
        return None
    s = unicodedata.normalize("NFKC", str(s)).strip()
    if s.isdigit():
        return int(s)
    table = {"30分?60分": 45, "1H?1H30": 75, "1H30?2H": 105, "2H?": 120}
    return table.get(s)


def parse_number(s):
    """'2000㎡以上' → 2000 / '80' → 80 / 空欄 → None"""
    if pd.isna(s):
        return None
    s = unicodedata.normalize("NFKC", str(s))
    m = re.search(r"\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def clean_text(s):
    """'S R C' や '1 L D K' のような文字間の空白・全角を整える"""
    if pd.isna(s):
        return None
    return unicodedata.normalize("NFKC", str(s)).replace(" ", "").strip() or None


# ---- メイン処理 ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="不動産取引価格CSVの前処理")
    parser.add_argument("input", nargs="?", default=str(RAW_DIR),
                        help="CSVファイル、またはCSVだけを入れたフォルダ(省略時は data/raw)")
    args = parser.parse_args()
    src = Path(args.input).expanduser()
    if src.is_file():
        files = [src]
    elif src.is_dir():
        files = sorted(src.glob("*.csv"))
    else:
        raise SystemExit(f"{src} が見つかりません。パスを確認してください。")
    if not files:
        raise SystemExit(f"{src} にCSVがありません。先にダウンロードして置いてください。")

    df = pd.concat([read_csv_auto(f) for f in files], ignore_index=True)
    log(f"[1] 読み込み: {len(files)}ファイル / {len(df):,}行")
    log(f"    列名: {list(df.columns)}")

    # 列名の表記ゆれ: 実CSVでは「取引時期」(画面や資料では「取引時点」)
    if "取引時点" not in df.columns and "取引時期" in df.columns:
        df = df.rename(columns={"取引時期": "取引時点"})

    required = ["種類", "都道府県名", "市区町村名", "取引価格（総額）", "面積（㎡）",
                "建築年", "取引時点", "最寄駅：距離（分）"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"列が見つかりません: {missing}\n上の列名を見て、このスクリプトの列名を直してください。")

    # 成約価格情報が混ざっていたら除外(列の作りが違うため)
    if "価格情報区分" in df.columns:
        before = len(df)
        df = df[df["価格情報区分"].str.contains("取引価格", na=False)]
        log(f"[2a] 不動産取引価格情報のみに絞り込み: {before - len(df):,}行除外 → {len(df):,}行")

    # 対象を絞る: 中古マンション等 × 東京23区
    df = df[df["種類"].str.contains("マンション", na=False)]
    log(f"[2] 中古マンション等に絞り込み: {len(df):,}行")
    df = df[(df["都道府県名"] == "東京都") & (df["市区町村名"].str.endswith("区", na=False))]
    log(f"[3] 東京23区に絞り込み: {len(df):,}行")

    # 数値化
    df = df.copy()
    df["price"] = pd.to_numeric(df["取引価格（総額）"], errors="coerce")
    df["area"] = df["面積（㎡）"].map(parse_number)
    df["build_year"] = df["建築年"].map(parse_build_year)
    df["station_min"] = df["最寄駅：距離（分）"].map(parse_station_minutes)
    ym = df["取引時点"].map(parse_trade_period)
    df["trade_year"] = ym.map(lambda x: x[0])
    df["trade_quarter"] = ym.map(lambda x: x[1])
    for src, dst in [("建ぺい率（％）", "coverage_ratio"), ("容積率（％）", "floor_area_ratio")]:
        if src in df.columns:
            df[dst] = df[src].map(parse_number)
    for src, dst in [("建物の構造", "structure"), ("用途", "usage"), ("都市計画", "zoning"),
                     ("改装", "renovation"), ("間取り", "layout"), ("取引の事情等", "circumstances")]:
        if src in df.columns:
            df[dst] = df[src].map(clean_text)

    before = len(df)
    df = df.dropna(subset=["price", "area", "build_year", "trade_year"])
    log(f"[4] 価格・面積・築年・取引時点のいずれかが欠損の行を除外: {before - len(df):,}行除外 → {len(df):,}行")

    # 派生列
    df["age"] = df["trade_year"] - df["build_year"]
    df["price_per_m2"] = df["price"] / df["area"]

    # 明らかな異常値を除外(基準はあとで見直す前提。除外件数は必ず記録する)
    before = len(df)
    df = df[(df["age"] >= 0) & (df["area"] >= 10) & (df["area"] <= 500)]
    lo, hi = df["price_per_m2"].quantile([0.001, 0.999])
    df = df[df["price_per_m2"].between(lo, hi)]
    log(f"[5] 異常値(築年マイナス・面積10〜500㎡外・㎡単価の上下0.1%)を除外: {before - len(df):,}行除外 → {len(df):,}行")

    keep = ["市区町村名", "地区名", "最寄駅：名称", "station_min", "layout", "area", "build_year",
            "age", "structure", "usage", "zoning", "coverage_ratio", "floor_area_ratio",
            "renovation", "circumstances", "trade_year", "trade_quarter", "price", "price_per_m2"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].reset_index(drop=True)
    out.to_parquet(OUT_DIR / "mansion_tokyo23.parquet", index=False)
    log(f"[6] 保存: {OUT_DIR / 'mansion_tokyo23.parquet'} ({len(out):,}行)")

    (OUT_DIR / "cleaning_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
