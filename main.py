"""
주식 스크리닝 시스템 - GitHub Actions용 단일 파일 버전
==========================================================
가치(저평가) + 성장/모멘텀 + 배당/현금흐름 3팩터로 한국(KOSPI/KOSDAQ)과
미국(S&P500) 종목 순위를 매기고, 엑셀 리포트 + 모바일 대시보드(docs/index.html)를 생성합니다.

이 파일은 GitHub Actions에서 매일 자동 실행되는 것을 전제로, 관리 편의를 위해
config/scoring/kr/us/dashboard 로직을 하나의 파일로 합쳤습니다.
(PC에서 직접 개발/커스터마이징하려면 Claude가 별도로 드린 모듈형 버전을 사용하세요.)

설정을 바꾸려면 아래 CONFIG 섹션의 값만 수정하면 됩니다.
==========================================================
"""
import argparse
import datetime as dt
import html
import os
import shutil
import time

import numpy as np
import pandas as pd

try:
    from pykrx import stock as pykrx_stock
except ImportError:
    pykrx_stock = None

try:
    import FinanceDataReader as fdr
except ImportError:
    fdr = None

try:
    import yfinance as yf
except ImportError:
    yf = None


# ==========================================================
# CONFIG - 여기 값만 바꾸면 스크리닝 기준을 조정할 수 있습니다.
# ==========================================================
WEIGHTS = {"value": 0.4, "momentum": 0.3, "dividend": 0.3}

KR_MARKETS = ["KOSPI", "KOSDAQ"]
KR_MIN_MARKETCAP = 100_000_000_000     # 1,000억원
KR_EXCLUDE_KEYWORDS = ["스팩", "리츠", "우선주"]

US_UNIVERSE = "sp500"                  # "sp500" | "custom"
US_CUSTOM_TICKERS = []                 # 예: ["AAPL", "MSFT"]
US_MIN_MARKETCAP = 2_000_000_000       # 20억달러

PER_MAX = 60
PBR_MAX = 15
EXCLUDE_NEGATIVE_EARNINGS = True

TOP_N = 30
OUTPUT_DIR = "output"


# ==========================================================
# 공통 스코어링 로직
# ==========================================================
def percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    s = series.copy()
    valid = s.notna()
    result = pd.Series(np.nan, index=s.index)
    if valid.sum() == 0:
        return result
    pct = s[valid].rank(pct=True, ascending=not ascending)
    result[valid] = pct * 100
    return result


def composite_score(df: pd.DataFrame, components: dict) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)
    for col, (w, ascending) in components.items():
        if col not in df.columns:
            continue
        pr = percentile_rank(df[col], ascending=ascending)
        mask = pr.notna()
        score[mask] += pr[mask] * w
        weight_sum[mask] += w
    return score / weight_sum.replace(0, np.nan)


VALUE_COMPONENTS = {"PER": (0.5, True), "PBR": (0.5, True)}
MOMENTUM_COMPONENTS = {
    "수익률_3M": (0.3, False), "수익률_6M": (0.35, False), "수익률_12M": (0.35, False),
}
DIVIDEND_COMPONENTS = {"배당수익률": (1.0, False)}
US_MOMENTUM_EXTRA = {"매출성장률": (0.15, False), "이익성장률": (0.15, False)}
US_DIVIDEND_EXTRA = {"FCF수익률": (0.5, False)}


def _normalize_pct(series: pd.Series) -> pd.Series:
    if series is None:
        return series
    s = series.copy()
    mask = s.notna() & (s.abs() < 1)
    s[mask] = s[mask] * 100
    return s


def score_market(df: pd.DataFrame, extra_momentum=None, extra_dividend=None) -> pd.DataFrame:
    df = df.copy()
    value_comp = dict(VALUE_COMPONENTS)
    mom_comp = dict(MOMENTUM_COMPONENTS)
    if extra_momentum:
        mom_comp.update(extra_momentum)
    div_comp = dict(DIVIDEND_COMPONENTS)
    if extra_dividend:
        div_comp.update(extra_dividend)

    df["가치점수"] = composite_score(df, value_comp)
    df["모멘텀점수"] = composite_score(df, mom_comp)
    df["배당점수"] = composite_score(df, div_comp)

    wsum = sum(WEIGHTS.values()) or 1.0
    df["종합점수"] = (
        df["가치점수"].fillna(0) * WEIGHTS["value"]
        + df["모멘텀점수"].fillna(0) * WEIGHTS["momentum"]
        + df["배당점수"].fillna(0) * WEIGHTS["dividend"]
    ) / wsum
    return df


# ==========================================================
# 한국 시장 데이터 수집
# ==========================================================
def _latest_trading_day() -> str:
    d = dt.date.today() - dt.timedelta(days=1)
    for i in range(10):
        cand = d - dt.timedelta(days=i)
        if cand.weekday() < 5:
            return cand.strftime("%Y%m%d")
    return d.strftime("%Y%m%d")


def fetch_kr_fundamentals(markets=None) -> pd.DataFrame:
    if pykrx_stock is None:
        raise ImportError("pykrx가 설치되어 있지 않습니다.")
    markets = markets or KR_MARKETS
    base_date = dt.datetime.strptime(_latest_trading_day(), "%Y%m%d").date()

    last_err = None
    for offset in range(7):
        cand = base_date - dt.timedelta(days=offset)
        if cand.weekday() >= 5:
            continue
        date_str = cand.strftime("%Y%m%d")
        try:
            frames = []
            for m in markets:
                fundamental = pykrx_stock.get_market_fundamental(date_str, market=m)
                if fundamental is None or fundamental.empty:
                    raise ValueError(f"{date_str} {m} 데이터 없음(공휴일 또는 미확정)")
                cap = pykrx_stock.get_market_cap(date_str, market=m)
                df = fundamental.join(cap, how="inner")
                df["시장"] = m
                frames.append(df)
            result = pd.concat(frames)
            result.index.name = "티커"
            result = result.reset_index()
            names = {}
            for t in result["티커"]:
                try:
                    names[t] = pykrx_stock.get_market_ticker_name(t)
                except Exception:
                    names[t] = t
            result["종목명"] = result["티커"].map(names)
            print(f"[KR] {date_str} 기준 데이터 사용")
            return result
        except Exception as e:
            last_err = e
            print(f"[KR] {date_str} 수집 실패({e}) — 이전 거래일 재시도")
            continue

    raise RuntimeError(f"최근 7일 내 유효한 KR 데이터를 찾지 못했습니다: {last_err}")


def fetch_kr_momentum(tickers, lookback_days=380) -> pd.DataFrame:
    if fdr is None:
        raise ImportError("FinanceDataReader가 설치되어 있지 않습니다.")
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    rows = []
    for t in tickers:
        try:
            px = fdr.DataReader(t, start, end)["Close"].dropna()
            if len(px) < 20:
                continue
            last = px.iloc[-1]

            def ret(days):
                cutoff = px.index[-1] - pd.Timedelta(days=days)
                past = px[px.index <= cutoff]
                return float(last / past.iloc[-1] - 1) if len(past) else np.nan

            rows.append({"티커": t, "수익률_3M": ret(90), "수익률_6M": ret(180), "수익률_12M": ret(365)})
        except Exception:
            continue
        time.sleep(0.05)
    return pd.DataFrame(rows)


def build_kr_universe() -> pd.DataFrame:
    fundamentals = fetch_kr_fundamentals()
    df = fundamentals[fundamentals["시가총액"] >= KR_MIN_MARKETCAP].copy()
    for kw in KR_EXCLUDE_KEYWORDS:
        df = df[~df["종목명"].astype(str).str.contains(kw, na=False)]

    if EXCLUDE_NEGATIVE_EARNINGS:
        df.loc[df["PER"] <= 0, "PER"] = np.nan
    df.loc[df["PER"] > PER_MAX, "PER"] = np.nan
    df.loc[df["PBR"] > PBR_MAX, "PBR"] = np.nan
    df.loc[df["PBR"] <= 0, "PBR"] = np.nan

    momentum = fetch_kr_momentum(df["티커"].tolist())
    df = df.merge(momentum, on="티커", how="left")
    df["시장구분"] = "KR"
    return df


# ==========================================================
# 미국 시장 데이터 수집
# ==========================================================
def get_sp500_tickers() -> list:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()


def get_universe_tickers() -> list:
    if US_UNIVERSE == "custom":
        return US_CUSTOM_TICKERS
    return get_sp500_tickers()


def fetch_us_fundamentals(tickers) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance가 설치되어 있지 않습니다.")
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()
        except Exception:
            continue
        if not info:
            continue
        market_cap = info.get("marketCap")
        if market_cap is None or market_cap < US_MIN_MARKETCAP:
            continue
        fcf = info.get("freeCashflow")
        rows.append({
            "티커": t, "종목명": info.get("shortName", t), "섹터": info.get("sector"),
            "시가총액": market_cap, "PER": info.get("trailingPE"), "PBR": info.get("priceToBook"),
            "부채비율": info.get("debtToEquity"), "ROE": info.get("returnOnEquity"),
            "매출성장률": info.get("revenueGrowth"), "이익성장률": info.get("earningsGrowth"),
            "배당수익률": info.get("dividendYield"), "배당성향": info.get("payoutRatio"),
            "잉여현금흐름": fcf, "FCF수익률": (fcf / market_cap) if (fcf is not None and market_cap) else np.nan,
        })
        time.sleep(0.05)
    return pd.DataFrame(rows)


def fetch_us_momentum(tickers) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance가 설치되어 있지 않습니다.")
    rows = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="13mo")["Close"].dropna()
            if len(hist) < 20:
                continue
            last = hist.iloc[-1]

            def ret(days):
                cutoff = hist.index[-1] - pd.Timedelta(days=days)
                past = hist[hist.index <= cutoff]
                return float(last / past.iloc[-1] - 1) if len(past) else np.nan

            rows.append({"티커": t, "수익률_3M": ret(90), "수익률_6M": ret(180), "수익률_12M": ret(365)})
        except Exception:
            continue
    return pd.DataFrame(rows)


def build_us_universe() -> pd.DataFrame:
    tickers = get_universe_tickers()
    fundamentals = fetch_us_fundamentals(tickers)
    if fundamentals.empty:
        return fundamentals
    if EXCLUDE_NEGATIVE_EARNINGS:
        fundamentals.loc[fundamentals["PER"] <= 0, "PER"] = np.nan
    fundamentals.loc[fundamentals["PER"] > PER_MAX, "PER"] = np.nan
    fundamentals.loc[fundamentals["PBR"] > PBR_MAX, "PBR"] = np.nan
    momentum = fetch_us_momentum(fundamentals["티커"].tolist())
    df = fundamentals.merge(momentum, on="티커", how="left")
    df["시장구분"] = "US"
    return df


# ==========================================================
# 엑셀 리포트 출력
# ==========================================================
def export(df: pd.DataFrame) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(OUTPUT_DIR, f"stock_screening_{ts}.xlsx")

    display_cols = [c for c in [
        "전체순위", "시장구분", "티커", "종목명", "섹터", "시가총액",
        "PER", "PBR", "ROE", "부채비율", "배당수익률", "FCF수익률",
        "수익률_3M", "수익률_6M", "수익률_12M", "매출성장률", "이익성장률",
        "가치점수", "모멘텀점수", "배당점수", "종합점수",
    ] if c in df.columns]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df[display_cols].head(TOP_N * 3).to_excel(writer, sheet_name="종합TOP", index=False)
        df.sort_values("가치점수", ascending=False)[display_cols].head(TOP_N).to_excel(writer, sheet_name="가치주TOP", index=False)
        df.sort_values("모멘텀점수", ascending=False)[display_cols].head(TOP_N).to_excel(writer, sheet_name="성장모멘텀TOP", index=False)
        df.sort_values("배당점수", ascending=False)[display_cols].head(TOP_N).to_excel(writer, sheet_name="배당TOP", index=False)
        if (df["시장구분"] == "KR").any():
            df[df["시장구분"] == "KR"][display_cols].to_excel(writer, sheet_name="KR_전체", index=False)
        if (df["시장구분"] == "US").any():
            df[df["시장구분"] == "US"][display_cols].to_excel(writer, sheet_name="US_전체", index=False)
        meta = pd.DataFrame({
            "항목": ["실행일시", "가치 가중치", "모멘텀 가중치", "배당 가중치", "PER 상한", "PBR 상한"],
            "값": [ts, WEIGHTS["value"], WEIGHTS["momentum"], WEIGHTS["dividend"], PER_MAX, PBR_MAX],
        })
        meta.to_excel(writer, sheet_name="실행설정", index=False)

    print(f"완료: {path}")
    return path


# ==========================================================
# 모바일 대시보드(docs/index.html) 생성
# ==========================================================
def _fmt(v, digits=1):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:.{digits}f}"


def _fmt_pct(v, digits=1):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v*100:+.{digits}f}%"


def _card(row) -> str:
    name = html.escape(str(row.get("종목명", "")))
    ticker = html.escape(str(row.get("티커", "")))
    flag = "\U0001F1F0\U0001F1F7" if row.get("시장구분") == "KR" else "\U0001F1FA\U0001F1F8"
    per, pbr, div = _fmt(row.get("PER")), _fmt(row.get("PBR")), _fmt(row.get("배당수익률"))
    m12, score = _fmt_pct(row.get("수익률_12M")), _fmt(row.get("종합점수"))
    return (f'<div class="card"><div class="card-top"><span class="flag">{flag}</span>'
            f'<span class="name">{name}</span><span class="ticker">{ticker}</span>'
            f'<span class="score">{score}</span></div><div class="metrics">'
            f'<span>PER {per}</span><span>PBR {pbr}</span><span>배당 {div}%</span>'
            f'<span>12개월 {m12}</span></div></div>')


def _section(anchor, title, df, sort_col, n) -> str:
    if sort_col not in df.columns or df.empty:
        rows_html = '<p class="empty">데이터 없음</p>'
    else:
        rows_html = "".join(_card(r) for _, r in df.sort_values(sort_col, ascending=False).head(n).iterrows())
    return f'<section id="{anchor}"><h2>{html.escape(title)}</h2><div class="cards">{rows_html}</div></section>'


def build_dashboard(df: pd.DataFrame, out_path: str = "docs/index.html") -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    w = WEIGHTS
    total = _section("top", "\U0001F3C6 종합 TOP 20", df, "종합점수", 20)
    value = _section("value", "\U0001F4B0 가치주 TOP 15", df, "가치점수", 15)
    mom = _section("momentum", "\U0001F680 성장·모멘텀 TOP 15", df, "모멘텀점수", 15)
    div_ = _section("dividend", "\U0001F4B5 배당·현금흐름 TOP 15", df, "배당점수", 15)

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta name="theme-color" content="#0b0f14">
<title>주식 스크리닝 리포트</title>
<style>
  :root {{ --bg:#0b0f14; --card:#151b22; --text:#e6edf3; --muted:#8b98a5; --accent:#4f8cff; --border:#232b33; }}
  * {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; -webkit-font-smoothing:antialiased; }}
  header {{ padding:18px 16px 10px; position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--border); z-index:10; }}
  header h1 {{ margin:0 0 4px; font-size:19px; }}
  header p {{ margin:0; color:var(--muted); font-size:12.5px; line-height:1.5; }}
  nav {{ display:flex; gap:8px; overflow-x:auto; padding:10px 16px; -webkit-overflow-scrolling:touch; }}
  nav a {{ flex:0 0 auto; padding:7px 14px; background:var(--card); border:1px solid var(--border); border-radius:999px; color:var(--text); text-decoration:none; font-size:13px; }}
  main {{ padding:0 16px 30px; max-width:640px; margin:0 auto; }}
  section {{ margin-top:22px; scroll-margin-top:110px; }}
  section h2 {{ font-size:15.5px; margin:0 0 10px; }}
  .cards {{ display:flex; flex-direction:column; gap:8px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 14px; }}
  .card-top {{ display:flex; align-items:center; gap:6px; }}
  .flag {{ font-size:15px; }}
  .name {{ font-weight:600; font-size:14px; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .ticker {{ color:var(--muted); font-size:11.5px; }}
  .score {{ background:var(--accent); color:#fff; font-size:12px; font-weight:700; padding:2px 9px; border-radius:999px; }}
  .metrics {{ margin-top:6px; display:flex; gap:10px; flex-wrap:wrap; color:var(--muted); font-size:12px; }}
  .empty {{ color:var(--muted); font-size:13px; }}
  footer {{ text-align:center; color:var(--muted); font-size:11px; padding:14px 16px 36px; line-height:1.7; }}
  a.dl {{ color:var(--accent); }}
</style>
</head>
<body>
<header>
  <h1>\U0001F4CA 주식 스크리닝 리포트</h1>
  <p>업데이트: {ts} (KST) · 가치 {w['value']*100:.0f}% · 모멘텀 {w['momentum']*100:.0f}% · 배당 {w['dividend']*100:.0f}%<br>
  <a class="dl" href="reports/latest.xlsx">엑셀 전체 데이터 다운로드</a></p>
</header>
<nav><a href="#top">종합</a><a href="#value">가치주</a><a href="#momentum">모멘텀</a><a href="#dividend">배당</a></nav>
<main>{total}{value}{mom}{div_}</main>
<footer>본 리포트는 투자자문이 아닙니다. 공개 데이터를 기계적으로 점수화한 참고 자료이며,<br>
투자 판단과 책임은 본인에게 있습니다. 매일 자동 갱신됩니다 (GitHub Actions).</footer>
</body>
</html>"""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


# ==========================================================
# 실행 진입점
# ==========================================================
def run(kr: bool = True, us: bool = True) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = []

    if kr:
        print("[KR] 데이터 수집 중...")
        kr_df = build_kr_universe().rename(columns={"DIV": "배당수익률"})
        if "배당수익률" in kr_df:
            kr_df["배당수익률"] = _normalize_pct(kr_df["배당수익률"])
        kr_df = score_market(kr_df)
        frames.append(kr_df)
        print(f"[KR] {len(kr_df)}개 종목 처리 완료")

    if us:
        print("[US] 데이터 수집 중...")
        us_df = build_us_universe()
        if not us_df.empty:
            if "배당수익률" in us_df:
                us_df["배당수익률"] = _normalize_pct(us_df["배당수익률"])
            us_df = score_market(us_df, extra_momentum=US_MOMENTUM_EXTRA, extra_dividend=US_DIVIDEND_EXTRA)
            frames.append(us_df)
            print(f"[US] {len(us_df)}개 종목 처리 완료")

    if not frames:
        print("수집된 데이터가 없습니다.")
        return ""

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["전체순위"] = combined["종합점수"].rank(ascending=False, method="min")
    combined = combined.sort_values("종합점수", ascending=False)

    xlsx_path = export(combined)
    build_dashboard(combined, out_path="docs/index.html")
    os.makedirs("docs/reports", exist_ok=True)
    if xlsx_path:
        shutil.copyfile(xlsx_path, os.path.join("docs", "reports", "latest.xlsx"))
    return xlsx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="가치/모멘텀/배당 3팩터 주식 스크리닝")
    parser.add_argument("--kr-only", action="store_true")
    parser.add_argument("--us-only", action="store_true")
    args = parser.parse_args()
    run(kr=not args.us_only, us=not args.kr_only)
