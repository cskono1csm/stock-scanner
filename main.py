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
import concurrent.futures
import datetime as dt
import html
import json
import os
import re
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

import requests
from bs4 import BeautifulSoup

_orig_request = requests.sessions.Session.request
def _patched_request(self, method, url, **kwargs):
    headers = kwargs.get("headers") or {}
    headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    )
    headers.setdefault("Referer", "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd")
    kwargs["headers"] = headers
    resp = _orig_request(self, method, url, **kwargs)
    if "data.krx.co.kr" in url:
        print(f"[DEBUG-KRX] {method} {url}")
        print(f"[DEBUG-KRX] status={resp.status_code} content-type={resp.headers.get('Content-Type')} len={len(resp.content)}")
        print(f"[DEBUG-KRX] body_head={resp.text[:300]!r}")
    return resp
requests.sessions.Session.request = _patched_request


# ==========================================================
# CONFIG - 여기 값만 바꾸면 스크리닝 기준을 조정할 수 있습니다.
# ==========================================================
WEIGHTS = {"value": 0.4, "momentum": 0.3, "dividend": 0.3}

KR_MARKETS = ["KOSPI", "KOSDAQ"]
KR_MIN_MARKETCAP = 100_000_000_000     # 1,000억원
# 종목명에 아래 문자열이 "포함"되면 제외(스팩/리츠). 우선주는 이름에 "우선주"라는 단어가 실제로
# 붙어 나오지 않고(GS우, 삼성전자우, 현대차2우B 처럼 "우"로 끝나는 표기이므로) 별도 함수
# (_is_kr_preferred_stock)로 코드 규칙 + 이름 패턴을 함께 봐서 걸러냅니다.
KR_EXCLUDE_KEYWORDS = ["스팩", "리츠"]

# 한국 데이터 소스: "yfinance"(기본) | "pykrx"
# - pykrx는 data.krx.co.kr를 스크래핑하는데, 클라우드/해외 IP(GitHub Actions 러너 포함) 차단 +
#   최근 버전은 일부 호출에 KRX 로그인(KRX_ID/KRX_PW)까지 요구하기 시작해 사실상 사용 불가 상태(2026.08).
#   문제가 해결되면 이 값을 "pykrx"로 되돌리면 기존 로직을 그대로 다시 쓸 수 있습니다.
# - "yfinance"는 티커에 .KS(코스피)/.KQ(코스닥) 접미사를 붙여 조회하며, 이미 미국 종목 수집에 쓰고
#   있는 것과 같은 경로라 클라우드 IP 차단 이슈가 없습니다. 다만 종목 리스트 자체는 여전히
#   FinanceDataReader(fdr.StockListing)로 가져오므로, 이쪽이 막히면 이 경로도 영향을 받습니다.
KR_DATA_SOURCE = "yfinance"

US_UNIVERSE = "sp500"                  # "sp500" | "custom"
US_CUSTOM_TICKERS = []                 # 예: ["AAPL", "MSFT"]
US_MIN_MARKETCAP = 2_000_000_000       # 20억달러

PER_MAX = 60
PBR_MAX = 15
EXCLUDE_NEGATIVE_EARNINGS = True
DIVIDEND_YIELD_MAX = 20        # 배당수익률 상한(%) — 이보다 크면 데이터 이상치로 보고 제외

# 적정주가(상대가치평가) 추정 설정
# - PER/PBR/EV-EBITDA 각 방식으로 "동종그룹(섹터 또는 시장) 중앙값 배수"를 적용해
#   적정주가를 추정한 뒤, 아래 가중치로 가중평균합니다.
# - 미래 현금흐름을 직접 추정하는 DCF가 아니라 "동종업종 대비 몇 배가 합리적인가"를
#   보는 상대가치평가이므로, 업종 전체가 고평가/저평가된 국면에서는 왜곡될 수 있습니다.
FAIR_VALUE_WEIGHTS = {"PER": 0.4, "PBR": 0.3, "EV_EBITDA": 0.3}
FAIR_VALUE_MIN_PEER_GROUP = 5   # 섹터/시장 내 표본이 이 수 미만이면 상위 그룹 중앙값으로 대체
FAIR_VALUE_MAX_UPSIDE = 3.0     # 괴리율 표시 상한(+300%) — 그 이상은 저신뢰로 간주해 클리핑

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
    s[s.notna() & ((s < 0) | (s > DIVIDEND_YIELD_MAX))] = np.nan
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
# 적정주가(상대가치평가) 추정 로직
# ==========================================================
def _peer_group_median(df: pd.DataFrame, group_cols: list, value_col: str) -> pd.Series:
    """그룹(섹터 등) 중앙값을 각 행에 매핑합니다.
    그룹 표본 수가 FAIR_VALUE_MIN_PEER_GROUP 미만이면 전체 시장 중앙값으로 대체합니다."""
    if value_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    market_median = df[value_col].median(skipna=True)
    valid_cols = [c for c in group_cols if c in df.columns]
    if not valid_cols:
        return pd.Series(market_median, index=df.index)
    grouped = df.groupby(valid_cols)[value_col]
    group_median = grouped.transform("median")
    group_size = grouped.transform("count")
    return group_median.where(group_size >= FAIR_VALUE_MIN_PEER_GROUP, market_median)


def estimate_fair_value(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """PER/PBR/EV-EBITDA 동종그룹(섹터·시장) 중앙값 배수를 이용한 상대가치평가 기반
    적정주가를 추정해 다음 컬럼을 추가합니다.
      - PER_적정주가 / PBR_적정주가 / EV_EBITDA_적정주가: 방식별 추정치
      - 적정주가: 이용 가능한 방식들의 FAIR_VALUE_WEIGHTS 가중평균
      - 괴리율: (적정주가 - 현재주가) / 현재주가  (+면 저평가, -면 고평가 추정)
      - 적정주가_모델수: 이번 추정에 실제로 반영된 방식 개수(신뢰도 참고용)
    데이터가 없는 방식은 자동으로 제외되고 남은 방식들끼리 가중치를 재정규화합니다.
    """
    df = df.copy()

    # 1) PER 기반: 동종그룹 PER 중앙값 x 자사 EPS
    if {"PER", "EPS"}.issubset(df.columns):
        peer_per = _peer_group_median(df, group_cols, "PER")
        eps = df["EPS"].where(df["EPS"] > 0)
        df["PER_적정주가"] = peer_per * eps
    else:
        df["PER_적정주가"] = np.nan

    # 2) PBR 기반: 동종그룹 PBR 중앙값 x 자사 BPS(주당순자산)
    if {"PBR", "BPS"}.issubset(df.columns):
        peer_pbr = _peer_group_median(df, group_cols, "PBR")
        bps = df["BPS"].where(df["BPS"] > 0)
        df["PBR_적정주가"] = peer_pbr * bps
    else:
        df["PBR_적정주가"] = np.nan

    # 3) EV/EBITDA 기반: 동종그룹 EV/EBITDA 배수 중앙값 x 자사 EBITDA -> 적정 기업가치
    #    적정주가 = (적정 기업가치 - 순부채) / 발행주식수
    needed = {"EBITDA", "기업가치", "순부채", "발행주식수"}
    if needed.issubset(df.columns):
        ebitda = df["EBITDA"].where(df["EBITDA"] > 0)
        own_multiple = (df["기업가치"] / ebitda).where(df["기업가치"] > 0)
        tmp = df.assign(_ev_ebitda_배수=own_multiple)
        peer_multiple = _peer_group_median(tmp, group_cols, "_ev_ebitda_배수")
        implied_ev = peer_multiple * ebitda
        shares = df["발행주식수"].where(df["발행주식수"] > 0)
        df["EV_EBITDA_적정주가"] = (implied_ev - df["순부채"]) / shares
    else:
        df["EV_EBITDA_적정주가"] = np.nan

    # 4) 가중평균 결합 (방식별 데이터 없으면 자동 제외 후 재정규화)
    method_cols = {"PER": "PER_적정주가", "PBR": "PBR_적정주가", "EV_EBITDA": "EV_EBITDA_적정주가"}
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_total = pd.Series(0.0, index=df.index)
    model_count = pd.Series(0, index=df.index)
    for key, col in method_cols.items():
        w = FAIR_VALUE_WEIGHTS.get(key, 0)
        est = pd.to_numeric(df[col], errors="coerce")
        valid = est.notna() & (est > 0) & np.isfinite(est)
        weighted_sum[valid] += est[valid] * w
        weight_total[valid] += w
        model_count[valid] += 1

    df["적정주가"] = weighted_sum / weight_total.replace(0, np.nan)
    df["적정주가_모델수"] = model_count

    if "현재주가" in df.columns:
        cur = df["현재주가"].where(df["현재주가"] > 0)
        df["괴리율"] = (df["적정주가"] - cur) / cur
        df["괴리율"] = df["괴리율"].clip(lower=-FAIR_VALUE_MAX_UPSIDE, upper=FAIR_VALUE_MAX_UPSIDE)
    else:
        df["괴리율"] = np.nan

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


def fetch_kr_fundamentals_pykrx(markets=None) -> pd.DataFrame:
    """[레거시/현재 미사용] KRX 공식 웹 엔드포인트를 pykrx로 스크래핑. 클라우드 IP 차단 및
    최근 버전의 KRX 로그인 요구 문제로 2026.08 기준 GitHub Actions에서 동작하지 않음.
    KR_DATA_SOURCE = "pykrx"로 되돌리면 다시 사용됩니다."""
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
            print(f"[KR] {date_str} 기준 데이터 사용(pykrx)")
            return result
        except Exception as e:
            last_err = e
            print(f"[KR] {date_str} 수집 실패({e}) — 이전 거래일 재시도")
            continue

    raise RuntimeError(f"최근 7일 내 유효한 KR 데이터를 찾지 못했습니다(pykrx): {last_err}")


def get_kr_ticker_universe() -> pd.DataFrame:
    """FinanceDataReader로 KOSPI/KOSDAQ 종목 리스트(티커·종목명·시가총액)를 가져옵니다.
    시가총액 컬럼이 있으면 KR_MIN_MARKETCAP 미만 종목을 미리 걸러내 이후 yfinance 조회 건수를
    (전종목 약 2,000~2,500개 -> 대형주 위주 수백 개로) 줄입니다."""
    if fdr is None:
        raise ImportError("FinanceDataReader가 설치되어 있지 않습니다.")
    frames = []
    for m in KR_MARKETS:
        try:
            listing = fdr.StockListing(m)
        except Exception as e:
            print(f"[KR] {m} 종목 리스트 수집 실패: {e}")
            continue
        if listing is None or listing.empty:
            continue
        code_col = next((c for c in ["Code", "Symbol"] if c in listing.columns), None)
        name_col = next((c for c in ["Name"] if c in listing.columns), None)
        cap_col = next((c for c in ["Marcap", "MarketCap"] if c in listing.columns), None)
        if code_col is None or name_col is None:
            print(f"[KR] {m} 종목 리스트 컬럼 인식 실패: {list(listing.columns)}")
            continue
        keep = [code_col, name_col] + ([cap_col] if cap_col else [])
        df = listing[keep].rename(columns={code_col: "티커", name_col: "종목명", **({cap_col: "시가총액_참고"} if cap_col else {})})
        df["시장"] = m
        frames.append(df)

    if not frames:
        raise RuntimeError("KOSPI/KOSDAQ 종목 리스트를 하나도 가져오지 못했습니다(FinanceDataReader).")
    combined = pd.concat(frames, ignore_index=True)
    if "시가총액_참고" in combined.columns:
        before = len(combined)
        combined = combined[pd.to_numeric(combined["시가총액_참고"], errors="coerce") >= KR_MIN_MARKETCAP]
        print(f"[KR] 종목 리스트 {before}개 -> 시가총액 사전 필터 후 {len(combined)}개")
    return combined.reset_index(drop=True)


_naver_debug_done = False


def _fetch_naver_valuation(code: str) -> dict:
    """yfinance가 한국 종목에는 PER/PBR/EPS/BPS를 거의 채워주지 못해(Yahoo의 KRX 커버리지 한계),
    네이버금융 종목 페이지(finance.naver.com/item/main.naver)에서 이 4개 값만 보강 수집합니다.
    처음 1건만 진단 로그를 남겨, 혹시 네이버 페이지 구조가 바뀌어도 GitHub Actions 로그에서
    바로 원인을 확인할 수 있게 합니다."""
    global _naver_debug_done
    result = {"PER": np.nan, "PBR": np.nan, "EPS": np.nan, "BPS": np.nan}
    try:
        resp = requests.get(
            f"https://finance.naver.com/item/main.naver?code={code}",
            headers={"Referer": "https://finance.naver.com/"},
            timeout=10,
        )
        soup = BeautifulSoup(resp.content, "html.parser")

        def _num(elem_id):
            el = soup.find("em", id=elem_id)
            if not el:
                return np.nan
            txt = el.get_text(strip=True).replace(",", "")
            try:
                return float(txt)
            except ValueError:
                return np.nan

        result["PER"] = _num("_per")
        result["PBR"] = _num("_pbr")
        result["EPS"] = _num("_eps")
        result["BPS"] = _num("_bps")

        if not _naver_debug_done:
            _naver_debug_done = True
            print(f"[DEBUG-NAVER] {code} status={resp.status_code} 결과={result}")
            if pd.isna(result["PER"]):
                print(f"[DEBUG-NAVER] _per 태그를 못 찾음. 페이지 앞부분={resp.text[:800]!r}")
    except Exception as e:
        if not _naver_debug_done:
            _naver_debug_done = True
            print(f"[DEBUG-NAVER] {code} 요청 실패: {e}")
    return result


def _fetch_naver_valuations_parallel(codes, max_workers=10) -> dict:
    """여러 종목의 네이버금융 PER/PBR/EPS/BPS를 동시에(병렬) 수집합니다.
    종목당 순차 요청은 1,300여 개 종목 기준으로 수십 분이 추가로 걸려
    GitHub Actions의 job timeout(45분)을 넘겨 실행이 취소되는 문제가 있었습니다.
    스레드풀로 동시에 요청해 전체 소요 시간을 크게 줄입니다."""
    results = {}
    codes = list(dict.fromkeys(str(c) for c in codes))  # 중복 제거, 순서 보존
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_fetch_naver_valuation, code): code for code in codes}
        for future in concurrent.futures.as_completed(future_map):
            code = future_map[future]
            try:
                results[code] = future.result()
            except Exception:
                results[code] = {"PER": np.nan, "PBR": np.nan, "EPS": np.nan, "BPS": np.nan}
    return results


def fetch_kr_fundamentals_yfinance(markets=None) -> pd.DataFrame:
    """yfinance로 .KS(코스피)/.KQ(코스닥) 접미사를 붙여 한국 종목 펀더멘털을 수집합니다.
    미국 종목 수집(fetch_us_fundamentals)과 동일한 경로/필드를 사용합니다."""
    if yf is None:
        raise ImportError("yfinance가 설치되어 있지 않습니다.")
    markets = markets or KR_MARKETS
    universe = get_kr_ticker_universe()
    universe = universe[universe["시장"].isin(markets)]
    suffix = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}

    # yfinance는 한국 종목에는 PER/PBR/EPS/BPS를 거의 채워주지 못해(Yahoo의 KRX 커버리지 한계),
    # 네이버금융에서 이 4개 값만 보강 수집합니다. 종목 수가 많아(1,000개 이상) 순차 요청 시
    # 실행 시간이 크게 늘어나므로, 아래에서 한 번에 병렬로 미리 수집해둡니다.
    print(f"[KR] 네이버금융 PER/PBR/EPS/BPS 병렬 수집 시작 ({len(universe)}개 종목)")
    naver_map = _fetch_naver_valuations_parallel(universe["티커"].tolist())
    print(f"[KR] 네이버금융 수집 완료 ({len(naver_map)}개)")

    empty_naver = {"PER": np.nan, "PBR": np.nan, "EPS": np.nan, "BPS": np.nan}
    rows = []
    for _, r in universe.iterrows():
        code, name, market = str(r["티커"]), r["종목명"], r["시장"]
        yf_ticker = f"{code}{suffix.get(market, '.KS')}"
        try:
            info = yf.Ticker(yf_ticker).get_info()
        except Exception:
            continue
        if not info:
            continue
        market_cap = info.get("marketCap")
        shares = info.get("sharesOutstanding")
        fcf = info.get("freeCashflow")
        total_debt = info.get("totalDebt")
        total_cash = info.get("totalCash")
        net_debt = (total_debt - total_cash) if (total_debt is not None and total_cash is not None) else np.nan
        naver_val = naver_map.get(code, empty_naver)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        per = naver_val["PER"] if pd.notna(naver_val["PER"]) else info.get("trailingPE")
        pbr = naver_val["PBR"] if pd.notna(naver_val["PBR"]) else info.get("priceToBook")
        eps = naver_val["EPS"] if pd.notna(naver_val["EPS"]) else info.get("trailingEps")
        bps = naver_val["BPS"] if pd.notna(naver_val["BPS"]) else info.get("bookValue")
        # 네이버금융 페이지에서 BPS만 별도 <em> 태그로 안 잡히는 경우가 있어(구조 차이 추정),
        # PBR = 현재가/BPS 관계를 거꾸로 이용해 현재가·PBR로 BPS를 역산하는 보강 로직을 둡니다.
        if pd.isna(bps) and price and pbr not in (None, 0) and pd.notna(pbr):
            bps = price / pbr
        rows.append({
            "티커": code, "종목명": name, "시장": market, "섹터": info.get("sector"),
            "시가총액": market_cap, "상장주식수": shares, "발행주식수": shares,
            "PER": per, "PBR": pbr,
            "EPS": eps, "BPS": bps,
            "DIV": info.get("dividendYield"), "배당성향": info.get("payoutRatio"),
            "부채비율": info.get("debtToEquity"), "ROE": info.get("returnOnEquity"),
            "매출성장률": info.get("revenueGrowth"), "이익성장률": info.get("earningsGrowth"),
            "잉여현금흐름": fcf, "FCF수익률": (fcf / market_cap) if (fcf is not None and market_cap) else np.nan,
            "EBITDA": info.get("ebitda"), "기업가치": info.get("enterpriseValue"),
            "순부채": net_debt,
            "현재주가": price,
        })
        time.sleep(0.05)

    if not rows:
        raise RuntimeError("yfinance로 KR 펀더멘털을 하나도 가져오지 못했습니다.")
    print(f"[KR] yfinance로 {len(rows)}개 종목 펀더멘털 수집 완료")
    return pd.DataFrame(rows)


def fetch_kr_fundamentals(markets=None) -> pd.DataFrame:
    if KR_DATA_SOURCE == "pykrx":
        return fetch_kr_fundamentals_pykrx(markets=markets)
    return fetch_kr_fundamentals_yfinance(markets=markets)


def fetch_kr_momentum(tickers, lookback_days=380):
    """모멘텀(3/6/12개월 수익률)과 함께, 이미 받아온 일별 시세에서 공짜로 뽑을 수 있는
    "가장 최근 거래일"의 거래량/거래대금(+그 날짜), 그리고 관심종목 추적(과거 날짜 지정 시
    그날의 실제 종가를 기준가로 쓰기 위한) 일별 종가 이력을 함께 수집합니다. 거래대금은
    여러 날짜를 평균 내지 않고 데이터가 실제로 존재하는 가장 최근 거래일 하루치 값을 그대로
    쓰고, 그 기준 날짜를 함께 저장해 화면에 명시합니다(평균을 쓰면 "오늘 기준" 감각과 어긋나
    혼란을 줄 수 있어 단일 최근일 값으로 통일). 추가 네트워크 호출 없이 기존에 이미 하던 시세
    조회 결과를 재활용하는 것이라 실행 시간에 미치는 영향은 없습니다."""
    if fdr is None:
        raise ImportError("FinanceDataReader가 설치되어 있지 않습니다.")
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    rows = []
    price_history = {}
    for t in tickers:
        try:
            hist_df = fdr.DataReader(t, start, end)
            px = hist_df["Close"].dropna()
            if len(px) < 20:
                continue
            last = px.iloc[-1]

            def ret(days):
                cutoff = px.index[-1] - pd.Timedelta(days=days)
                past = px[px.index <= cutoff]
                return float(last / past.iloc[-1] - 1) if len(past) else np.nan

            vol = hist_df["Volume"].dropna() if "Volume" in hist_df.columns else pd.Series(dtype=float)
            last_volume = float(vol.iloc[-1]) if len(vol) else np.nan
            last_value = float(last * last_volume) if len(vol) and pd.notna(last_volume) else np.nan
            last_trade_date = px.index[-1].strftime("%Y-%m-%d")

            rows.append({
                "티커": t, "수익률_3M": ret(90), "수익률_6M": ret(180), "수익률_12M": ret(365),
                "거래량": last_volume, "거래대금": last_value, "거래기준일": last_trade_date,
            })
            price_history[f"KR_{t}"] = {
                idx.strftime("%Y-%m-%d"): round(float(v), 2) for idx, v in px.items()
            }
        except Exception:
            continue
        time.sleep(0.05)
    return pd.DataFrame(rows), price_history


_PREFERRED_NAME_RE = re.compile(r"\d?우[A-Z]?$")


def _is_kr_preferred_stock(code, name) -> bool:
    """한국 우선주 여부를 판별합니다.
    우선주는 종목명에 "우선주"라는 단어가 그대로 붙어있지 않고 GS우/삼성전자우/현대차2우B 처럼
    이름이 "우"(+시리즈 숫자/구분 알파벳)로 끝나는 표기를 씁니다. 다만 "미래에셋대우"처럼
    이름이 우연히 "우"로 끝나는 보통주도 있어 이름만으로는 오탐이 날 수 있습니다.
    이를 보완하기 위해 KRX 종목코드 관행(보통주는 뒷자리가 "0"으로 끝나고, 우선주 등 특수 종류
    주식은 그 외 숫자/문자로 끝남)을 함께 확인해, 두 조건이 동시에 맞을 때만 우선주로 판단합니다."""
    code = str(code).strip()
    name = str(name).strip()
    if not code or code[-1] == "0":
        return False
    return bool(_PREFERRED_NAME_RE.search(name))


def build_kr_universe():
    fundamentals = fetch_kr_fundamentals()
    df = fundamentals[fundamentals["시가총액"] >= KR_MIN_MARKETCAP].copy()
    for kw in KR_EXCLUDE_KEYWORDS:
        df = df[~df["종목명"].astype(str).str.contains(kw, na=False)]
    is_preferred = df.apply(lambda r: _is_kr_preferred_stock(r["티커"], r["종목명"]), axis=1)
    if is_preferred.any():
        print(f"[KR] 우선주로 판단되어 제외: {df.loc[is_preferred, '종목명'].tolist()}")
    df = df[~is_preferred]

    if EXCLUDE_NEGATIVE_EARNINGS:
        df.loc[df["PER"] <= 0, "PER"] = np.nan
    df.loc[df["PER"] > PER_MAX, "PER"] = np.nan
    df.loc[df["PBR"] > PBR_MAX, "PBR"] = np.nan
    df.loc[df["PBR"] <= 0, "PBR"] = np.nan

    momentum, price_history = fetch_kr_momentum(df["티커"].tolist())
    df = df.merge(momentum, on="티커", how="left")
    df["시장구분"] = "KR"

    if {"시가총액", "상장주식수"}.issubset(df.columns):
        shares = df["상장주식수"].where(df["상장주식수"] > 0)
        derived_price = df["시가총액"] / shares
        # yfinance 등에서 이미 현재주가를 받아온 경우 그 값을 우선 쓰고, 없을 때만 역산값으로 채움
        df["현재주가"] = df["현재주가"].fillna(derived_price) if "현재주가" in df.columns else derived_price
    df = estimate_fair_value(df, group_cols=["시장"])
    return df, price_history


# ==========================================================
# 미국 시장 데이터 수집
# ==========================================================
def get_sp500_tickers() -> list:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
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
        total_debt = info.get("totalDebt")
        total_cash = info.get("totalCash")
        net_debt = (total_debt - total_cash) if (total_debt is not None and total_cash is not None) else np.nan
        rows.append({
            "티커": t, "종목명": info.get("shortName", t), "섹터": info.get("sector"),
            "시가총액": market_cap, "PER": info.get("trailingPE"), "PBR": info.get("priceToBook"),
            "부채비율": info.get("debtToEquity"), "ROE": info.get("returnOnEquity"),
            "매출성장률": info.get("revenueGrowth"), "이익성장률": info.get("earningsGrowth"),
            "배당수익률": info.get("dividendYield"), "배당성향": info.get("payoutRatio"),
            "잉여현금흐름": fcf, "FCF수익률": (fcf / market_cap) if (fcf is not None and market_cap) else np.nan,
            # 적정주가(상대가치평가) 추정에 사용
            "현재주가": info.get("currentPrice") or info.get("regularMarketPrice"),
            "EPS": info.get("trailingEps"), "BPS": info.get("bookValue"),
            "EBITDA": info.get("ebitda"), "기업가치": info.get("enterpriseValue"),
            "순부채": net_debt, "발행주식수": info.get("sharesOutstanding"),
        })
        time.sleep(0.05)
    return pd.DataFrame(rows)


def fetch_us_momentum(tickers):
    """모멘텀과 함께, 이미 받아온 일별 시세에서 "가장 최근 거래일"의 거래량/거래대금(+그 날짜)과
    관심종목 추적용 일별 종가 이력을 추가 네트워크 호출 없이 함께 수집합니다. 거래대금은 평균이
    아니라 데이터가 존재하는 가장 최근 거래일 하루치 값을 그대로 쓰고 기준 날짜를 함께 기록합니다."""
    if yf is None:
        raise ImportError("yfinance가 설치되어 있지 않습니다.")
    rows = []
    price_history = {}
    for t in tickers:
        try:
            hist_df = yf.Ticker(t).history(period="13mo")
            hist = hist_df["Close"].dropna()
            if len(hist) < 20:
                continue
            last = hist.iloc[-1]

            def ret(days):
                cutoff = hist.index[-1] - pd.Timedelta(days=days)
                past = hist[hist.index <= cutoff]
                return float(last / past.iloc[-1] - 1) if len(past) else np.nan

            vol = hist_df["Volume"].dropna() if "Volume" in hist_df.columns else pd.Series(dtype=float)
            last_volume = float(vol.iloc[-1]) if len(vol) else np.nan
            last_value = float(last * last_volume) if len(vol) and pd.notna(last_volume) else np.nan
            last_trade_date = hist.index[-1].strftime("%Y-%m-%d")

            rows.append({
                "티커": t, "수익률_3M": ret(90), "수익률_6M": ret(180), "수익률_12M": ret(365),
                "거래량": last_volume, "거래대금": last_value, "거래기준일": last_trade_date,
            })
            price_history[f"US_{t}"] = {
                idx.strftime("%Y-%m-%d"): round(float(v), 2) for idx, v in hist.items()
            }
        except Exception:
            continue
    return pd.DataFrame(rows), price_history


def build_us_universe():
    tickers = get_universe_tickers()
    fundamentals = fetch_us_fundamentals(tickers)
    if fundamentals.empty:
        return fundamentals, {}
    if EXCLUDE_NEGATIVE_EARNINGS:
        fundamentals.loc[fundamentals["PER"] <= 0, "PER"] = np.nan
    fundamentals.loc[fundamentals["PER"] > PER_MAX, "PER"] = np.nan
    fundamentals.loc[fundamentals["PBR"] > PBR_MAX, "PBR"] = np.nan
    fundamentals.loc[fundamentals["PBR"] <= 0, "PBR"] = np.nan
    momentum, price_history = fetch_us_momentum(fundamentals["티커"].tolist())
    df = fundamentals.merge(momentum, on="티커", how="left")
    df["시장구분"] = "US"
    df = estimate_fair_value(df, group_cols=["섹터"])
    return df, price_history


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
        "현재주가", "적정주가", "괴리율", "적정주가_모델수",
        "가치점수", "모멘텀점수", "배당점수", "종합점수",
    ] if c in df.columns]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df[display_cols].head(TOP_N * 3).to_excel(writer, sheet_name="종합TOP", index=False)
        df.sort_values("가치점수", ascending=False)[display_cols].head(TOP_N).to_excel(writer, sheet_name="가치주TOP", index=False)
        df.sort_values("모멘텀점수", ascending=False)[display_cols].head(TOP_N).to_excel(writer, sheet_name="성장모멘텀TOP", index=False)
        df.sort_values("배당점수", ascending=False)[display_cols].head(TOP_N).to_excel(writer, sheet_name="배당TOP", index=False)
        if "괴리율" in df.columns:
            fv = df[df["적정주가_모델수"] >= 2] if "적정주가_모델수" in df.columns else df
            fv.sort_values("괴리율", ascending=False)[display_cols].head(TOP_N).to_excel(
                writer, sheet_name="적정주가_저평가TOP", index=False)
        if (df["시장구분"] == "KR").any():
            df[df["시장구분"] == "KR"][display_cols].to_excel(writer, sheet_name="KR_전체", index=False)
        if (df["시장구분"] == "US").any():
            df[df["시장구분"] == "US"][display_cols].to_excel(writer, sheet_name="US_전체", index=False)
        meta = pd.DataFrame({
            "항목": [
                "실행일시", "가치 가중치", "모멘텀 가중치", "배당 가중치", "PER 상한", "PBR 상한",
                "적정주가-PER 가중치", "적정주가-PBR 가중치", "적정주가-EV/EBITDA 가중치",
                "적정주가 최소 동종그룹 표본수",
            ],
            "값": [
                ts, WEIGHTS["value"], WEIGHTS["momentum"], WEIGHTS["dividend"], PER_MAX, PBR_MAX,
                FAIR_VALUE_WEIGHTS["PER"], FAIR_VALUE_WEIGHTS["PBR"], FAIR_VALUE_WEIGHTS["EV_EBITDA"],
                FAIR_VALUE_MIN_PEER_GROUP,
            ],
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


def _fmt_money(v, market, is_price=False) -> str:
    """시가총액/현금흐름/적정주가 등 금액을 시장에 맞는 단위(원/달러)로 보기 좋게 표시합니다."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if market == "KR":
        if is_price:
            return f"{v:,.0f}원"
        eok = v / 1e8
        if abs(eok) >= 10000:
            return f"{eok/10000:,.1f}조원"
        return f"{eok:,.0f}억원"
    if is_price:
        return f"${v:,.2f}"
    if abs(v) >= 1e9:
        return f"${v/1e9:,.1f}B"
    return f"${v/1e6:,.0f}M"


def _card_key(row) -> str:
    return f"{row.get('시장구분', 'US')}_{row.get('티커', '')}"


def _card(row, show_fair_value=False, show_volume=False) -> str:
    name = html.escape(str(row.get("종목명", "")))
    ticker = html.escape(str(row.get("티커", "")))
    market = row.get("시장구분", "US")
    flag = "\U0001F1F0\U0001F1F7" if market == "KR" else "\U0001F1FA\U0001F1F8"
    key = html.escape(_card_key(row))
    per, pbr, div = _fmt(row.get("PER")), _fmt(row.get("PBR")), _fmt(row.get("배당수익률"))
    m12, score = _fmt_pct(row.get("수익률_12M")), _fmt(row.get("종합점수"))
    extra_metrics = f'<span>PER {per}</span><span>PBR {pbr}</span><span>배당 {div}%</span><span>12개월 {m12}</span>'
    if show_fair_value:
        fv, gap = _fmt(row.get("적정주가"), 0), _fmt_pct(row.get("괴리율"))
        n = row.get("적정주가_모델수")
        n_str = f"{int(n)}개 모델" if pd.notna(n) else "-"
        extra_metrics = f'<span>적정주가 {fv}</span><span>괴리율 {gap}</span><span>{n_str}</span>'
    elif show_volume:
        value_str = _fmt_money(row.get("거래대금"), market)
        vol = row.get("거래량")
        vol_str = f"{vol:,.0f}주" if pd.notna(vol) else "-"
        base_date = row.get("거래기준일")
        date_str = f"({base_date} 기준)" if pd.notna(base_date) and base_date else ""
        extra_metrics = (f'<span>거래대금{date_str} {value_str}</span>'
                          f'<span>거래량 {vol_str}</span><span>12개월 {m12}</span>')
    return (f'<div class="card" data-market="{market}" data-key="{key}">'
            f'<div class="card-top"><span class="flag">{flag}</span>'
            f'<span class="name">{name}</span><span class="ticker">{ticker}</span>'
            f'<button class="star-btn" data-key="{key}" aria-label="관심종목 추가">☆</button>'
            f'<span class="score">{score}</span></div><div class="metrics">'
            f'{extra_metrics}</div></div>')


# 관심종목 추적 UI(카드 탭 -> 추적 시작)는 오직 대시보드에 실제로 카드로 뜬 종목에서만 열 수 있어
# ("1차로 추천/취합된 종목 중에서만 2차 선택"), 여기서는 그 카드가 "어떤 근거로 노출됐는지" 라벨만
# 추가로 계산합니다. 추적 시작 시 이 라벨을 함께 저장해두면, 나중에 "가치주 추천이 맞았는지",
# "모멘텀 추천이 맞았는지"처럼 카테고리별 추천 정확도를 되짚어볼 수 있습니다.
_LABEL_SECTIONS = [
    ("종합", "종합점수", 20, 0),
    ("가치주", "가치점수", 15, 0),
    ("모멘텀", "모멘텀점수", 15, 0),
    ("배당", "배당점수", 15, 0),
    ("적정주가", "괴리율", 15, 2),
    ("거래대금상위", "거래대금", 20, 0),
]


def _todays_recommend_labels(df: pd.DataFrame) -> dict:
    labels = {}
    for label, sort_col, n, min_models in _LABEL_SECTIONS:
        if sort_col not in df.columns or "시장구분" not in df.columns:
            continue
        base = df
        if min_models and "적정주가_모델수" in df.columns:
            base = base[base["적정주가_모델수"] >= min_models]
        for market in ("KR", "US"):
            sub = base[base["시장구분"] == market]
            if sub.empty:
                continue
            top = sub.sort_values(sort_col, ascending=False).head(n)
            for _, r in top.iterrows():
                labels.setdefault(_card_key(r), []).append(label)
    return {k: ", ".join(v) for k, v in labels.items()}


def _build_prices_json(df: pd.DataFrame, labels: dict) -> str:
    """개인 관심종목(추적) 기능은 브라우저(localStorage)에만 저장되어, '어떤 종목을 언제부터
    추적하는지'는 서버가 알 수 없습니다. 대신 배포 때마다 전 종목의 최신 가격표(+오늘 기준 추천
    근거 라벨)를 여기 담아 두면, 각자의 기기가 자기 추적목록에 대해 알아서 최신 수익률을 계산해
    보여줄 수 있습니다."""
    prices = {}
    if df.empty:
        return json.dumps(prices, ensure_ascii=False)
    dedup = df.drop_duplicates(subset=["시장구분", "티커"])
    for _, r in dedup.iterrows():
        price = r.get("현재주가")
        key = _card_key(r)
        prices[key] = {
            "종목명": str(r.get("종목명", "")),
            "시장구분": str(r.get("시장구분", "")),
            "티커": str(r.get("티커", "")),
            "현재주가": float(price) if pd.notna(price) else None,
            "추천구분": labels.get(key, ""),
        }
    return json.dumps(prices, ensure_ascii=False)


def _build_price_history_json(price_history: dict, selectable_keys) -> str:
    """관심종목 추적을 과거 날짜로 시작할 때 그 날짜의 실제 종가를 기준가로 쓸 수 있도록,
    카드로 노출되어 실제로 선택 가능한 종목에 한해서만 일별 종가 이력을 대시보드에 담습니다
    (전체 유니버스 대상으로 담으면 페이지 용량이 지나치게 커져 선택 가능한 종목만 추립니다)."""
    if not price_history or not selectable_keys:
        return json.dumps({}, ensure_ascii=False)
    trimmed = {k: v for k, v in price_history.items() if k in selectable_keys}
    return json.dumps(trimmed, ensure_ascii=False)


def _section(anchor, title, df, sort_col, n, show_fair_value=False, min_models=0, show_volume=False) -> str:
    base = df
    if min_models and "적정주가_모델수" in df.columns:
        base = df[df["적정주가_모델수"] >= min_models]
    if sort_col not in base.columns:
        base = base.iloc[0:0]
    else:
        base = base[base[sort_col].notna()]

    groups = []
    for market, label in (("KR", "\U0001F1F0\U0001F1F7 한국"), ("US", "\U0001F1FA\U0001F1F8 미국")):
        sub = base[base["시장구분"] == market] if "시장구분" in base.columns else base.iloc[0:0]
        if sub.empty:
            cards_html = '<p class="empty">데이터 없음</p>'
        else:
            cards_html = "".join(
                _card(r, show_fair_value=show_fair_value, show_volume=show_volume)
                for _, r in sub.sort_values(sort_col, ascending=False).head(n).iterrows()
            )
        groups.append(
            f'<div class="market-group" data-market="{market}">'
            f'<h3 class="market-h">{label}</h3><div class="cards">{cards_html}</div></div>'
        )
    return f'<section id="{anchor}"><h2>{html.escape(title)}</h2>{"".join(groups)}</section>'


def _detail_row(label, value) -> str:
    return f'<div class="drow"><span class="dlabel">{html.escape(label)}</span><span class="dvalue">{value}</span></div>'


def _detail_group(title, rows_html) -> str:
    return f'<div class="dgroup"><h4>{html.escape(title)}</h4>{rows_html}</div>'


def _detail_payload(row) -> dict:
    """카드를 클릭했을 때 보여줄 종목 세부정보(회계/재무 지표) HTML 조각을 만듭니다."""
    market = row.get("시장구분", "US")
    ticker = str(row.get("티커", ""))
    name = html.escape(str(row.get("종목명", "")))
    sector_val = row.get("섹터")
    sector = html.escape(str(sector_val)) if pd.notna(sector_val) else "-"

    basic = "".join([
        _detail_row("시장", "코스피/코스닥" if market == "KR" else "S&amp;P500(미국)"),
        _detail_row("섹터", sector),
        _detail_row("시가총액", _fmt_money(row.get("시가총액"), market)),
        _detail_row("현재주가", _fmt_money(row.get("현재주가"), market, is_price=True)),
    ])
    valuation = "".join([
        _detail_row("PER", _fmt(row.get("PER"))),
        _detail_row("PBR", _fmt(row.get("PBR"))),
        _detail_row("EPS(주당순이익)", _fmt(row.get("EPS"))),
        _detail_row("BPS(주당순자산)", _fmt(row.get("BPS"))),
    ])
    health = "".join([
        _detail_row("ROE", _fmt_pct(row.get("ROE"))),
        _detail_row("부채비율", _fmt(row.get("부채비율"))),
    ])
    growth = "".join([
        _detail_row("매출성장률", _fmt_pct(row.get("매출성장률"))),
        _detail_row("이익성장률", _fmt_pct(row.get("이익성장률"))),
    ])
    div_html = "".join([
        _detail_row("배당수익률", f"{_fmt(row.get('배당수익률'))}%"),
        _detail_row("배당성향", _fmt_pct(row.get("배당성향"))),
        _detail_row("FCF수익률", _fmt_pct(row.get("FCF수익률"))),
        _detail_row("잉여현금흐름", _fmt_money(row.get("잉여현금흐름"), market)),
    ])
    momentum = "".join([
        _detail_row("3개월 수익률", _fmt_pct(row.get("수익률_3M"))),
        _detail_row("6개월 수익률", _fmt_pct(row.get("수익률_6M"))),
        _detail_row("12개월 수익률", _fmt_pct(row.get("수익률_12M"))),
    ])
    fair = "".join([
        _detail_row("PER 모델 적정주가", _fmt_money(row.get("PER_적정주가"), market, is_price=True)),
        _detail_row("PBR 모델 적정주가", _fmt_money(row.get("PBR_적정주가"), market, is_price=True)),
        _detail_row("EV/EBITDA 모델 적정주가", _fmt_money(row.get("EV_EBITDA_적정주가"), market, is_price=True)),
        _detail_row("종합 적정주가", _fmt_money(row.get("적정주가"), market, is_price=True)),
        _detail_row("괴리율", _fmt_pct(row.get("괴리율"))),
    ])
    scores = "".join([
        _detail_row("가치점수", _fmt(row.get("가치점수"))),
        _detail_row("모멘텀점수", _fmt(row.get("모멘텀점수"))),
        _detail_row("배당점수", _fmt(row.get("배당점수"))),
        _detail_row("종합점수", _fmt(row.get("종합점수"))),
    ])

    body = "".join([
        _detail_group("기본정보", basic),
        _detail_group("밸류에이션", valuation),
        _detail_group("수익성·재무건전성", health),
        _detail_group("성장성", growth),
        _detail_group("배당·현금흐름", div_html),
        _detail_group("가격 모멘텀", momentum),
        _detail_group("적정주가(상대가치평가)", fair),
        _detail_group("스코어", scores),
    ])

    if market == "KR":
        ext_url = f"https://finance.naver.com/item/main.naver?code={ticker}"
        ext_label = "네이버 금융에서 재무제표 전체 보기 →"
    else:
        ext_url = f"https://finance.yahoo.com/quote/{ticker}/financials"
        ext_label = "Yahoo Finance에서 재무제표 전체 보기 →"
    link = f'<a class="ext-link" href="{html.escape(ext_url)}" target="_blank" rel="noopener">{html.escape(ext_label)}</a>'

    flag = "\U0001F1F0\U0001F1F7" if market == "KR" else "\U0001F1FA\U0001F1F8"
    rank = row.get("전체순위")
    rank_str = f" · 종합순위 {int(rank)}위" if pd.notna(rank) else ""
    return {
        "title": f"{flag} {name}",
        "sub": f"{ticker}{rank_str}",
        "body": body,
        "link": link,
    }


def _build_details_json(df: pd.DataFrame) -> str:
    details = {}
    dedup = df.drop_duplicates(subset=["시장구분", "티커"])
    for _, r in dedup.iterrows():
        details[_card_key(r)] = _detail_payload(r)
    return json.dumps(details, ensure_ascii=False)


def build_dashboard(df: pd.DataFrame, price_history: dict = None, out_path: str = "docs/index.html") -> str:
    # GitHub Actions 러너는 UTC로 동작하므로, 화면에 "(KST)"라고 표시하려면 명시적으로 9시간을
    # 더해 실제 한국시간으로 변환해야 합니다(이전에는 UTC 시각을 KST라고 잘못 표시하던 버그가
    # 있었습니다 - 최대 9시간까지 실제 갱신시각과 화면 표시가 어긋났습니다).
    ts = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
    w = WEIGHTS
    total = _section("top", "\U0001F3C6 종합 TOP 20", df, "종합점수", 20)
    value = _section("value", "\U0001F4B0 가치주 TOP 15", df, "가치점수", 15)
    mom = _section("momentum", "\U0001F680 성장·모멘텀 TOP 15", df, "모멘텀점수", 15)
    div_ = _section("dividend", "\U0001F4B5 배당·현금흐름 TOP 15", df, "배당점수", 15)
    fair = _section("fairvalue", "\U0001F3AF 적정주가 저평가 TOP 15", df, "괴리율", 15,
                     show_fair_value=True, min_models=2)
    volume = _section("volume", "\U0001F4B9 거래대금 TOP 20 (전체 종목 대상, 추천점수와 무관, 최근 거래일 기준)",
                       df, "거래대금", 20, show_volume=True)
    details_json = _build_details_json(df)
    labels = _todays_recommend_labels(df)
    prices_json = _build_prices_json(df, labels)
    price_history_json = _build_price_history_json(price_history or {}, set(labels.keys()))

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
  .filterbar {{ display:flex; gap:8px; padding:10px 16px; -webkit-overflow-scrolling:touch; }}
  .filter-btn {{ flex:1; padding:8px 10px; background:var(--card); border:1px solid var(--border); border-radius:10px; color:var(--muted); font-size:13px; font-weight:600; white-space:nowrap; }}
  .filter-btn.active {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
  .cat-filter {{ overflow-x:auto; }}
  .cat-filter .filter-btn {{ flex:0 0 auto; padding:8px 14px; }}
  main {{ padding:0 16px 30px; max-width:640px; margin:0 auto; }}
  section {{ margin-top:22px; scroll-margin-top:150px; }}
  section h2 {{ font-size:15.5px; margin:0 0 10px; }}
  .market-h {{ font-size:12.5px; color:var(--muted); margin:14px 0 8px; font-weight:600; }}
  .market-group:first-child .market-h {{ margin-top:0; }}
  body.filter-KR .market-group[data-market="US"] {{ display:none; }}
  body.filter-US .market-group[data-market="KR"] {{ display:none; }}
  .cards {{ display:flex; flex-direction:column; gap:8px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 14px; cursor:pointer; }}
  .card:active {{ opacity:.7; }}
  .card-top {{ display:flex; align-items:center; gap:6px; }}
  .flag {{ font-size:15px; }}
  .name {{ font-weight:600; font-size:14px; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .ticker {{ color:var(--muted); font-size:11.5px; }}
  .score {{ background:var(--accent); color:#fff; font-size:12px; font-weight:700; padding:2px 9px; border-radius:999px; }}
  .star-btn {{ background:none; border:none; font-size:17px; line-height:1; padding:2px 4px; cursor:pointer; color:var(--muted); }}
  .star-btn.tracked {{ color:#ffd166; }}
  .metrics {{ margin-top:6px; display:flex; gap:10px; flex-wrap:wrap; color:var(--muted); font-size:12px; }}
  .empty {{ color:var(--muted); font-size:13px; }}
  .perf-scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; border:1px solid var(--border); border-radius:12px; }}
  .perf-table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  .perf-table th {{ text-align:left; color:var(--muted); font-weight:600; font-size:11px; padding:8px; border-bottom:1px solid var(--border); white-space:nowrap; background:var(--card); }}
  .perf-table td {{ padding:8px; border-bottom:1px solid var(--border); white-space:nowrap; }}
  .perf-table tr:last-child td {{ border-bottom:none; }}
  .pname {{ display:block; font-weight:600; }}
  .pticker {{ display:block; color:var(--muted); font-size:11px; font-weight:400; margin-top:2px; }}
  .pos {{ color:#3ecf8e; font-weight:700; }}
  .neg {{ color:#ff6b6b; font-weight:700; }}
  .track-x {{ background:none; border:none; color:var(--muted); font-size:14px; padding:4px 8px; cursor:pointer; }}
  footer {{ text-align:center; color:var(--muted); font-size:11px; padding:14px 16px 36px; line-height:1.7; }}
  a.dl {{ color:var(--accent); }}
  .overlay {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.55); z-index:100;
    align-items:flex-end; justify-content:center; }}
  .overlay.open {{ display:flex; }}
  .sheet {{ background:var(--card); width:100%; max-width:640px; max-height:82vh; overflow-y:auto;
    border-radius:16px 16px 0 0; border:1px solid var(--border); border-bottom:none; padding:16px 16px 24px; }}
  .sheet-header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:10px;
    position:sticky; top:-16px; background:var(--card); padding:0 0 10px; margin:-16px 0 4px; }}
  .sheet-title {{ font-size:16.5px; font-weight:700; }}
  .sheet-sub {{ font-size:12px; color:var(--muted); margin-top:2px; }}
  .close-btn {{ background:none; border:none; color:var(--muted); font-size:18px; padding:4px 8px; cursor:pointer; }}
  .dgroup {{ margin-top:14px; }}
  .dgroup h4 {{ font-size:12.5px; color:var(--accent); margin:0 0 6px; }}
  .drow {{ display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid var(--border); font-size:13px; }}
  .drow:last-child {{ border-bottom:none; }}
  .dlabel {{ color:var(--muted); }}
  .dvalue {{ font-weight:600; text-align:right; }}
  .ext-link {{ display:block; text-align:center; margin-top:18px; padding:11px; background:var(--accent);
    color:#fff; text-decoration:none; border-radius:10px; font-size:13.5px; font-weight:600; }}
  .track-box {{ background:rgba(79,140,255,.08); border:1px solid var(--border); border-radius:10px; padding:12px; margin:12px 0 4px; }}
  .track-label {{ display:block; font-size:12px; color:var(--muted); margin-bottom:8px; }}
  .track-add-row {{ display:flex; gap:8px; }}
  .track-add-row input[type="date"] {{ flex:1; background:var(--bg); border:1px solid var(--border); color:var(--text); border-radius:8px; padding:8px; font-size:13px; }}
  .track-btn {{ border:none; border-radius:8px; padding:9px 14px; font-size:13px; font-weight:600; cursor:pointer; }}
  .track-add {{ background:var(--accent); color:#fff; }}
  .track-remove {{ background:transparent; color:var(--muted); border:1px solid var(--border); width:100%; margin-top:8px; }}
  .track-row {{ display:flex; justify-content:space-between; padding:4px 0; font-size:13px; }}
  .track-error {{ color:#ff6b6b; font-size:12px; margin:8px 0 0; }}
</style>
</head>
<body>
<header>
  <h1>\U0001F4CA 주식 스크리닝 리포트</h1>
  <p>업데이트: {ts} (KST) · 가치 {w['value']*100:.0f}% · 모멘텀 {w['momentum']*100:.0f}% · 배당 {w['dividend']*100:.0f}%<br>
  <a class="dl" href="reports/latest.xlsx">엑셀 전체 데이터 다운로드</a></p>
</header>
<div class="filterbar cat-filter">
  <button class="filter-btn active" data-cat="all">전체</button>
  <button class="filter-btn" data-cat="top">종합</button>
  <button class="filter-btn" data-cat="value">가치주</button>
  <button class="filter-btn" data-cat="momentum">모멘텀</button>
  <button class="filter-btn" data-cat="dividend">배당</button>
  <button class="filter-btn" data-cat="fairvalue">적정주가</button>
  <button class="filter-btn" data-cat="volume">\U0001F4B9 거래대금</button>
  <button class="filter-btn" data-cat="performance">\U0001F4CC 관심종목</button>
</div>
<div class="filterbar market-filter">
  <button class="filter-btn active" data-filter="all">전체</button>
  <button class="filter-btn" data-filter="KR">\U0001F1F0\U0001F1F7 한국만</button>
  <button class="filter-btn" data-filter="US">\U0001F1FA\U0001F1F8 미국만</button>
</div>
<main>{total}{value}{mom}{div_}{fair}{volume}<section id="performance"><h2>\U0001F4CC 관심종목 추적</h2><div id="perfContainer"></div></section></main>
<footer>본 리포트는 투자자문이 아닙니다. 공개 데이터를 기계적으로 점수화한 참고 자료이며,<br>
투자 판단과 책임은 본인에게 있습니다. 매일 자동 갱신됩니다 (GitHub Actions).<br><br>
"적정주가"는 동종 섹터/시장의 PER·PBR·EV-EBITDA 중앙값 배수를 자사 실적에 대입한
<b>상대가치평가</b> 추정치입니다(미래 현금흐름을 직접 추정하는 DCF가 아님). 동종군 표본이 적거나
실적이 일시적으로 왜곡된 경우 오차가 커질 수 있어, 사용된 모델 수가 2개 이상인 종목 위주로
참고하시고 최종 투자 판단 전 재무제표 원본을 확인하세요.<br><br>
"거래대금 TOP"은 추천점수와 무관하게, 데이터가 있는 가장 최근 거래일 하루의 거래대금(종가×거래량)
기준으로 코스피·코스닥·S&amp;P500 전체에서 가장 활발히 거래된 종목을 모은 목록입니다 - 여러 날을
평균 내지 않고 그날 하루치 값을 그대로 쓰며, 각 카드에 몇 월 며칠 기준인지 날짜를 함께 표시합니다
(한국·미국 장 마감/개장 시간 차이로 두 시장의 기준일이 하루 정도 다르게 보일 수 있습니다 - 예를
들어 주말에는 한국·미국 모두 지난주 마지막 거래일인 금요일 데이터가 보이는 게 정상입니다). 저희
추천 리스트에 없는 종목도 여기서 확인할 수 있습니다.<br><br>
카드 오른쪽 위 별표(☆)를 누르면 오늘 날짜·오늘 가격 기준으로 바로 "관심종목"에 추가됩니다.
카드를 탭해서 열리는 세부정보 화면에서는 추적 시작일을 과거 날짜로도 지정할 수 있고, 이 경우
실제 그 날짜의 종가(최근 약 1년치 보관)를 기준가로 사용합니다. 추가된 종목은 어느 리스트
(종합/가치주/모멘텀/배당/적정주가/거래대금상위)에서 골라졌는지와 함께 기록되어, 나중에 실제로
분석한 방향대로 주가가 움직였는지 확인할 수 있습니다. 추적 목록은 이 기기의 브라우저에만
저장됩니다(서버에는 저장되지 않아 다른 기기/브라우저에는 보이지 않고, 브라우저 데이터를 지우면
사라집니다).<br><br>
위 카테고리 탭(종합/가치주/모멘텀/배당/적정주가/거래대금/관심종목)을 누르면 해당 목록만 바로 볼 수 있습니다.</footer>

<div class="overlay" id="detailOverlay">
  <div class="sheet">
    <div class="sheet-header">
      <div>
        <div class="sheet-title" id="dTitle"></div>
        <div class="sheet-sub" id="dSub"></div>
      </div>
      <button class="close-btn" id="dClose">✕</button>
    </div>
    <div id="dTrack"></div>
    <div id="dBody"></div>
    <div id="dLink"></div>
  </div>
</div>

<script>
const DETAILS = {details_json};
const PRICES = {prices_json};
const PRICE_HISTORY = {price_history_json};
const WATCHLIST_KEY = 'stockScreenerWatchlist';

function loadWatchlist() {{
  try {{ return JSON.parse(localStorage.getItem(WATCHLIST_KEY) || '{{}}'); }} catch (e) {{ return {{}}; }}
}}
function saveWatchlist(w) {{
  try {{ localStorage.setItem(WATCHLIST_KEY, JSON.stringify(w)); }} catch (e) {{ /* 저장 불가(프라이빗 모드 등) - 무시 */ }}
}}
function escapeHtml(s) {{
  const div = document.createElement('div');
  div.textContent = String(s == null ? '' : s);
  return div.innerHTML;
}}
function fmtMoney(v, market) {{
  if (v === null || v === undefined || isNaN(v)) return '-';
  if (market === 'KR') return Math.round(v).toLocaleString('ko-KR') + '원';
  return '$' + Number(v).toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
}}
function fmtRet(v) {{
  if (v === null || v === undefined || isNaN(v)) return '-';
  return (v > 0 ? '+' : '') + v.toFixed(1) + '%';
}}
function todayStr() {{ return new Date().toISOString().slice(0, 10); }}
function daysBetween(a, b) {{
  const d1 = new Date(a + 'T00:00:00');
  const d2 = new Date(b + 'T00:00:00');
  return Math.round((d2 - d1) / 86400000);
}}
function historyDates(key) {{
  const series = PRICE_HISTORY[key];
  if (!series) return [];
  return Object.keys(series).sort();
}}
function findHistoricalPrice(key, dateStr) {{
  const series = PRICE_HISTORY[key];
  if (!series) return null;
  if (series[dateStr] !== undefined) return series[dateStr];
  // 정확히 그 날짜의 데이터가 없으면(주말/휴장일 등) 그 이전의 가장 가까운 거래일 종가를 사용
  const dates = historyDates(key);
  let found = null;
  for (let i = 0; i < dates.length; i++) {{
    if (dates[i] <= dateStr) found = dates[i]; else break;
  }}
  return found ? series[found] : null;
}}

function addToWatchlist(key, date) {{
  const info = PRICES[key];
  const box = document.getElementById('dTrack');
  if (!info || info.현재주가 === null || info.현재주가 === undefined) {{
    if (box) box.insertAdjacentHTML('beforeend', '<p class="track-error">현재가 정보가 없어 추적을 시작할 수 없습니다.</p>');
    return;
  }}
  const chosenDate = date || todayStr();
  let basePrice = info.현재주가;
  if (chosenDate !== todayStr()) {{
    const hist = findHistoricalPrice(key, chosenDate);
    if (hist === null) {{
      if (box) box.insertAdjacentHTML('beforeend', '<p class="track-error">해당 날짜의 과거 가격 데이터가 없어 오늘 가격을 기준가로 사용합니다(최근 약 1년 이내 날짜만 지원).</p>');
    }} else {{
      basePrice = hist;
    }}
  }}
  const w = loadWatchlist();
  w[key] = {{
    종목명: info.종목명, 시장구분: info.시장구분, 티커: info.티커,
    추천구분: info.추천구분 || '', 추적일: chosenDate, 추적가: basePrice
  }};
  saveWatchlist(w);
  renderTrackControls(key);
  renderPerformance();
  refreshStarButtons();
}}
function removeFromWatchlist(key) {{
  const w = loadWatchlist();
  delete w[key];
  saveWatchlist(w);
  renderTrackControls(key);
  renderPerformance();
  refreshStarButtons();
}}
function quickToggleTrack(key) {{
  const w = loadWatchlist();
  if (w[key]) {{
    removeFromWatchlist(key);
    return;
  }}
  addToWatchlist(key, todayStr());
}}
function refreshStarButtons() {{
  const w = loadWatchlist();
  document.querySelectorAll('.star-btn').forEach(function(btn) {{
    const key = btn.dataset.key;
    if (w[key]) {{ btn.textContent = '★'; btn.classList.add('tracked'); }}
    else {{ btn.textContent = '☆'; btn.classList.remove('tracked'); }}
  }});
}}

function renderTrackControls(key) {{
  const el = document.getElementById('dTrack');
  if (!el) return;
  const w = loadWatchlist();
  const entry = w[key];
  const today = todayStr();
  if (entry) {{
    const info = PRICES[key] || {{}};
    const cur = (info.현재주가 !== undefined && info.현재주가 !== null) ? info.현재주가 : null;
    const days = daysBetween(entry.추적일, today);
    const ret = (cur !== null && entry.추적가) ? (cur / entry.추적가 - 1) * 100 : null;
    const cls = ret > 0 ? 'pos' : (ret < 0 ? 'neg' : '');
    el.innerHTML =
      '<div class="track-box">' +
      (entry.추천구분 ? '<div class="track-row"><span>추천 근거</span><span>' + escapeHtml(entry.추천구분) + '</span></div>' : '') +
      '<div class="track-row"><span>추적일</span><span>' + entry.추적일 + ' (' + days + '일 경과)</span></div>' +
      '<div class="track-row"><span>추적 시작가</span><span>' + fmtMoney(entry.추적가, entry.시장구분) + '</span></div>' +
      '<div class="track-row"><span>현재가</span><span>' + (cur !== null ? fmtMoney(cur, entry.시장구분) : '데이터없음') + '</span></div>' +
      '<div class="track-row"><span>수익률</span><span class="' + cls + '">' + fmtRet(ret) + '</span></div>' +
      '<button class="track-btn track-remove" id="trackRemoveBtn">추적 해제</button></div>';
    const btn = document.getElementById('trackRemoveBtn');
    if (btn) btn.addEventListener('click', function() {{ removeFromWatchlist(key); }});
  }} else {{
    const dates = historyDates(key);
    const minDate = dates.length ? dates[0] : today;
    const hint = dates.length
      ? '추적 시작일(과거 날짜를 고르면 그날의 실제 종가를 기준가로 사용, ' + minDate + '부터 가능)'
      : '추적 시작일(과거 시세 데이터가 없어 오늘 날짜만 가능)';
    el.innerHTML =
      '<div class="track-box"><label class="track-label">' + hint + '</label>' +
      '<div class="track-add-row"><input type="date" id="trackDateInput" value="' + today + '" max="' + today + '" min="' + minDate + '">' +
      '<button class="track-btn track-add" id="trackAddBtn">추적 시작</button></div></div>';
    const btn = document.getElementById('trackAddBtn');
    if (btn) btn.addEventListener('click', function() {{
      const dateVal = document.getElementById('trackDateInput').value || today;
      addToWatchlist(key, dateVal);
    }});
  }}
}}

function renderPerformance() {{
  const container = document.getElementById('perfContainer');
  if (!container) return;
  const w = loadWatchlist();
  const keys = Object.keys(w);
  if (keys.length === 0) {{
    container.innerHTML = '<p class="empty">추적 중인 종목이 없습니다. 관심 있는 종목 카드를 눌러 상세정보에서 "추적 시작"을 눌러보세요.</p>';
    return;
  }}
  const today = todayStr();
  const groups = {{ KR: [], US: [] }};
  keys.forEach(function(key) {{
    const entry = w[key];
    const m = entry.시장구분 === 'KR' ? 'KR' : 'US';
    groups[m].push([key, entry]);
  }});
  ['KR', 'US'].forEach(function(m) {{
    groups[m].sort(function(a, b) {{ return daysBetween(b[1].추적일, today) - daysBetween(a[1].추적일, today); }});
  }});
  const labels = {{ KR: '\U0001F1F0\U0001F1F7 한국', US: '\U0001F1FA\U0001F1F8 미국' }};
  let out = '';
  ['KR', 'US'].forEach(function(m) {{
    const list = groups[m];
    let body;
    if (list.length === 0) {{
      body = '<p class="empty">추적 중인 종목 없음</p>';
    }} else {{
      const rows = list.map(function(item) {{
        const key = item[0], entry = item[1];
        const info = PRICES[key] || {{}};
        const cur = (info.현재주가 !== undefined && info.현재주가 !== null) ? info.현재주가 : null;
        const days = daysBetween(entry.추적일, today);
        const ret = (cur !== null && entry.추적가) ? (cur / entry.추적가 - 1) * 100 : null;
        const cls = ret > 0 ? 'pos' : (ret < 0 ? 'neg' : '');
        const basis = entry.추천구분 ? (entry.티커 + ' · ' + entry.추천구분) : entry.티커;
        return '<tr><td><span class="pname">' + escapeHtml(entry.종목명) + '</span><span class="pticker">' + escapeHtml(basis) + '</span></td>' +
          '<td>' + entry.추적일 + '<span class="pticker">' + days + '일 경과</span></td>' +
          '<td>' + fmtMoney(entry.추적가, m) + '</td>' +
          '<td>' + (cur !== null ? fmtMoney(cur, m) : '데이터없음') + '</td>' +
          '<td class="' + cls + '">' + fmtRet(ret) + '</td>' +
          '<td><button class="track-x" data-key="' + escapeHtml(key) + '">✕</button></td></tr>';
      }}).join('');
      body = '<div class="perf-scroll"><table class="perf-table"><thead><tr><th>종목</th><th>추적일</th><th>추적가</th><th>현재가</th><th>수익률</th><th></th></tr></thead><tbody>' + rows + '</tbody></table></div>';
    }}
    out += '<div class="market-group" data-market="' + m + '"><h3 class="market-h">' + labels[m] + ' (' + list.length + '개)</h3>' + body + '</div>';
  }});
  container.innerHTML = out;
  container.querySelectorAll('.track-x').forEach(function(btn) {{
    btn.addEventListener('click', function() {{ removeFromWatchlist(btn.dataset.key); }});
  }});
}}

const overlay = document.getElementById('detailOverlay');
function openDetail(key) {{
  const d = DETAILS[key];
  if (!d) return;
  document.getElementById('dTitle').innerHTML = d.title;
  document.getElementById('dSub').innerHTML = d.sub;
  document.getElementById('dBody').innerHTML = d.body;
  document.getElementById('dLink').innerHTML = d.link;
  renderTrackControls(key);
  overlay.classList.add('open');
}}
function closeDetail() {{ overlay.classList.remove('open'); }}
document.getElementById('dClose').addEventListener('click', closeDetail);
overlay.addEventListener('click', function(e) {{ if (e.target === overlay) closeDetail(); }});
document.querySelector('main').addEventListener('click', function(e) {{
  const starBtn = e.target.closest('.star-btn');
  if (starBtn) {{
    e.stopPropagation();
    quickToggleTrack(starBtn.dataset.key);
    return;
  }}
  const card = e.target.closest('.card');
  if (card && card.dataset.key) openDetail(card.dataset.key);
}});

let marketFilter = 'all';
let catFilter = 'all';
function applyFilters() {{
  document.body.className = (marketFilter === 'all') ? '' : 'filter-' + marketFilter;
  document.querySelectorAll('main > section').forEach(function(sec) {{
    sec.style.display = (catFilter === 'all' || sec.id === catFilter) ? '' : 'none';
  }});
}}
document.querySelectorAll('.market-filter .filter-btn').forEach(function(btn) {{
  btn.addEventListener('click', function() {{
    document.querySelectorAll('.market-filter .filter-btn').forEach(function(b) {{ b.classList.remove('active'); }});
    btn.classList.add('active');
    marketFilter = btn.dataset.filter;
    applyFilters();
  }});
}});
document.querySelectorAll('.cat-filter .filter-btn').forEach(function(btn) {{
  btn.addEventListener('click', function() {{
    document.querySelectorAll('.cat-filter .filter-btn').forEach(function(b) {{ b.classList.remove('active'); }});
    btn.classList.add('active');
    catFilter = btn.dataset.cat;
    applyFilters();
    window.scrollTo(0, 0);
  }});
}});

renderPerformance();
refreshStarButtons();
</script>
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
    price_history = {}

    if kr:
        print("[KR] 데이터 수집 중...")
        kr_df, kr_price_history = build_kr_universe()
        kr_df = kr_df.rename(columns={"DIV": "배당수익률"})
        if "배당수익률" in kr_df:
            kr_df["배당수익률"] = _normalize_pct(kr_df["배당수익률"])
        kr_df = score_market(kr_df)
        frames.append(kr_df)
        price_history.update(kr_price_history)
        print(f"[KR] {len(kr_df)}개 종목 처리 완료")

    if us:
        print("[US] 데이터 수집 중...")
        us_df, us_price_history = build_us_universe()
        if not us_df.empty:
            if "배당수익률" in us_df:
                us_df["배당수익률"] = _normalize_pct(us_df["배당수익률"])
            us_df = score_market(us_df, extra_momentum=US_MOMENTUM_EXTRA, extra_dividend=US_DIVIDEND_EXTRA)
            frames.append(us_df)
            price_history.update(us_price_history)
            print(f"[US] {len(us_df)}개 종목 처리 완료")

    if not frames:
        print("수집된 데이터가 없습니다.")
        return ""

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["전체순위"] = combined["종합점수"].rank(ascending=False, method="min")
    combined = combined.sort_values("종합점수", ascending=False)

    xlsx_path = export(combined)
    build_dashboard(combined, price_history, out_path="docs/index.html")
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
