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
import io
import json
import os
import random
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

# Yahoo Finance는 2024년 이후 봇 탐지를 TLS 핑거프린트 수준까지 강화해서, 파이썬 표준
# requests 세션으로 yfinance를 쓰면 (특히 짧은 시간에 많은 티커를 조회할 때) quoteSummary
# 요청이 전부 "Quote not found" 404로 막히는 사례가 보고되어 있습니다(실제로 이 증상 발생
# 확인: 2026.09 KR 펀더멘털 수집이 이 오류로 대량 실패하며 실행시간이 급증 -> 타임아웃).
# yfinance 공식 문서가 권장하는 대로 curl_cffi로 브라우저(Chrome) TLS 핑거프린트를 흉내 낸
# 세션을 만들어 모든 yfinance 호출에 재사용합니다. curl_cffi가 없거나 세션 생성이 실패해도
# 예외를 던지지 않고 기본 세션(예전 동작)으로 조용히 되돌아갑니다.
# 개별 요청에 타임아웃(YF_REQUEST_TIMEOUT_SEC)을 걸어, 한 요청이 무한정 걸리는 상황을 막습니다.
YF_MAX_WORKERS = 3          # yfinance 동시 요청 수(네이버 스크래핑보다 훨씬 보수적으로)
YF_REQUEST_TIMEOUT_SEC = 12
YF_FETCH_TIME_BUDGET_SEC = 900  # 이 시간(15분)이 지나면 남은 종목은 건너뛰고 계속 진행
try:
    from curl_cffi import requests as _curl_requests
    _YF_SESSION = _curl_requests.Session(impersonate="chrome", timeout=YF_REQUEST_TIMEOUT_SEC)
    print("[YF] curl_cffi(Chrome 흉내) 세션으로 yfinance 호출을 진행합니다.")
except Exception as _yf_session_err:
    _YF_SESSION = None
    print(f"[YF] curl_cffi 세션 생성 실패, 기본 세션으로 진행합니다: {_yf_session_err}")


def _yf_ticker(symbol: str):
    """모든 yfinance Ticker 생성을 이 함수로 통일해, curl_cffi 세션 적용 여부를 한 곳에서
    관리합니다."""
    if _YF_SESSION is not None:
        return yf.Ticker(symbol, session=_YF_SESSION)
    return yf.Ticker(symbol)


_yf_session_primed = False


def _prime_yf_session(sample_symbols):
    """yfinance는 첫 호출 시 내부적으로 Yahoo에 쿠키/crumb(세션 인증 토큰)를 발급받는데,
    이 발급 절차가 아직 안 끝난 상태에서 여러 스레드가 동시에 처음 호출하면 각자 crumb
    발급을 따로 시도하게 됩니다. 2026.09.10 실행에서 이 경쟁 상태 때문에 crumb 발급
    자체가 429(rate-limited)로 막히고, 그 뒤 모든 요청이 "Invalid Crumb" 401로 연쇄
    실패하는 사고가 있었습니다. 병렬 조회를 시작하기 전에 단 한 번, 순차적으로 먼저 호출해
    crumb/쿠키를 확보해둡니다(실패해도 예외를 던지지 않음 - 병렬 조회에서 다시 시도됨)."""
    global _yf_session_primed
    if _yf_session_primed or yf is None:
        return
    for symbol in list(sample_symbols)[:3]:
        try:
            info = _yf_ticker(symbol).get_info()
            if info:
                print(f"[YF] 세션 준비 완료(crumb 확보, 확인용 종목: {symbol})")
                _yf_session_primed = True
                return
        except Exception as e:
            print(f"[YF] 세션 준비 시도 실패({symbol}): {e}")
    print("[YF] 경고: 세션 준비(crumb 확보)에 계속 실패했습니다 - 이후 병렬 조회도 대부분 "
          "실패할 가능성이 높습니다.")


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

# 한국 데이터 소스: "naver"(기본) | "yfinance" | "pykrx"
# - pykrx는 data.krx.co.kr를 스크래핑하는데, 클라우드/해외 IP(GitHub Actions 러너 포함) 차단 +
#   최근 버전은 일부 호출에 KRX 로그인(KRX_ID/KRX_PW)까지 요구하기 시작해 사실상 사용 불가 상태(2026.08).
#   문제가 해결되면 이 값을 "pykrx"로 되돌리면 기존 로직을 그대로 다시 쓸 수 있습니다.
# - "yfinance"는 티커에 .KS(코스피)/.KQ(코스닥) 접미사를 붙여 조회하는 예전 방식입니다. 2026.09에
#   Yahoo Finance가 한국 종목 대량 조회를 반복적으로 차단해(1차: quoteSummary 404 대량,
#   2차: 병렬화 후 crumb 인증 401 대량) 더 이상 기본값이 아닙니다. 코드/로직은 그대로 남겨뒀으니
#   Naver 쪽이 막히는 날이 오면 이 값으로 되돌려 임시 대응할 수 있습니다.
# - "naver"(기본, 2026.09.10부터): 이 프로젝트에서 이미 안정적으로 동작해온 네이버금융
#   스크래핑(PER/PBR/EPS/BPS/사업개요)을 확장해 한국 종목 펀더멘털 전체를 여기서만 가져옵니다.
#   무료 공식 API(Finnhub/FMP/Twelve Data 등)도 검토했으나, 한국(KRX) 데이터는 대부분 무료
#   플랜에서 아예 지원하지 않거나(유료 전용) 무료 호출 한도가 한국 전종목(1,300여 개)을
#   감당하지 못해(예: FMP 250건/일) 채택하지 않았습니다. 트레이드오프: EBITDA/기업가치/순부채/
#   잉여현금흐름은 네이버금융에 없어 확보하지 못하고(적정주가 3방식 중 PER·PBR 2방식만 반영 -
#   기존에도 "모델수 2개 이상"이면 정상 노출되므로 동작에는 문제없음), ROE/부채비율은 별도
#   엔드포인트로 참고용 시도만 합니다(실패해도 표시만 비고 스코어링에는 영향 없음). 핵심 추천
#   점수(가치·모멘텀·배당 3팩터)에 쓰이는 PER/PBR/모멘텀/배당수익률은 모두 그대로 확보됩니다.
KR_DATA_SOURCE = "naver"

# ROE/부채비율 + 매출액/영업이익(성장률)을 별도 엔드포인트(companyinfo.stock.naver.com)에서
# 추가로 가져올지 여부. 2026.09.10 KR_DATA_SOURCE="naver" 첫 배포 직후 이 엔드포인트가
# 원인으로 의심되는 45분 타임아웃이 있어 한동안 기본 비활성화했었습니다. 2026.09.11부터는
# "정책 관련 섹터 안에서 매출·영업이익 실적 기준으로 추천"이라는 새 추천 방식의 핵심 데이터가
# 되어(POLICY_RANK_COMPONENTS 참고) 다시 기본 활성화합니다 - 이 엔드포인트가 실제로 그
# 타임아웃의 원인이었는지는 끝내 로그로 확인되지 못했고, 이미 적용된 시간예산 안전장치
# (NAVER_FETCH_TIME_BUDGET_SEC)가 최소한 "전체를 무한정 기다리다 45분을 다 쓰는" 최악의
# 상황은 막아줍니다. 다음 실행 로그를 꼭 확인해, 타임아웃이 재발하면 다시 False로 되돌리세요.
KR_FETCH_FINANCIALS = True

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

# 사업분야/국가정책 연계 태그(참고용) 설정
# - 공식 공시나 정부 정책 문서를 직접 매칭하는 게 아니라, yfinance의 업종(영문 GICS 섹터/업종)과
#   종목명·사업개요 텍스트에 특정 키워드가 있는지로 "관련 가능성이 있는" 정책 테마를 자동으로
#   태깅하는 단순 규칙 기반 로직입니다. 오탐/누락이 있을 수 있어 투자 판단의 근거가 아니라
#   "이 종목이 어떤 산업 흐름과 엮여 있는지 빠르게 훑어보는 참고 표시"로만 사용하세요.
POLICY_THEMES = [
    ("반도체 (K-반도체 전략·미국 CHIPS Act)",
     ["semiconductor", "semiconductors", "foundry", "chip equipment", "반도체", "파운드리", "웨이퍼"]),
    ("2차전지·배터리 (K-배터리 산업·美 IRA)",
     ["battery", "batteries", "lithium", "2차전지", "배터리", "양극재", "음극재", "전해질"]),
    ("신재생에너지 (그린뉴딜·美 IRA)",
     ["renewable", "solar", "wind power", "hydrogen", "태양광", "풍력", "수소", "재생에너지", "연료전지"]),
    ("원자력 (원전 생태계 지원정책)",
     ["nuclear", "원자력", "원전", "SMR"]),
    ("방위산업 (K-방산 수출전략)",
     ["aerospace & defense", "defense", "방위산업", "방산", "국방"]),
    ("바이오·제약 (바이오헬스 국가전략기술)",
     ["biotechnology", "pharmaceutical", "drug manufacturer", "바이오", "제약", "신약", "임상"]),
    ("AI·로봇·반도체장비 (디지털/AI 육성정책)",
     ["software - infrastructure", "semiconductor equipment", "robotics", "인공지능", "로봇", "자율주행"]),
    ("전기차·모빌리티 (미래차 산업·美 IRA EV 세액공제)",
     ["auto manufacturers", "auto parts", "electric vehicle", "전기차", "모빌리티"]),
    ("조선·해운 (친환경 선박 정책)",
     ["marine shipping", "shipbuilding", "조선", "선박", "해운"]),
    ("우주항공 (우주산업 육성정책)",
     ["aerospace", "우주", "위성", "발사체"]),
]


def _match_policy_themes(name, sector, industry, profile_text) -> list:
    haystack = " ".join(str(x) for x in (name, sector, industry, profile_text) if x and pd.notna(x)).lower()
    if not haystack.strip():
        return []
    matched = []
    for label, keywords in POLICY_THEMES:
        if any(kw.lower() in haystack for kw in keywords):
            matched.append(label)
    return matched


# ==========================================================
# 정책테마 추천 (2026.09.11 신설)
# ==========================================================
# 배경: "밸류에이션·보조지표 위주 스코어링이 숫자 왜곡으로 이해하기 힘든 종목을 추천한다"는
# 피드백에 따라, 대시보드의 1번째(기본) 화면을 다음 순서로 다시 설계했습니다.
#   1) POLICY_THEMES에 매칭되는(=정부 정책·산업 확장과 관련 있다고 키워드로 추정되는) 종목만
#      후보로 남김 (1차 필터 - 기존 10개 테마 키워드 매핑을 그대로 승격해서 사용)
#   2) 그 후보 안에서만 매출성장률·영업이익성장률(=실제 사업이 얼마나 성장하고 있는지) 기준으로
#      순위를 매김 - PER/PBR 같은 밸류에이션과 모멘텀/배당은 이 섹션의 순위에는 반영하지 않음
# 기존 "종합/가치주/모멘텀/배당/적정주가" 섹션은 그대로 남겨두어(탭으로 계속 접근 가능) 서로
# 다른 관점의 결과를 비교해볼 수 있게 했습니다 - 다만 첫 화면(기본 탭)은 이제 "정책테마"입니다.
POLICY_TOP_N = 10           # 테마별로 보여줄 상위 종목 수(시장별)
POLICY_RANK_COMPONENTS = {
    # 한국은 네이버 재무정보 표에서 얻은 매출성장률/영업이익성장률을, 미국은 yfinance의
    # 매출성장률/순이익성장률(이익성장률)을 씁니다. composite_score는 종목별로 값이 없는
    # 항목은 자동으로 빼고 나머지 항목끼리 가중치를 재정규화하므로, 시장마다 사용 가능한
    # 항목이 달라도(예: 한국은 영업이익성장률, 미국은 이익성장률) 하나의 계산식으로 처리됩니다.
    "매출성장률": (0.5, False),
    "영업이익성장률": (0.25, False),
    "이익성장률": (0.25, False),
}

# ==========================================================
# 매크로 지표 (2026.09.11 신설, 참고 표시 전용 - 스코어링에는 반영하지 않음)
# ==========================================================
# 기준금리·환율·원자재 가격을 대시보드 상단에 참고 정보로만 보여줍니다. 별도 API 키 없이 접근
# 가능한 FRED(세인트루이스 연은)의 CSV 다운로드 엔드포인트를 사용합니다(공식 API 키 발급 없이
# 그래프용 CSV를 받는 방식으로, 일부 시리즈는 몇 주 지연이 있을 수 있음). 한국은행 기준금리를
# 정확히 집계하는 무료·무인증 API는 찾지 못해, 실제 콜금리(초단기 금리로 기준금리와 거의 같이
# 움직임)로 대체하고 화면에 "근사치"임을 명시합니다 - 정확한 한국은행 기준금리가 꼭 필요하면
# 한국은행 ECOS(https://ecos.bok.or.kr)에서 무료 API 키를 발급받아 연동을 확장할 수 있습니다.
MACRO_SERIES = [
    ("US_RATE", "미국 기준금리 (Fed, %)", "FEDFUNDS", False),
    ("KR_RATE", "한국 콜금리 (기준금리 근사치, %)", "IRSTCI01KRM156N", False),
    ("USDKRW", "원/달러 환율", "DEXKOUS", False),
    ("WTI", "WTI 유가 (달러/배럴)", "DCOILWTICO", False),
]

TOP_N = 30
OUTPUT_DIR = "output"


def fetch_macro_indicators() -> dict:
    """기준금리·환율·원자재 가격을 FRED CSV 엔드포인트에서 가져옵니다(참고 표시 전용,
    스코어링에는 전혀 반영하지 않음). 시리즈 하나가 실패해도 나머지에는 영향이 없도록
    항목별로 개별 예외 처리합니다 - 이 프로젝트의 다른 외부 데이터 수집과 동일한 원칙입니다."""
    result = {}
    for key, label, series_id, _ in MACRO_SERIES:
        try:
            resp = requests.get(
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=10,
            )
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            series = pd.to_numeric(df[series_id], errors="coerce").dropna()
            date_col = df.columns[0]
            if len(series):
                last_idx = series.index[-1]
                result[key] = {
                    "label": label, "value": float(series.iloc[-1]),
                    "as_of": str(df.loc[last_idx, date_col]),
                }
            else:
                result[key] = {"label": label, "value": None, "as_of": None}
        except Exception as e:
            print(f"[MACRO] {label}({series_id}) 조회 실패(참고용 항목이라 무시하고 계속 진행): {e}")
            result[key] = {"label": label, "value": None, "as_of": None}
    return result


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


def _fetch_kr_listing_kind(market: str) -> pd.DataFrame:
    """FinanceDataReader의 모든 경로가 막혔을 때 쓰는 최종 대체 경로: KRX 상장공시시스템(KIND)의
    "상장법인목록" 다운로드를 직접 스크래핑합니다. 시가총액 정보가 없어 이 경로로 얻은 종목은
    시가총액 사전 필터를 건너뛰고(수가 많아 이후 yfinance 조회가 다소 늘어날 수 있음) 최종
    시가총액 필터(build_kr_universe 단계)에서 걸러지도록 둡니다. 실패해도 예외를 던지지 않고
    빈 DataFrame을 반환합니다(이 경로마저 막혀도 다른 시장 처리는 계속 진행되도록)."""
    market_type = "stockMkt" if market == "KOSPI" else "kosdaqMkt"
    url = f"https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&marketType={market_type}"
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        tables = pd.read_html(resp.content, encoding="euc-kr")
        raw = tables[0].rename(columns={"회사명": "종목명", "종목코드": "티커"})
        if "티커" not in raw.columns or "종목명" not in raw.columns:
            print(f"[KR] {market} KIND 대체 경로 컬럼 인식 실패: {list(raw.columns)}")
            return pd.DataFrame()
        raw["티커"] = raw["티커"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        df = raw[["티커", "종목명"]].dropna()
        df["시장"] = market
        print(f"[KR] {market} KIND 대체 경로로 {len(df)}개 종목 확보(시가총액 사전필터 미적용)")
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"[KR] {market} KIND 대체 경로 요청 실패: {e}")
        return pd.DataFrame()


def get_kr_ticker_universe() -> pd.DataFrame:
    """FinanceDataReader로 KOSPI/KOSDAQ 종목 리스트(티커·종목명·시가총액)를 가져옵니다.
    시가총액 컬럼이 있으면 KR_MIN_MARKETCAP 미만 종목을 미리 걸러내 이후 yfinance 조회 건수를
    (전종목 약 2,000~2,500개 -> 대형주 위주 수백 개로) 줄입니다.

    2026.09월 KRX 쪽 이슈로 fdr.StockListing("KOSPI")/("KOSDAQ")가 HTTP 404를 반환해 종목
    리스트를 하나도 못 가져오는 장애가 있었습니다(재실행해도 재현). 원인은 저희 코드가 아니라
    FinanceDataReader가 내부적으로 의존하는 KRX 데이터 소스 쪽 변경/장애로 추정되며, 언제든
    다시 발생할 수 있어 아래처럼 3단계 대체 경로를 둡니다:
      1) fdr.StockListing(market) - 기존 방식(정상이면 시가총액까지 바로 확보)
      2) fdr.StockListing("KRX") - 시장 구분 없이 통합 조회 후 여기서 시장별로 나눔
         (개별 시장 요청 경로만 막혔을 때 다른 내부 경로를 타 우회될 수 있음)
      3) KIND(상장공시시스템) 상장법인목록 직접 스크래핑 - 1)·2)가 모두 FinanceDataReader에
         의존하므로, 그 라이브러리 자체가 완전히 막혔을 때를 대비한 최종 수단(시가총액 정보는
         없음)
    """
    if fdr is None:
        raise ImportError("FinanceDataReader가 설치되어 있지 않습니다.")
    frames = []
    krx_combined = None  # fdr.StockListing("KRX") 결과 캐시(두 시장에서 재사용, 필요할 때만 조회)

    for m in KR_MARKETS:
        listing = None
        try:
            listing = fdr.StockListing(m)
        except Exception as e:
            print(f"[KR] {m} 종목 리스트 수집 실패(1차, StockListing('{m}')): {e}")

        if listing is None or listing.empty:
            if krx_combined is None:
                try:
                    krx_combined = fdr.StockListing("KRX")
                except Exception as e:
                    print(f"[KR] StockListing('KRX') 통합 조회(2차 대체 경로) 실패: {e}")
                    krx_combined = pd.DataFrame()
            if not krx_combined.empty and "Market" in krx_combined.columns:
                sub = krx_combined[krx_combined["Market"].astype(str).str.upper() == m]
                if not sub.empty:
                    print(f"[KR] {m} StockListing('KRX') 통합 조회로 {len(sub)}개 확보(2차 대체 경로)")
                    listing = sub

        if listing is None or listing.empty:
            listing = _fetch_kr_listing_kind(m)

        if listing is None or listing.empty:
            print(f"[KR] {m} 종목 리스트를 모든 경로(1차/2차/3차)에서 확보하지 못해 이번 실행에서는 건너뜁니다.")
            continue

        code_col = next((c for c in ["Code", "Symbol", "티커"] if c in listing.columns), None)
        name_col = next((c for c in ["Name", "종목명"] if c in listing.columns), None)
        cap_col = next((c for c in ["Marcap", "MarketCap"] if c in listing.columns), None)
        if code_col is None or name_col is None:
            print(f"[KR] {m} 종목 리스트 컬럼 인식 실패: {list(listing.columns)}")
            continue
        keep = [code_col, name_col] + ([cap_col] if cap_col else [])
        df = listing[keep].rename(columns={code_col: "티커", name_col: "종목명", **({cap_col: "시가총액_참고"} if cap_col else {})})
        df["시장"] = m
        frames.append(df)

    if not frames:
        raise RuntimeError("KOSPI/KOSDAQ 종목 리스트를 하나도 가져오지 못했습니다(FinanceDataReader/KIND 대체 경로 모두 실패).")
    combined = pd.concat(frames, ignore_index=True)
    if "시가총액_참고" in combined.columns:
        before = len(combined)
        # KIND 대체 경로로 확보한 행은 시가총액_참고가 NaN이라 아래 필터에 안 걸리고 그대로
        # 남습니다(정상 경로로 얻은 종목만 사전 필터링, 나머지는 build_kr_universe의 최종
        # 시가총액 필터 단계에서 걸러짐).
        cap_num = pd.to_numeric(combined["시가총액_참고"], errors="coerce")
        combined = combined[cap_num.isna() | (cap_num >= KR_MIN_MARKETCAP)]
        print(f"[KR] 종목 리스트 {before}개 -> 시가총액 사전 필터 후 {len(combined)}개")
    return combined.reset_index(drop=True)


NAVER_MAX_WORKERS = 10
NAVER_FETCH_TIME_BUDGET_SEC = 900  # 15분 - 이 시간을 넘기면 남은 종목은 건너뛰고 계속 진행


def _run_parallel_with_budget(codes, fetch_fn, empty_result, max_workers=NAVER_MAX_WORKERS,
                               time_budget_sec=NAVER_FETCH_TIME_BUDGET_SEC, label="네이버") -> dict:
    """네이버금융 스크래핑 함수들(밸류에이션/사업개요/재무비율)이 공통으로 쓰는 "시간 예산이
    있는 병렬 수집" 헬퍼입니다.

    2026.09.10에 yfinance 쪽에서 "병렬 조회 단계 하나가 예상보다 오래 걸려 45분 잡 타임아웃
    전체를 날려버리는" 사고를 두 번 겪은 뒤(_fetch_yfinance_info_parallel에 시간 예산을
    추가해 대응), 네이버금융 쪽 병렬 수집 함수들에는 아직 같은 안전장치가 없다는 걸 뒤늦게
    발견했습니다. 네이버금융이 지금까지는 안정적으로 동작해왔지만(PER/PBR/EPS/BPS/사업개요는
    이미 여러 차례 라이브 검증됨), 이 프로젝트에서 반복적으로 확인된 패턴(외부 스크래핑
    대상은 언제든 느려지거나 막힐 수 있음)을 감안해 모든 외부망 병렬 수집 단계에 동일한
    시간 예산 안전장치를 두는 것으로 통일합니다. 예산을 넘기면 남은 종목은 건너뛰고 지금까지
    모은 결과만으로 계속 진행합니다(부분 데이터라도 완주가 완전 실패보다 낫다는 원칙,
    _fetch_yfinance_info_parallel과 동일)."""
    if not codes:
        return {}
    codes = list(dict.fromkeys(str(c) for c in codes))
    results = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    future_map = {executor.submit(fetch_fn, code): code for code in codes}
    done_count = 0
    timed_out = False
    try:
        for future in concurrent.futures.as_completed(future_map, timeout=time_budget_sec):
            code = future_map[future]
            try:
                results[code] = future.result()
            except Exception:
                results[code] = dict(empty_result) if isinstance(empty_result, dict) else empty_result
            done_count += 1
    except concurrent.futures.TimeoutError:
        timed_out = True
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    total = len(codes)
    skipped = total - done_count
    if timed_out or skipped:
        print(f"[{label}] 시간 예산({time_budget_sec}초) 초과로 {skipped}개 종목 조회를 "
              f"건너뛰고 지금까지 모은 결과({len(results)}개)로 계속 진행합니다.")
    return results


_naver_debug_done = False


def _fetch_naver_valuation(code: str) -> dict:
    """yfinance가 한국 종목에는 PER/PBR/EPS/BPS를 거의 채워주지 못해(Yahoo의 KRX 커버리지 한계),
    네이버금융 종목 페이지(finance.naver.com/item/main.naver)에서 이 값들을 보강 수집합니다.

    2026.09.10부터는 KR_DATA_SOURCE 기본값이 "naver"로 바뀌면서(yfinance의 반복적인 차단
    문제로) 이 함수가 한국 종목 펀더멘털의 유일한 소스가 됐습니다. 그래서 기존 PER/PBR/EPS/BPS
    (정확한 <em id="..."> 태그로 파싱 - 이미 실제 운영에서 검증됨) 외에, 배당수익률·시가총액·
    상장주식수도 같은 페이지에서 함께 추출합니다(추가 요청 없음). 다만 이 3개는 정확한 태그
    id를 이 세션에서 확인할 방법이 없어(라이브 접근 불가), 태그 구조가 바뀌어도 잘 견디도록
    "라벨 텍스트 다음에 숫자가 나온다"는 패턴을 페이지 전체 텍스트에서 정규식으로 찾는 방식을
    씁니다 - 태그 구조에 덜 의존적인 대신, 라벨 문구 자체가 바뀌면(가능성 낮음) 실패할 수
    있습니다. 실패해도 예외 없이 NaN을 반환해 파이프라인은 계속 진행됩니다."""
    global _naver_debug_done
    result = {
        "PER": np.nan, "PBR": np.nan, "EPS": np.nan, "BPS": np.nan,
        "DIV": np.nan, "시가총액_naver": np.nan, "상장주식수_naver": np.nan,
    }
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

        page_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

        def _regex_num(pattern):
            m = re.search(pattern, page_text)
            if not m:
                return np.nan
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return np.nan

        result["DIV"] = _regex_num(r"배당수익률\s*([\d.]+)\s*%")
        market_sum_eok = _regex_num(r"시가총액\s+([\d,]+)\s*억원")  # 억원 단위 -> 원 단위로 환산
        if pd.notna(market_sum_eok):
            result["시가총액_naver"] = market_sum_eok * 100_000_000
        result["상장주식수_naver"] = _regex_num(r"상장주식수\s+([\d,]+)")

        if not _naver_debug_done:
            _naver_debug_done = True
            print(f"[DEBUG-NAVER] {code} status={resp.status_code} 결과={result}")
            if pd.isna(result["PER"]):
                print(f"[DEBUG-NAVER] _per 태그를 못 찾음. 페이지 앞부분={resp.text[:800]!r}")
            if pd.isna(result["DIV"]) and pd.isna(result["시가총액_naver"]):
                print("[DEBUG-NAVER] 배당수익률/시가총액 보조 필드 정규식 추출 실패 - "
                      "페이지 라벨 문구가 예상과 다를 수 있음(참고용/보조 필드라 동작에는 영향 없음).")
    except Exception as e:
        if not _naver_debug_done:
            _naver_debug_done = True
            print(f"[DEBUG-NAVER] {code} 요청 실패: {e}")
    return result


def _fetch_naver_valuations_parallel(codes, max_workers=NAVER_MAX_WORKERS) -> dict:
    """여러 종목의 네이버금융 PER/PBR/EPS/BPS/배당수익률 등을 동시에(병렬) 수집합니다.
    종목당 순차 요청은 1,300여 개 종목 기준으로 수십 분이 추가로 걸려
    GitHub Actions의 job timeout(45분)을 넘겨 실행이 취소되는 문제가 있었습니다.
    스레드풀로 동시에 요청해 전체 소요 시간을 크게 줄이고, 시간 예산(_run_parallel_with_budget)
    으로 이 단계 자체가 무한정 늘어나는 것도 막습니다."""
    empty = {
        "PER": np.nan, "PBR": np.nan, "EPS": np.nan, "BPS": np.nan,
        "DIV": np.nan, "시가총액_naver": np.nan, "상장주식수_naver": np.nan,
    }
    return _run_parallel_with_budget(codes, _fetch_naver_valuation, empty,
                                      max_workers=max_workers, label="NAVER-VAL")


_profile_debug_done = False


def _fetch_naver_company_profile(code: str) -> str:
    """네이버금융과 연동된 WiseReport 기업개요 페이지에서 "회사의 주요 사업/제품" 설명 문단을
    스크래핑합니다. yfinance의 longBusinessSummary는 한국 종목에는 거의 채워지지 않아(Yahoo의
    KRX 커버리지 한계), 국문 사업개요는 이 경로로 별도 보강합니다.
    사이트 구조가 바뀌면 실패할 수 있으므로(다른 네이버 스크래핑 함수들과 동일한 위험), 실패 시
    빈 문자열을 반환할 뿐 파이프라인을 중단시키지 않습니다. 처음 1건만 진단 로그를 남깁니다."""
    global _profile_debug_done
    text = ""
    try:
        resp = requests.get(
            f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}",
            headers={"Referer": "https://finance.naver.com/"},
            timeout=10,
        )
        soup = BeautifulSoup(resp.content, "html.parser")
        header = soup.find(string=re.compile("기업개요"))
        if header:
            node = header.find_parent(["th", "td", "span", "div", "b", "strong", "p"])
            hops = 0
            while node is not None and hops < 6:
                sib = node.find_next(["td", "div", "p", "li"])
                if sib is not None:
                    candidate = sib.get_text(" ", strip=True)
                    if len(candidate) > 40:
                        text = candidate
                        break
                node = node.parent
                hops += 1
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 300:
            text = text[:300].rstrip() + "…"
        if not _profile_debug_done:
            _profile_debug_done = True
            print(f"[DEBUG-PROFILE] {code} status={resp.status_code} 추출길이={len(text)}")
            if not text:
                print(f"[DEBUG-PROFILE] 사업개요 추출 실패. 페이지 앞부분={resp.text[:500]!r}")
    except Exception as e:
        if not _profile_debug_done:
            _profile_debug_done = True
            print(f"[DEBUG-PROFILE] {code} 요청 실패: {e}")
    return text


def _fetch_naver_profiles_parallel(codes, max_workers=NAVER_MAX_WORKERS) -> dict:
    """여러 종목의 국문 사업개요를 동시에(병렬) 수집합니다. 원리는
    _fetch_naver_valuations_parallel과 동일합니다(순차 요청 시 타임아웃 위험 + 시간 예산 적용)."""
    return _run_parallel_with_budget(codes, _fetch_naver_company_profile, "",
                                      max_workers=max_workers, label="NAVER-PROFILE")


_ratio_debug_done = False


_EMPTY_NAVER_FINANCIALS = {
    "ROE": np.nan, "부채비율": np.nan,
    "매출액": np.nan, "영업이익": np.nan,
    "매출성장률": np.nan, "영업이익성장률": np.nan,
}


def _fetch_naver_financial_ratio(code: str) -> dict:
    """네이버금융 산하 기업정보(companyinfo.stock.naver.com)의 연간 실적 요약 표에서
    ROE·부채비율(참고용)과 매출액·영업이익(및 전년 대비 성장률 - 2026.09.11부터 "정책 관련
    섹터 안에서 매출·영업이익 실적 기준으로 추천"의 핵심 데이터로 사용)을 함께 수집합니다.

    이 표는 보통 종목별로 연도 컬럼이 여러 개(작년/올해/추정치 등) 나열된 형태입니다. 정확한
    컬럼 순서/개수는 종목마다, 그리고 이 세션에서는 라이브로 직접 확인할 방법이 없어(외부망
    차단) 다를 수 있어, "레이블 텍스트가 포함된 행에서 왼쪽부터 유효한 숫자를 순서대로 모두
    추출한 뒤, 맨 뒤 두 개(가장 최근 두 시점)로 증감률을 계산"하는 방식으로 표 구조 변화에
    최대한 강건하게 만들었습니다. 그래도 실패하면 NaN만 반환하고 예외를 던지지 않습니다 -
    ROE/부채비율은 여전히 참고 표시용이라 실패해도 무방하지만, 매출액/영업이익(성장률)은
    이제 정책테마 섹션 순위에 쓰이므로 이 값이 자주 비면 그 섹션 정확도가 떨어집니다. 다음
    실행 로그의 [DEBUG-RATIO] 줄로 실제 파싱 성공 여부를 확인하세요."""
    global _ratio_debug_done
    result = dict(_EMPTY_NAVER_FINANCIALS)
    try:
        resp = requests.get(
            "https://companyinfo.stock.naver.com/v1/company/ajax/cF1001.aspx",
            params={"cmp_cd": code, "fin_typ": "0", "freq_typ": "Y"},
            headers={"Referer": "https://finance.naver.com/"},
            timeout=10,
        )
        tables = pd.read_html(io.StringIO(resp.text))
        if tables:
            table = tables[0]
            first_col = table.iloc[:, 0].astype(str)

            def _row_numbers(row_idx):
                row = table.iloc[row_idx, 1:]
                nums = pd.to_numeric(
                    row.astype(str).str.replace(",", "").str.replace("%", "").str.strip(),
                    errors="coerce",
                ).dropna()
                return list(nums)

            def _last_numeric(row_idx):
                nums = _row_numbers(row_idx)
                return float(nums[-1]) if nums else np.nan

            def _yoy_growth(row_idx):
                """맨 뒤 두 시점(가장 최근 확정치 기준)으로 전년 대비 증감률을 "비율"(예: 8%
                성장 -> 0.08)로 계산합니다. yfinance의 revenueGrowth 등 이 프로젝트의 다른 모든
                성장률 컬럼과 단위를 맞춰야 _fmt_pct()(내부에서 *100을 해 "%"로 표시) 등
                공용 포맷 함수를 그대로 재사용할 수 있습니다(퍼센트로 이미 변환해서 반환하면
                화면에 8%가 아니라 800%로 표시되는 단위 중복 버그가 납니다).
                직전 값이 0이거나 부호가 바뀌는 경우(적자->흑자 등)는 비율 자체가 의미가 없어
                NaN으로 둡니다."""
                nums = _row_numbers(row_idx)
                if len(nums) < 2:
                    return np.nan, (nums[-1] if nums else np.nan)
                prev, latest = nums[-2], nums[-1]
                if prev == 0 or (prev < 0) != (latest < 0):
                    return np.nan, latest
                return (latest - prev) / abs(prev), latest

            for i, label in first_col.items():
                label_norm = label.replace(" ", "")
                if pd.isna(result["ROE"]) and "ROE" in label.upper():
                    result["ROE"] = _last_numeric(i)
                elif pd.isna(result["부채비율"]) and "부채비율" in label_norm:
                    result["부채비율"] = _last_numeric(i)
                elif pd.isna(result["매출액"]) and label_norm.startswith("매출액"):
                    growth, latest = _yoy_growth(i)
                    result["매출액"], result["매출성장률"] = latest, growth
                elif pd.isna(result["영업이익"]) and label_norm.startswith("영업이익") \
                        and "률" not in label_norm:
                    growth, latest = _yoy_growth(i)
                    result["영업이익"], result["영업이익성장률"] = latest, growth
        if not _ratio_debug_done:
            _ratio_debug_done = True
            print(f"[DEBUG-RATIO] {code} status={resp.status_code} 결과={result}")
            if all(pd.isna(v) for v in result.values()):
                print(f"[DEBUG-RATIO] 재무정보 행을 하나도 못 찾음(표 구조가 예상과 다를 수 "
                      f"있음). 응답 앞부분={resp.text[:500]!r}")
    except Exception as e:
        if not _ratio_debug_done:
            _ratio_debug_done = True
            print(f"[DEBUG-RATIO] {code} 요청/파싱 실패: {e}")
    return result


def _fetch_naver_financial_ratios_parallel(codes, max_workers=NAVER_MAX_WORKERS) -> dict:
    """여러 종목의 ROE/부채비율/매출액/영업이익(및 성장률)을 동시에(병렬) 수집합니다. 원리는
    _fetch_naver_valuations_parallel과 동일합니다(시간 예산 적용).
    KR_FETCH_FINANCIALS=False면 fetch_kr_fundamentals_naver()에서 아예 호출하지 않습니다."""
    return _run_parallel_with_budget(codes, _fetch_naver_financial_ratio, dict(_EMPTY_NAVER_FINANCIALS),
                                      max_workers=max_workers, label="NAVER-RATIO")




def _fetch_yfinance_info(symbol: str):
    """yfinance Ticker.get_info() 하나를 안전하게 호출합니다(실패 시 None). 병렬 수집에서
    워커로 재사용합니다. 매 호출 전 짧은 무작위 지연을 둬 여러 워커의 요청이 같은 순간에
    몰리는 것을 피하고, "crumb"/401 관련 오류로 실패한 경우에는 한 번만(조금 더 긴 지연 후)
    재시도합니다 - crumb 발급 직후 아주 짧은 기간의 일시적 실패인 경우가 많기 때문입니다."""
    try:
        time.sleep(random.uniform(0.1, 0.4))
        return _yf_ticker(symbol).get_info()
    except Exception as e:
        msg = str(e).lower()
        if "crumb" in msg or "401" in msg:
            try:
                time.sleep(random.uniform(1.5, 3.0))
                return _yf_ticker(symbol).get_info()
            except Exception:
                return None
        return None


def _fetch_yfinance_info_parallel(symbols, max_workers=YF_MAX_WORKERS,
                                   time_budget_sec=YF_FETCH_TIME_BUDGET_SEC) -> dict:
    """여러 종목의 yfinance 펀더멘털(get_info)을 동시에(병렬) 수집합니다.

    주의: 동시성을 과하게 높이면(예전 8) Yahoo가 더 빠르게 차단하는 것으로 보여(2026.09.10
    실행에서 crumb 발급 자체가 429로 막히고 이후 전 요청이 "Invalid Crumb" 401로 실패하는
    사고 확인), 네이버금융 병렬 수집보다 훨씬 낮은 동시성(YF_MAX_WORKERS)을 씁니다. 그래도
    Yahoo 쪽 차단이 심한 날에는 대부분 실패할 수 있으므로, 이 함수가 통째로 시간을 너무 오래
    끌어 45분 잡 타임아웃으로 전체 실행이 죽는 일이 없도록 시간 예산(time_budget_sec)을 두고,
    예산을 넘기면 남은 종목 조회를 포기하고 지금까지 모은 결과만으로 계속 진행합니다."""
    if not symbols:
        return {}
    _prime_yf_session(symbols)
    results = {}
    fail_count = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    future_map = {executor.submit(_fetch_yfinance_info, s): s for s in symbols}
    done_count = 0
    timed_out = False
    try:
        for future in concurrent.futures.as_completed(future_map, timeout=time_budget_sec):
            try:
                info = future.result()
            except Exception:
                info = None
            if info:
                results[future_map[future]] = info
            else:
                fail_count += 1
            done_count += 1
    except concurrent.futures.TimeoutError:
        timed_out = True
    finally:
        # wait=False로 즉시 반환합니다: 아직 시작 안 한 작업은 취소되고, 이미 실행 중이던
        # 소수의 요청은 각자의 개별 타임아웃(YF_REQUEST_TIMEOUT_SEC) 내에 백그라운드에서
        # 스스로 끝나도록 둡니다(뒤이어 실행되는 다른 단계들이 어차피 그보다 오래 걸리므로
        # 실질적인 지연 없음). wait=True로 두면 이미 실행 중인 요청이 응답 없이 오래
        # 걸리는 경우 시간 예산을 넘겨서까지 여기서 계속 대기하게 되어 예산을 둔 목적이
        # 무의미해집니다(Python 3.9+ 필요).
        executor.shutdown(wait=False, cancel_futures=True)
    total = len(symbols)
    skipped = total - done_count
    if timed_out or skipped:
        print(f"[YF] 시간 예산({time_budget_sec}초) 초과로 {skipped}개 종목 조회를 건너뛰고 "
              f"지금까지 모은 결과({len(results)}개 성공)로 계속 진행합니다.")
    print(f"[YF] {total}개 요청 -> 성공 {len(results)}개 / 실패 {fail_count}개 / 건너뜀 {skipped}개")
    if total and len(results) < total * 0.2:
        print("[YF] 경고: 성공률이 20% 미만입니다 - Yahoo Finance 쪽 차단/장애 가능성이 높습니다 "
              "(curl_cffi 세션 적용 여부는 실행 로그 맨 위 [YF] 줄에서 확인하세요).")
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

    # 종목 클릭 팝업에 보여줄 "사업분야 요약"용 국문 사업개요도 같은 방식(병렬)으로 미리 수집합니다.
    print(f"[KR] 사업개요(기업정보) 병렬 수집 시작 ({len(universe)}개 종목)")
    profile_map = _fetch_naver_profiles_parallel(universe["티커"].tolist())
    print(f"[KR] 사업개요 수집 완료 ({sum(1 for v in profile_map.values() if v)}개 종목에서 텍스트 확보)")

    # yfinance 펀더멘털(get_info)도 병렬로 미리 수집합니다(과거엔 종목별 순차 요청이었음).
    yf_symbol_of = {str(r["티커"]): f"{str(r['티커'])}{suffix.get(r['시장'], '.KS')}" for _, r in universe.iterrows()}
    print(f"[KR] yfinance 펀더멘털 병렬 수집 시작 ({len(yf_symbol_of)}개 종목)")
    info_by_symbol = _fetch_yfinance_info_parallel(list(yf_symbol_of.values()))

    empty_naver = {"PER": np.nan, "PBR": np.nan, "EPS": np.nan, "BPS": np.nan}
    rows = []
    for _, r in universe.iterrows():
        code, name, market = str(r["티커"]), r["종목명"], r["시장"]
        yf_ticker = yf_symbol_of[code]
        info = info_by_symbol.get(yf_ticker)
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
        # 사업개요: 네이버(국문) 우선, 없으면 yfinance의 longBusinessSummary(영문, 있는 경우만)로 대체
        profile_text = profile_map.get(code) or info.get("longBusinessSummary") or ""
        industry = info.get("industry")
        rows.append({
            "티커": code, "종목명": name, "시장": market, "섹터": info.get("sector"),
            "업종": industry,
            "사업개요": profile_text,
            "정책연계": _match_policy_themes(name, info.get("sector"), industry, profile_text),
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
        # 네트워크 호출은 위에서 이미 병렬로 끝났으므로(info_by_symbol), 여기서는 순수 계산만
        # 하는 루프라 예전처럼 요청 간 대기(time.sleep)를 둘 필요가 없습니다.

    if not rows:
        raise RuntimeError("yfinance로 KR 펀더멘털을 하나도 가져오지 못했습니다.")
    print(f"[KR] yfinance로 {len(rows)}개 종목 펀더멘털 수집 완료")
    return pd.DataFrame(rows)


def fetch_kr_fundamentals_naver(markets=None) -> pd.DataFrame:
    """한국 종목 펀더멘털을 네이버금융에서만 수집합니다(yfinance 미사용, 2026.09.10 기본값).

    2026.09에 yfinance(Yahoo Finance)가 한국 종목 대량 조회 시 반복적으로 차단되는 문제를
    겪었습니다(1차: quoteSummary "Quote not found" 404 대량 실패, 2차: 병렬화 후 오히려
    crumb 인증 자체가 429로 막혀 이후 요청이 전부 401로 실패). "무료 공식 API로 전환"도
    검토했으나 Finnhub/Financial Modeling Prep/Twelve Data 등 주요 무료 API가 한국(KRX)
    데이터를 아예 지원하지 않거나(유료 플랜 전용 - 예: Twelve Data는 한국거래소가 Pro+ 이상
    유료 플랜에서만 열림) 무료 호출 한도가 한국 전종목(1,300여 개)을 하루에 감당하지 못해서
    (예: FMP 무료 250건/일) 이 규모에는 채택할 수 없었습니다.

    대신 이 프로젝트에서 이미 여러 차례 라이브로 검증되며 안정적으로 동작해온(PER/PBR/EPS/BPS,
    사업개요) 네이버금융 스크래핑을 확장해 한국 종목 펀더멘털 전체를 여기서 가져옵니다.

    알려진 트레이드오프(사용자에게 문서로 안내됨):
    - EBITDA/기업가치/순부채/잉여현금흐름은 네이버금융의 일반 페이지에 없어 확보하지 못합니다.
      적정주가 3가지 방식(PER/PBR/EV·EBITDA) 중 한국 종목은 PER·PBR 2가지만 반영되지만,
      기존에도 "모델수 2개 이상"이면 정상 노출되는 구조라 동작 자체에는 문제가 없습니다.
    - ROE/부채비율은 별도 엔드포인트(_fetch_naver_financial_ratios_parallel)로 참고용 시도만
      하며, 이 세션에서는 라이브 검증을 못 했습니다. 실패해도 해당 필드만 비고, 스코어링
      (가치·모멘텀·배당 3팩터)에는 전혀 쓰이지 않는 정보라 추천 순위에는 영향이 없습니다.
    - 업종(GICS 스타일 영문 분류)도 yfinance 없이는 확보할 수 없어 비워둡니다. 국가정책 연계
      태그 매칭(POLICY_THEMES)은 종목명 + 국문 사업개요만으로 계속 동작합니다(오히려 한국어
      키워드 매칭에는 국문 텍스트가 더 적합할 수 있음).
    - 현재주가는 이 함수 단계에서는 비워두고, build_kr_universe()가 모멘텀 수집(이미
      FinanceDataReader 기반이라 yfinance와 무관) 단계에서 얻는 실제 최근 종가로 채웁니다.
    """
    markets = markets or KR_MARKETS
    universe = get_kr_ticker_universe()
    universe = universe[universe["시장"].isin(markets)]

    print(f"[KR] 네이버금융 PER/PBR/EPS/BPS/배당수익률 병렬 수집 시작 ({len(universe)}개 종목)")
    naver_map = _fetch_naver_valuations_parallel(universe["티커"].tolist())
    print(f"[KR] 네이버금융 수집 완료 ({len(naver_map)}개)")

    print(f"[KR] 사업개요(기업정보) 병렬 수집 시작 ({len(universe)}개 종목)")
    profile_map = _fetch_naver_profiles_parallel(universe["티커"].tolist())
    print(f"[KR] 사업개요 수집 완료 ({sum(1 for v in profile_map.values() if v)}개 종목에서 텍스트 확보)")

    if KR_FETCH_FINANCIALS:
        print(f"[KR] 재무정보(ROE/부채비율/매출액/영업이익) 병렬 수집 시작 ({len(universe)}개 종목)")
        ratio_map = _fetch_naver_financial_ratios_parallel(universe["티커"].tolist())
        rev_found = sum(1 for v in ratio_map.values() if pd.notna(v.get("매출성장률")))
        op_found = sum(1 for v in ratio_map.values() if pd.notna(v.get("영업이익성장률")))
        print(f"[KR] 재무정보 수집 완료(매출성장률 확보 {rev_found}개, 영업이익성장률 확보 "
              f"{op_found}개 / {len(universe)}개 중 - 정책테마 섹션 순위에 사용)")
    else:
        print("[KR] 재무정보(ROE/부채비율/매출액/영업이익) 수집은 이번 실행에서 비활성화 상태입니다 "
              "(CONFIG의 KR_FETCH_FINANCIALS=True로 바꾸면 다시 시도합니다 - 단, 꺼두면 정책테마 "
              "섹션에서 한국 종목 순위를 매길 실적 데이터가 없어집니다).")
        ratio_map = {}

    empty_naver = {
        "PER": np.nan, "PBR": np.nan, "EPS": np.nan, "BPS": np.nan,
        "DIV": np.nan, "시가총액_naver": np.nan, "상장주식수_naver": np.nan,
    }
    empty_ratio = dict(_EMPTY_NAVER_FINANCIALS)
    rows = []
    for _, r in universe.iterrows():
        code, name, market = str(r["티커"]), r["종목명"], r["시장"]
        naver_val = naver_map.get(code, empty_naver)
        ratio = ratio_map.get(code, empty_ratio)

        # 시가총액: FinanceDataReader 상장 리스트에서 얻은 값을 우선 사용합니다(전종목을 한 번에
        # 이미 받아온 값이라 정확도가 높고 추가 요청도 필요 없음). 그 값이 없는 종목(3단계 KIND
        # 대체 경로로 들어와 시가총액 정보 자체가 없는 경우)만 네이버금융 페이지 값으로 보강합니다.
        market_cap = pd.to_numeric(r.get("시가총액_참고"), errors="coerce")
        if pd.isna(market_cap):
            market_cap = naver_val.get("시가총액_naver", np.nan)

        shares = naver_val.get("상장주식수_naver", np.nan)
        profile_text = profile_map.get(code) or ""
        # 업종(영문 GICS 분류)은 yfinance 없이는 대체 소스가 없어 비워둡니다.
        # _match_policy_themes는 종목명 + 국문 사업개요만으로도 동작합니다.
        industry = None
        rows.append({
            "티커": code, "종목명": name, "시장": market, "섹터": None,
            "업종": industry,
            "사업개요": profile_text,
            "정책연계": _match_policy_themes(name, None, industry, profile_text),
            "시가총액": market_cap, "상장주식수": shares, "발행주식수": shares,
            "PER": naver_val["PER"], "PBR": naver_val["PBR"],
            "EPS": naver_val["EPS"], "BPS": naver_val["BPS"],
            "DIV": naver_val.get("DIV", np.nan), "배당성향": np.nan,
            "부채비율": ratio.get("부채비율", np.nan), "ROE": ratio.get("ROE", np.nan),
            "매출액": ratio.get("매출액", np.nan), "영업이익": ratio.get("영업이익", np.nan),
            "매출성장률": ratio.get("매출성장률", np.nan),
            "영업이익성장률": ratio.get("영업이익성장률", np.nan),
            "이익성장률": np.nan,
            "잉여현금흐름": np.nan, "FCF수익률": np.nan,
            "EBITDA": np.nan, "기업가치": np.nan,
            "순부채": np.nan,
            # 현재주가는 아직 모름 - build_kr_universe()가 모멘텀 수집 단계에서 실제 최근
            # 종가로 채웁니다(2026.09.10부터: 기존에는 yfinance 현재가 -> 없으면 시가총액/
            # 상장주식수 역산이었으나, 이제는 실제 종가를 우선 소스로 씀 - 더 정확함).
            "현재주가": np.nan,
        })

    if not rows:
        raise RuntimeError("네이버금융으로 KR 펀더멘털을 하나도 가져오지 못했습니다.")
    df = pd.DataFrame(rows)
    with_cap = df["시가총액"].notna().sum()
    print(f"[KR] 네이버금융 기반으로 {len(df)}개 종목 펀더멘털 조립 완료 "
          f"(시가총액 확보 {with_cap}/{len(df)}개 - 대부분 FinanceDataReader 리스트에서 옴)")
    return df


def fetch_kr_fundamentals(markets=None) -> pd.DataFrame:
    if KR_DATA_SOURCE == "pykrx":
        return fetch_kr_fundamentals_pykrx(markets=markets)
    if KR_DATA_SOURCE == "yfinance":
        return fetch_kr_fundamentals_yfinance(markets=markets)
    return fetch_kr_fundamentals_naver(markets=markets)


def fetch_kr_momentum(tickers, lookback_days=380):
    """모멘텀(3/6/12개월 수익률)과 함께, 이미 받아온 일별 시세에서 공짜로 뽑을 수 있는
    "가장 최근 거래일"의 거래량/거래대금(+그 날짜), 그리고 관심종목 추적(과거 날짜 지정 시
    그날의 실제 종가를 기준가로 쓰기 위한) 일별 종가 이력을 함께 수집합니다. 거래대금은
    여러 날짜를 평균 내지 않고 데이터가 실제로 존재하는 가장 최근 거래일 하루치 값을 그대로
    쓰고, 그 기준 날짜를 함께 저장해 화면에 명시합니다(평균을 쓰면 "오늘 기준" 감각과 어긋나
    혼란을 줄 수 있어 단일 최근일 값으로 통일). 추가 네트워크 호출 없이 기존에 이미 하던 시세
    조회 결과를 재활용하는 것이라 실행 시간에 미치는 영향은 없습니다.

    2026.09.10부터(KR_DATA_SOURCE="naver" 기본값) 현재주가도 여기서 함께 채웁니다: 이전에는
    yfinance의 실시간 현재가를 우선 쓰고 없으면 시가총액/상장주식수로 역산했지만, 이제 한국
    종목은 yfinance를 아예 쓰지 않으므로 이 함수가 이미 받아온 최근 종가(last)를 현재주가의
    1차 소스로 씁니다(실시간은 아니지만 실제 거래된 값이라 역산치보다 정확함) - 추가 네트워크
    호출 없음."""
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
                "현재주가_모멘텀": float(last),
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

    # 현재주가: 모멘텀 수집 단계에서 얻은 실제 최근 종가(현재주가_모멘텀)를 1차 소스로 쓰고,
    # 그마저 없는 경우에만(모멘텀 수집 자체가 실패한 극소수 종목) 시가총액/상장주식수 역산으로
    # 채웁니다. 2026.09.10 이전에는 yfinance 실시간 현재가가 1차 소스였으나, 한국 종목은
    # yfinance를 더 이상 쓰지 않으므로 순서를 바꿨습니다(실제 거래된 종가가 역산치보다 정확함).
    if "현재주가_모멘텀" in df.columns:
        df["현재주가"] = df["현재주가"].fillna(df["현재주가_모멘텀"]) if "현재주가" in df.columns \
            else df["현재주가_모멘텀"]
    if {"시가총액", "상장주식수"}.issubset(df.columns):
        shares = df["상장주식수"].where(df["상장주식수"] > 0)
        derived_price = df["시가총액"] / shares
        df["현재주가"] = df["현재주가"].fillna(derived_price) if "현재주가" in df.columns else derived_price
    if "현재주가_모멘텀" in df.columns:
        df = df.drop(columns=["현재주가_모멘텀"])

    # BPS 보강: 네이버금융 페이지에서 BPS만 못 잡히는 경우가 있어(구조 차이 추정), 이제 실제
    # 최근 종가를 확보한 뒤 PBR = 현재가/BPS 관계를 거꾸로 이용해 역산합니다(2026.08.31에
    # yfinance 버전에서 쓰던 것과 같은 보강 로직 - 이제는 모멘텀 병합 이후에 수행).
    if {"BPS", "PBR", "현재주가"}.issubset(df.columns):
        need_bps = df["BPS"].isna() & df["현재주가"].notna() & df["PBR"].notna() & (df["PBR"] != 0)
        df.loc[need_bps, "BPS"] = df.loc[need_bps, "현재주가"] / df.loc[need_bps, "PBR"]

    # EPS 보강(2026.09.11 신규): 네이버금융 페이지의 <em id="_eps"> 태그는 이 프로젝트에서
    # 실거래로 검증된 적이 없어(기존에 "검증됨"으로 표시했던 것은 PER/PBR 자체 값이었지, EPS는
    # 아니었습니다 - 적정주가 계산 전까지는 EPS를 실제로 쓰는 곳이 없었기 때문), 대시보드에서
    # "적정주가 저평가 TOP 15" 한국 종목이 전부 "데이터 없음"으로 뜨는 문제의 유력한 원인으로
    # 지목됩니다. BPS와 완전히 동일한 방식으로 PER = 현재가/EPS 관계를 거꾸로 이용해
    # EPS = 현재가/PER로 역산해두면, 이미 검증된 PER·현재가 값만으로 EPS 없이도 PER 기반
    # 적정주가 모델을 항상 계산할 수 있어 이 문제를 근본적으로 우회합니다.
    if {"EPS", "PER", "현재주가"}.issubset(df.columns):
        need_eps = df["EPS"].isna() & df["현재주가"].notna() & df["PER"].notna() & (df["PER"] != 0)
        df.loc[need_eps, "EPS"] = df.loc[need_eps, "현재주가"] / df.loc[need_eps, "PER"]

    df = estimate_fair_value(df, group_cols=["시장"])
    if "적정주가_모델수" in df.columns:
        dist = df["적정주가_모델수"].value_counts().sort_index().to_dict()
        print(f"[KR] 적정주가 모델수 분포(0/1/2개 방식 반영된 종목 수): {dist} "
              f"(대시보드 '적정주가 저평가 TOP'은 모델수 2 이상만 노출)")
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


def fetch_nasdaq_listing_lite() -> pd.DataFrame:
    """나스닥 상장 종목 "검색"용 가벼운 리스트(종목명·티커만)를 수집합니다. 저희가 매일
    점수를 매기는 S&P500 유니버스와는 별개로, 대시보드 검색창에서 S&P500 밖의 나스닥 종목도
    찾을 수 있게 하기 위한 용도입니다. 목록 하나를 한 번에 받아오는 방식이라(종목별 개별 요청
    없음) 종목 수가 수천 개여도 실행 시간에 미치는 영향이 거의 없습니다 - 다만 재무 지표는
    전혀 없어서, 검색으로 열어도 "상세 재무 정보 없음 + 외부 링크"만 보여줍니다(대시보드
    안내문에도 명시). 실패해도 예외를 던지지 않고 빈 목록을 반환해 나머지 실행에는 영향이
    없도록 합니다."""
    if fdr is None:
        return pd.DataFrame(columns=["티커", "종목명"])
    try:
        listing = fdr.StockListing("NASDAQ")
    except Exception as e:
        print(f"[US] 나스닥 검색용 종목 리스트 수집 실패: {e}")
        return pd.DataFrame(columns=["티커", "종목명"])
    if listing is None or listing.empty:
        print("[US] 나스닥 검색용 종목 리스트가 비어 있습니다.")
        return pd.DataFrame(columns=["티커", "종목명"])
    code_col = next((c for c in ["Symbol", "Code"] if c in listing.columns), None)
    name_col = next((c for c in ["Name"] if c in listing.columns), None)
    if code_col is None or name_col is None:
        print(f"[US] 나스닥 검색용 종목 리스트 컬럼 인식 실패: {list(listing.columns)}")
        return pd.DataFrame(columns=["티커", "종목명"])
    out = listing[[code_col, name_col]].rename(columns={code_col: "티커", name_col: "종목명"})
    out = out.dropna(subset=["티커"]).drop_duplicates(subset=["티커"])
    print(f"[US] 나스닥 검색용 종목 리스트 {len(out)}개 확보(재무데이터 없이 이름·티커만)")
    return out.reset_index(drop=True)


def fetch_us_fundamentals(tickers) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance가 설치되어 있지 않습니다.")
    print(f"[US] yfinance 펀더멘털 병렬 수집 시작 ({len(tickers)}개 종목)")
    info_by_symbol = _fetch_yfinance_info_parallel(tickers)
    rows = []
    for t in tickers:
        info = info_by_symbol.get(t)
        if not info:
            continue
        market_cap = info.get("marketCap")
        if market_cap is None or market_cap < US_MIN_MARKETCAP:
            continue
        fcf = info.get("freeCashflow")
        total_debt = info.get("totalDebt")
        total_cash = info.get("totalCash")
        net_debt = (total_debt - total_cash) if (total_debt is not None and total_cash is not None) else np.nan
        industry = info.get("industry")
        profile_text = info.get("longBusinessSummary") or ""
        revenue = info.get("totalRevenue")
        op_margin = info.get("operatingMargins")
        # 영업이익 자체는 yfinance info에 직접 없어, 매출×영업이익률로 근사합니다(참고용 표시).
        # 영업이익 "성장률"은 이 근사치로는 신뢰하기 어려워(전년도 마진을 모름) 별도로 만들지
        # 않고, 정책테마 순위에는 기존처럼 매출성장률/이익성장률(순이익 성장률)을 사용합니다.
        op_income = (revenue * op_margin) if (revenue is not None and op_margin is not None) else np.nan
        rows.append({
            "티커": t, "종목명": info.get("shortName", t), "섹터": info.get("sector"),
            "업종": industry,
            "사업개요": profile_text,
            "정책연계": _match_policy_themes(info.get("shortName", t), info.get("sector"), industry, profile_text),
            "시가총액": market_cap, "PER": info.get("trailingPE"), "PBR": info.get("priceToBook"),
            "부채비율": info.get("debtToEquity"), "ROE": info.get("returnOnEquity"),
            "매출액": revenue, "영업이익": op_income, "영업이익성장률": np.nan,
            "매출성장률": info.get("revenueGrowth"), "이익성장률": info.get("earningsGrowth"),
            "배당수익률": info.get("dividendYield"), "배당성향": info.get("payoutRatio"),
            "잉여현금흐름": fcf, "FCF수익률": (fcf / market_cap) if (fcf is not None and market_cap) else np.nan,
            # 적정주가(상대가치평가) 추정에 사용
            "현재주가": info.get("currentPrice") or info.get("regularMarketPrice"),
            "EPS": info.get("trailingEps"), "BPS": info.get("bookValue"),
            "EBITDA": info.get("ebitda"), "기업가치": info.get("enterpriseValue"),
            "순부채": net_debt, "발행주식수": info.get("sharesOutstanding"),
        })
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
            hist_df = _yf_ticker(t).history(period="13mo")
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
    # 2026.09.11: 동종그룹 기준을 GICS 대분류 "섹터"(11개, 예: Communication Services 하나에
    # 저성장 통신사와 고PER 빅테크가 뒤섞임)에서 훨씬 좁은 "업종"(yfinance industry, 100여개)
    # 으로 변경. 실제 배포된 대시보드에서 Charter Communications·Berkshire Hathaway·FIS 등
    # 서로 업종이 전혀 다른 종목들이 나란히 괴리율 +300%(상한 클리핑)로 뜨는 것을 확인 -
    # "섹터" 단위로 묶으면 그룹 안에 고PER 종목이 하나만 섞여도 중앙값이 크게 왜곡돼(중앙값은
    # 평균보다 이상치에 강하지만, 그룹 표본이 작을수록 극단치 1~2개가 중앙값 자체를 바꿔버림)
    # 실제로는 정상 밸류에이션인 종목까지 "심각한 저평가"로 잘못 표시되는 것으로 추정됨. "업종"
    # 그룹 표본이 5개(FAIR_VALUE_MIN_PEER_GROUP) 미만이면 기존처럼 전체 시장 중앙값으로
    # 자동 대체되므로, 그룹이 너무 잘게 쪼개지는 부작용은 이미 방어돼 있음.
    df = estimate_fair_value(df, group_cols=["업종"])
    if "적정주가_모델수" in df.columns:
        dist = df["적정주가_모델수"].value_counts().sort_index().to_dict()
        gap = df["괴리율"].dropna()
        extreme = int((gap.abs() >= FAIR_VALUE_MAX_UPSIDE * 0.99).sum()) if len(gap) else 0
        print(f"[US] 적정주가 모델수 분포: {dist}, 괴리율 상한(±{FAIR_VALUE_MAX_UPSIDE*100:.0f}%) "
              f"근접/클리핑 종목 수: {extreme}개")
    return df, price_history


# ==========================================================
# 엑셀 리포트 출력
# ==========================================================
def export(df: pd.DataFrame) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(OUTPUT_DIR, f"stock_screening_{ts}.xlsx")

    display_cols = [c for c in [
        "전체순위", "시장구분", "티커", "종목명", "섹터", "업종", "시가총액",
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


def _card(row, show_fair_value=False, show_volume=False, show_policy=False) -> str:
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
    elif show_policy:
        rev_g = _fmt_pct(row.get("매출성장률"))
        # 한국은 영업이익성장률, 미국은 (영업이익성장률이 없어) 이익성장률(순이익성장률)을 보여줍니다.
        op_g = row.get("영업이익성장률")
        op_label = "영업이익성장률"
        if pd.isna(op_g):
            op_g = row.get("이익성장률")
            op_label = "이익성장률"
        op_g_str = _fmt_pct(op_g)
        pscore = _fmt(row.get("정책실적점수"))
        extra_metrics = (f'<span>매출성장률 {rev_g}</span><span>{op_label} {op_g_str}</span>'
                          f'<span>실적점수 {pscore}</span>')
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

    # 정책테마(2026.09.11 신설)는 다른 섹션과 달리 "정책연계 태그가 있는 종목 중에서만" 순위를
    # 매기므로, 위 공통 루프(_LABEL_SECTIONS)로는 표현할 수 없어 별도로 처리합니다.
    if {"정책연계", "정책실적점수", "시장구분"}.issubset(df.columns):
        has_theme = df["정책연계"].apply(lambda v: isinstance(v, (list, tuple)) and len(v) > 0)
        base = df[has_theme & df["정책실적점수"].notna()]
        for market in ("KR", "US"):
            sub = base[base["시장구분"] == market]
            if sub.empty:
                continue
            top = sub.sort_values("정책실적점수", ascending=False).head(POLICY_TOP_N)
            for _, r in top.iterrows():
                labels.setdefault(_card_key(r), []).append("정책테마")
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


def _build_extra_nasdaq_search_json(existing_us_tickers) -> str:
    """검색 전용 나스닥 라이트 목록(fetch_nasdaq_listing_lite)에서, 이미 우리가 점수까지 매겨
    DETAILS/PRICES에 들어있는 S&P500 종목은 제외하고 나머지만 {키: {종목명, 티커}} 형태로
    담습니다. 재무 데이터가 없으므로 현재주가 등은 아예 넣지 않고(클라이언트에서 "상세정보
    없음"으로 처리), 페이지 용량을 최소화합니다."""
    listing = fetch_nasdaq_listing_lite()
    if listing.empty:
        return json.dumps({}, ensure_ascii=False)
    extra = {}
    for _, r in listing.iterrows():
        ticker = str(r.get("티커", "")).strip()
        if not ticker or ticker in existing_us_tickers:
            continue
        key = f"US_{ticker}"
        if key in extra:
            continue
        extra[key] = {"종목명": str(r.get("종목명", "")), "티커": ticker}
    return json.dumps(extra, ensure_ascii=False)


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


def _business_summary_html(row) -> str:
    """사업분야 요약 팝업 내용: 업종, 사업개요(주력분야/주요제품 텍스트), 국가정책 연계 태그.
    "주거래처"는 무료로 안정적으로 얻을 수 있는 자동화 데이터 소스가 없어(공시 원문을 매번
    파싱해야 함) 이번 버전에는 포함하지 않았습니다 - 사업개요 본문에 주요 고객사가 언급된
    경우에는 그 문장이 그대로 보이니 참고할 수 있습니다."""
    market = row.get("시장구분", "US")
    profile = row.get("사업개요")
    profile_html = html.escape(str(profile)).replace("\n", "<br>") if profile and pd.notna(profile) else \
        "아직 확보하지 못했습니다(다음 자동 갱신 때 다시 시도합니다)."
    lang_note = "" if market == "KR" or not profile else '<p class="biz-note">※ Yahoo Finance 원문(영문)입니다.</p>'

    themes = row.get("정책연계")
    if isinstance(themes, (list, tuple)) and len(themes) > 0:
        tags_html = "".join(f'<span class="tag-pill">{html.escape(t)}</span>' for t in themes)
    else:
        tags_html = '<span class="tag-pill tag-empty">매칭된 테마 없음</span>'

    rows = _detail_row("업종", html.escape(str(row.get("업종"))) if pd.notna(row.get("업종")) else "-")
    return (
        f'<div class="dgroup"><h4>사업분야</h4>{rows}'
        f'<div class="biz-summary">{profile_html}</div>{lang_note}'
        f'<div class="policy-tags">{tags_html}</div>'
        f'<p class="policy-note">※ 국가정책 연계는 업종·사업개요 텍스트의 키워드를 자동으로 매칭한 '
        f'참고용 태그입니다(공식 정책 연관성 확인 아님). 투자 판단 전 최근 정책 뉴스/공시를 '
        f'직접 확인하세요.</p></div>'
    )


def _detail_payload(row) -> dict:
    """카드를 클릭했을 때 보여줄 종목 세부정보(사업분야 요약 + 회계/재무 지표) HTML 조각을 만듭니다."""
    market = row.get("시장구분", "US")
    ticker = str(row.get("티커", ""))
    name = html.escape(str(row.get("종목명", "")))
    sector_val = row.get("섹터")
    sector = html.escape(str(sector_val)) if pd.notna(sector_val) else "-"

    business = _business_summary_html(row)

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
        _detail_row("매출액", _fmt_money(row.get("매출액"), market)),
        _detail_row("매출성장률", _fmt_pct(row.get("매출성장률"))),
        _detail_row("영업이익", _fmt_money(row.get("영업이익"), market)),
        _detail_row("영업이익성장률", _fmt_pct(row.get("영업이익성장률"))),
        _detail_row("이익성장률(순이익)", _fmt_pct(row.get("이익성장률"))),
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
        _detail_row("정책실적점수(매출·영업이익 성장 기준)", _fmt(row.get("정책실적점수"))),
    ])

    body = "".join([
        business,
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


def _macro_bar_html(macro: dict) -> str:
    """기준금리·환율·원자재 참고 지표를 헤더 아래에 작은 정보 띠로 보여줍니다.
    조회에 실패한 항목은 조용히 건너뛰고(대시보드 전체가 깨지지 않도록), 하나라도 값이
    있으면 표시하고 전부 실패했으면 이 영역 자체를 비웁니다."""
    if not macro:
        return ""
    chips = []
    for key, label, series_id, _ in MACRO_SERIES:
        item = macro.get(key) or {}
        value = item.get("value")
        if value is None:
            continue
        if key == "USDKRW":
            value_str = f"{value:,.1f}원"
        elif key == "WTI":
            value_str = f"${value:,.1f}"
        else:
            value_str = f"{value:.2f}%"
        as_of = item.get("as_of") or ""
        chips.append(f'<span class="macro-chip">{html.escape(label)} <b>{value_str}</b>'
                     f'<i>({html.escape(as_of)})</i></span>')
    if not chips:
        return ""
    return f'<div class="macro-bar">{"".join(chips)}</div>'


def _policy_sections_html(df: pd.DataFrame) -> str:
    """정책테마별로 매출성장률·영업이익성장률(정책실적점수) 기준 TOP 종목을 보여주는,
    2026.09.11 신설 "정책테마" 섹션의 본문을 만듭니다. 테마 하나당 하나의 하위 블록으로
    묶어 렌더링하며, 어떤 테마에도 매칭된 종목이 없으면 그 테마는 건너뜁니다."""
    if "정책연계" not in df.columns:
        return '<section id="policy"><h2>\U0001F3DB️ 정책테마 추천</h2><p class="empty">데이터 없음</p></section>'

    blocks = []
    for label, _keywords in POLICY_THEMES:
        mask = df["정책연계"].apply(lambda v, lbl=label: isinstance(v, (list, tuple)) and lbl in v)
        theme_df = df[mask & df["정책실적점수"].notna()]
        market_groups = []
        any_data = False
        for market, mlabel in (("KR", "\U0001F1F0\U0001F1F7 한국"), ("US", "\U0001F1FA\U0001F1F8 미국")):
            sub = theme_df[theme_df["시장구분"] == market] if "시장구분" in theme_df.columns else theme_df.iloc[0:0]
            if sub.empty:
                cards_html = '<p class="empty">데이터 없음</p>'
            else:
                any_data = True
                cards_html = "".join(
                    _card(r, show_policy=True)
                    for _, r in sub.sort_values("정책실적점수", ascending=False).head(POLICY_TOP_N).iterrows()
                )
            market_groups.append(
                f'<div class="market-group" data-market="{market}">'
                f'<h3 class="market-h">{mlabel}</h3><div class="cards">{cards_html}</div></div>'
            )
        if any_data:
            blocks.append(
                f'<div class="policy-theme"><h3 class="policy-theme-title">{html.escape(label)}</h3>'
                f'{"".join(market_groups)}</div>'
            )
    body = "".join(blocks) if blocks else '<p class="empty">정책테마와 매칭되면서 실적 데이터도 확보된 종목이 아직 없습니다.</p>'
    return (f'<section id="policy"><h2>\U0001F3DB️ 정책테마 추천 (매출·영업이익 성장 기준)</h2>'
            f'<p class="policy-section-note">아래 10개 정책·산업 테마에 매칭된 종목만 모아, 그 안에서'
            f' 매출성장률·영업이익성장률이 높은 순으로 보여줍니다. PER/PBR 같은 밸류에이션이나'
            f' 주가 모멘텀은 이 순위에 반영되지 않습니다.</p>{body}</section>')


def build_dashboard(df: pd.DataFrame, price_history: dict = None, macro: dict = None,
                     out_path: str = "docs/index.html") -> str:
    # GitHub Actions 러너는 UTC로 동작하므로, 화면에 "(KST)"라고 표시하려면 명시적으로 9시간을
    # 더해 실제 한국시간으로 변환해야 합니다(이전에는 UTC 시각을 KST라고 잘못 표시하던 버그가
    # 있었습니다 - 최대 9시간까지 실제 갱신시각과 화면 표시가 어긋났습니다).
    ts = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
    w = WEIGHTS
    policy = _policy_sections_html(df)
    macro_bar = _macro_bar_html(macro or {})
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
    existing_us_tickers = set(df.loc[df["시장구분"] == "US", "티커"].astype(str)) if "시장구분" in df.columns else set()
    extra_us_json = _build_extra_nasdaq_search_json(existing_us_tickers)

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta name="theme-color" content="#0b0f14">
<!-- 모바일/PC에서 서로 다른(오래된) 실행 결과가 보인다는 문의가 있어 추가함(2026.09.11):
     브라우저가 이 페이지를 공격적으로 캐싱하지 않도록 명시적으로 재검증을 요청합니다.
     GitHub Pages(Fastly CDN) 자체도 몇 분간 캐싱하지만 보통 곧 갱신되고, 이 메타 태그는
     "브라우저가 어제 열어봤던 걸 그대로 재사용하는" 흔한 케이스를 줄여줍니다. 그래도 오래된
     내용이 보이면 새로고침(강제 새로고침/캐시 지우고 재접속)을 한 번 해보세요. -->
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>주식 스크리닝 리포트</title>
<style>
  :root {{ --bg:#0b0f14; --card:#151b22; --text:#e6edf3; --muted:#8b98a5; --accent:#4f8cff; --border:#232b33; }}
  * {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; -webkit-font-smoothing:antialiased; }}
  header {{ padding:18px 16px 10px; position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--border); z-index:10; }}
  header h1 {{ margin:0 0 4px; font-size:19px; }}
  header p {{ margin:0; color:var(--muted); font-size:12.5px; line-height:1.5; }}
  .searchbar {{ position:relative; margin-top:10px; }}
  .searchbar input {{ width:100%; background:var(--card); border:1px solid var(--border); color:var(--text);
    border-radius:10px; padding:10px 12px; font-size:14px; }}
  .searchbar input:focus {{ outline:none; border-color:var(--accent); }}
  .search-results {{ display:none; position:absolute; left:0; right:0; top:calc(100% + 4px); z-index:20;
    background:var(--card); border:1px solid var(--border); border-radius:10px; max-height:340px;
    overflow-y:auto; box-shadow:0 8px 20px rgba(0,0,0,.35); }}
  .search-results.open {{ display:block; }}
  .search-item {{ display:flex; align-items:center; gap:8px; padding:10px 12px; border-bottom:1px solid var(--border); cursor:pointer; }}
  .search-item:last-child {{ border-bottom:none; }}
  .search-item:active {{ background:rgba(79,140,255,.12); }}
  .search-item .name {{ flex:1; }}
  .search-item .price {{ color:var(--muted); font-size:12px; white-space:nowrap; }}
  .search-empty {{ padding:12px; color:var(--muted); font-size:13px; text-align:center; }}
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
  .biz-summary {{ font-size:13px; line-height:1.6; color:var(--text); margin-top:6px; }}
  .biz-note {{ font-size:11px; color:var(--muted); margin:4px 0 0; }}
  .policy-tags {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
  .tag-pill {{ background:rgba(79,140,255,.15); color:var(--accent); border:1px solid var(--accent);
    border-radius:999px; padding:4px 10px; font-size:11.5px; font-weight:600; }}
  .tag-pill.tag-empty {{ background:transparent; color:var(--muted); border-color:var(--border); font-weight:400; }}
  .policy-note {{ font-size:11px; color:var(--muted); margin:8px 0 0; line-height:1.5; }}
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
  .macro-bar {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
  .macro-chip {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
    padding:5px 9px; font-size:11.5px; color:var(--muted); }}
  .macro-chip b {{ color:var(--text); margin:0 3px; }}
  .macro-chip i {{ font-style:normal; opacity:.7; }}
  .policy-section-note {{ color:var(--muted); font-size:12.5px; margin:0 0 14px; line-height:1.5; }}
  .policy-theme {{ margin-bottom:22px; padding-bottom:6px; border-bottom:1px solid var(--border); }}
  .policy-theme-title {{ font-size:14.5px; margin:0 0 8px; color:var(--accent); }}
</style>
</head>
<body>
<header>
  <h1>\U0001F4CA 주식 스크리닝 리포트</h1>
  <p>업데이트: {ts} (KST) · 가치 {w['value']*100:.0f}% · 모멘텀 {w['momentum']*100:.0f}% · 배당 {w['dividend']*100:.0f}%<br>
  <a class="dl" href="reports/latest.xlsx">엑셀 전체 데이터 다운로드</a></p>
  {macro_bar}
  <div class="searchbar">
    <input type="text" id="stockSearch" placeholder="\U0001F50D 종목명·티커 검색 (코스피·코스닥·S&amp;P500)" autocomplete="off">
    <div id="searchResults" class="search-results"></div>
  </div>
</header>
<div class="filterbar cat-filter">
  <button class="filter-btn" data-cat="all">전체</button>
  <button class="filter-btn active" data-cat="policy">\U0001F3DB️ 정책테마</button>
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
<main>{policy}{total}{value}{mom}{div_}{fair}{volume}<section id="performance"><h2>\U0001F4CC 관심종목 추적</h2><div id="perfContainer"></div></section></main>
<footer>본 리포트는 투자자문이 아닙니다. 공개 데이터를 기계적으로 점수화한 참고 자료이며,<br>
투자 판단과 책임은 본인에게 있습니다. 매일 자동 갱신됩니다 (GitHub Actions).<br><br>
<b>"정책테마 추천"(2026.09.11 신설, 기본 화면)</b>은 기존 밸류에이션·모멘텀 위주 점수가 숫자
왜곡으로 실제 사업과 동떨어진 종목을 추천한다는 피드백에 따라 방식을 바꾼 섹션입니다. 반도체·
2차전지·신재생에너지 등 10개 정책·산업 테마 키워드(하단 "사업분야" 설명 참고)에 매칭되는
종목만 후보로 추리고, 그 안에서는 PER·PBR 같은 밸류에이션이 아니라 <b>매출성장률·영업이익
성장률(실제 사업이 얼마나 커지고 있는지)</b>만으로 순위를 매깁니다. 한국은 네이버금융의 연간
실적 표에서, 미국은 Yahoo Finance에서 이 값을 가져오며, 실적 데이터가 아직 확보되지 않은
종목은 이 섹션에서 자동으로 제외됩니다(다른 섹션에는 계속 나타날 수 있음). 정책 테마 매칭은
공식 정책 문서를 매번 대조한 것이 아니라 업종·사업개요 텍스트의 키워드 매칭이라는 점,
매출·영업이익 성장이 곧 주가 상승을 보장하지 않는다는 점은 여전히 유의하세요.<br><br>
헤더의 기준금리·환율·유가 수치는 참고용으로만 표시되며 추천 순위 계산에는 전혀 반영되지
않습니다. 한국 기준금리는 정확한 무료·무인증 공개 API를 찾지 못해 실제 콜금리(초단기 금리로
기준금리와 거의 같이 움직임)로 대체해 표시하고 있습니다.<br><br>
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
위 카테고리 탭(정책테마/종합/가치주/모멘텀/배당/적정주가/거래대금/관심종목)을 누르면 해당 목록만 바로 볼 수 있습니다. 정책테마가 기본(첫 화면) 탭입니다.<br><br>
카드를 탭하면 뜨는 세부정보 상단에 "사업분야" 섹션이 추가되어 업종, 사업개요(주력 제품·사업),
국가정책 연계 태그를 볼 수 있습니다. 국가정책 연계 태그는 업종·사업개요 텍스트에 특정 키워드
(반도체, 2차전지, 신재생에너지 등)가 있는지 자동으로 매칭한 참고용 표시일 뿐, 정부 정책과의
실제 연관성을 확인한 것이 아닙니다. 사업개요는 한국 종목은 네이버금융, 미국 종목은 Yahoo
Finance(영문 원문)에서 가져오며, 소형주 등 일부 종목은 원본 데이터가 없어 비어 있을 수
있습니다. "주요 매출처"는 매번 안정적으로 자동 수집할 free 데이터 소스가 없어 이번 버전에는
넣지 않았습니다(사업개요 문장에 고객사가 언급된 경우 그 문장으로 참고).<br><br>
상단 검색창에서는 저희가 매일 점수까지 매기는 전체 종목(코스피·코스닥 시가총액 1,000억원
이상 + S&amp;P500)은 물론, 나스닥 상장 종목 전체(이름·티커만, 재무 데이터는 없음)까지
종목명·티커로 찾아볼 수 있습니다 - 추천 리스트(TOP 15/20)에 없는 종목도 검색되면 조회
가능합니다. 다만 코스피·코스닥 소형주(시가총액 1,000억원 미만)는 검색 대상이 아니고,
S&amp;P500이 아닌 나스닥 종목은 이름·티커 검색만 되고 PER·PBR 같은 상세 재무 지표와
실시간 현재가는 제공되지 않습니다(대신 Yahoo Finance 링크를 바로 보여드립니다). 뉴욕증권거래소
(NYSE) 상장 종목은 이번 검색 범위에 포함되어 있지 않습니다.</footer>

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
const EXTRA_US = {extra_us_json};
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

// 종목 검색: 서버가 매일 점수까지 매기는 전체 종목 가격표(PRICES, 코스피/코스닥 + S&P500)에
// 더해, 재무 데이터 없이 이름·티커만 가진 나스닥 전체 목록(EXTRA_US)도 검색 대상에 포함합니다
// - 카드로 추천되지 않은 종목도 이 안에 있으면 검색·조회할 수 있습니다. EXTRA_US 쪽 종목은
// 상세 재무 지표/실시간 현재가가 없어 검색 결과에 가격이 "-"로 표시되고, 눌러도 "상세정보
// 없음 + 외부 링크"만 보여줍니다(openDetail의 EXTRA_US 처리 참고).
const SEARCH_INDEX = Object.keys(PRICES).map(function(key) {{
  const p = PRICES[key];
  return {{ key: key, name: p.종목명 || '', ticker: p.티커 || '', market: p.시장구분 || '', price: p.현재주가 }};
}}).concat(Object.keys(EXTRA_US).map(function(key) {{
  const e = EXTRA_US[key];
  return {{ key: key, name: e.종목명 || '', ticker: e.티커 || '', market: 'US', price: null }};
}}));
function searchStocks(query) {{
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return SEARCH_INDEX.filter(function(item) {{
    return item.name.toLowerCase().includes(q) || item.ticker.toLowerCase().includes(q);
  }}).slice(0, 20);
}}
function renderSearchResults(items, query) {{
  const box = document.getElementById('searchResults');
  if (!box) return;
  if (!query.trim()) {{ box.classList.remove('open'); box.innerHTML = ''; return; }}
  if (items.length === 0) {{
    box.innerHTML = '<div class="search-empty">검색 결과가 없습니다(수집 대상 종목이 아니거나 철자를 확인해주세요).</div>';
  }} else {{
    box.innerHTML = items.map(function(item) {{
      const flag = item.market === 'KR' ? '\U0001F1F0\U0001F1F7' : '\U0001F1FA\U0001F1F8';
      return '<div class="search-item" data-key="' + escapeHtml(item.key) + '">' +
        '<span class="flag">' + flag + '</span>' +
        '<span class="name">' + escapeHtml(item.name) + ' <span style="color:var(--muted);font-size:11.5px;">' + escapeHtml(item.ticker) + '</span></span>' +
        '<span class="price">' + fmtMoney(item.price, item.market) + '</span></div>';
    }}).join('');
  }}
  box.classList.add('open');
}}
const searchInput = document.getElementById('stockSearch');
if (searchInput) {{
  searchInput.addEventListener('input', function() {{
    renderSearchResults(searchStocks(searchInput.value), searchInput.value);
  }});
  searchInput.addEventListener('focus', function() {{
    if (searchInput.value.trim()) renderSearchResults(searchStocks(searchInput.value), searchInput.value);
  }});
  document.getElementById('searchResults').addEventListener('click', function(e) {{
    const item = e.target.closest('.search-item');
    if (!item) return;
    openDetail(item.dataset.key);
    document.getElementById('searchResults').classList.remove('open');
    searchInput.value = '';
    searchInput.blur();
  }});
  document.addEventListener('click', function(e) {{
    if (!e.target.closest('.searchbar')) document.getElementById('searchResults').classList.remove('open');
  }});
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
  }} else if (!PRICES[key]) {{
    // EXTRA_US(나스닥 검색 전용, 재무 데이터 없음) 종목 - 현재가가 없어 추적을 시작할 수 없음을
    // 안내하고, 추적 시작 폼 자체는 보여주지 않습니다(눌러도 실패할 게 뻔한 폼을 안 보여주는 편이 나음).
    el.innerHTML = '<div class="track-box"><span class="track-label">이 종목은 실시간 현재가를 제공하지 않아 관심종목 추적을 시작할 수 없습니다.</span></div>';
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

function buildExtraDetail(entry) {{
  // EXTRA_US(나스닥 검색 전용, 재무 데이터 없음) 종목의 상세정보를 그때그때 만듭니다 -
  // 서버가 수천 개 종목 각각에 똑같은 안내문 HTML을 미리 만들어 보내면 페이지 용량만
  // 커지므로, 필요할 때 클라이언트에서 조립합니다.
  const yahooUrl = 'https://finance.yahoo.com/quote/' + encodeURIComponent(entry.티커) + '/';
  return {{
    title: '\U0001F1FA\U0001F1F8 ' + escapeHtml(entry.종목명),
    sub: entry.티커,
    body: '<div class="dgroup"><h4>안내</h4><div class="biz-summary">이 종목은 나스닥 상장 종목 ' +
      '검색을 위해 이름·티커만 가져온 항목으로, 저희가 매일 점수를 매기는 S&amp;P500 대상에는 ' +
      '포함되어 있지 않아 PER·PBR 같은 상세 재무 지표와 실시간 현재가를 제공하지 않습니다. ' +
      '아래 링크에서 최신 시세와 재무정보를 직접 확인해주세요.</div></div>',
    link: '<a class="ext-link" href="' + escapeHtml(yahooUrl) + '" target="_blank" rel="noopener">Yahoo Finance에서 시세·재무제표 보기 →</a>',
  }};
}}
const overlay = document.getElementById('detailOverlay');
function openDetail(key) {{
  let d = DETAILS[key];
  if (!d && EXTRA_US[key]) d = buildExtraDetail(EXTRA_US[key]);
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
let catFilter = 'policy';  // 2026.09.11: 정책테마 섹션을 기본(첫 화면) 탭으로 변경
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

    # 정책테마 추천(2026.09.11 신설): 매출성장률/영업이익성장률(한국)·이익성장률(미국) 기준으로
    # 실적 성장 순위를 매겨두고, 대시보드에서는 POLICY_THEMES에 매칭된 종목만 이 점수로 정렬해
    # 보여줍니다(정책연계가 없는 종목도 참고용으로 이 점수 자체는 계산해둠).
    combined["정책실적점수"] = composite_score(combined, POLICY_RANK_COMPONENTS)
    has_theme = combined["정책연계"].apply(lambda v: isinstance(v, (list, tuple)) and len(v) > 0)
    print(f"[정책테마] 정책 관련 키워드가 매칭된 종목: {int(has_theme.sum())} / {len(combined)}개 "
          f"(KR {int((has_theme & (combined['시장구분'] == 'KR')).sum())}개, "
          f"US {int((has_theme & (combined['시장구분'] == 'US')).sum())}개)")

    print("[매크로] 기준금리/환율/원자재 참고 지표 수집 중...")
    macro = fetch_macro_indicators()

    xlsx_path = export(combined)
    build_dashboard(combined, price_history, macro=macro, out_path="docs/index.html")
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
