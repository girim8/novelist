"""
문체 서재 (Style Studio) — Streamlit Community Cloud 전용
=========================================================

Cloud는 파일시스템이 휘발성이다. 재부팅·재배포·12시간 슬립이면 저장한 것이 사라진다.
그래서 이 앱은 **디스크에 아무것도 쓰지 않는다.** 모든 상태는 세션에 두고,
작업 결과는 `.json` 백업 파일로 사용자가 직접 내려받아 보관한다.

배포 준비
---------
리포 루트에 다음 세 파일을 둔다.

    app.py
    requirements.txt
    .streamlit/config.toml   (선택)

Advanced settings → Secrets 에 아래를 붙여넣는다.

    app_password  = "원하는_비밀번호"
    GOOGLE_API_KEY = "..."
    ANTHROPIC_API_KEY = "..."

파이썬 버전은 배포 시 Advanced settings 드롭다운에서 고른다.
runtime.txt는 Cloud에서 무시된다.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.util import find_spec

import numpy as np
import pandas as pd
import streamlit as st

try:  # 그래프는 있으면 좋고, 없어도 앱은 살아 있어야 한다
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ModuleNotFoundError:  # pragma: no cover
    go = None
    HAS_PLOTLY = False

HAS_LIGHTRAG = find_spec("lightrag") is not None

# ═════════════════════════════════════════════════════════════
# 설정
# ═════════════════════════════════════════════════════════════

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 1536
BACKUP_VERSION = 3

MODELS = {
    "Gemini": {"분석": "gemini-2.5-flash", "집필": "gemini-2.5-pro"},
    "Claude": {"분석": "claude-haiku-4-5-20251001", "집필": "claude-sonnet-4-6"},
}
PREFIX = {"Gemini": "gemini", "Claude": "claude"}

PLATFORM_LIMITS = {
    "문피아 (5,500자)": 5500,
    "카카오페이지 (5,000자)": 5000,
    "네이버 시리즈 (5,000자)": 5000,
    "노벨피아 (5,000자)": 5000,
    "직접 지정": 5000,
}

# Cloud 메모리는 약 1GB다. 원고 총량이 이 선을 넘으면 경고한다.
SOFT_CHAR_LIMIT = 3_000_000

AI_TELLS = [
    "다름 아니었다", "다름 아닌", "에 지나지 않았다", "라 할 수 있다",
    "하는 것이었다", "인 것이었다", "그러나 그것도 잠시", "하지만 그것도 잠시",
    "그것도 잠시", "무언가", "그야말로", "이는 곧", "동시에 그",
    "말로 형용할 수 없는", "라는 듯이", "듯한 느낌", "알 수 없는 감정",
    "에 의해", "로 인해", "을 가지고 있", "중 하나였다", "할 수밖에 없었다",
    "묘한 기분", "형언할 수 없", "그 순간이었다", "찰나의 순간",
]

ENTITY_GUIDANCE = """등장 개체를 아래 유형 중 하나로 분류한다. 맞는 것이 없으면 `기타`.

- 인물: 이름이 주어진 등장인물
- 문파: 문파, 세가, 조직, 상단, 관부 등 집단
- 무공: 무공, 심법, 초식, 검법, 진법
- 물건: 병기, 영약, 비급, 신물
- 장소: 지명, 산문, 객잔, 건물, 지역
- 사건: 전투, 회합, 배신, 죽음, 혼약, 밀약 등 서사상의 발생
- 개념: 강호의 규칙, 맹세, 예언, 가문의 비밀"""

STYLE_DNA_PROMPT = """아래 원고에서 작가의 문체를 재현 가능한 사양서로 분해하라.
추측 없이 관찰된 것만, 원고 속 실제 예시를 붙여서 적는다.

## 1. 문장 — 평균 길이, 단문 연쇄 구간과 장문 구간의 구분 기준
## 2. 어미 — 종결어미 분포와 이 작가만의 반복 패턴
## 3. 시점 — 인칭, 초점화 인물, 서술자와 인물의 심리적 거리
## 4. 장면 — 액션·대화·묘사의 호흡과 비율, 장면 전환 신호
## 5. 어휘 — 선호 어휘 20개, 한자어 대 고유어 비율, 결코 쓰지 않는 표현
## 6. 대사 — 대사 대 지문 비율, 대사 태그 처리, 인물별 말투 차이
## 7. 재현 규칙 — 이 문체를 흉내 낼 때 지킬 명령형 규칙 10개

[측정된 통계 — 사양서에 반드시 반영할 것]
{stats}

--- 원고 ---
{corpus}"""

EVENT_PROMPT = """아래는 소설 원고의 한 회차다. 실제로 벌어진 사건만 뽑아라.

- 생각, 회상, 묘사는 사건이 아니다. 서사가 움직인 것만 뽑는다.
- 회상 속 과거 사건은 회상을 true로 둔다.
- 한 회차에 보통 1~5개다. 억지로 채우지 않는다.
- JSON 배열만 출력한다. 설명이나 마크다운 울타리 금지.

형식: {{"사건":"20자 이내","인물":["이름"],"유형":"전투|밀약|배신|이동|각성|죽음|만남|폭로","회상":false,"복선":"심은 떡밥 또는 빈 문자열"}}

--- {title} ---
{body}"""

CODEX_PROMPT = """아래 원고에서 등장인물 사전을 만들어라. JSON 배열만 출력한다.

형식: {{"이름":"","분류":"주역|조역|적대|단역","소속":"","무공":"","첫등장":"{title}","특징":"40자 이내","관계":"다른 인물과의 관계"}}

원고에 근거가 없는 항목은 빈 문자열로 둔다. 추측 금지.

--- {title} ---
{body}"""

WRITE_SYSTEM = """너는 이 작가 본인의 필기구다. 문체를 설명하지 말고 그 문체로 써라.

[문체 사양서]
{dna}

[원고에서 검색한 관련 설정]
{context}

지킬 것:
- 사양서의 어미 분포와 문장 길이 분포를 통계적으로 맞춘다.
- 위 설정과 충돌하는 내용을 쓰지 않는다. 없는 설정은 기존과 어긋나지 않게 만든다.
- 번역투, 설명체, AI 특유의 대구와 마무리 훈계를 쓰지 않는다.
- 요청 분량을 채운다. 줄거리 요약으로 대체하지 않는다.
- 결과물만 출력한다."""


# ═════════════════════════════════════════════════════════════
# 텍스트 유틸
# ═════════════════════════════════════════════════════════════

SENT_SPLIT = re.compile(r"(?<=[.!?…”\"』」])\s+|\n+")
DIALOG = re.compile(r"[“\"']([^”\"']{1,300})[”\"']|「([^」]{1,300})」")
WORD = re.compile(r"[가-힣]{2,}")


def chapter_order(name: str) -> tuple[int, str]:
    """파일명 속 첫 숫자를 회차로 삼는다. 없으면 뒤로 민다."""
    found = re.search(r"\d+", name)
    return (int(found.group()) if found else 10**6, name)


def strip_frontmatter(text: str) -> str:
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 1]


def ending_of(sentence: str) -> str:
    body = re.sub(r"[^가-힣]", "", sentence)
    if not body:
        return "기타"
    for suffix in ("었다", "였다", "했다", "린다", "는다", "이다", "았다"):
        if body.endswith(suffix):
            return suffix
    if body.endswith("다"):
        return "다"
    for suffix in ("까", "냐", "지", "요", "군", "라", "네", "야"):
        if body.endswith(suffix):
            return suffix
    return "명사종결"


@st.cache_data(show_spinner=False, max_entries=64)
def analyze(text: str) -> dict:
    """원고 한 덩어리의 문체 지표. 매 rerun마다 재계산하지 않도록 캐싱한다."""
    sents = sentences(text)
    lengths = [len(s) for s in sents] or [0]
    dialog_chars = sum(len(m.group()) for m in DIALOG.finditer(text))
    words = WORD.findall(text)
    return {
        "글자": len(text),
        "문장": len(sents),
        "평균문장": round(float(np.mean(lengths)), 1),
        "중앙문장": int(np.median(lengths)),
        "최장문장": int(max(lengths)),
        "단문비율": round(sum(1 for x in lengths if x <= 20) / max(len(lengths), 1), 3),
        "장문비율": round(sum(1 for x in lengths if x >= 60) / max(len(lengths), 1), 3),
        "대사비율": round(dialog_chars / max(len(text), 1), 3),
        "어휘다양도": round(len(set(words)) / max(len(words), 1), 3),
        "어미분포": dict(Counter(ending_of(s) for s in sents).most_common(8)),
    }


def stats_block(stats: dict) -> str:
    endings = ", ".join(f"{k} {v}회" for k, v in stats["어미분포"].items())
    return (
        f"- 평균 문장 길이 {stats['평균문장']}자 (중앙값 {stats['중앙문장']}, 최장 {stats['최장문장']})\n"
        f"- 20자 이하 단문 {stats['단문비율']:.0%}, 60자 이상 장문 {stats['장문비율']:.0%}\n"
        f"- 대사가 전체 분량의 {stats['대사비율']:.0%}\n"
        f"- 어휘 다양도 {stats['어휘다양도']}\n"
        f"- 종결어미 분포: {endings}"
    )


@st.cache_data(show_spinner=False, max_entries=32)
def repeated_phrases(text: str, n: int = 3, floor: int = 4) -> pd.DataFrame:
    tokens = re.sub(r"[^가-힣 ]", " ", text).split()
    grams = Counter(" ".join(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))
    rows = [
        {"표현": g, "횟수": c} for g, c in grams.most_common(200) if c >= floor and len(g) > 6
    ]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=32)
def find_tells(text: str) -> pd.DataFrame:
    rows = []
    for phrase in AI_TELLS:
        hits = [m.start() for m in re.finditer(re.escape(phrase), text)]
        if hits:
            near = text[max(hits[0] - 25, 0) : hits[0] + 25].replace("\n", " ")
            rows.append({"표현": phrase, "횟수": len(hits), "첫 용례": f"…{near}…"})
    return pd.DataFrame(sorted(rows, key=lambda r: -r["횟수"]))


def _break_long(block: str, limit: int) -> list[str]:
    """한 문단이 기준을 넘으면 문장 경계로, 그래도 넘으면 글자 수로 자른다."""
    if len(block) <= limit:
        return [block]
    pieces, buf = [], ""
    for sent in sentences(block) or [block]:
        candidate = f"{buf} {sent}".strip() if buf else sent
        if len(candidate) <= limit:
            buf = candidate
        else:
            if buf:
                pieces.append(buf)
            while len(sent) > limit:
                pieces.append(sent[:limit])
                sent = sent[limit:]
            buf = sent
    if buf:
        pieces.append(buf)
    return pieces


def split_for_platform(text: str, limit: int) -> list[str]:
    """문단 경계를 우선 지키되, 기준을 넘는 문단은 문장 단위로 쪼갠다."""
    chunks, buf = [], ""
    for para in re.split(r"\n{2,}", text):
        if not para.strip():
            continue
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= limit:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        parts = _break_long(para, limit)
        chunks.extend(parts[:-1])
        buf = parts[-1]
    if buf:
        chunks.append(buf)
    return chunks


def compare_style(base: dict, draft: dict) -> pd.DataFrame:
    rows = []
    for k in ["평균문장", "단문비율", "장문비율", "대사비율", "어휘다양도"]:
        a, b = base[k], draft[k]
        gap = (b - a) / a if a else 0.0
        rows.append(
            {
                "지표": k, "기존 원고": a, "새 원고": b, "격차": f"{gap:+.0%}",
                "판정": "일치" if abs(gap) < 0.15 else ("주의" if abs(gap) < 0.35 else "이탈"),
            }
        )
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    """tabulate가 없어도 동작하는 마크다운 표 변환."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        rule = "|" + "|".join(["---"] * len(df.columns)) + "|"
        body = [
            "| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join([head, rule, *body])


def parse_json_array(raw: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


# ═════════════════════════════════════════════════════════════
# 작품 (디스크에 쓰지 않는다)
# ═════════════════════════════════════════════════════════════

@dataclass
class Project:
    name: str = "새 작품"
    docs: dict[str, str] = field(default_factory=dict)
    dna: str = ""
    timeline: list[dict] = field(default_factory=list)
    codex: list[dict] = field(default_factory=list)
    indexed: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(v) for v in self.docs.values())

    def ordered(self) -> list[str]:
        return sorted(self.docs, key=chapter_order)

    def sample(self, limit: int) -> str:
        text = "\n\n".join(f"### {k}\n\n{self.docs[k]}" for k in self.ordered())
        if len(text) <= limit:
            return text
        cut = limit // 3
        mid = len(text) // 2
        return "\n\n[…]\n\n".join([text[:cut], text[mid : mid + cut], text[-cut:]])

    def to_backup(self) -> str:
        payload = asdict(self)
        payload["_version"] = BACKUP_VERSION
        payload["_saved"] = datetime.now().isoformat(timespec="seconds")
        return json.dumps(payload, ensure_ascii=False, indent=1)

    @classmethod
    def from_backup(cls, raw: str) -> "Project":
        data = json.loads(raw)
        allowed = {f for f in asdict(cls()).keys()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def touch() -> None:
    """백업되지 않은 변경이 있음을 표시한다."""
    st.session_state.dirty = st.session_state.get("dirty", 0) + 1


# ═════════════════════════════════════════════════════════════
# 모델 어댑터
# ═════════════════════════════════════════════════════════════

def secret(name: str) -> str:
    """secrets.toml이 없어도 죽지 않게 감싼다."""
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:  # noqa: BLE001 — secrets 파일 자체가 없는 경우
        return ""


def key_of(vendor: str) -> str:
    return st.session_state.get(f"key_{vendor}", "") or ""


def _need_key(vendor: str) -> str:
    key = key_of(vendor)
    if not key:
        raise ValueError(f"사이드바에 {vendor} API 키를 먼저 넣는다.")
    return key


def check_model(vendor: str, model: str) -> None:
    if model and not model.lower().startswith(PREFIX[vendor]):
        raise ValueError(
            f"공급자는 {vendor}인데 모델이 `{model}`이다. "
            f"`{PREFIX[vendor]}`로 시작하는 모델을 넣거나 공급자를 바꾼다."
        )


async def _gemini(prompt, system_prompt, history, model, temperature) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_need_key("Gemini"))
    contents = [
        types.Content(
            role="user" if m.get("role") == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in history or []
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    resp = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=16384,
            temperature=temperature,
        ),
    )
    return resp.text or ""


async def _claude(prompt, system_prompt, history, model, temperature) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_need_key("Claude"))
    system = (
        [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        if system_prompt
        else anthropic.NOT_GIVEN
    )
    resp = await asyncio.to_thread(
        client.messages.create,
        model=model,
        max_tokens=16384,
        system=system,
        messages=(history or []) + [{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return "".join(b.text for b in resp.content if b.type == "text")


async def llm(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """LightRAG가 요구하는 시그니처를 그대로 만족하는 통합 호출부."""
    vendor = kwargs.pop("vendor", None) or st.session_state.vendor
    model = kwargs.pop("model", None) or st.session_state.model_analyze
    temp = kwargs.pop("temperature", 0.4)
    check_model(vendor, model)
    _need_key(vendor)
    fn = _gemini if vendor == "Gemini" else _claude
    try:
        return await fn(prompt, system_prompt, history_messages, model, temp)
    except ModuleNotFoundError as err:
        pkg = "google-genai" if vendor == "Gemini" else "anthropic"
        raise RuntimeError(f"{pkg} 패키지가 없다. requirements.txt를 확인한다.") from err


async def embed(texts: list[str]) -> np.ndarray:
    from google import genai
    from google.genai import types

    key = key_of("Gemini")
    if not key:
        raise ValueError("지식그래프 색인에는 Gemini 키가 필요하다.")
    client = genai.Client(api_key=key)
    resp = await asyncio.to_thread(
        client.models.embed_content,
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    return np.array([e.values for e in resp.embeddings], dtype=np.float32)


def ask(prompt: str, system: str = "", model: str | None = None, temperature: float = 0.4) -> str:
    return asyncio.run(llm(prompt, system_prompt=system or None, model=model, temperature=temperature))


def get_rag():
    """Cloud에서는 대개 쓸 수 없다. 로컬 실행용으로 남겨둔다."""
    if not HAS_LIGHTRAG:
        raise RuntimeError("LightRAG가 없다. 지식그래프는 로컬 실행에서만 쓸 수 있다.")
    if st.session_state.get("rag") is None:
        from lightrag import LightRAG
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from lightrag.utils import EmbeddingFunc

        async def _build():
            rag = LightRAG(
                working_dir="./rag_storage",
                llm_model_func=llm,
                llm_model_name=st.session_state.model_analyze,
                llm_model_max_async=4,
                embedding_func=EmbeddingFunc(
                    embedding_dim=EMBED_DIM, max_token_size=2048, func=embed
                ),
                addon_params={"language": "Korean", "entity_types_guidance": ENTITY_GUIDANCE},
                summary_max_tokens=1500,
            )
            await rag.initialize_storages()
            await initialize_pipeline_status()
            return rag

        st.session_state.rag = asyncio.run(_build())
    return st.session_state.rag


# ═════════════════════════════════════════════════════════════
# 연표 렌더
# ═════════════════════════════════════════════════════════════

PALETTE = {
    "전투": "#B4402F", "배신": "#6B2D5C", "밀약": "#3D5A6C", "죽음": "#2B2B2B",
    "각성": "#C08B2E", "폭로": "#7A6A9B", "만남": "#4A7C59", "이동": "#8A8577",
}


def render_timeline(df: pd.DataFrame) -> None:
    if not HAS_PLOTLY:
        st.warning("plotly가 없어 표로 대체한다. requirements.txt를 확인한다.")
        st.dataframe(
            df.sort_values("회차")[["회차", "인물", "사건", "유형", "회상"]],
            width="stretch", hide_index=True,
        )
        return
    work = df.assign(주역=df["인물"].astype(str).str.split(",").str[0].str.strip())
    lanes = work.groupby("주역")["회차"].min().sort_values().index.tolist()
    fig = go.Figure()
    for kind, group in work.groupby("유형"):
        fig.add_trace(
            go.Scatter(
                x=group["회차"], y=group["주역"], mode="markers", name=str(kind),
                marker=dict(
                    size=15, color=PALETTE.get(str(kind), "#9A9A9A"),
                    symbol=["diamond" if r else "circle" for r in group["회상"]],
                    line=dict(width=1, color="white"),
                ),
                customdata=np.stack([group["사건"], group["출처"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>%{y} · %{x}화<br>%{customdata[1]}<extra></extra>",
            )
        )
    fig.update_layout(
        height=max(340, 44 * max(len(lanes), 1)),
        xaxis=dict(title="회차", dtick=1, gridcolor="#EEEEEE", zeroline=False),
        yaxis=dict(categoryorder="array", categoryarray=lanes[::-1], title=""),
        plot_bgcolor="white", margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, width="stretch")


def to_mermaid(df: pd.DataFrame) -> str:
    lines = ["```mermaid", "timeline", "    title 플롯 연표"]
    for chapter, group in df.sort_values("회차").groupby("회차"):
        joined = " : ".join(
            f"{r.사건}({str(r.인물).split(',')[0].strip()})" for r in group.itertuples()
        )
        lines.append(f"    {chapter}화 : {joined}")
    lines.append("```")
    return "\n".join(lines)


def to_obsidian(df: pd.DataFrame, project: str) -> str:
    out = [
        "---", f"작품: {project}", "tags: [플롯, 연표]",
        f"갱신: {datetime.now():%Y-%m-%d}", "---", "",
        f"# {project} 플롯 연표", "", to_mermaid(df), "",
    ]
    hooks = df[df["복선"].astype(str).str.strip() != ""]
    if not hooks.empty:
        out += ["## 심어둔 복선", ""]
        out += [f"- [ ] **{r.회차}화** {r.복선} — `{r.사건}`" for r in hooks.itertuples()]
        out += [""]
    out += ["## 전체 사건", "", df_to_markdown(df)]
    return "\n".join(out)


# ═════════════════════════════════════════════════════════════
# 세션 초기화 · 인증
# ═════════════════════════════════════════════════════════════

st.set_page_config(page_title="문체 서재", page_icon="✒️", layout="wide")
st.markdown(
    """
    <style>
      .stApp { font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif; }
      .stChatMessage p { line-height: 1.95; }
      div[data-testid="stMetricValue"] { font-size: 1.35rem; }
      section[data-testid="stSidebar"] { min-width: 335px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def sync_models() -> None:
    vendor = st.session_state.vendor
    st.session_state.model_analyze = MODELS[vendor]["분석"]
    st.session_state.model_write = MODELS[vendor]["집필"]


def stash_key(vendor: str) -> None:
    """Streamlit은 렌더되지 않은 위젯 상태를 정리한다. 별도 슬롯에 복사해 둔다."""
    st.session_state[f"key_{vendor}"] = st.session_state.get(f"_kin_{vendor}", "") or ""


def key_input(vendor: str, label: str, help_text: str) -> None:
    st.text_input(
        label, type="password", key=f"_kin_{vendor}",
        value=st.session_state.get(f"key_{vendor}", ""),
        on_change=stash_key, args=(vendor,), help=help_text,
    )
    stash_key(vendor)


for k, v in {
    "project": Project(),
    "proj_rev": 0,   # 작품이 교체되면 올린다. 이름 입력란을 새 위젯으로 갈아끼우기 위함.
    "chat": [],
    "rag": None,
    "dirty": 0,
    "authed": False,
    "key_Gemini": secret("GOOGLE_API_KEY"),
    "key_Claude": secret("ANTHROPIC_API_KEY"),
    "vendor": "Gemini",
    "model_analyze": MODELS["Gemini"]["분석"],
    "model_write": MODELS["Gemini"]["집필"],
}.items():
    st.session_state.setdefault(k, v)

APP_PASSWORD = secret("app_password")

if APP_PASSWORD and not st.session_state.authed:
    st.title("✒️ 문체 서재")
    st.caption("원고를 다루는 도구다. 비밀번호를 넣어야 열린다.")
    with st.form("gate"):
        entered = st.text_input("비밀번호", type="password")
        if st.form_submit_button("열기", type="primary"):
            if entered == APP_PASSWORD:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("비밀번호가 다르다.")
    st.stop()

proj: Project = st.session_state.project

# ═════════════════════════════════════════════════════════════
# 사이드바
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ✒️ 문체 서재")

    if not APP_PASSWORD:
        st.error(
            "**이 앱은 잠겨 있지 않다.** URL을 아는 누구나 원고를 볼 수 있다. "
            "Settings → Secrets 에 `app_password`를 넣는다."
        )

    # 작품을 교체하면 위젯 키가 바뀌어 새 이름이 제대로 반영된다.
    pname_key = f"_pname_{st.session_state.proj_rev}"
    st.text_input("작품 이름", key=pname_key, value=proj.name)
    proj.name = (st.session_state.get(pname_key) or "").strip() or "새 작품"

    st.divider()
    st.markdown("**원고 올리기**")
    st.caption("`.md` `.txt` 여러 개 선택. 문체 학습과 분석에 함께 쓰인다.")
    uploads = st.file_uploader(
        "원고 파일", type=["md", "markdown", "txt"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploads and st.button(f"{len(uploads)}개 넣기", type="primary", width="stretch"):
        for f in uploads:
            proj.docs[f.name] = strip_frontmatter(f.getvalue().decode("utf-8", errors="ignore"))
        touch()
        st.rerun()

    st.divider()
    st.markdown("**백업**")
    st.caption("Cloud는 재부팅하면 전부 지워진다. 작업 후 반드시 내려받는다.")
    dirty = st.session_state.dirty
    if dirty:
        st.warning(f"백업하지 않은 변경 {dirty}건")
    st.download_button(
        "💾 백업 내려받기 (.json)", proj.to_backup(),
        file_name=f"{proj.name}_{datetime.now():%Y%m%d_%H%M}.json",
        mime="application/json", width="stretch", type="primary" if dirty else "secondary",
        on_click=lambda: st.session_state.update(dirty=0),
    )
    restore = st.file_uploader("백업 불러오기", type=["json"], key="_restore")
    if restore and st.button("불러오기", width="stretch"):
        try:
            st.session_state.project = Project.from_backup(
                restore.getvalue().decode("utf-8")
            )
            st.session_state.proj_rev += 1   # 이름 입력란을 새로 그린다
            st.session_state.dirty = 0
            st.session_state.chat = []
            st.rerun()
        except (json.JSONDecodeError, TypeError, ValueError) as err:
            st.error(f"백업 파일을 읽을 수 없다: {err}")

    st.divider()
    st.markdown("**모델**")
    vendor = st.radio("공급자", list(MODELS), horizontal=True, key="vendor", on_change=sync_models)
    key_input(vendor, f"{vendor} API 키",
              "Secrets에 넣어두면 매번 입력하지 않아도 된다.")
    if vendor == "Claude" and HAS_LIGHTRAG:
        key_input("Gemini", "Gemini 키 (색인용)", "Claude는 임베딩을 제공하지 않는다.")

    ca, cb = st.columns(2)
    with ca:
        st.text_input("분석 모델", key="model_analyze")
    with cb:
        st.text_input("집필 모델", key="model_write")

    bad = [
        label for label, name in (("분석", st.session_state.model_analyze),
                                  ("집필", st.session_state.model_write))
        if not name.lower().startswith(PREFIX[vendor])
    ]
    if bad:
        st.warning(f"{'·'.join(bad)} 모델이 {vendor} 것이 아니다.")
        st.button(f"{vendor} 기본 모델로 되돌리기", width="stretch", on_click=sync_models)

    st.divider()
    st.metric("원고", f"{proj.chars:,}자")
    st.caption(f"{len(proj.docs)}개 파일 · 사건 {len(proj.timeline)} · 인물 {len(proj.codex)}")
    if proj.chars > SOFT_CHAR_LIMIT:
        st.error("원고가 너무 많다. Cloud 메모리 한도에 걸릴 수 있으니 나눠서 작업한다.")
    if proj.dna:
        st.download_button("문체 사양서 (.md)", proj.dna,
                           file_name=f"{proj.name}_문체.md", width="stretch")

# ═════════════════════════════════════════════════════════════
# 본문
# ═════════════════════════════════════════════════════════════

st.title(proj.name)

TAB_NAMES = ["원고", "문체", "점검", "연표", "인물", "집필", "내보내기"]
if HAS_LIGHTRAG:
    TAB_NAMES.insert(5, "심문")
tabs = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

# ── 원고 ──
with tabs["원고"]:
    if not proj.docs:
        st.info("왼쪽 사이드바에서 원고를 올린다. API 키 없이도 분석은 돌아간다.")
    else:
        whole = "\n\n".join(proj.docs[n] for n in proj.ordered())
        stats = analyze(whole)
        cols = st.columns(5)
        cols[0].metric("총 분량", f"{proj.chars:,}자")
        cols[1].metric("회차", f"{len(proj.docs)}화")
        cols[2].metric("평균 문장", f"{stats['평균문장']}자")
        cols[3].metric("대사 비율", f"{stats['대사비율']:.0%}")
        cols[4].metric("어휘 다양도", f"{stats['어휘다양도']}")

        per = pd.DataFrame(
            [
                {"회차": i, "파일": n, "글자": len(proj.docs[n])}
                for i, n in enumerate(proj.ordered(), start=1)
            ]
        )
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**회차별 분량**")
            if HAS_PLOTLY:
                fig = go.Figure(
                    go.Bar(x=per["회차"], y=per["글자"], marker_color="#3D5A6C",
                           hovertext=per["파일"])
                )
                fig.add_hline(y=5000, line_dash="dot", line_color="#B4402F",
                              annotation_text="연재 분량 5,000자")
                fig.update_layout(height=260, plot_bgcolor="white",
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis=dict(title="회차", dtick=1),
                                  yaxis=dict(title="글자", gridcolor="#EEE"))
                st.plotly_chart(fig, width="stretch")
            else:
                st.bar_chart(per.set_index("회차")["글자"], height=260)
        with right:
            st.markdown("**종결어미 분포**")
            end_df = pd.DataFrame(sorted(stats["어미분포"].items(), key=lambda x: -x[1]),
                                  columns=["어미", "빈도"])
            if HAS_PLOTLY:
                fig2 = go.Figure(go.Bar(x=end_df["빈도"], y=end_df["어미"],
                                        orientation="h", marker_color="#7A6A9B"))
                fig2.update_layout(height=260, plot_bgcolor="white",
                                   margin=dict(l=10, r=10, t=10, b=10),
                                   yaxis=dict(autorange="reversed"),
                                   xaxis=dict(gridcolor="#EEE"))
                st.plotly_chart(fig2, width="stretch")
            else:
                st.bar_chart(end_df.set_index("어미")["빈도"], height=260)

        st.dataframe(per, width="stretch", hide_index=True)

        chosen = st.selectbox("원고 열기", proj.ordered())
        if chosen:
            edited = st.text_area("본문", value=proj.docs[chosen], height=340)
            e1, e2 = st.columns([1, 4])
            if e1.button("저장", type="primary"):
                proj.docs[chosen] = edited
                touch()
                st.success("세션에 반영했다. 사이드바에서 백업을 내려받는다.")
            if e2.button("이 회차 빼기"):
                proj.docs.pop(chosen, None)
                touch()
                st.rerun()

# ── 문체 ──
with tabs["문체"]:
    st.subheader("문체 사양서")
    st.caption("한 번만 뽑으면 된다. 집필 요청마다 자동으로 붙는다.")
    if not proj.docs:
        st.info("원고를 먼저 올린다.")
    else:
        base = analyze("\n\n".join(proj.docs[n] for n in proj.ordered()))
        st.code(stats_block(base), language="text")
        if st.button("문체 분석 실행", type="primary"):
            with st.spinner("원고를 읽는 중"):
                try:
                    proj.dna = ask(
                        STYLE_DNA_PROMPT.format(stats=stats_block(base),
                                                corpus=proj.sample(200_000)),
                        system="너는 문체 분석가다. 관찰된 사실만 적는다.",
                        model=st.session_state.model_write, temperature=0.2,
                    )
                    touch()
                except Exception as err:  # noqa: BLE001
                    st.error(f"실행 실패: {err}")
        edited_dna = st.text_area("사양서 (직접 고쳐도 된다)", value=proj.dna, height=440)
        if st.button("사양서 저장"):
            proj.dna = edited_dna
            touch()
            st.success("세션에 반영했다.")

# ── 점검 ──
with tabs["점검"]:
    st.subheader("원고 점검")
    st.caption("API 없이 즉시 돌아간다. 입버릇과 번역투를 잡는 용도.")
    if not proj.docs:
        st.info("원고를 먼저 올린다.")
    else:
        scope = st.selectbox("대상", ["전체 원고"] + proj.ordered())
        text = ("\n\n".join(proj.docs[n] for n in proj.ordered())
                if scope == "전체 원고" else proj.docs[scope])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**반복 표현**")
            n = st.slider("어절 수", 2, 5, 3)
            floor = st.slider("최소 횟수", 2, 20, 4)
            rep = repeated_phrases(text, n, floor)
            if rep.empty:
                st.success("기준을 넘는 반복 표현이 없다.")
            else:
                st.dataframe(rep.head(40), width="stretch", hide_index=True)
        with c2:
            st.markdown("**번역투 · AI 상투구**")
            tells = find_tells(text)
            if tells.empty:
                st.success("잡힌 표현이 없다.")
            else:
                st.dataframe(tells, width="stretch", hide_index=True)

        st.divider()
        st.markdown("**새 원고 문체 대조**")
        draft = st.text_area("갓 쓴 원고를 붙여넣는다", height=170, key="draft")
        if draft.strip():
            st.dataframe(
                compare_style(analyze("\n\n".join(proj.docs[n] for n in proj.ordered())),
                              analyze(draft)),
                width="stretch", hide_index=True,
            )

# ── 연표 ──
with tabs["연표"]:
    st.subheader("플롯 연표")
    if not proj.docs:
        st.info("원고를 먼저 올린다.")
    else:
        done = {r["출처"] for r in proj.timeline}
        todo = [n for n in proj.ordered() if n not in done]
        st.caption(
            "Cloud는 한 번에 오래 도는 작업에 약하다. 배치로 나눠 돌리고, "
            "중간 결과는 그대로 남으니 이어서 진행하면 된다."
        )
        b1, b2 = st.columns([1, 2])
        b1.metric("미처리", f"{len(todo)}화")
        size = b2.select_slider("한 번에 처리할 회차", [5, 10, 20, 30], value=10)

        if todo and st.button(f"다음 {min(size, len(todo))}화 추출", type="primary"):
            bar = st.progress(0.0)
            batch = todo[:size]
            for i, name in enumerate(batch, start=1):
                bar.progress(i / len(batch), text=f"{name} 읽는 중")
                try:
                    raw = ask(EVENT_PROMPT.format(title=name, body=proj.docs[name][:60_000]),
                              system="너는 서사 구조 분석가다. JSON만 출력한다.", temperature=0.1)
                except Exception as err:  # noqa: BLE001
                    st.error(f"{name}에서 멈췄다. 여기까지는 보존된다. ({err})")
                    break
                idx = proj.ordered().index(name) + 1
                for ev in parse_json_array(raw):
                    proj.timeline.append({
                        "회차": idx, "출처": name,
                        "사건": str(ev.get("사건", ""))[:40],
                        "인물": ", ".join(ev.get("인물") or ["미상"]),
                        "유형": ev.get("유형", "기타"),
                        "회상": bool(ev.get("회상", False)),
                        "복선": str(ev.get("복선", "")),
                    })
                touch()
            bar.empty()
            st.rerun()

        tl = pd.DataFrame(proj.timeline)
        if tl.empty:
            st.caption("아직 추출된 사건이 없다.")
        else:
            f1, f2 = st.columns([3, 1])
            picks = f1.multiselect("유형", sorted(tl["유형"].unique()),
                                   default=sorted(tl["유형"].unique()))
            keep_flash = f2.checkbox("회상 포함", value=True)
            view = tl[tl["유형"].isin(picks)]
            if not keep_flash:
                view = view[~view["회상"]]
            if view.empty:
                st.caption("조건에 맞는 사건이 없다.")
            else:
                render_timeline(view)
                st.caption("마름모는 회상 속 사건. 세로축은 첫 등장 순.")

            hooks = tl[tl["복선"].astype(str).str.strip() != ""]
            if not hooks.empty:
                st.markdown("**심어둔 복선**")
                st.dataframe(hooks[["회차", "사건", "복선"]], width="stretch", hide_index=True)
            with st.expander("전체 사건표"):
                st.dataframe(tl, width="stretch", hide_index=True)

            d1, d2, d3 = st.columns(3)
            d1.download_button("옵시디언 노트", to_obsidian(tl, proj.name),
                               file_name=f"{proj.name}_연표.md", width="stretch")
            d2.download_button("머메이드", to_mermaid(tl),
                               file_name="timeline.md", width="stretch")
            d3.download_button("CSV", tl.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{proj.name}_연표.csv", width="stretch")

# ── 인물 ──
with tabs["인물"]:
    st.subheader("인물 사전")
    if not proj.docs:
        st.info("원고를 먼저 올린다.")
    else:
        seen = {p.get("_출처") for p in proj.codex}
        todo = [n for n in proj.ordered() if n not in seen]
        c1, c2 = st.columns([1, 2])
        c1.metric("미처리", f"{len(todo)}화")
        size = c2.select_slider("한 번에 처리할 회차", [5, 10, 20, 30], value=10, key="cx_size")

        if todo and st.button(f"다음 {min(size, len(todo))}화 추출", type="primary", key="cx_run"):
            bar = st.progress(0.0)
            merged = {p["이름"]: p for p in proj.codex if p.get("이름")}
            batch = todo[:size]
            for i, name in enumerate(batch, start=1):
                bar.progress(i / len(batch), text=f"{name} 읽는 중")
                try:
                    raw = ask(CODEX_PROMPT.format(title=name, body=proj.docs[name][:60_000]),
                              system="너는 설정 관리자다. JSON만 출력한다.", temperature=0.1)
                except Exception as err:  # noqa: BLE001
                    st.error(f"{name}에서 멈췄다. 여기까지는 보존된다. ({err})")
                    break
                for person in parse_json_array(raw):
                    key = str(person.get("이름", "")).strip()
                    if not key:
                        continue
                    person["_출처"] = name
                    person.setdefault("첫등장", name)
                    if key in merged:
                        person["첫등장"] = merged[key].get("첫등장", name)
                        merged[key].update({k: v for k, v in person.items() if v})
                    else:
                        merged[key] = person
                # 처리 완료 표시용 더미가 아니라, 실제 처리한 회차를 기록한다
                merged.setdefault(f"__done__{name}", {"이름": "", "_출처": name})
                touch()
            proj.codex = list(merged.values())
            bar.empty()
            st.rerun()

        real = [p for p in proj.codex if p.get("이름")]
        if not real:
            st.caption("아직 추출된 인물이 없다.")
        else:
            cx = pd.DataFrame(real).drop(columns=["_출처"], errors="ignore")
            st.dataframe(cx, width="stretch", hide_index=True)
            note = "\n\n".join(
                f"## {r.get('이름','')}\n- 분류:: {r.get('분류','')}\n"
                f"- 소속:: {r.get('소속','')}\n- 무공:: {r.get('무공','')}\n"
                f"- 첫등장:: {r.get('첫등장','')}\n- 관계:: {r.get('관계','')}\n\n"
                f"{r.get('특징','')}"
                for r in cx.to_dict("records")
            )
            st.download_button(
                "옵시디언 인물 노트",
                f"---\n작품: {proj.name}\ntags: [설정, 인물]\n---\n\n# 인물 사전\n\n{note}",
                file_name=f"{proj.name}_인물.md", width="stretch",
            )

# ── 심문 (LightRAG 있을 때만) ──
if "심문" in tabs:
    with tabs["심문"]:
        st.subheader("원고에 묻기")
        pending = [n for n in proj.ordered() if n not in proj.indexed]
        c1, c2 = st.columns([1, 3])
        c1.metric("색인 대기", f"{len(pending)}개")
        if pending and c2.button("그래프 색인 실행", type="primary"):
            bar = st.progress(0.0)
            try:
                rag = get_rag()
                for i, name in enumerate(pending, start=1):
                    bar.progress(i / len(pending), text=f"{name} 색인 중")
                    asyncio.run(rag.ainsert(proj.docs[name], file_paths=name, ids=name))
                    proj.indexed.append(name)
                touch()
                st.success("색인 완료")
            except Exception as err:  # noqa: BLE001
                st.error(f"색인 실패: {err}")
            bar.empty()

        mode = st.selectbox("검색 방식", ["mix", "hybrid", "global", "local", "naive"])
        q = st.text_area("질문", height=90, label_visibility="collapsed")
        if st.button("묻기") and q.strip():
            with st.spinner("그래프를 훑는 중"):
                try:
                    from lightrag import QueryParam

                    st.markdown(asyncio.run(get_rag().aquery(q, param=QueryParam(mode=mode))))
                except Exception as err:  # noqa: BLE001
                    st.error(f"실행 실패: {err}")

# ── 집필 ──
with tabs["집필"]:
    if not proj.dna:
        st.info("문체 탭에서 사양서를 먼저 만든다.")
    else:
        o1, o2 = st.columns([2, 1])
        use_graph = o1.checkbox("그래프에서 설정 끌어오기",
                                value=bool(proj.indexed), disabled=not HAS_LIGHTRAG)
        temp = o2.slider("온도", 0.0, 1.5, 1.0, 0.1)

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("예: 12화, 주인공이 검을 처음 뽑는 장면을 4천 자로")
        if prompt:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            context = ""
            if use_graph and HAS_LIGHTRAG and proj.indexed:
                try:
                    from lightrag import QueryParam

                    context = asyncio.run(
                        get_rag().aquery(prompt, param=QueryParam(mode="mix", only_need_context=True))
                    )
                except Exception as err:  # noqa: BLE001
                    st.warning(f"설정 검색을 건너뛴다: {err}")

            with st.chat_message("assistant"):
                with st.spinner("쓰는 중"):
                    try:
                        out = ask(prompt,
                                  system=WRITE_SYSTEM.format(dna=proj.dna, context=context or "(없음)"),
                                  model=st.session_state.model_write, temperature=temp)
                        st.markdown(out)
                        st.session_state.chat.append({"role": "assistant", "content": out})
                        m = analyze(out)
                        st.caption(
                            f"{m['글자']:,}자 · 평균 문장 {m['평균문장']}자 · 대사 {m['대사비율']:.0%}"
                        )
                    except Exception as err:  # noqa: BLE001
                        st.error(f"실행 실패: {err}")

        if st.session_state.chat:
            last = next((m["content"] for m in reversed(st.session_state.chat)
                         if m["role"] == "assistant"), "")
            g1, g2, g3 = st.columns(3)
            g1.download_button(
                "대화 전체 (.md)",
                "\n\n".join(f"**{m['role']}**\n\n{m['content']}" for m in st.session_state.chat),
                file_name="집필기록.md", width="stretch",
            )
            if last:
                g2.download_button("마지막 원고만", last, file_name="초고.md", width="stretch")
            if g3.button("대화 비우기", width="stretch"):
                st.session_state.chat = []
                st.rerun()

# ── 내보내기 ──
with tabs["내보내기"]:
    st.subheader("연재 분량으로 자르기")
    st.caption("문단 경계를 지키며 플랫폼 기준에 맞춰 나눈다.")
    source = st.radio("원본", ["원고에서 고르기", "직접 붙여넣기"], horizontal=True)
    if source == "원고에서 고르기" and proj.docs:
        text = proj.docs[st.selectbox("회차", proj.ordered())]
    else:
        text = st.text_area("본문", height=200, key="export_text")

    p1, p2 = st.columns(2)
    platform = p1.selectbox("플랫폼", list(PLATFORM_LIMITS))
    manual = p2.number_input("기준 글자", 1000, 20000, PLATFORM_LIMITS[platform], step=500,
                             disabled=platform != "직접 지정")
    limit = int(manual if platform == "직접 지정" else PLATFORM_LIMITS[platform])

    if text and text.strip():
        parts = split_for_platform(text, limit)
        st.success(f"{len(parts)}개 분량으로 나뉜다.")
        st.dataframe(
            pd.DataFrame([
                {"편": i, "글자": len(p), "첫 줄": p.strip().split("\n")[0][:40]}
                for i, p in enumerate(parts, start=1)
            ]),
            width="stretch", hide_index=True,
        )
        for i, part in enumerate(parts, start=1):
            with st.expander(f"{i}편 · {len(part):,}자"):
                st.text_area("본문", value=part, height=250, key=f"part_{i}",
                             label_visibility="collapsed")
        st.download_button(
            "전체 내려받기 (.md)",
            "\n\n---\n\n".join(f"## {i}편\n\n{p}" for i, p in enumerate(parts, start=1)),
            file_name=f"{proj.name}_분할.md", width="stretch",
        )
