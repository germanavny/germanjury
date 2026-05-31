"""
News / Catalyst layer — runs autonomously (yfinance headlines, no external keys).
Scores recent headlines per ticker and returns a sentiment signal that gates or
confirms a technical entry. Catalysts watched: earnings beats/misses, layoffs,
cloud/AI demand, guidance, downgrades/upgrades, regulation/lawsuits.

Deliberately kept OUT of the technical backtest (no historical headlines) — it
only modifies LIVE entries in run_trader, so it never introduces look-ahead bias.
"""

import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone

import yfinance as yf

# Keyword lexicon → weight. Headline sentiment = sum of matched weights, clamped.
_POSITIVE = {
    "beat": 2.0, "beats": 2.0, "tops estimate": 2.5, "record": 1.5, "all-time high": 1.5,
    "surge": 1.5, "soar": 1.5, "rally": 1.0, "jumps": 1.2, "upgrade": 2.0, "raised": 1.5,
    "raises guidance": 2.5, "buyback": 1.5, "dividend increase": 1.5, "strong demand": 2.0,
    "ai demand": 1.5, "cloud growth": 2.0, "azure": 1.0, "outperform": 1.5, "bullish": 1.2,
    "partnership": 1.0, "expansion": 1.0, "wins": 1.0, "new high": 1.5, "growth": 0.8,
}
_NEGATIVE = {
    "miss": -2.0, "misses": -2.0, "misses estimate": -2.5, "layoff": -2.0, "layoffs": -2.0,
    "job cuts": -2.0, "downgrade": -2.0, "cuts guidance": -2.5, "lowered": -1.5, "plunge": -2.0,
    "plummet": -2.0, "tumble": -1.5, "slump": -1.5, "lawsuit": -1.5, "probe": -1.5,
    "investigation": -1.5, "antitrust": -1.5, "regulation": -1.0, "fine": -1.0, "warning": -1.5,
    "recall": -1.5, "weak demand": -2.0, "slowdown": -1.5, "bearish": -1.2, "selloff": -1.5,
    "underperform": -1.5, "data breach": -2.0, "delay": -1.0, "halt": -1.5,
}


# Company aliases — a headline only counts if it actually names the company,
# otherwise generic market noise pollutes the score.
_ALIASES = {
    "MSFT":  ["microsoft", "msft", "azure"],
    "AAPL":  ["apple", "aapl", "iphone"],
    "NVDA":  ["nvidia", "nvda"],
    "GOOGL": ["google", "alphabet", "googl", "gOOg"],
    "AMZN":  ["amazon", "amzn", "aws"],
}


def _mentions_company(text: str, ticker: str) -> bool:
    t = text.lower()
    return any(alias.lower() in t for alias in _ALIASES.get(ticker, [ticker.lower()]))


def _headline_text(item: dict) -> tuple[str, str]:
    """Extract (title, summary) from a yfinance news item (schema-tolerant)."""
    c = item.get("content", item) if isinstance(item, dict) else {}
    title   = (c.get("title") or "").strip()
    summary = (c.get("summary") or c.get("description") or "").strip()
    return title, summary


def _score_text(text: str) -> tuple[float, list[str]]:
    t = text.lower()
    score, hits = 0.0, []
    for kw, w in {**_POSITIVE, **_NEGATIVE}.items():
        if kw in t:
            score += w
            hits.append(kw)
    return score, hits


def get_news_sentiment(ticker: str, max_items: int = 10) -> dict:
    """
    Returns:
      sentiment : float in roughly [-1, +1]  (avg headline score, normalised)
      label     : human-readable bias
      headlines : list of {title, score, hits}
      flags     : notable catalyst keywords found
    """
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        return {"sentiment": 0.0, "label": "news unavailable", "headlines": [], "flags": [],
                "error": str(e)}

    scored, all_flags, total = [], set(), 0.0
    for it in items[:max_items]:
        title, summary = _headline_text(it)
        if not title:
            continue
        full = f"{title}. {summary}"
        # only score headlines that actually name the company → kills market noise
        if not _mentions_company(full, ticker):
            continue
        s, hits = _score_text(full)
        if hits:
            all_flags.update(hits)
        scored.append({"title": title[:120], "score": round(s, 1), "hits": hits})
        total += s

    n = max(len([s for s in scored if s["hits"]]), 1)
    # normalise: average over headlines that actually matched, squash to [-1,1]
    raw = total / n
    sentiment = max(-1.0, min(1.0, raw / 3.0))

    if sentiment >= 0.4:
        label = "POSITIVE catalysts"
    elif sentiment <= -0.4:
        label = "NEGATIVE catalysts"
    else:
        label = "neutral / mixed"

    return {
        "sentiment": round(sentiment, 2),
        "label": label,
        "headlines": scored[:5],
        "flags": sorted(all_flags),
    }


# Unambiguous, company-specific catalysts that justify overriding a technical entry.
_HARD_NEGATIVE = {"layoff", "layoffs", "job cuts", "downgrade", "cuts guidance",
                  "lawsuit", "probe", "investigation", "antitrust", "data breach", "recall"}
_HARD_POSITIVE = {"beat", "beats", "tops estimate", "upgrade", "raises guidance"}


def news_gate(signal: str, news: dict, strong: float = 0.45) -> tuple[str, str]:
    """
    Conservative news gate. Because keyword sentiment is noisy, only override a
    technical signal when EITHER a hard company-specific catalyst is present OR the
    aggregate sentiment is strongly opposed. Advisory by design — never invents trades.
    Returns (possibly_modified_signal, reason).
    """
    sentiment = news.get("sentiment", 0.0)
    flags = set(news.get("flags", []))
    hard_neg = flags & _HARD_NEGATIVE
    hard_pos = flags & _HARD_POSITIVE

    if signal == "LONG" and (hard_neg or sentiment <= -strong):
        why = f"hard catalyst {sorted(hard_neg)}" if hard_neg else f"sentiment {sentiment:+.2f}"
        return "HOLD", f"LONG blocked by news — {why}"
    if signal == "SHORT" and (hard_pos or sentiment >= strong):
        why = f"hard catalyst {sorted(hard_pos)}" if hard_pos else f"sentiment {sentiment:+.2f}"
        return "HOLD", f"SHORT blocked by news — {why}"
    return signal, ""


if __name__ == "__main__":
    for tk in ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN"]:
        r = get_news_sentiment(tk)
        print(f"\n{tk}: {r['sentiment']:+.2f} ({r['label']})  flags={r['flags']}")
        for h in r["headlines"]:
            print(f"   [{h['score']:+.1f}] {h['title']}")
