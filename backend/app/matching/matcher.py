"""
Optimized matcher.py
Target: medium dataset (2k-10k entries)
Features:
 - exact phrase
 - subphrase (split alternatives)
 - token-level IDF-weighted scoring
 - multiword-aware token filtering
 - generic token blacklist
 - single-char / numeric token filtering
 - fuzzy matching using token_set_ratio (with fast prefilter)
 - precomputations at startup for speed

Save as: backend/app/matching/matcher.py
"""

from typing import Dict, Any, List, Set, Tuple
import math
import re
from rapidfuzz import fuzz
from .datastore import KeywordStore, KeywordEntry
from .preprocess import normalize_text, tokenize_and_lemmatize
from . import config

# Tunables (can also live in config.py)
_MIN_IDF_FACTOR = 0.6
_MAX_IDF_FACTOR = 3.5
_MAX_TOKEN_WEIGHT = 10.0

_SUBPHRASE_MIN_TOKENS = getattr(config, "SUBPHRASE_MIN_TOKENS", 2)
_SUBPHRASE_WEIGHT = getattr(config, "SUBPHRASE_WEIGHT", 2.5)

_MIN_MEANINGFUL_TOKENS_FOR_RELEVANT = getattr(config, "MIN_MEANINGFUL_TOKENS_FOR_RELEVANT", 2)
_SINGLE_TOKEN_MAX_SCORE = getattr(config, "SINGLE_TOKEN_MAX_SCORE", 10)
_GENERIC_TOKEN_FREQ_RATIO = getattr(config, "GENERIC_TOKEN_FREQ_RATIO", 0.30)
_GENERIC_TOKEN_BLACKLIST = set(getattr(config, "GENERIC_TOKEN_BLACKLIST", ["fiber", "media", "converter", "system", "device", "unit", "module", "kit", "tool"]))

# Fuzzy prefilter: minimal token-overlap fraction (relative to entry tokens) to bother calling rapidfuzz
# For medium dataset we allow a conservative threshold to avoid most expensive calls
_FUZZY_PREFILTER_MIN_OVERLAP_RATIO = getattr(config, "FUZZY_PREFILTER_MIN_OVERLAP_RATIO", 0.35)


def _is_input_token_valid(tok: str) -> bool:
    """Filter noisy input tokens: ignore single-char tokens, pure digits."""
    if not tok:
        return False
    if len(tok) < 2:
        return False
    if tok.isdigit():
        return False
    return True


def _split_alternatives_from_phrase(raw_phrase: str) -> List[str]:
    """Split phrase into alternatives using common separators."""
    s = raw_phrase
    s = re.sub(r"\s+or\s+|\s+OR\s+|\s+Or\s+", "|", s)
    s = s.replace("/", "|").replace(";", "|").replace("| |", "|")
    s = re.sub(r"\|+", "|", s)
    alts = [a.strip() for a in s.split("|") if a.strip()]
    return alts


class OptimizedEntry:
    """
    Lightweight cached representation of a KeywordEntry for fast requests.
    """
    __slots__ = ("phrase", "norm", "category", "tokens", "alt_norms", "alt_token_sets", "token_count")

    def __init__(self, entry: KeywordEntry):
        self.phrase: str = entry.phrase
        self.norm: str = entry.norm
        self.category: str = entry.category
        # token set from datastore entry (already lemmatized/normalized)
        self.tokens: Set[str] = set(entry.tokens) if entry.tokens else set()
        self.token_count: int = len(self.tokens)
        # alternatives: normalized alternative strings and their token sets
        alts = _split_alternatives_from_phrase(entry.phrase)
        self.alt_norms: List[str] = [normalize_text(a, keep_hyphen=True) for a in alts] if alts else [entry.norm]
        self.alt_token_sets: List[Set[str]] = [set(tokenize_and_lemmatize(a, keep_hyphen=True)) for a in self.alt_norms]


class Matcher:
    def __init__(self, store: KeywordStore):
        self.store = store
        self.total_entries = max(1, self.store.size())

        # Precompute token frequencies and IDF factors once
        self.token_freq: Dict[str, int] = {t: len(idxs) for t, idxs in self.store.token_index.items()}
        self.idf_factor: Dict[str, float] = {}
        for tok, freq in self.token_freq.items():
            factor = 1.0 + math.log((self.total_entries) / (1 + freq)) if freq >= 0 else 1.0
            self.idf_factor[tok] = max(_MIN_IDF_FACTOR, min(_MAX_IDF_FACTOR, factor))

        # Build single-word vs multiword token sets
        single_tokens: Set[str] = set()
        multi_tokens: Set[str] = set()
        for e in self.store.all_entries():
            toks = e.tokens or set()
            if len(toks) <= 1:
                single_tokens.update(toks)
            else:
                multi_tokens.update(toks)
        self.single_word_tokens = single_tokens
        self.multiword_tokens = multi_tokens

        # Precompute lightweight entries for fast access (list preserves store index ordering)
        self.entries: List[OptimizedEntry] = [OptimizedEntry(e) for e in self.store.all_entries()]

        # Precompute a mapping entry_index -> optimized entry (for direct token index lookups)
        self.indexed_entries: Dict[int, OptimizedEntry] = {i: self.entries[i] for i in range(len(self.entries))}

        # Small helper caches for frequently-used config values
        self.exact_weight = getattr(config, "EXACT_PHRASE_WEIGHT", 10.0)
        self.token_weight = getattr(config, "TOKEN_WEIGHT", 1.0)
        self.fuzzy_strong_w = getattr(config, "FUZZY_STRONG_WEIGHT", 5.0)
        self.fuzzy_weak_w = getattr(config, "FUZZY_WEAK_WEIGHT", 2.0)

    # ---------- internal helpers ----------
    def _is_generic_token(self, tok: str) -> bool:
        if tok in _GENERIC_TOKEN_BLACKLIST:
            return True
        freq = self.token_freq.get(tok, 0)
        if freq / self.total_entries > _GENERIC_TOKEN_FREQ_RATIO:
            return True
        return False

    def _add_match(self, collector_matches: List[Dict[str, Any]], matched_phrases: Set[str],
                   entry: OptimizedEntry, match_type: str, weight: float, matched_text: str,
                   category_scores: Dict[str, float], raw_score_ref: Dict[str, float]):
        if entry.phrase in matched_phrases:
            return
        collector_matches.append({
            "phrase": entry.phrase,
            "category": entry.category,
            "match_type": match_type,
            "weight": weight,
            "matched_text": matched_text
        })
        matched_phrases.add(entry.phrase)
        raw_score_ref["v"] += weight
        category_scores[entry.category] = category_scores.get(entry.category, 0.0) + weight

    # ---------- main analyze ----------
    def analyze(self, text: str, category_filter: str = "all") -> Dict[str, Any]:
        if not text or not text.strip():
            return {"relevant": False, "score_pct": 0, "matches": [], "category_scores": {}, "raw_score": 0.0, "matched_count": 0}

        cf = (category_filter or "all").strip().lower()

        norm_text = normalize_text(text, keep_hyphen=True)
        # tokenized input (raw) and filtered tokens for matching
        text_tokens_raw = list(set(tokenize_and_lemmatize(text, keep_hyphen=True)))
        text_tokens = set([t for t in text_tokens_raw if _is_input_token_valid(t)])
        if not text_tokens:
            # nothing meaningful to match
            return {"relevant": False, "score_pct": 0, "matches": [], "category_scores": {}, "raw_score": 0.0, "matched_count": 0}

        matches: List[Dict[str, Any]] = []
        matched_phrases: Set[str] = set()
        raw_score_ref = {"v": 0.0}
        category_scores: Dict[str, float] = {}

        # ---------- 1) Exact phrase matches (fast substring check on normalized text) ----------
        # Use precomputed normalized phrase map from datastore
        for norm_phrase, entry in list(self.store.phrase_map.items()):
            if cf != "all" and entry.category.lower() != cf:
                continue
            if not norm_phrase:
                continue
            if norm_phrase in norm_text:
                opt_entry = OptimizedEntry(entry)  # small wrapper for this hit (cheap)
                self._add_match(matches, matched_phrases, opt_entry, "exact", self.exact_weight, entry.phrase, category_scores, raw_score_ref)

        # ---------- 2) Subphrase matching: evaluate alternatives precomputed via OptimizedEntry.alt_token_sets ----------
        # We iterate over store entries directly (fast)
        for i, opt in enumerate(self.entries):
            if cf != "all" and opt.category.lower() != cf:
                continue
            if opt.phrase in matched_phrases:
                continue
            # For each normalized alternative, check token overlap with input tokens quickly
            for alt_tokens in opt.alt_token_sets:
                if not alt_tokens:
                    continue
                # skip alt if it has no intersection
                overlap = alt_tokens & text_tokens
                if len(overlap) >= _SUBPHRASE_MIN_TOKENS:
                    # add subphrase match
                    self._add_match(matches, matched_phrases, opt, "subphrase", _SUBPHRASE_WEIGHT, ", ".join(sorted(overlap)), category_scores, raw_score_ref)
                    break  # matched this entry, move to next entry

        # ---------- 3) Candidate selection using token index (fast) ----------
        candidate_indices: Set[int] = set()
        for tok in text_tokens:
            idxs = self.store.token_index.get(tok)
            if idxs:
                candidate_indices.update(idxs)

        if not candidate_indices:
            # fallback: consider all entries (small datasets only)
            candidate_list = [self.entries[i] for i in range(len(self.entries)) if (cf == "all" or self.entries[i].category.lower() == cf)]
        else:
            candidate_list = [self.indexed_entries[i] for i in candidate_indices if (cf == "all" or self.indexed_entries[i].category.lower() == cf)]

        # ---------- 4) Token-level scoring using precomputed IDF factors ----------
        meaningful_tokens_matched: Set[str] = set()
        for opt in candidate_list:
            if opt.phrase in matched_phrases:
                continue
            overlap = opt.tokens & text_tokens
            if not overlap:
                continue

            # multiword-aware: allow token only if it's standalone in CSV or not exclusively multiword-only
            allowed_overlap = {t for t in overlap if (t in self.single_word_tokens) or (t not in self.multiword_tokens)}
            if not allowed_overlap:
                continue

            # sum IDF-weighted token contributions (fast lookup)
            token_weight_sum = 0.0
            for t in allowed_overlap:
                idf_f = self.idf_factor.get(t, 1.0)
                token_weight_sum += self.token_weight * idf_f
            weight = min(token_weight_sum, _MAX_TOKEN_WEIGHT)
            self._add_match(matches, matched_phrases, opt, "token", weight, ", ".join(sorted(allowed_overlap)), category_scores, raw_score_ref)

            for t in allowed_overlap:
                if not self._is_generic_token(t):
                    meaningful_tokens_matched.add(t)

        # ---------- 5) Fuzzy matching (expensive) but only when prefilter suggests likelihood ----------
        # Prefilter reduces number of fuzzy calls dramatically:
        for opt in candidate_list:
            if opt.phrase in matched_phrases:
                continue
            # compute token overlap fraction (relative to entry size)
            if opt.token_count == 0:
                continue
            inter = len(opt.tokens & text_tokens)
            overlap_ratio = inter / opt.token_count
            # only consider fuzzy if overlap >= threshold OR if there is at least one non-generic token match
            consider_fuzzy = (overlap_ratio >= _FUZZY_PREFILTER_MIN_OVERLAP_RATIO) or (inter >= 1)
            if not consider_fuzzy:
                continue
            # now perform accurate fuzzy check (token_set_ratio)
            try:
                ratio = fuzz.token_set_ratio(opt.norm, norm_text)
            except Exception:
                ratio = fuzz.ratio(opt.norm, norm_text)
            if ratio >= getattr(config, "FUZZY_STRONG_RATIO", 90):
                self._add_match(matches, matched_phrases, opt, "fuzzy_strong", self.fuzzy_strong_w, f"ratio:{ratio}", category_scores, raw_score_ref)
            elif ratio >= getattr(config, "FUZZY_WEAK_RATIO", 75):
                self._add_match(matches, matched_phrases, opt, "fuzzy_weak", self.fuzzy_weak_w, f"ratio:{ratio}", category_scores, raw_score_ref)

        # ---------- 6) Category boost ----------
        for cat in list(category_scores.keys()):
            count_matches_for_cat = sum(1 for m in matches if m["category"] == cat)
            if count_matches_for_cat >= 2:
                bonus = 0.5 * (count_matches_for_cat - 1)
                category_scores[cat] = category_scores.get(cat, 0.0) + bonus

        # ---------- 7) Normalize score to percentage ----------
        matched_count = max(1, len({m["phrase"] for m in matches}))
        base_for_denominator = max(3, matched_count)
        denom_keywords = min(getattr(config, "MAX_KEYWORDS_CONSIDERED", 8), base_for_denominator)
        max_possible = self.exact_weight * denom_keywords
        raw_score = raw_score_ref["v"]
        score_pct = int(min(100, round(100.0 * raw_score / max(1.0, max_possible))))

        # ---------- 8) Single-token / generic safeguard ----------
        has_exact_or_strong = any(m["match_type"] in ("exact", "fuzzy_strong", "subphrase") for m in matches)
        meaningful_count = len(meaningful_tokens_matched)
        if meaningful_count < _MIN_MEANINGFUL_TOKENS_FOR_RELEVANT and not has_exact_or_strong:
            final_relevant = False
            final_score = min(score_pct, _SINGLE_TOKEN_MAX_SCORE)
        else:
            final_relevant = score_pct >= getattr(config, "RELEVANT_SCORE_THRESHOLD", 50)
            final_score = score_pct

        # sort matches and format category scores
        matches_sorted = sorted(matches, key=lambda m: (-m["weight"], m["match_type"], m["phrase"]))
        category_scores_out = {k: (int(v) if float(v).is_integer() else round(v, 2)) for k, v in category_scores.items()}

        return {
            "relevant": bool(final_relevant),
            "score_pct": int(final_score),
            "matches": matches_sorted,
            "category_scores": category_scores_out,
            "raw_score": raw_score,
            "matched_count": len(matched_phrases),
            "meaningful_tokens_matched": sorted(list(meaningful_tokens_matched)),
            "text_tokens_raw": sorted(text_tokens_raw),
            "text_tokens_filtered": sorted(list(text_tokens))
        }
