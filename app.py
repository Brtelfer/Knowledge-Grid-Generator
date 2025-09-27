import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request
import pandas as pd
import random



"""
Semantic word generators: similar + dissimilar.

Usage:
  - semantically_similar_generator(seed_word, frequency='medium', min_len=None, max_len=None)
  - semantically_dissimilar_generator(seed_word, frequency='medium', min_len=None, max_len=None)

Frequency options: 'high', 'medium', 'low', 'any'
Length constraints are character counts.

The script prefers to use `wordfreq` for frequency info. If wordfreq is missing, frequency filter will be a no-op and you'll see a warning.
Requires: nltk (WordNet). If you don't have nltk data for wordnet, run:
    import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')
"""

from typing import Generator, Optional, Iterable, List, Tuple
import warnings

# Try to import wordfreq for frequency control — optional
try:
    from wordfreq import zipf_frequency, top_n_list
    _HAS_WORDFREQ = True
except Exception:
    _HAS_WORDFREQ = False

# NLTK WordNet (used for semantics)
try:
    import nltk
    from nltk.corpus import wordnet as wn
except Exception as e:
    raise ImportError("nltk and wordnet are required. Install nltk and download wordnet (nltk.download('wordnet')).") from e

# If wordfreq present, build candidate pool from top list; else fallback to WordNet lemma names
POOL_SIZE = 50000

if _HAS_WORDFREQ:
    try:
        CANDIDATE_POOL = top_n_list("en", n=POOL_SIZE)
        # Precache zipf frequencies for the pool
        _ZIPF_CACHE = {w: zipf_frequency(w, "en") for w in CANDIDATE_POOL}
    except Exception:
        _HAS_WORDFREQ = False

if not _HAS_WORDFREQ:
    warnings.warn("wordfreq not available — frequency filtering will be ignored. Using WordNet lemma list as candidate pool.")
    # Create a candidate list from WordNet lemma names (unique, normalized)
    lemmas = set()
    for syn in wn.all_synsets():
        for l in syn.lemmas():
            lemmas.add(l.name().replace('_', ' '))
    CANDIDATE_POOL = sorted(lemmas)  # deterministic ordering
    _ZIPF_CACHE = {}  # empty

# Frequency bucket thresholds (zipf)
FREQ_BUCKETS = {
    "high": (5.0, 10.0),
    "medium": (3.0, 5.0),
    "low": (-10.0, 3.0),
    "any": (-10.0, 10.0)
}

def frequency_filter(word: str, bucket: str) -> bool:
    if bucket == "any" or not _HAS_WORDFREQ:
        return True
    z = _ZIPF_CACHE.get(word)
    if z is None:
        # if not cached, try computing; if that fails, pass
        try:
            from wordfreq import zipf_frequency as _zf
            z = _zf(word, "en")
        except Exception:
            return True
    lo, hi = FREQ_BUCKETS.get(bucket, FREQ_BUCKETS["any"])
    return lo <= z < hi

# WordNet helpers
def synsets_for_word(word: str):
    return wn.synsets(word)

def max_path_similarity_between(word1: str, word2: str) -> Optional[float]:
    """Return the maximum path_similarity (0..1) between any synset pair, or None."""
    s1 = synsets_for_word(word1)
    s2 = synsets_for_word(word2)
    best = None
    for a in s1:
        for b in s2:
            try:
                sim = a.path_similarity(b)
            except Exception:
                sim = None
            if sim is not None:
                if best is None or sim > best:
                    best = sim
    return best

def antonyms_for_word(word: str) -> List[str]:
    ants = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                ants.add(ant.name().replace('_',' '))
    return sorted(ants)

# Ranking collectors
def collect_similar_candidates(seed: str, pool: Iterable[str], max_candidates: int = 1000) -> List[str]:
    scores: List[Tuple[str, float]] = []
    for w in pool:
        if w.lower() == seed.lower(): 
            continue
        sim = max_path_similarity_between(seed, w)
        if sim is None:
            # weak proxy: character overlap scaled down
            shared = len(set(seed.lower()) & set(w.lower()))
            sim = (shared / max(1, len(set(seed.lower()) | set(w.lower())))) * 0.25
        scores.append((w, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scores[:max_candidates]]

def collect_dissimilar_candidates(seed: str, pool: Iterable[str], max_candidates: int = 1000) -> List[str]:
    scores: List[Tuple[str, float]] = []
    for w in pool:
        if w.lower() == seed.lower():
            continue
        sim = max_path_similarity_between(seed, w)
        if sim is None:
            shared = len(set(seed.lower()) & set(w.lower()))
            sim = (shared / max(1, len(set(seed.lower()) | set(w.lower())))) * 0.25
        scores.append((w, sim))
    scores.sort(key=lambda x: x[1])  # ascending => least similar first
    return [w for w, _ in scores[:max_candidates]]

# Generators
def semantically_similar_generator(seed_word: str, frequency: str = 'medium', min_len: Optional[int] = None, max_len: Optional[int] = None) -> Generator[str, None, None]:
    """
    Yields words semantically similar to seed_word.
    frequency: 'high'|'medium'|'low'|'any' (best-effort; ignored if wordfreq isn't available)
    min_len, max_len: optional integer length bounds (characters)
    """
    # filter pool
    def length_ok(w: str) -> bool:
        if min_len is not None and len(w) < min_len: return False
        if max_len is not None and len(w) > max_len: return False
        return True

    pool = [w for w in CANDIDATE_POOL if frequency_filter(w, frequency) and length_ok(w)]
    ranked = collect_similar_candidates(seed_word, pool, max_candidates=len(pool))
    for w in ranked:
        yield w

def semantically_dissimilar_generator(seed_word: str, frequency: str = 'medium', min_len: Optional[int] = None, max_len: Optional[int] = None) -> Generator[str, None, None]:
    """
    Yields words semantically dissimilar to seed_word.
    Prefers explicit antonyms (if any), then least-similar words from pool.
    """
    def length_ok(w: str) -> bool:
        if min_len is not None and len(w) < min_len: return False
        if max_len is not None and len(w) > max_len: return False
        return True

    # yield antonyms first
    for a in antonyms_for_word(seed_word):
        if frequency_filter(a, frequency) and length_ok(a):
            yield a

    pool = [w for w in CANDIDATE_POOL if frequency_filter(w, frequency) and length_ok(w)]
    ranked = collect_dissimilar_candidates(seed_word, pool, max_candidates=len(pool))
    yielded = set()
    for a in antonyms_for_word(seed_word):
        yielded.add(a.lower())
    for w in ranked:
        if w.lower() in yielded: 
            continue
        yield w

# Quick demo function for convenience
def demo(seed: str, frequency='medium', min_len=None, max_len=None, n=12):
    print(f"DEMO — seed={seed!r}, frequency={frequency}, min_len={min_len}, max_len={max_len}\n")
    sim = []
    dis = []
    sim_gen = semantically_similar_generator(seed, frequency=frequency, min_len=min_len, max_len=max_len)
    dis_gen = semantically_dissimilar_generator(seed, frequency=frequency, min_len=min_len, max_len=max_len)
    for _ in range(n):
        try:
            sim.append(next(sim_gen))
        except StopIteration:
            break
    for _ in range(n):
        try:
            dis.append(next(dis_gen))
        except StopIteration:
            break
    print("Similar (top):", sim)
    print("Dissimilar (top):", dis)


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    df = None
    if request.method == "POST":
        # Collect form inputs
        word = request.form.get("word")
        frequency = request.form.get("frequency", "high")
        min_len = int(request.form.get("min_len", 3))
        max_len = int(request.form.get("max_len", 7))
        word_num = int(request.form.get("word_num", 5))

        # Build dictionary
        Word_Freq = {word: {'Similar': [], 'Dissimilar': []}}

        # Generate similar words
        g = semantically_similar_generator(word, frequency=frequency, min_len=min_len, max_len=max_len)
        for _ in range(word_num):
            Word_Freq[word]['Similar'].append(next(g))

        # Generate dissimilar words
        h = semantically_dissimilar_generator(word, frequency=frequency, min_len=min_len, max_len=max_len)
        for _ in range(word_num):
            Word_Freq[word]['Dissimilar'].append(next(h))

        # Combine and shuffle
        words_with_labels = (
            [(w, "Similar") for w in Word_Freq[word]['Similar']] +
            [(w, "Dissimilar") for w in Word_Freq[word]['Dissimilar']]
        )
        random.shuffle(words_with_labels)

        df = pd.DataFrame(words_with_labels, columns=["Word", "Key"])
        df["Related to Topic"] = "○"
        df["Not Related to Topic"] = "○"
        df["I don't know"] = "○"

    return render_template("index.html", table=df.to_html(classes="table table-striped", index=False) if df is not None else None)


if __name__ == "__main__":
    app.run(debug=True)
