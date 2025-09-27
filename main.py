import io
import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)  # optional, for multilingual/lemma support

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, session, send_file
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

from nltk.corpus import wordnet as wn
from functools import lru_cache
import random

# -------------------------------
# Caching functions
# -------------------------------

# Cache synsets for each word
@lru_cache(maxsize=5000)
def synsets_for_word(word):
    return tuple(wn.synsets(word))

# Cache path similarity for each pair of words
similarity_cache = {}

def max_path_similarity_between(word1, word2):
    key = tuple(sorted([word1, word2]))
    if key in similarity_cache:
        return similarity_cache[key]

    s1 = synsets_for_word(word1)
    s2 = synsets_for_word(word2)

    max_sim = 0
    for a in s1:
        for b in s2:
            sim = a.path_similarity(b)
            if sim is not None and sim > max_sim:
                max_sim = sim

    similarity_cache[key] = max_sim
    return max_sim

# -------------------------------
# Collect similar candidates
# -------------------------------

def collect_similar_candidates(seed_word, pool, max_candidates=None):
    scored = []
    for w in pool:
        if w == seed_word:
            continue
        sim = max_path_similarity_between(seed_word, w)
        scored.append((w, sim or 0))
    scored.sort(key=lambda x: x[1], reverse=True)
    if max_candidates:
        scored = scored[:max_candidates]
    for w, _ in scored:
        yield w

# -------------------------------
# Collect dissimilar candidates
# -------------------------------

def collect_dissimilar_candidates(seed_word, pool, max_candidates=None):
    scored = []
    for w in pool:
        if w == seed_word:
            continue
        sim = max_path_similarity_between(seed_word, w)
        scored.append((w, sim or 0))
    scored.sort(key=lambda x: x[1])  # low similarity first
    if max_candidates:
        scored = scored[:max_candidates]
    for w, _ in scored:
        yield w

# -------------------------------
# Generators
# -------------------------------

def semantically_similar_generator(seed_word, pool, frequency='high', min_len=3, max_len=7):
    g = collect_similar_candidates(seed_word, pool)
    for word in g:
        if min_len <= len(word) <= max_len:
            yield word

def semantically_dissimilar_generator(seed_word, pool, frequency='high', min_len=3, max_len=7):
    g = collect_dissimilar_candidates(seed_word, pool)
    for word in g:
        if min_len <= len(word) <= max_len:
            yield word

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
app.secret_key = "replace_this_with_a_random_string"

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

        SMALL_POOL = random.sample(CANDIDATE_POOL, 10000)
        
        # Generate similar words
        g = semantically_similar_generator(word, SMALL_POOL, frequency=frequency, min_len=min_len, max_len=max_len)
        for _ in range(word_num):
            Word_Freq[word]['Similar'].append(next(g))

        # Generate dissimilar words
        h = semantically_dissimilar_generator(word, SMALL_POOL, frequency=frequency, min_len=min_len, max_len=max_len)
        for _ in range(word_num):
            Word_Freq[word]['Dissimilar'].append(next(h))

        # Combine and shuffle
        words_with_labels = (
            [(w, "Similar") for w in Word_Freq[word]['Similar']] +
            [(w, "Dissimilar") for w in Word_Freq[word]['Dissimilar']]
        )
        random.shuffle(words_with_labels)

        df = pd.DataFrame(words_with_labels, columns=["Word", "Key"])
        df["Related to Topic"] = " "
        df["Not Related to Topic"] = " "
        df["I don't know"] = " "

        # Save CSV in session
        session['csv_data'] = df.to_csv(index=False, encoding='utf-8-sig')
        session['topic'] = topic

    return render_template(
        "index.html",
        table=df.to_html(classes="table table-striped", index=False, header=True) if df is not None else None,
        topic=topic
    )
    return render_template(
        "index.html",
        table=df.to_html(classes="table table-striped", index=False) if df is not None else None
    )

@app.route("/download_csv")
def download_csv():
    if 'csv_data' not in session or 'topic' not in session:
        return "No data to download", 400

    csv_bytes = io.BytesIO(session['csv_data'].encode('utf-8-sig'))
    csv_bytes.seek(0)

    filename = f"{session['topic']}_words.csv"
    return send_file(
        csv_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

if __name__ == "__main__":
    app.run(debug=True)
