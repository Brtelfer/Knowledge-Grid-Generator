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
import re
from collections import Counter

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
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
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

# -------------------------------
# Text processing functions
# -------------------------------

def extract_vocabulary_from_texts(texts, topic, min_word_length=3, max_words=50):
    """Extract relevant vocabulary from input texts related to the topic."""
    stop_words = set(stopwords.words('english'))
    all_words = []
    
    for text in texts:
        if not text.strip():
            continue
            
        # Tokenize and clean text
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and len(word) >= min_word_length]
        words = [word for word in words if word not in stop_words]
        
        all_words.extend(words)
    
    # Count word frequencies
    word_freq = Counter(all_words)
    
    # Filter words that are semantically related to the topic
    relevant_words = []
    for word, freq in word_freq.most_common(max_words * 2):  # Get more candidates for filtering
        if word == topic.lower():
            relevant_words.append((word, freq))
            continue
            
        # Check semantic similarity with topic
        similarity = max_path_similarity_between(topic.lower(), word)
        if similarity > 0.1:  # Threshold for semantic relevance
            relevant_words.append((word, freq))
    
    # Return top relevant words by frequency
    relevant_words.sort(key=lambda x: x[1], reverse=True)
    return [word for word, freq in relevant_words[:max_words]]

def generate_related_words_not_in_texts(topic, texts_vocab, pool, num_words=25, frequency='high', min_len=3, max_len=7):
    """Generate words related to topic but not in the texts vocabulary."""
    generated_words = []
    text_vocab_set = set(texts_vocab)
    
    similar_gen = semantically_similar_generator(topic, pool, frequency=frequency, min_len=min_len, max_len=max_len)
    
    for word in similar_gen:
        if word not in text_vocab_set and word != topic:
            generated_words.append(word)
            if len(generated_words) >= num_words:
                break
                
    return generated_words

def generate_knowledge_grid(topic, texts, total_words=20, frequency='high', min_len=3, max_len=7):
    """Generate the knowledge grid with the specified proportions."""
    
    # Extract vocabulary from texts
    texts_vocabulary = extract_vocabulary_from_texts(texts, topic)
    
    # Calculate word counts based on proportions
    total_related = total_words // 2
    words_from_texts_count = total_related // 2  # 25% of total
    related_not_in_texts_count = total_related // 2  # 25% of total
    dissimilar_count = total_words // 2  # 50% of total
    
    SMALL_POOL = random.sample(CANDIDATE_POOL, 20000)
    
    # 1. Words from texts (25%)
    words_from_texts = random.sample(texts_vocabulary, min(words_from_texts_count, len(texts_vocabulary)))
    
    # 2. Related words not in texts (25%)
    related_not_in_texts = generate_related_words_not_in_texts(
        topic, texts_vocabulary, SMALL_POOL, 
        num_words=related_not_in_texts_count,
        frequency=frequency, min_len=min_len, max_len=max_len
    )
    
    # 3. Dissimilar words (50%)
    dissimilar_words = []
    dis_gen = semantically_dissimilar_generator(topic, SMALL_POOL, frequency=frequency, min_len=min_len, max_len=max_len)
    for _ in range(dissimilar_count):
        try:
            word = next(dis_gen)
            if word not in texts_vocabulary:
                dissimilar_words.append(word)
        except StopIteration:
            break
    
    # Combine all words with labels
    words_with_labels = (
        [(w, "From Texts") for w in words_from_texts] +
        [(w, "Related Not in Texts") for w in related_not_in_texts] +
        [(w, "Dissimilar") for w in dissimilar_words]
    )
    
    # Randomize order
    random.shuffle(words_with_labels)
    
    return words_with_labels

app = Flask(__name__)
app.secret_key = "replace_this_with_a_random_string"

@app.route("/", methods=["GET", "POST"])
def index():
    df = None
    topic = None

    if request.method == "POST":
        # Collect form inputs
        topic = request.form.get("topic")
        text1 = request.form.get("text1", "")
        text2 = request.form.get("text2", "")
        text3 = request.form.get("text3", "")
        
        frequency = request.form.get("frequency", "high")
        min_len = int(request.form.get("min_len", 3))
        max_len = int(request.form.get("max_len", 7))
        word_num = int(request.form.get("word_num", 20))
        
        texts = [text1, text2, text3]
        texts = [text for text in texts if text.strip()]  # Remove empty texts
        
        if not topic:
            return "Please provide a topic", 400
        if not texts:
            return "Please provide at least one text", 400
        
        # Generate knowledge grid
        words_with_labels = generate_knowledge_grid(
            topic, texts, 
            total_words=word_num,
            frequency=frequency,
            min_len=min_len,
            max_len=max_len
        )
        
        # Create DataFrame
        df = pd.DataFrame(words_with_labels, columns=["Word", "Type"])
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

@app.route("/download_csv")
def download_csv():
    if 'csv_data' not in session or 'topic' not in session:
        return "No data to download", 400

    csv_bytes = io.BytesIO(session['csv_data'].encode('utf-8-sig'))
    csv_bytes.seek(0)

    filename = f"{session['topic']}_knowledge_grid.csv"
    return send_file(
        csv_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

if __name__ == "__main__":
    app.run(debug=True)
