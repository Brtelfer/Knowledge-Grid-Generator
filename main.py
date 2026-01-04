import nltk
import os

try:
    nltk_data_dir = '/tmp/nltk_data'
    if os.path.exists(nltk_data_dir):
        nltk.data.path.append(nltk_data_dir)
except:
    pass  # If it fails, NLTK will use default paths

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        print("punkt_tab not available, using standard tokenizer")

import sys
import os
import pandas as pd
import random
import re
from collections import Counter
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
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
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

def simple_tokenize(text):
    """Simple tokenization that doesn't rely on complex NLTK tokenizers"""
    # Use regex to split on non-alphanumeric characters
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text.lower())

def extract_vocabulary_from_texts(texts, topic, min_word_length=3, max_words=50):
    """Extract relevant vocabulary from input texts related to the topic."""
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback stopwords if NLTK stopwords not available
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    all_words = []
    
    for text in texts:
        if not text.strip():
            continue
            
        # Use simple tokenization
        words = simple_tokenize(text)
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
        try:
            similarity = max_path_similarity_between(topic.lower(), word)
            if similarity > 0.7:  # Threshold for semantic relevance
                relevant_words.append((word, freq))
        except:
            # If similarity check fails, include the word anyway
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
    """Generate the knowledge grid with guaranteed proportions."""
    
    # Extract vocabulary from texts
    texts_vocabulary = extract_vocabulary_from_texts(texts, topic)
    
    # Calculate exact word counts
    words_from_texts_count = total_words // 4  # 25%
    related_not_in_texts_count = total_words // 4  # 25% 
    dissimilar_count = total_words // 2  # 50%
    
    SMALL_POOL = random.sample(CANDIDATE_POOL, 5000)
    
    # 1. Words from texts (25%) - with fallback
    available_text_words = min(words_from_texts_count, len(texts_vocabulary))
    words_from_texts = random.sample(texts_vocabulary, available_text_words) if available_text_words > 0 else []
    
    # If we don't have enough text words, adjust other categories
    text_words_shortfall = words_from_texts_count - available_text_words
    if text_words_shortfall > 0:
        # Add the shortfall to related-not-in-texts
        related_not_in_texts_count += text_words_shortfall
    
    # 2. Related words not in texts (25%) - with guaranteed generation
    related_not_in_texts = []
    similar_gen = semantically_similar_generator(topic, SMALL_POOL, frequency=frequency, min_len=min_len, max_len=max_len)
    text_vocab_set = set(texts_vocabulary)
    
    for word in similar_gen:
        if word not in text_vocab_set and word != topic:
            related_not_in_texts.append(word)
            if len(related_not_in_texts) >= related_not_in_texts_count:
                break
    
    # If we don't have enough related words, adjust dissimilar count
    related_shortfall = related_not_in_texts_count - len(related_not_in_texts)
    if related_shortfall > 0:
        dissimilar_count += related_shortfall
    
    # 3. Dissimilar words (50%) - generate exactly what we need
    dissimilar_words = []
    dis_gen = semantically_dissimilar_generator(topic, SMALL_POOL, frequency=frequency, min_len=min_len, max_len=max_len)
    
    # Use a set to avoid duplicates from the generator
    seen_words = set(words_from_texts + related_not_in_texts)
    
    for word in dis_gen:
        if word not in seen_words and word != topic:
            dissimilar_words.append(word)
            seen_words.add(word)
            if len(dissimilar_words) >= dissimilar_count:
                break
    
    # Final validation and padding if necessary
    final_words_from_texts = words_from_texts
    final_related_not_in_texts = related_not_in_texts
    final_dissimilar = dissimilar_words
    
    # If we're still short, pad with extra dissimilar words
    total_actual = len(final_words_from_texts) + len(final_related_not_in_texts) + len(final_dissimilar)
    if total_actual < total_words:
        extra_needed = total_words - total_actual
        # Generate more dissimilar words to pad
        for word in dis_gen:
            if word not in seen_words and word != topic:
                final_dissimilar.append(word)
                seen_words.add(word)
                if len(final_dissimilar) >= len(dissimilar_words) + extra_needed:
                    break
    
    # Combine all words with labels
    words_with_labels = (
        [(w, "From Texts") for w in final_words_from_texts] +
        [(w, "Related Not in Texts") for w in final_related_not_in_texts] +
        [(w, "Dissimilar") for w in final_dissimilar]
    )
    
    # Final check - truncate to exact total if we somehow got too many
    if len(words_with_labels) > total_words:
        words_with_labels = words_with_labels[:total_words]
    
    # Randomize order
    random.shuffle(words_with_labels)
    
    return words_with_labels
