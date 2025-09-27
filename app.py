import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request
import pandas as pd
import random
import semgens
from semgens import semantically_similar_generator, semantically_dissimilar_generator


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
