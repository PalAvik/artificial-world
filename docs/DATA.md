# Source corpora

`freeflow/data/wordlist.tsv` and `freeflow/data/synonyms.tsv` are **generated**
and committed, so a checkout can run without network access. Their *inputs* are
not committed — `/data/` is gitignored, and these are third-party corpora with
their own licences. Fetch them before regenerating:

```bash
mkdir -p data/sources && cd data/sources
curl -sSLO https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt
mv Concreteness_ratings_Brysbaert_et_al_BRM.txt bry.txt
curl -sSLO https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt
curl -sSL -o badwords.txt https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en
cd ../.. && python scripts/build_vocab.py && python scripts/build_synonyms.py
```

`build_synonyms.py` additionally needs WordNet. NLTK's downloader refuses a
proxied fetch, so take the corpus directly rather than disabling that check:

```bash
mkdir -p ~/nltk_data/corpora
curl -sSL -o /tmp/wordnet.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip
unzip -q -o /tmp/wordnet.zip -d ~/nltk_data/corpora/
```

| file | source | supplies |
|---|---|---|
| `bry.txt` | Brysbaert, Warriner & Kuperman (2014), *Concreteness ratings for 40 thousand generally known English word lemmas*, Behavior Research Methods 46(3) | concreteness (1–5), share of raters who knew the word, SUBTLEX frequency, dominant part of speech |
| `google-10000-english-usa.txt` | `first20hours/google-10000-english` | frequency ordering, to prefer words the model has plausibly seen |
| `badwords.txt` | `LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words`, English list | excluding slurs and profanity from a corpus that gets rendered into images |
| WordNet | Princeton WordNet via `nltk_data` | the synonym control |

The concreteness norms are what make the concrete/abstract split a measurement
rather than an author's intuition, and they are citable. **Cite Brysbaert et al.
wherever those classes are used.**
