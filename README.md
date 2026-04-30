## deepnewsdef-text-metric-based
Metric-based generated text detection (i.e. feature-based).

## Version 1
A small dataset to see if everything runs smoothly has been added to `data/`. Files ending with `*_paraphrase.txt` and `*_rewrite.txt` have been generated with GPT-5.3, starting from their corresponding human-written articles.

Tuning needs to be done.

We now have classifiers based on dependency trees and SVMs. Running: `python experiment.py`

## Version 1.1

Added cross-validation (i.e. mean accuracy and standard deviation over 10 runs), and document multi-processing: `python processing.py -n 4 data\news_2`
