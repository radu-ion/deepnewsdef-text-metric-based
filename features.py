import os
from typing import List, Dict, Tuple, Optional
import re
from typing import Dict, List, Set, Union
import string
from collections import Counter, defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import math
from conllu.models import SentenceList
import numpy as np

# Minimum word frequency so that the word is considered
# when computing features
conf_min_freq = 3
# How many segments to split the text into for certain features
conf_segments = 10
# Local entropy is computed over this many words
conf_window_size = 50
# For n-gram feature computation
conf_max_ngram = 3

content_poses = {"NOUN", "VERB", "ADJ", "ADV"}
function_poses = {"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"}
clause_dep_labels = {"ccomp", "xcomp", "advcl", "acl", "relcl"}
clause_heads_uposes = {"VERB", "AUX"}


def remove_diacs(word: str) -> str:
    word = word.replace('ă', 'a')
    word = word.replace('â', 'a')
    word = word.replace('î', 'i')
    word = word.replace('ș', 's')
    word = word.replace('ț', 't')

    return word


def _read_romanian_function_words() -> Set[str]:
    result = set()

    with open(os.path.join('resources', 'func_words_ro.txt'),
              mode='r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            result.add(word)
            result.add(remove_diacs(word))
        # end for
    # end with

    return result


def _read_romanian_stop_words() -> Set[str]:
    result = set()

    with open(os.path.join('resources', 'stop_words_ro.txt'),
              mode='r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            result.add(word)
            result.add(remove_diacs(word))
        # end for
    # end with

    return result


def read_func_stop_words() -> Tuple[Set[str], Set[str]]:
    f_words = _read_romanian_function_words()
    s_words = _read_romanian_stop_words()
    f_words.difference_update(s_words)

    return f_words, s_words


class TextFeatures:
    def __init__(self, sentences: SentenceList):
        """Input text is already tokenized by a BaseProcessor text processor.
        The BaseProcessor returns the `sentences`."""
        self.sentences = sentences
        self.tokens = self._extract_tokens(make_lower=True)
        self.raw_tokens = self._extract_tokens(make_lower=False)
        self.N = len(self.tokens)

    def _extract_tokens(self, make_lower: bool) -> List[str]:
        tokens = []

        for sent in self.sentences:
            for tok in sent:
                if isinstance(tok["id"], int):
                    # Ignore multiword tokens
                    if make_lower:
                        tokens.append(tok["form"].lower())
                    else:
                        tokens.append(tok["form"])
                    # end if
                # end if
            # end for

        return tokens
    
    def _split_segments(self, K=conf_segments) -> List[List[str]]:
        if self.N == 0:
            return []
        # end if

        seg_size = max(1, self.N // K)

        return [
            self.tokens[i:i + seg_size]
            for i in range(0, self.N, seg_size)
        ]

    def _safe_div(self, num: float, den: float) -> float:
        return num / den if den != 0 else 0.0

    def features(self) -> Dict[str, float]:
        """Collects and executes all feature methods starting with 'feat_'"""
        
        feats = {}

        for attr_name in dir(self):
            if not attr_name.startswith("feat_"):
                continue
            # end if

            method = getattr(self, attr_name)

            if not callable(method):
                continue
            # end if

            feat_name = attr_name[len("feat_"):]
            feat_value = method()

            if type(feat_value) is float:
                feats[feat_name] = feat_value
            elif type(feat_value) is dict:
                feats.update(feat_value)
            # end if
        # end for

        return feats


class BasicLexicalFeatures(TextFeatures):
    def __init__(self, sentences: SentenceList):
        super().__init__(sentences=sentences)

        self.function_words, self.stop_words = read_func_stop_words()
        self.punctuation = set(string.punctuation)

        self.word_tokens = [
            tok for tok in self.tokens
            if not self._is_punctuation(tok)
        ]

        self.word_N = len(self.word_tokens)
        self.counter = Counter(self.word_tokens)
        self.V = len(self.counter)

    def _is_punctuation(self, tok: str) -> bool:
        return all(ch in self.punctuation for ch in tok)

    def _sentence_token_lengths(self) -> List[int]:
        lengths = []

        for sent in self.sentences:
            n = 0
            for tok in sent:
                if isinstance(tok["id"], int):
                    form = tok["form"].lower()
                    if not self._is_punctuation(form):
                        n += 1
                    # end if
                # end if
            # end for
            lengths.append(n)
        # end for

        return lengths

    def _sentence_char_lengths(self) -> List[int]:
        lengths = []

        for sent in self.sentences:
            n = 0
            for tok in sent:
                if isinstance(tok["id"], int):
                    form = tok["form"]
                    if not self._is_punctuation(form):
                        n += len(form)
                    # end if
                # end if
            # end for
            lengths.append(n)
        # end for

        return lengths

    # ------------------------------------------------------------------
    # Basic length features
    # ------------------------------------------------------------------

    def feat_average_word_length(self) -> float:
        return self._safe_div(
            sum(len(tok) for tok in self.word_tokens),
            self.word_N,
        )

    def feat_average_token_length_all(self) -> float:
        return self._safe_div(
            sum(len(tok) for tok in self.tokens),
            self.N,
        )

    def feat_average_sentence_length_tokens(self) -> float:
        lengths = self._sentence_token_lengths()
        return self._safe_div(sum(lengths), len(lengths))

    def feat_average_sentence_length_chars(self) -> float:
        lengths = self._sentence_char_lengths()
        return self._safe_div(sum(lengths), len(lengths))

    # ------------------------------------------------------------------
    # Lexical diversity
    # ------------------------------------------------------------------

    def feat_ttr(self) -> float:
        return self._safe_div(self.V, self.word_N)

    def feat_rttr(self) -> float:
        return self._safe_div(self.V, math.sqrt(self.word_N))

    def feat_cttr(self) -> float:
        return self._safe_div(self.V, math.sqrt(2 * self.word_N))

    def feat_herdan_c(self) -> float:
        if self.word_N <= 1 or self.V <= 1:
            return 0.0
        # end if

        return math.log(self.V) / math.log(self.word_N)

    def feat_guiraud_r(self) -> float:
        return self.feat_rttr()

    def feat_repetition_rate(self) -> float:
        return 1.0 - self.feat_ttr() if self.word_N > 0 else 0.0

    def feat_mean_token_frequency(self) -> float:
        return self._safe_div(self.word_N, self.V)

    # ------------------------------------------------------------------
    # Hapax / frequency spectrum
    # ------------------------------------------------------------------

    def feat_hapax_legomena_ratio(self) -> float:
        hapax = sum(1 for _, f in self.counter.items() if f == 1)
        return self._safe_div(hapax, self.word_N)

    def feat_hapax_dislegomena_ratio(self) -> float:
        dis = sum(1 for _, f in self.counter.items() if f == 2)
        return self._safe_div(dis, self.word_N)

    def feat_frequency_class_ratios(self, max_k: int = 5) -> Dict[str, float]:
        feats = {}

        for k in range(1, max_k):
            count = sum(1 for _, f in self.counter.items() if f == k)

            feats[f"frequency_class_{k}_ratio"] = self._safe_div(
                count,
                self.word_N,
            )
        # end for

        count_ge = sum(1 for _, f in self.counter.items() if f >= max_k)
        feats[f"frequency_class_ge_{max_k}_ratio"] = self._safe_div(
            count_ge,
            self.word_N,
        )

        return feats

    def feat_top_k_coverage(self, ks: Tuple[int, ...] = (10, 50, 100)) -> Dict[str, float]:
        feats = {}

        freqs = sorted(self.counter.values(), reverse=True)

        for k in ks:
            feats[f"top_{k}_coverage"] = self._safe_div(
                sum(freqs[:k]),
                self.word_N,
            )
        # end for

        return feats

    # ------------------------------------------------------------------
    # Stopwords / function words
    # ------------------------------------------------------------------

    def feat_stopword_ratio(self) -> float:
        count = sum(1 for tok in self.word_tokens if tok in self.stop_words)
        return self._safe_div(count, self.word_N)

    def feat_function_word_ratio(self) -> float:
        count = sum(1 for tok in self.word_tokens if tok in self.function_words)
        return self._safe_div(count, self.word_N)

    def feat_content_word_ratio(self) -> float:
        return 1.0 - self.feat_function_word_ratio() if self.word_N > 0 else 0.0

    # ------------------------------------------------------------------
    # Punctuation
    # ------------------------------------------------------------------

    def feat_punctuation_token_ratio(self) -> float:
        punct_count = sum(
            1 for tok in self.tokens if self._is_punctuation(tok))
        return self._safe_div(punct_count, self.N)

    def feat_punctuation_frequencies(self) -> Dict[str, float]:
        marks = {
            ",": "comma",
            ".": "period",
            ";": "semicolon",
            ":": "colon",
            "!": "exclamation",
            "?": "question",
            "-": "hyphen",
            "—": "emdash",
            "(": "left_paren",
            ")": "right_paren",
            '"': "quote",
            "'": "apostrophe",
        }

        feats = {}

        for mark, name in marks.items():
            count = sum(1 for tok in self.tokens if tok == mark)
            feats[f"punct_{name}_freq"] = self._safe_div(count, self.N)
        # end for

        return feats

    # ------------------------------------------------------------------
    # Character-level ratios
    # ------------------------------------------------------------------

    def feat_character_type_ratios(self) -> Dict[str, float]:
        chars = "".join(self.raw_tokens)

        C = len(chars)

        alpha = sum(1 for ch in chars if ch.isalpha())
        digit = sum(1 for ch in chars if ch.isdigit())
        upper = sum(1 for ch in chars if ch.isupper())

        return {
            "alpha_char_ratio": self._safe_div(alpha, C),
            "digit_char_ratio": self._safe_div(digit, C),
            "uppercase_char_ratio": self._safe_div(upper, C),
        }

    # ------------------------------------------------------------------
    # Word length distribution
    # ------------------------------------------------------------------

    def feat_word_length_variance(self) -> float:
        if self.word_N == 0:
            return 0.0

        lengths = [len(tok) for tok in self.word_tokens]
        mu = sum(lengths) / self.word_N

        return sum((x - mu) ** 2 for x in lengths) / self.word_N

    def feat_word_length_cv(self) -> float:
        mu = self.feat_average_word_length()

        if mu == 0.0:
            return 0.0

        var = self.feat_word_length_variance()
        return math.sqrt(var) / mu

    def feat_short_word_ratio(self, threshold: int = 3) -> float:
        count = sum(1 for tok in self.word_tokens if len(tok) <= threshold)
        return self._safe_div(count, self.word_N)

    def feat_long_word_ratio(self, threshold: int = 7) -> float:
        count = sum(1 for tok in self.word_tokens if len(tok) >= threshold)
        return self._safe_div(count, self.word_N)

    # ------------------------------------------------------------------
    # Sentence length distribution
    # ------------------------------------------------------------------

    def feat_sentence_token_length_variance(self) -> float:
        lengths = self._sentence_token_lengths()

        if len(lengths) == 0:
            return 0.0
        # end if

        mu = sum(lengths) / len(lengths)

        return sum((x - mu) ** 2 for x in lengths) / len(lengths)


class DistribVariabFeatures(TextFeatures):
    def __init__(self, sentences: SentenceList):
        super().__init__(sentences=sentences)
        
        self.s_lengths = self._sentence_lengths()
        self.w_lengths = self._word_lengths()

    def _sentence_lengths(self) -> List[int]:
        return [
            sum(1 for tok in sent if isinstance(tok["id"], int))
            for sent in self.sentences
        ]

    def _word_lengths(self) -> List[int]:
        return [len(w) for w in self.tokens]

    def feat_sent_len_variance(self) -> float:
        """Variance of sentence length"""
        
        return float(np.var(self.s_lengths)) if self.s_lengths else 0.0

    def feat_entropy(self) -> float:
        """Token entropy"""
        
        if self.N == 0:
            return 0.0
        # end if

        counts = Counter(self.tokens)
        probs = np.array(list(counts.values())) / self.N

        return float(-np.sum(probs * np.log(probs)))

    def feat_burstiness_vmr(self,
                            K=conf_segments,
                            min_freq=conf_min_freq) -> float:
        """Burstiness (VMR) -- average over `K` segments"""

        segments = self._split_segments(K)
        
        if not segments:
            return 0.0
        # end if

        word_counts_per_seg = [Counter(seg) for seg in segments]
        global_counts = Counter(self.tokens)

        burst_values = []

        for w, total in global_counts.items():
            if total < min_freq:
                continue
            # end if

            freqs = np.array([c[w] for c in word_counts_per_seg])
            mu = np.mean(freqs)
            
            if mu == 0:
                continue
            # end if

            var = np.var(freqs)
            burst_values.append(var / mu)
        # end for

        return float(np.mean(burst_values)) if burst_values else 0.0

    def feat_burstiness_interarrival(self, min_freq=conf_min_freq) -> float:
        """Burstiness (inter-arrival) -- average over `K` segments"""
        
        positions: Dict[str, List[int]] = {}

        for i, w in enumerate(self.tokens):
            if w not in positions:
                positions[w] = []
            # end if

            positions[w].append(i)
        # end for

        burst_values = []

        for w, pos in positions.items():
            if len(pos) < min_freq:
                continue
            # end if

            gaps = np.diff(pos)

            if len(gaps) == 0:
                continue
            # end if

            mu = np.mean(gaps)
            sigma = np.std(gaps)

            if sigma + mu == 0:
                continue
            # end if

            B = (sigma - mu) / (sigma + mu)
            burst_values.append(B)
        # end for

        return float(np.mean(burst_values)) if burst_values else 0.0

    def feat_zipf_deviation(self) -> float:
        """Zipf deviation"""
        
        if self.N == 0:
            return 0.0
        # end if

        counts = Counter(self.tokens)
        freqs = np.array(sorted(counts.values(), reverse=True))
        ranks = np.arange(1, len(freqs) + 1)

        log_r = np.log(ranks)
        log_f = np.log(freqs)

        a, b = np.polyfit(log_r, log_f, 1)
        pred = a * log_r + b

        mse = np.mean((log_f - pred) ** 2)
        return float(mse)

    def feat_sent_len_cv(self) -> float:
        """Coefficient of variation -- sentences"""
        
        if not self.s_lengths:
            return 0.0
        # end if

        mu = np.mean(self.s_lengths)
        sigma = np.std(self.s_lengths)
        return float(sigma / mu) if mu != 0 else 0.0

    def feat_local_entropy_var(self, window_size=conf_window_size) -> float:
        """Local entropy over a window size -- variance"""
        
        if self.N < window_size:
            return 0.0
        # end if

        entropies = []

        for i in range(0, self.N - window_size + 1, window_size):
            window = self.tokens[i:i + window_size]
            counts = Counter(window)
            probs = np.array(list(counts.values())) / len(window)
            H = -np.sum(probs * np.log(probs))
            entropies.append(H)

        return float(np.var(entropies)) if entropies else 0.0

    def feat_js_divergence(self, K=conf_segments) -> float:
        """Jensen–Shannon divergence over segments of `K` tokens"""
        
        segments = self._split_segments(K)
        
        if len(segments) < 2:
            return 0.0
        # end if

        vocab = list(set(self.tokens))
        vocab_index = {w: i for i, w in enumerate(vocab)}

        def dist(seg):
            vec = np.zeros(len(vocab))
            c = Counter(seg)

            for w, cnt in c.items():
                vec[vocab_index[w]] = cnt
            # end for

            if vec.sum() == 0:
                return vec
            # end if

            return vec / vec.sum()
        # end def

        dists = [dist(seg) for seg in segments]

        def kl(p, q):
            mask = (p > 0) & (q > 0)
            return np.sum(p[mask] * np.log(p[mask] / q[mask]))
        # end def

        js_values = []

        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                p, q = dists[i], dists[j]
                m = 0.5 * (p + q)
                js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
                js_values.append(js)
            # end for
        # end for

        return float(np.mean(js_values)) if js_values else 0.0

    def feat_type_dispersion(self, K: int = conf_segments,
                             min_freq: int = conf_min_freq) -> float:
        """Token Type dispersion"""

        segments = self._split_segments(K)

        if not segments:
            return 0.0

        global_counts = Counter(self.tokens)
        seg_sets = [set(seg) for seg in segments]

        dispersions = []

        for w, total in global_counts.items():
            if total < min_freq:
                continue
            # end if

            count = sum(1 for s in seg_sets if w in s)
            d = count / len(segments)
            dispersions.append(d)
        # end for

        return float(np.mean(dispersions)) if dispersions else 0.0


class RepeatRedundFeatures(TextFeatures):
    def __init__(self, sentences: SentenceList):
        super().__init__(sentences=sentences)

    def _ngrams(self, n: int):
        return [tuple(self.tokens[i:i + n]) for i in range(self.N - n + 1)]

    def _ngram_counts(self, n: int):
        return Counter(self._ngrams(n))

    def feat_ngram_rep_rate(self, max_n: int = conf_max_ngram) -> Dict[str, float]:
        """N-gram repetition rate (coverage)"""
        feats = {}

        for n in range(1, max_n + 1):
            G_n = self._ngrams(n)

            if not G_n:
                feats[f"rep_{n}"] = 0.0
                continue
            # end if

            U_n = set(G_n)
            feats[f"rep_{n}"] = 1.0 - len(U_n) / len(G_n)
        # end for

        return feats

    def feat_ngram_gini(self, max_n: int = conf_max_ngram) -> Dict[str, float]:
        """N-gram concentration (Gini-style)"""

        feats = {}
        
        for n in range(1, max_n + 1):
            counts = self._ngram_counts(n)
            total = sum(counts.values())

            if total == 0:
                feats[f"gini_{n}"] = 0.0
                continue
            # end if

            gini = sum((f / total) ** 2 for f in counts.values())
            feats[f"gini_{n}"] = gini
        # end for

        return feats

    def feat_ngram_entropy(self, max_n: int = conf_max_ngram) -> Dict[str, float]:
        """N-gram entropy (normalized)"""

        feats = {}

        for n in range(1, max_n + 1):
            counts = self._ngram_counts(n)
            total = sum(counts.values())

            if total == 0 or len(counts) <= 1:
                feats[f"entropy_{n}"] = 0.0
                continue
            # end if

            probs = [f / total for f in counts.values()]
            H = -sum(p * math.log(p) for p in probs)
            H_max = math.log(len(counts))

            feats[f"entropy_{n}"] = H / H_max if H_max > 0 else 0.0
        # end for
        return feats

    def feat_self_bleu(self, K: int = conf_segments) -> Dict[str, float]:
        """Self-BLEU"""
        
        if self.N < K:
            return {"self_bleu": 0.0}
        # end if

        segments = self._split_segments(K=K)
        smoothie = SmoothingFunction().method1
        scores = []

        for i, hyp in enumerate(segments):
            refs = segments[:i] + segments[i+1:]
            
            if not refs or not hyp:
                continue
            # end if

            score = sentence_bleu(
                refs,
                hyp,
                smoothing_function=smoothie
            )
            scores.append(score)
        # end for

        return {
            "self_bleu": sum(scores) / len(scores) if scores else 0.0
        }

    def feat_self_bleu_pairwise(self, K: int = conf_segments) -> Dict[str, float]:
        if self.N < K:
            return {"self_bleu_pairwise": 0.0}
        # end if

        segments = self._split_segments(K=K)
        smoothie = SmoothingFunction().method1
        scores = []

        for i in range(len(segments)):
            for j in range(len(segments)):
                if i == j:
                    continue
                # end if

                if not segments[i] or not segments[j]:
                    continue
                # end if

                score = sentence_bleu(
                    [segments[j]],
                    segments[i],
                    smoothing_function=smoothie
                )
                scores.append(score)
            # end for
        # end for

        return {
            "self_bleu_pairwise": sum(scores) / len(scores) if scores else 0.0
        }

    def feat_repeat_freq(self, max_n: int = conf_max_ngram) -> Dict[str, float]:
        """Frequency of repeated n-grams"""

        feats = {}

        for n in range(1, max_n + 1):
            counts = self._ngram_counts(n)
            total = sum(counts.values())
            
            if total == 0:
                feats[f"repeat_freq_{n}"] = 0.0
                continue
            # end if

            repeated = sum(1 for f in counts.values() if f > 1)
            feats[f"repeat_freq_{n}"] = repeated / total
        # end for

        return feats

    def feat_lrs(self) -> Dict[str, float]:
        """Longest Repeated Substring (token-based)"""
        
        n = self.N
        
        if n == 0:
            return {"lrs_len": 0, "lrs_norm": 0.0}
        # end if

        dp = [[0] * (n + 1) for _ in range(n + 1)]
        lrs = 0

        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if self.tokens[i - 1] == self.tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    lrs = max(lrs, dp[i][j])
                else:
                    dp[i][j] = 0
                # end if
            # end for
        # end for

        return {
            "lrs_len": lrs,
            "lrs_norm": lrs / n
        }


class SyntacticFeatures(TextFeatures):
    def __init__(self, sentences: SentenceList):
        super().__init__(sentences=sentences)

    def _iter_tokens(self):
        """Yield (sentence_index, token_index, token_dict)"""

        for si, sent in enumerate(self.sentences):
            for ti, tok in enumerate(sent):
                if isinstance(tok["id"], int):
                    yield si, ti, tok
                # end if
            # end for
        # end for

    def _sentence_tokens(self, sent):
        return [tok for tok in sent if isinstance(tok["id"], int)]

    def feat_pos_distribution(self) -> Dict[str, float]:
        """POS distribution"""
        
        cnt = Counter()

        for _, _, tok in self._iter_tokens():
            cnt[tok["upos"]] += 1
        # end for

        total = sum(cnt.values())
        return {f"pos_{k.lower()}": self._safe_div(v, total) for k, v in cnt.items()}

    def feat_content_function_ratio(self) -> float:
        """Content function ratio"""

        c = f = 0

        for _, _, tok in self._iter_tokens():
            if tok["upos"] in content_poses:
                c += 1
            elif tok["upos"] in function_poses:
                f += 1
            # end if
        # end for

        return self._safe_div(f, c)

    def feat_pos_bigram_entropy(self) -> float:
        """POS n-gram entropy(bigrams)"""
        
        bigrams = Counter()
        total = 0

        for sent in self.sentences:
            toks = self._sentence_tokens(sent)
            tags = [t["upos"] for t in toks]

            for i in range(len(tags) - 1):
                bigrams[(tags[i], tags[i + 1])] += 1
                total += 1
            # end for
        # end for

        if total == 0:
            return 0.0
        # end if

        H = 0.0

        for v in bigrams.values():
            p = v / total
            H -= p * math.log(p + 1e-12)
        # end for

        return H

    def _tree_depth(self, tid: int,
              head_map: Dict[int, int], max_descent: int = 30) -> int:
        d = 0

        while tid != 0 and tid in head_map:
            tid = head_map[tid]
            d += 1

            if d > max_descent:
                # Do not descend more
                # than max_descent nodes
                break
            # end if
        # end while

        return d

    def _depths_sentence(self, sent) -> List[int]:
        toks = self._sentence_tokens(sent)
        head_map = {t["id"]: t["head"] for t in toks}

        return [self._tree_depth(t["id"], head_map=head_map) for t in toks]

    def feat_tree_depth_stats(self) -> Dict[str, float]:
        """Parse tree depth"""

        depths = []

        for sent in self.sentences:
            d = self._depths_sentence(sent)
            
            if d:
                depths.append(max(d))
            # end if
        # end for

        if not depths:
            return {"tree_depth_mean": 0.0, "tree_depth_var": 0.0}
        # end if

        depths = np.array(depths, dtype=float)

        return {
            "tree_depth_mean": float(np.mean(depths)),
            "tree_depth_var": float(np.var(depths))
        }

    def feat_dependency_distance(self) -> Dict[str, float]:
        """Dependency distance"""
        
        dists = []

        for _, _, tok in self._iter_tokens():
            if tok["head"] is not None:
                dists.append(abs(tok["id"] - tok["head"]))
            # end if
        # end for

        if not dists:
            return {"dep_dist_mean": 0.0, "dep_dist_var": 0.0}
        # end if

        dists = np.array(dists, dtype=float)

        return {
            "dep_dist_mean": float(np.mean(dists)),
            "dep_dist_var": float(np.var(dists))
        }

    def feat_clause_density(self) -> float:
        """Clause density"""
        
        densities = []

        for sent in self.sentences:
            toks = self._sentence_tokens(sent)
            
            if not toks:
                continue
            # end if

            clauses = sum(1 for t in toks if t["deprel"] in clause_dep_labels)
            densities.append(clauses / len(toks))
        # end for

        return sum(densities) / len(densities) if densities else 0.0

    def feat_dep_label_distribution(self) -> Dict[str, float]:
        """Dependency label distribution"""

        cnt = Counter()

        for _, _, tok in self._iter_tokens():
            cnt[tok["deprel"]] += 1
        # end for

        total = sum(cnt.values())
        return {f"dep_{k}": self._safe_div(v, total) for k, v in cnt.items()}

    def feat_branching_factor(self) -> Dict[str, float]:
        """Branching factor"""

        child_counts: Dict[int, int] = defaultdict(int)

        for _, _, tok in self._iter_tokens():
            head = tok["head"]

            if head is not None:
                child_counts[head] += 1
            # end if
        # end for

        vals = list(child_counts.values())

        if not vals:
            return {"branch_mean": 0.0, "branch_var": 0.0}
        # end if

        vals = np.array(vals, dtype=float)

        return {
            "branch_mean": float(np.mean(vals)),
            "branch_var": float(np.var(vals))
        }

    def feat_dependency_direction(self) -> Dict[str, float]:
        """Left/right dependency ratio"""
        
        left = right = 0

        for _, _, tok in self._iter_tokens():
            if tok["head"] is None:
                continue
            # end if

            if tok["id"] < tok["head"]:
                left += 1
            elif tok["id"] > tok["head"]:
                right += 1
            # end if
        # end for

        total = left + right

        return {
            "dep_left_ratio": self._safe_div(left, total),
            "dep_right_ratio": self._safe_div(right, total),
        }

    def feat_root_pos(self) -> Dict[str, float]:
        """Root POS distribution"""

        cnt = Counter()

        for sent in self.sentences:
            for tok in sent:
                if isinstance(tok["id"], int) and tok["head"] == 0:
                    cnt[tok["upos"]] += 1
                # end if
            # end for
        # end for

        total = sum(cnt.values())
        return {f"root_{k.lower()}": self._safe_div(v, total) for k, v in cnt.items()}

    def feat_sentence_variance(self) -> Dict[str, float]:
        """Sentence-level variance of clause dependency relations"""
        
        depths = []
        clauses = []

        clause_labels = {"ccomp", "xcomp", "advcl", "acl", "relcl"}

        for sent in self.sentences:
            toks = self._sentence_tokens(sent)
            
            if not toks:
                continue
            # end if

            d = self._depths_sentence(sent)
            depths.append(max(d) if d else 0)

            c = sum(1 for t in toks if t["deprel"] in clause_labels)
            clauses.append(c)
        # end for

        return {
            "sent_depth_var": float(np.var(depths)),
            "sent_clause_var": float(np.var(clauses))
        }

    def feat_pos_transition_entropy(self) -> float:
        """POS transition entropy, POS bigrams P(t_i | t_i-1)"""
        
        transitions = Counter()
        total = 0

        for sent in self.sentences:
            toks = self._sentence_tokens(sent)
            tags = [t["upos"] for t in toks]

            for i in range(len(tags) - 1):
                transitions[(tags[i], tags[i + 1])] += 1
                total += 1
            # end for

        if total == 0:
            return 0.0
        # end if

        H = 0.0

        for v in transitions.values():
            p = v / total
            H -= p * math.log(p + 1e-12)
        # end for

        return H

    def feat_coord_subord_ratio(self) -> float:
        """Coordination vs subordination"""
        
        coord = subord = 0

        for _, _, tok in self._iter_tokens():
            if tok["deprel"] == "conj":
                coord += 1
            elif tok["deprel"] in clause_dep_labels:
                subord += 1
            # end if
        # end for

        return self._safe_div(coord, subord + 1e-6)

    def feat_passive_ratio(self) -> float:
        passive_clauses = 0
        total_clauses = 0

        for sent in self.sentences:
            toks = self._sentence_tokens(sent)

            # find clause heads (root + clausal deps)
            for tok in toks:
                if tok["upos"] not in clause_heads_uposes:
                    continue
                # end if

                total_clauses += 1

                # check if this clause has passive markers
                children = [t for t in toks if t["head"] == tok["id"]]

                is_passive = any(
                    child["deprel"] in {"aux:pass", "nsubj:pass"}
                    for child in children
                )

                if is_passive:
                    passive_clauses += 1
                # end if
            # end for
        # end for

        return self._safe_div(passive_clauses, total_clauses)

    def feat_dep_distance_entropy(self) -> float:
        """Dependency distance entropy"""
        
        cnt = Counter()

        for _, _, tok in self._iter_tokens():
            d = abs(tok["id"] - tok["head"])
            cnt[d] += 1
        # end for

        total = sum(cnt.values())

        if total == 0:
            return 0.0
        # end if

        H = 0.0

        for v in cnt.values():
            p = v / total
            H -= p * math.log(p + 1e-12)
        # end for

        return H

    def feat_segment_syntax_variation(self) -> Dict[str, float]:
        segments = self._split_segments()

        if len(segments) < 2:
            return {
                "seg_dep_dist_var": 0.0,
                "seg_clause_density_var": 0.0,
                "seg_pos_entropy_var": 0.0,
                "seg_tree_depth_var": 0.0,
                "seg_dep_label_entropy_var": 0.0,
            }
        # end if

        dep_means = []
        clause_densities = []
        pos_entropies = []
        tree_depths = []
        dep_label_entropies = []
        flat_tokens = []
        
        for sent in self.sentences:
            for tok in sent:
                if isinstance(tok["id"], int):
                    flat_tokens.append(tok)
                # end if
            # end for
        # end for

        idx = 0

        for seg in segments:
            seg_len = len(seg)
            seg_toks = flat_tokens[idx: idx + seg_len]
            idx += seg_len

            if not seg_toks:
                continue
            # end if

            # --- Dependency distance mean
            dists = [
                abs(tok["id"] - tok["head"])
                for tok in seg_toks
                if tok["head"] is not None
            ]
            dep_means.append(np.mean(dists) if dists else 0.0)

            # --- Clause density
            clauses = sum(1 for tok in seg_toks if tok["deprel"] in clause_dep_labels)
            clause_densities.append(clauses / len(seg_toks))

            # --- POS entropy
            pos_cnt = Counter(tok["upos"] for tok in seg_toks)
            total = sum(pos_cnt.values())

            H_pos = 0.0

            for v in pos_cnt.values():
                p = v / total
                H_pos -= p * math.log(p + 1e-12)
            # end for

            pos_entropies.append(H_pos)

            # --- Tree depth (max per segment)
            # reuse head map locally
            head_map = {tok["id"]: tok["head"] for tok in seg_toks}
            depths = [self._tree_depth(tok["id"], head_map=head_map) for tok in seg_toks]
            tree_depths.append(max(depths) if depths else 0)

            # --- Dependency label entropy
            dep_cnt = Counter(tok["deprel"] for tok in seg_toks)
            total_dep = sum(dep_cnt.values())

            H_dep = 0.0

            for v in dep_cnt.values():
                p = v / total_dep
                H_dep -= p * math.log(p + 1e-12)
            # end for

            dep_label_entropies.append(H_dep)
        # end for segments

        return {
            "seg_dep_dist_var": float(np.var(dep_means)),
            "seg_clause_density_var": float(np.var(clause_densities)),
            "seg_pos_entropy_var": float(np.var(pos_entropies)),
            "seg_tree_depth_var": float(np.var(tree_depths)),
            "seg_dep_label_entropy_var": float(np.var(dep_label_entropies)),
        }


class StylometricFeatures(TextFeatures):
    NOUN_POS = {"NOUN", "PROPN"}
    VERB_POS = {"VERB", "AUX"}
    ROMANIAN_LETTERS = "a-zA-ZăâîșțĂÂÎȘȚşţŞŢ"
    ROMANIAN_VOWELS = "aeiouăâîAEIOUĂÂÎ"
    PUNCTUATION = set(string.punctuation) | {"“", "‘", "’", "—", "–", "…", "„", "”"}

    def __init__(self, sentences: SentenceList):
        super().__init__(sentences=sentences)

        self.function_words, self.stop_words = read_func_stop_words()
        self.pos_tags = self._extract_pos_tags()
        self.text = self._reconstruct_text()
        self.chars = list(self.text)
        self.sentence_lengths = self._sentence_lengths()

    # ------------------------------------------------------------------
    # Basic extraction helpers
    # ------------------------------------------------------------------

    def _real_tokens(self):
        for sent in self.sentences:
            for tok in sent:
                if isinstance(tok["id"], int):
                    yield tok
                # end if
            # end for
        # end for

    def _extract_pos_tags(self) -> List[str]:
        return [
            tok["upos"]
            for tok in self._real_tokens()
            if tok.get("upos") is not None
        ]

    def _reconstruct_text(self) -> str:
        parts = []

        for sent in self.sentences:
            for tok in sent:
                if not isinstance(tok["id"], int):
                    continue
                # end if

                form = tok["form"]
                parts.append(form)

                misc = tok.get("misc") or {}

                if misc.get("SpaceAfter") != "No":
                    parts.append(" ")
                # end if
            # end for
        # end for

        return "".join(parts).strip()

    def _sentence_lengths(self) -> List[int]:
        lengths = []

        for sent in self.sentences:
            n = sum(1 for tok in sent if isinstance(tok["id"], int))

            if n > 0:
                lengths.append(n)
            # end if
        # end for

        return lengths

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    def _entropy(self, items: List) -> float:
        if not items:
            return 0.0

        counts = Counter(items)
        n = len(items)

        return -sum(
            (c / n) * math.log2(c / n)
            for c in counts.values()
            if c > 0
        )

    def _ngrams(self, items: List, n: int) -> List[Tuple]:
        if len(items) < n:
            return []
        # end if

        return [
            tuple(items[i:i + n])
            for i in range(len(items) - n + 1)
        ]

    # ------------------------------------------------------------------
    # Lexical stylometry
    # ------------------------------------------------------------------

    def feat_function_word_entropy(self) -> float:
        fwords = [w for w in self.tokens if w in self.function_words]
        return self._entropy(fwords)

    def feat_stopword_entropy(self) -> float:
        stops = [w for w in self.tokens if w in self.stop_words]
        return self._entropy(stops)

    def feat_yules_k(self) -> float:
        if self.N == 0:
            return 0.0
        # end if

        counts = Counter(self.tokens)
        freq_of_freq = Counter(counts.values())

        m1 = self.N
        m2 = sum((m ** 2) * vm for m, vm in freq_of_freq.items())

        return 10_000.0 * (m2 - m1) / (m1 ** 2)

    def feat_honores_r(self, eps: float = 1e-12) -> float:
        if self.N <= 1:
            return 0.0

        counts = Counter(self.tokens)
        V = len(counts)
        V1 = sum(1 for c in counts.values() if c == 1)

        if V == 0:
            return 0.0
        # end if

        denominator = 1.0 - (V1 / V) + eps

        return 100.0 * math.log(self.N) / denominator

    def feat_simpson_index(self) -> float:
        if self.N < 2:
            return 0.0
        # end if

        counts = Counter(self.tokens)

        numerator = sum(c * (c - 1) for c in counts.values())
        denominator = self.N * (self.N - 1)

        return numerator / denominator

    # ------------------------------------------------------------------
    # Character-level stylometry
    # ------------------------------------------------------------------

    def feat_char_entropy(self) -> float:
        return self._entropy(self.chars)

    def feat_char_3gram_entropy(self) -> float:
        return self._entropy(self._ngrams(self.chars, 3))

    def feat_char_4gram_entropy(self) -> float:
        return self._entropy(self._ngrams(self.chars, 4))

    def feat_char_5gram_entropy(self) -> float:
        return self._entropy(self._ngrams(self.chars, 5))

    def feat_uppercase_ratio(self) -> float:
        alpha = [c for c in self.chars if c.isalpha()]
        upper = [c for c in alpha if c.isupper()]

        return self._safe_div(len(upper), len(alpha))

    def feat_digit_ratio(self) -> float:
        return self._safe_div(
            sum(1 for c in self.chars if c.isdigit()),
            len(self.chars)
        )

    def feat_special_char_ratio(self) -> float:
        romanian_alpha = re.compile(rf"[{self.ROMANIAN_LETTERS}]", re.UNICODE)

        special = [
            c for c in self.chars
            if not c.isdigit()
            and not c.isspace()
            and c not in self.PUNCTUATION
            and not romanian_alpha.fullmatch(c)
        ]

        return self._safe_div(len(special), len(self.chars))

    # ------------------------------------------------------------------
    # Punctuation / orthographic stylometry
    # ------------------------------------------------------------------

    def feat_punctuation_char_ratio(self) -> float:
        return self._safe_div(
            sum(1 for c in self.chars if c in self.PUNCTUATION),
            len(self.chars)
        )

    def feat_punctuation_entropy(self) -> float:
        punct = [c for c in self.chars if c in self.PUNCTUATION]
        return self._entropy(punct)

    def feat_contraction_ratio(self) -> float:
        """
        Romanian does not use English-style contractions productively.
        This feature instead captures hyphenated clitic/auxiliary forms:
        m-am, te-ai, s-a, l-am, într-o, dintr-un, c-a, n-am, etc.
        """
        clitic_pattern = re.compile(
            rf"^[{self.ROMANIAN_LETTERS}]+[-’'][{self.ROMANIAN_LETTERS}]+$",
            re.UNICODE
        )

        return self._safe_div(
            sum(1 for w in self.raw_tokens if clitic_pattern.match(w)),
            self.N
        )
    
    def feat_hyphenated_word_ratio(self) -> float:
        pattern = re.compile(
            rf"^[{self.ROMANIAN_LETTERS}]+(?:-[{self.ROMANIAN_LETTERS}]+)+$",
            re.UNICODE
        )

        return self._safe_div(
            sum(1 for w in self.raw_tokens if pattern.match(w)),
            self.N
        )

    # ------------------------------------------------------------------
    # POS-based stylometry
    # ------------------------------------------------------------------

    def feat_pos_entropy(self) -> float:
        return self._entropy(self.pos_tags)

    def feat_pos_trigram_entropy(self) -> float:
        return self._entropy(self._ngrams(self.pos_tags, 3))

    def feat_noun_ratio(self) -> float:
        return self._safe_div(
            sum(1 for p in self.pos_tags if p in self.NOUN_POS),
            len(self.pos_tags)
        )

    def feat_verb_ratio(self) -> float:
        return self._safe_div(
            sum(1 for p in self.pos_tags if p in self.VERB_POS),
            len(self.pos_tags)
        )

    def feat_adjective_ratio(self) -> float:
        return self._safe_div(
            sum(1 for p in self.pos_tags if p == "ADJ"),
            len(self.pos_tags)
        )

    def feat_adverb_ratio(self) -> float:
        return self._safe_div(
            sum(1 for p in self.pos_tags if p == "ADV"),
            len(self.pos_tags)
        )

    def feat_pronoun_ratio(self) -> float:
        return self._safe_div(
            sum(1 for p in self.pos_tags if p == "PRON"),
            len(self.pos_tags)
        )

    # ------------------------------------------------------------------
    # Dependency stylometry
    # ------------------------------------------------------------------

    def _dependency_depths_and_children(self) -> Tuple[List[int], List[int]]:
        all_depths = []
        all_children_counts = []

        for sent in self.sentences:
            tokens = [tok for tok in sent if isinstance(tok["id"], int)]
            ids = {tok["id"] for tok in tokens}

            heads = {
                tok["id"]: tok.get("head")
                for tok in tokens
            }

            children = defaultdict(list)

            for tok in tokens:
                tid = tok["id"]
                head = tok.get("head")

                if isinstance(head, int) and head in ids:
                    children[head].append(tid)

            def depth(tid: int) -> int:
                seen = set()
                d = 0
                cur = tid

                while True:
                    if cur in seen:
                        return d

                    seen.add(cur)
                    head = heads.get(cur)

                    if not isinstance(head, int) or \
                            head == 0 or head not in ids:
                        return d
                    # end if

                    d += 1
                    cur = head
                # end while
            # end def

            for tok in tokens:
                tid = tok["id"]
                all_depths.append(depth(tid))
                all_children_counts.append(len(children.get(tid, [])))
            # end for
        # end for all sentences

        return all_depths, all_children_counts

    def feat_avg_dependency_depth(self) -> float:
        depths, _ = self._dependency_depths_and_children()
        return self._safe_div(sum(depths), len(depths))

    def feat_max_dependency_depth(self) -> float:
        depths, _ = self._dependency_depths_and_children()
        return float(max(depths)) if depths else 0.0

    def feat_dependency_depth_variance(self) -> float:
        depths, _ = self._dependency_depths_and_children()

        if not depths:
            return 0.0

        mean = sum(depths) / len(depths)

        return sum((d - mean) ** 2 for d in depths) / len(depths)

    def feat_avg_dependents_per_token(self) -> float:
        _, child_counts = self._dependency_depths_and_children()
        return self._safe_div(sum(child_counts), len(child_counts))

    def feat_leaf_token_ratio(self) -> float:
        _, child_counts = self._dependency_depths_and_children()
        return self._safe_div(
            sum(1 for c in child_counts if c == 0),
            len(child_counts)
        )

    # ------------------------------------------------------------------
    # Sentence-level stylometry
    # ------------------------------------------------------------------

    def feat_sentence_length_skewness(self) -> float:
        lengths = self.sentence_lengths
        M = len(lengths)

        if M == 0:
            return 0.0

        mean = sum(lengths) / M
        var = sum((x - mean) ** 2 for x in lengths) / M
        std = math.sqrt(var)

        if std == 0:
            return 0.0

        return sum(((x - mean) / std) ** 3 for x in lengths) / M

    def feat_sentence_length_kurtosis(self) -> float:
        lengths = self.sentence_lengths
        M = len(lengths)

        if M == 0:
            return 0.0

        mean = sum(lengths) / M
        var = sum((x - mean) ** 2 for x in lengths) / M
        std = math.sqrt(var)

        if std == 0:
            return 0.0

        kurtosis = sum(((x - mean) / std) ** 4 for x in lengths) / M

        return kurtosis - 3.0

    def feat_short_sentence_ratio(self, threshold: int = 8) -> float:
        return self._safe_div(
            sum(1 for x in self.sentence_lengths if x <= threshold),
            len(self.sentence_lengths)
        )

    def feat_long_sentence_ratio(self, threshold: int = 30) -> float:
        return self._safe_div(
            sum(1 for x in self.sentence_lengths if x >= threshold),
            len(self.sentence_lengths)
        )

    def feat_sentence_initial_function_word_ratio(self) -> float:
        first_tokens = []

        for sent in self.sentences:
            real = [tok for tok in sent if isinstance(tok["id"], int)]
            if real:
                first_tokens.append(real[0]["form"].lower())

        return self._safe_div(
            sum(1 for w in first_tokens if w in self.function_words),
            len(first_tokens)
        )

    def feat_repeated_sentence_opening_token_ratio(self, k: int = 3) -> float:
        openings = []

        for sent in self.sentences:
            real = [
                tok["form"].lower()
                for tok in sent
                if isinstance(tok["id"], int)
            ]

            if real:
                openings.append(tuple(real[:k]))

        return 1.0 - self._safe_div(len(set(openings)), len(openings)) if openings else 0.0

    def feat_repeated_sentence_opening_pos_ratio(self, k: int = 3) -> float:
        openings = []

        for sent in self.sentences:
            real = [
                tok.get("upos")
                for tok in sent
                if isinstance(tok["id"], int) and tok.get("upos") is not None
            ]

            if real:
                openings.append(tuple(real[:k]))

        return 1.0 - self._safe_div(len(set(openings)), len(openings)) if openings else 0.0

    # ------------------------------------------------------------------
    # Readability helpers
    # ------------------------------------------------------------------

    def _count_syllables(self, word: str) -> int:
        """
        Approximate Romanian syllable counter.

        Core heuristic:
        Romanian syllables are roughly centered around vowel groups.
        Diphthongs/triphthongs make exact counting difficult, so this returns
        a robust approximation useful as a stylometric feature.
        """
        word = word.lower()
        word = re.sub(
            rf"[^{self.ROMANIAN_LETTERS}]",
            "",
            word,
            flags=re.UNICODE
        )

        if not word:
            return 0
        # end if

        vowels = set(StylometricFeatures.ROMANIAN_VOWELS)
        count = 0
        prev_is_vowel = False

        for ch in word:
            is_vowel = ch in vowels

            if is_vowel and not prev_is_vowel:
                count += 1
            # end if

            prev_is_vowel = is_vowel
        # end for

        # Romanian final -i can be non-syllabic in many forms:
        # pomi, buni, mulți, etc.
        # This is only a heuristic; avoid subtracting for very short words.
        if len(word) > 3 and word.endswith("i") and count > 1:
            if not word.endswith(("ai", "ei", "oi", "ui", "ăi", "âi")):
                count -= 1
            # end if
        # end if

        return max(count, 1)

    def _readability_counts(self) -> Tuple[int, int, int, int, int]:
        word_pattern = re.compile(rf"[{self.ROMANIAN_LETTERS}]", re.UNICODE)

        words = [
            w for w in self.raw_tokens
            if word_pattern.search(w)
        ]

        n_words = len(words)
        n_sentences = len(self.sentence_lengths)

        n_chars = sum(
            len(re.findall(rf"[{self.ROMANIAN_LETTERS}0-9]", w, re.UNICODE))
            for w in words
        )

        syllables = [self._count_syllables(w) for w in words]
        n_syllables = sum(syllables)

        n_complex_words = sum(1 for s in syllables if s >= 3)

        return n_words, n_sentences, n_chars, n_syllables, n_complex_words
    
    # ------------------------------------------------------------------
    # Readability features
    # ------------------------------------------------------------------

    def feat_flesch_reading_ease(self) -> float:
        nw, ns, _, nsy, _ = self._readability_counts()

        if nw == 0 or ns == 0:
            return 0.0

        return (
            206.835
            - 1.015 * (nw / ns)
            - 84.6 * (nsy / nw)
        )

    def feat_flesch_kincaid_grade(self) -> float:
        nw, ns, _, nsy, _ = self._readability_counts()

        if nw == 0 or ns == 0:
            return 0.0

        return (
            0.39 * (nw / ns)
            + 11.8 * (nsy / nw)
            - 15.59
        )

    def feat_gunning_fog_index(self) -> float:
        nw, ns, _, _, ncw = self._readability_counts()

        if nw == 0 or ns == 0:
            return 0.0

        return 0.4 * ((nw / ns) + 100.0 * (ncw / nw))

    def feat_automated_readability_index(self) -> float:
        nw, ns, nc, _, _ = self._readability_counts()

        if nw == 0 or ns == 0:
            return 0.0

        return (
            4.71 * (nc / nw)
            + 0.5 * (nw / ns)
            - 21.43
        )

    def feat_avg_syllables_per_word(self) -> float:
        nw, _, _, nsy, _ = self._readability_counts()
        return self._safe_div(nsy, nw)

    def feat_complex_word_ratio(self) -> float:
        nw, _, _, _, ncw = self._readability_counts()
        return self._safe_div(ncw, nw)
