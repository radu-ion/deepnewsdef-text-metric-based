from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import math
from typing import List, Dict
from collections import Counter
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


class TextFeatures:
    def __init__(self, sentences: SentenceList):
        """Input text is already tokenized by a BaseProcessor text processor.
        The BaseProcessor returns the `sentences`."""
        self.sentences = sentences
        self.tokens = self._extract_tokens()
        self.N = len(self.tokens)

    def _extract_tokens(self) -> List[str]:
        tokens = []

        for sent in self.sentences:
            for tok in sent:
                if isinstance(tok["id"], int):
                    # Ignore multiword tokens
                    tokens.append(tok["form"].lower())
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

    def feat_word_len_variance(self) -> float:
        """Variance of word length"""

        return float(np.var(self.w_lengths)) if self.w_lengths else 0.0

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

    def feat_word_len_cv(self) -> float:
        """Coefficient of variation -- words"""

        if not self.w_lengths:
            return 0.0
        # end if

        mu = np.mean(self.w_lengths)
        sigma = np.std(self.w_lengths)

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

    def feat_type_dispersion(self, K=10, min_freq=3) -> float:
        """Type dispersion"""

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

    def feat_ngram_rep_rate(self, max_n: int = conf_max_ngram):
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

    def feat_ngram_gini(self, max_n: int = conf_max_ngram):
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

    def feat_ngram_entropy(self, max_n: int = conf_max_ngram):
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

    def feat_self_bleu(self, K: int = conf_segments):
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

    def feat_self_bleu_pairwise(self, K: int = conf_segments):
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

    def feat_repeat_freq(self, max_n: int = conf_max_ngram):
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

    def feat_lrs(self):
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
