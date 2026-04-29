from typing import List, Dict
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer


class FeatureNormalizer:
    def __init__(self, method: str = "minmax"):
        """
        Dataset-level feature normalizer.

        Parameters
        ----------
        method : str
            - "minmax"        -> [0, 1]
            - "minmax_sym"    -> [-1, 1]
            - "zscore"        -> mean=0, std=1
            - "robust"        -> median/IQR
            - "l2"            -> per-sample L2 normalization
            - "log_minmax"    -> log(1+x) then minmax
            - "log_zscore"    -> log(1+x) then zscore
            - "none"
        """
        self.method = method
        self.scaler = None
        self._feature_names: List[str] = []
        self._use_log = method.startswith("log_")

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def _strip_method(self):
        if self._use_log:
            return self.method.replace("log_", "")
        # end if

        return self.method

    def _build_matrix(self, samples: List[Dict[str, float]]) -> np.ndarray:
        """
        Convert list of dicts → matrix [n_samples, n_features]
        Missing features are filled with 0.
        """

        # Union of all keys
        if not self._feature_names:
            # Init self._feature_names when we first read
            # a data set
            keys = set()

            for s in samples:
                keys.update(s.keys())
            # end for

            self._feature_names = sorted(keys)
        # end if

        X = np.zeros((len(samples), len(self._feature_names)),
                      dtype=np.float32)

        for i, sample in enumerate(samples):
            for j, fname in enumerate(self._feature_names):
                val = sample.get(fname, 0.0)

                if not np.isfinite(val):
                    val = 0.0
                # end if

                X[i, j] = val
            # end for
        # end for

        return X

    def _apply_log(self, X: np.ndarray) -> np.ndarray:
        return np.log1p(X)

    def _get_scaler(self):
        method = self._strip_method()

        if method == "minmax":
            return MinMaxScaler((0, 1))
        elif method == "minmax_sym":
            return MinMaxScaler((-1, 1))
        elif method == "zscore":
            return StandardScaler()
        elif method == "robust":
            return RobustScaler()
        elif method == "l2":
            return Normalizer(norm="l2")
        elif method == "none":
            return None
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        # end if

    def fit(self, samples: List[Dict[str, float]]):
        X = self._build_matrix(samples)

        if self._use_log:
            X = self._apply_log(X)
        # end if

        self.scaler = self._get_scaler()

        if self.scaler is not None:
            self.scaler.fit(X)
        # end if

        return self

    def transform(self, samples: List[Dict[str, float]]) -> List[Dict[str, float]]:
        X = self._build_matrix(samples)

        if self._use_log:
            X = self._apply_log(X)
        # end if

        if self.scaler is not None:
            X = self.scaler.transform(X)
        # end if

        return self._matrix_to_dicts(X)

    def fit_transform(self, samples: List[Dict[str, float]]) -> List[Dict[str, float]]:
        self.fit(samples)
        return self.transform(samples)

    def _matrix_to_dicts(self, X: np.ndarray) -> List[Dict[str, float]]:
        out = []

        for row in X:
            d = {fname: float(val)
                 for fname, val in zip(self._feature_names, row)} # type: ignore
            out.append(d)
        # end for

        return out
