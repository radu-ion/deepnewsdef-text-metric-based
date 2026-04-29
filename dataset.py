import re
from pathlib import Path
from typing import Dict, List, \
    Tuple, Literal, Union, Type
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from features import TextFeatures
from processing import TextToCoNLLU


TGLabel = Literal["human", "generated"]


class TextClassifierDataset:
    """
    Loads human/generated text pairs from a folder.

    Expected format:
        1.txt
        2.txt
        ...
        1_paraphrase.txt
        2_paraphrase.txt
        ...
    """

    HUMAN_RE = re.compile(r"^(\d+)\.txt$")

    def __init__(self, dataset_dir: Union[str, Path]):
        self.dataset_dir = Path(dataset_dir)

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}")
        # end if

    def load(
        self,
        generation_method: str,
    ) -> Tuple[List[str], List[TGLabel], List[str]]:
        """
        Returns:
            texts: raw text contents
            labels: 'human' or 'generated'
            file_ids: file identifiers, useful for debugging
        """

        texts: List[str] = []
        labels: List[TGLabel] = []
        file_ids: List[str] = []

        text_files = sorted(
            self.dataset_dir.glob("*.txt"),
            key=lambda p: self._numeric_sort_key(p.name),
        )

        for txt_path in text_files:
            match = TextClassifierDataset.HUMAN_RE.match(txt_path.name)
            
            if not match:
                continue
            # end if

            article_id = match.group(1)
            generated_path = self.dataset_dir / \
                f"{article_id}_{generation_method}.txt"

            if not generated_path.exists():
                continue
            # end if

            human_text = txt_path.read_text(encoding="utf-8")
            generated_text = generated_path.read_text(encoding="utf-8")

            texts.append(human_text)
            labels.append("human")
            file_ids.append(article_id)

            texts.append(generated_text)
            labels.append("generated")
            file_ids.append(f"{article_id}_{generation_method}")
        # end for

        if not texts:
            raise ValueError(
                f"No human/generated pairs found for generation method: {generation_method}"
            )
        # end if

        return texts, labels, file_ids

    @staticmethod
    def _numeric_sort_key(filename: str) -> int:
        match = re.match(r"^(\d+)", filename)
        return int(match.group(1)) if match else 10**5


class FeatureExtractor:
    def __init__(self, processor: TextToCoNLLU,
                 feature_classes: List[Type[TextFeatures]]):
        self.processor = processor
        self.feature_classes = feature_classes
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def extract_dicts(self, texts: List[str]) -> List[Dict[str, float]]:
        rows = []

        for text in tqdm(texts, desc="Processing"):
            sentences = self.processor.process(text)
            feats = {}

            for fty in self.feature_classes:
                obj = fty(sentences)
                feats.update(obj.features())
            # end for

            rows.append(feats)
        # end for

        return rows

    def fit_feature_space(self, samples: List[Dict[str, float]]):
        self._feature_names = sorted(
            {k for s in samples for k in s.keys()}
        )

    def set_feature_space(self, f_names: List[str]):
        """When we use our FeatureNormalizer, it has its own feature names vector."""
        self._feature_names = f_names

    def dicts_to_matrix(self, samples: List[Dict[str, float]]) -> np.ndarray:
        X = np.array([
            [s.get(f, 0.0) for f in self._feature_names]
            for s in samples
        ], dtype=np.float32)

        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def make_train_dev_split(texts: List[str], labels: List[TGLabel],
                         file_ids: List[str],
                         dev_size: float = 0.2,
                         random_state: int = 1234):
    return train_test_split(texts, labels, file_ids,
                            test_size=dev_size, random_state=random_state,
                            stratify=labels)
