import os
from typing import List, Tuple
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from processing import TextToCoNLLU, StanzaProcessor, SpacyProcessor, TrankitProcessor
from features import BasicLexicalFeatures, \
    DistribVariabFeatures, RepeatRedundFeatures, \
    SyntacticFeatures, StylometricFeatures
from dataset import TGLabel, TextClassifierDataset, \
    FeatureExtractor, make_train_dev_split
from normalize import FeatureNormalizer


def get_permutation_importance(
    clf,
    X_dev,
    y_dev,
    feature_names: List[str],
    random_state: int,
    top_k: int = 10,
    n_repeats: int = 10
) -> List[Tuple[str, float]]:
    result = permutation_importance(
        clf,
        X_dev,
        y_dev,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="accuracy"
    )

    pairs = list(zip(feature_names, result.importances_mean)) # type: ignore
    pairs.sort(key=lambda x: x[1], reverse=True)

    return pairs[:top_k]

def train_and_evaluate_decision_tree(X_train: np.ndarray, y_train: List[TGLabel],
                                     X_dev: np.ndarray, y_dev: List[TGLabel],
                                     f_names: List[str], r_seed: int):
    clf = DecisionTreeClassifier(criterion="gini", max_depth=20,
                                 random_state=r_seed,
                                 class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)

    print("Decision Tree")
    print(classification_report(y_dev, y_pred, digits=3))

    f_scores = get_permutation_importance(
        clf, X_dev, y_dev, feature_names=f_names, random_state=r_seed)
    
    print("Discriminative features:")

    for i, (f_name, f_imp) in enumerate(f_scores):
        print(f"{i + 1}. {f_name}: {f_imp:.5f}")
    # end for

    print()

    return clf


def train_and_evaluate_svm(X_train: np.ndarray, y_train: List[TGLabel],
                           X_dev: np.ndarray, y_dev: List[TGLabel],
                           r_seed: int):
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=r_seed
                )
            ),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)

    print("SVM")
    print(classification_report(y_dev, y_pred, digits=3))

    return clf


def run_experiment(dataset_dir, generation_method, processor,
                   feature_classes, normalization_method: str = '',
                   dev_size=0.25, random_state=1234):
    dataset = TextClassifierDataset(dataset_dir)
    texts, labels, file_ids = dataset.load(generation_method)

    (
        texts_train,
        texts_dev,
        y_train,
        y_dev,
        ids_train,
        ids_dev
    ) = make_train_dev_split(texts, labels, file_ids,
                             dev_size=dev_size, random_state=random_state)

    extractor = FeatureExtractor(processor, feature_classes)

    # 1. Extract raw features
    train_dicts = extractor.extract_dicts(texts_train)
    dev_dicts = extractor.extract_dicts(texts_dev)

    if normalization_method:
        # 2. Normalize
        normalizer = FeatureNormalizer(method=normalization_method)

        # 2.1 But fit the scaler only on the train set!
        train_dicts_norm = normalizer.fit_transform(train_dicts)
        dev_dicts_norm = normalizer.transform(dev_dicts)

        # 3 Fix feature space
        extractor.set_feature_space(f_names=normalizer.feature_names)
        
        X_train = extractor.dicts_to_matrix(train_dicts_norm)
        X_dev = extractor.dicts_to_matrix(dev_dicts_norm)
        feature_names = normalizer.feature_names
    else:
        extractor.fit_feature_space(samples=train_dicts)
        
        X_train = extractor.dicts_to_matrix(train_dicts)
        X_dev = extractor.dicts_to_matrix(dev_dicts)
        feature_names = extractor.feature_names
    # end if

    # 4. Train models and evaluate models
    dt = train_and_evaluate_decision_tree(X_train, y_train,
                                          X_dev, y_dev,
                                          f_names=feature_names,
                                          r_seed=random_state)

    svm = train_and_evaluate_svm(X_train, y_train,
                                 X_dev, y_dev,
                                 r_seed=random_state)


if __name__ == '__main__':
    feature_set = [
        BasicLexicalFeatures,
        DistribVariabFeatures,
        RepeatRedundFeatures,
        SyntacticFeatures,
        StylometricFeatures
    ]
    crt_time = time.time()
    rnd_seed = int(1e5 * (crt_time - int(crt_time)))

    run_experiment(dataset_dir=os.path.join('data', 'news_2'),
                   feature_classes=feature_set, generation_method='paraphrase',
                   processor=TextToCoNLLU(processor=StanzaProcessor()),
                   normalization_method='minmax', random_state=rnd_seed)
