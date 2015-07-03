from pickle import load
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes, tree, metrics
from functools import partial
import numpy as np
import logging

scoring_metrics = [
    ("f1", metrics.f1_score),
    ("recall", metrics.recall_score),
]

def get_entry_dict(entry, features_to_remove):
    res = entry._get_dict()
    for feature in features_to_remove:
        del res[feature]
    return res

feature_extractors = [
    ("discriminatory features", partial(get_entry_dict, features_to_remove=["wage"])),
    ("ethical features", partial(get_entry_dict, features_to_remove="wage sex ancestry state".split()))
]

estimators = [
    ("MultinomialNB", naive_bayes.MultinomialNB()),
    ("Decision tree", tree.DecisionTreeClassifier(max_depth=7, class_weight="auto"))
]

PICKLE_FILE = "pus.pickle"

def scoring_function(estimator, X, y):
    predicted = estimator.predict(X)
    return {metric_name: metric(y, predicted) \
        for metric_name, metric in scoring_metrics}

logging.basicConfig(level="INFO")
logging.info("unpickling")
with open(PICKLE_FILE, "rb") as i:
    entries = load(i)
logging.info("extracting wages")
wages = [entry.wage for entry in entries]
logging.info("calculating 95th percentile of wages")
threshold = np.percentile(wages, 95)
del wages
logging.info("calculating labels")
labels = [int(entry.wage >= threshold) for entry in entries]
# del entries
for features_name, extractor in feature_extractors:
    # logging.info("unpicking")
    # with open(PICKLE_FILE, "rb") as i:
    #     entries = load(i)
    logging.info("vectorizing {}".format(features_name))
    features = DictVectorizer().fit_transform(map(extractor, entries))
    logging.info("vectorizing done")
    # del entries
    for estimator_name, estimator in estimators:
        logging.info("scoring {} {}".format(features_name, estimator_name))
        scores = cross_val_score(
            estimator,
            features,
            labels,
            scoring = scoring_function,
            cv=StratifiedKFold(labels, n_folds=10, shuffle=True)
        )
        print("scores for", features_name, estimator_name)
        print("\n".join(map(str, scores)))
