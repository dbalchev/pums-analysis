from pickle import load
from sklearn import naive_bayes, tree, metrics, linear_model, decomposition, svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
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
    ("no sex features", partial(get_entry_dict, features_to_remove="wage sex".split())),
    ("no ancestry features", partial(get_entry_dict, features_to_remove="wage ancestry".split())),
    ("ethical +state features", partial(get_entry_dict, features_to_remove="wage sex ancestry".split())),
    ("ethical features", partial(get_entry_dict, features_to_remove="wage sex ancestry state".split()))
]

estimators = [
    ("MultinomialNB", naive_bayes.MultinomialNB()),
    ("BernoulliNB", naive_bayes.BernoulliNB()),
    # ("Decision tree", tree.DecisionTreeClassifier(max_depth=7, class_weight="auto")),
    # ("linear SVM", svm.LinearSVC(class_weight="auto")),
    # ("LogisticRegression", linear_model.LogisticRegression(class_weight="auto")),
]

PICKLE_FILE = "pus.pickle"

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
        scores = []
        def f1_recall(estimator, X, y):
            logging.info("scoring")
            predicted = estimator.predict(X)
            cu_score =  {metric_name: metric(y, predicted) \
                for metric_name, metric in scoring_metrics}
            scores.append(cu_score)
            return np.mean(list(cu_score.values()))
        cross_val_score(
            estimator,
            features,
            labels,
            scoring = f1_recall,
            cv=StratifiedKFold(labels, n_folds=10, shuffle=True)
        )
        print("scores for", features_name, estimator_name)
        f1 = np.mean([score["f1"] for score in scores])
        recall = np.mean([score["recall"] for score in scores])
        print("f1 = {}; recall = {}".format(f1, recall))
