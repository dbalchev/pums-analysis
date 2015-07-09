from pickle import load, dump
from sklearn import naive_bayes, tree, metrics, linear_model, decomposition, svm, ensemble
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from functools import partial
import numpy as np
import logging
import pathlib

scoring_metrics = [
    ("f1", metrics.f1_score),
    ("precision", metrics.precision_score),
    ("recall", metrics.recall_score),
]

def get_entry_dict(entry, features_to_remove):
    res = entry._get_dict()
    for feature in features_to_remove:
        del res[feature]
    return res

feature_extractors = [
    ("discriminatory features", partial(get_entry_dict, features_to_remove=["wage"])),
    ("ethical features", partial(get_entry_dict, features_to_remove="wage sex ancestry state".split())),
    ("ethical +state features", partial(get_entry_dict, features_to_remove="wage sex ancestry".split())),
    ("no sex features", partial(get_entry_dict, features_to_remove="wage sex".split())),
    ("no ancestry features", partial(get_entry_dict, features_to_remove="wage ancestry".split())),
]

estimators = [
    ("MultinomialNB", naive_bayes.MultinomialNB()),
    ("BernoulliNB", naive_bayes.BernoulliNB()),
    ("SGD unweighted", linear_model.SGDClassifier()),
    ("SGD auto-weighted", linear_model.SGDClassifier(class_weight="auto")),
    ("LogisticRegression", linear_model.LogisticRegression(class_weight="auto")),
    # ("bagged BernoulliNB", ensemble.BaggingRegressor(naive_bayes.BernoulliNB())), #MemoryError
    # ("TruncatedSVD 17 + BernoulliNB", make_pipeline(decomposition.TruncatedSVD(17, n_iter=32), naive_bayes.BernoulliNB())), #NaN
    # ("Decision tree", tree.DecisionTreeClassifier(max_depth=7, class_weight="auto")), # too much time
    # ("linear SVM", svm.LinearSVC(class_weight="auto")), # too much estimated time
]

PICKLE_FILE = "pus.pickle"

logging.basicConfig(level="INFO")
if not pathlib.Path(PICKLE_FILE).exists():
    from .reader import read_records_from_file
    from .entries import Entry
    logging.info("creating {}".format(PICKLE_FILE))
    entries = list(read_records_from_file("csv_pus.zip", Entry))
    with open(PICKLE_FILE, "wb") as o:
        dump(entries, o)
else:
    logging.info("unpickling")
    with open(PICKLE_FILE, "rb") as i:
        entries = load(i)
logging.info("{} entries unpickled".format(len(entries)))
logging.info("extracting wages")
wages = [entry.wage for entry in entries]
logging.info("calculating 95th percentile of wages")
threshold = np.percentile(wages, 95)
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
            logging.info("scoring {}".format(len(scores)))
            predicted = estimator.predict(X)
            predicted = [int(entry.wage >= threshold) for entry in predicted]
            target    = [int(entry.wage >= threshold) for entry in y]
            cu_score =  {metric_name: metric(target, predicted) \
                for metric_name, metric in scoring_metrics}
            scores.append(cu_score)
            return np.mean(list(cu_score.values()))
        cross_val_score(
            estimator,
            features,
            wages,
            scoring = f1_recall,
            cv=StratifiedKFold(labels, n_folds=10, shuffle=True)
        )
        scores_mean = \
            {metric_name: np.mean([score[metric_name] for score in scores])
             for metric_name, _ in scoring_metrics}
        score_str = "scores for {} {}\n{}\n" \
            .format(features_name, estimator_name,
                " ".join(map(str, scores_mean.items())))
        print(score_str)
        with open("results.txt", "a") as res_out:
            res_out.write(score_str)
