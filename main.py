import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Imbalanced-learn modules
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

# Metrics
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Preprocessing
from sklearn.preprocessing import RobustScaler

# Model selection
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score

# Load dataset
dataset = pd.read_csv("creditcard.csv")

print("Fraudulent cases:", len(dataset[dataset['Class'] == 1]))
print("Normal cases:", len(dataset[dataset['Class'] == 0]))

# Feature distribution plots (commented as per request)
# f, axes = plt.subplots(1, 2, figsize=(18,4), sharex=True)
# amount_value = dataset['Amount'].values
# time_value = dataset['Time'].values
# sns.distplot(amount_value, hist=False, color="m", kde_kws={"shade": True}, ax=axes[0]).set_title('Distribution of Amount')
# sns.distplot(time_value, hist=False, color="m", kde_kws={"shade": True}, ax=axes[1]).set_title('Distribution of Time')
# plt.show()

print("Summary of Amount:", dataset["Amount"].describe())

# Standardize Time and Amount
scaler = RobustScaler().fit(dataset[["Time", "Amount"]])
dataset[["Time", "Amount"]] = scaler.transform(dataset[["Time", "Amount"]])

# Separate features and target
y = dataset["Class"]
X = dataset.iloc[:, 0:30]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# Function to train model and compute metrics
def get_model_best_estimator_and_metrics(estimator, params, kf=kf, X_train=X_train,
                                         y_train=y_train, X_test=X_test,
                                         y_test=y_test, is_grid_search=True,
                                         sampling=None, scoring="f1",
                                         n_jobs=2):
    if sampling is None:
        pipeline = make_pipeline(estimator)
    else:
        pipeline = make_pipeline(sampling, estimator)

    estimator_name = estimator.__class__.__name__.lower()
    new_params = {f'{estimator_name}__{key}': params[key] for key in params}

    if is_grid_search:
        search = GridSearchCV(pipeline, param_grid=new_params, cv=kf, return_train_score=True, n_jobs=n_jobs, verbose=2)
    else:
        search = RandomizedSearchCV(pipeline, param_distributions=new_params,
                                    cv=kf, scoring=scoring, return_train_score=True,
                                    n_jobs=n_jobs, verbose=1)

    # Fit model
    search.fit(X_train, y_train)

    # Cross-validation score
    cv_score = cross_val_score(search, X_train, y_train, scoring=scoring, cv=kf)

    # Predictions
    y_pred = search.best_estimator_.named_steps[estimator_name].predict(X_test)
    y_proba = search.best_estimator_.named_steps[estimator_name].predict_proba(X_test)[:, 1]

    # Metrics
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    return {
        "best_estimator": search.best_estimator_,
        "estimator_name": estimator_name,
        "cv_score": cv_score,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }

# Create results table
res_table = pd.DataFrame(columns=['classifiers', 'sampling', 'fpr', 'tpr', 'auc'])

# Define classifiers and hyperparameters
classifiers = [
    (LogisticRegression(), {"penalty": ['l1', 'l2'], 'C': [0.01, 0.1, 1, 100], 'solver': ['liblinear']}),
    (RandomForestClassifier(), {"n_estimators": [50, 100], "max_depth": [5, 10, None]}),
    (DecisionTreeClassifier(), {"max_depth": [5, 10, None]}),
    (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
    (SVC(probability=True), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]})
]

# Sampling strategies
sampling_strategies = {"NearMiss": NearMiss(), "SMOTE": SMOTE(), "None": None}

# Loop through classifiers and sampling strategies
for clf, params in classifiers:
    for sampling_name, sampling_method in sampling_strategies.items():
        print(f"Running {clf.__class__.__name__} with sampling={sampling_name}")
        results = get_model_best_estimator_and_metrics(
            estimator=clf,
            params=params,
            sampling=sampling_method
        )
        res_table = res_table.append({
            "classifiers": results["estimator_name"],
            "sampling": sampling_name,
            "fpr": results["fpr"],
            "tpr": results["tpr"],
            "auc": results["auc"]
        }, ignore_index=True)
        print(f"{clf.__class__.__name__} ({sampling_name}): Accuracy={results['accuracy']:.4f}, Recall={results['recall']:.4f}, F1={results['f1_score']:.4f}\n")

# Plot ROC curves
res_table.set_index(['classifiers', 'sampling'], inplace=True)
fig = plt.figure(figsize=(17, 7))

for idx in res_table.index:
    plt.plot(res_table.loc[idx]['fpr'],
             res_table.loc[idx]['tpr'],
             label=f"{idx[0]} ({idx[1]}), AUC={res_table.loc[idx]['auc']:.3f}")

plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curves for Classifiers with Different Sampling', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 12}, loc='lower right')
plt.show()
