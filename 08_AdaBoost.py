# 08_AdaBoost.py
# Supplementary example: the AdaBoost code is blank in the PDF.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)
# A separate depth-1 tree is used as the ensemble's weak learner.
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50, learning_rate=1.0, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Single weak classifier accuracy:', accuracy_score(y_test, stump.predict(X_test)))
print('AdaBoost accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names,
                            zero_division=0))
