# 07_Support_Vector_Machine.py
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
X, y = iris.data, iris.target
print('Feature names:', iris.feature_names)
print('\nTarget names:', iris.target_names)
print('\nFirst five records:', X[:5])
print('\nFirst five target values:', y[:5])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = SVC(kernel='linear', C=1.0)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print('\nAccuracy:', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))
plt.figure(figsize=(8, 6))
for class_value in range(3):
    plt.scatter(X[y == class_value, 2], X[y == class_value, 3],
                label=iris.target_names[class_value])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Dataset - SVM')
plt.legend()
plt.show()
