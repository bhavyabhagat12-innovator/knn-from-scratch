import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from knn import KNN

# Load data and split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ks = [1, 3, 5, 7, 9, 11]
my_scores, sk_scores = [], []

print(f"{'k':<4}{'Mine':<10}{'sklearn':<10}")
for k in ks:
    # My implementation
    mine = KNN(k)
    mine.fit(X_train, y_train)
    my_acc = (mine.predict(X_test) == y_test).mean()

    # scikit-learn
    sk = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    sk_acc = sk.score(X_test, y_test)

    my_scores.append(my_acc)
    sk_scores.append(sk_acc)
    print(f"{k:<4}{my_acc:<10.3f}{sk_acc:<10.3f}")

# Plot accuracy vs k
plt.plot(ks, my_scores, marker="o", label="My KNN")
plt.plot(ks, sk_scores, marker="x", linestyle="--", label="sklearn")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("KNN accuracy vs k (Iris)")
plt.legend()
plt.savefig("accuracy_vs_k.png", dpi=150)
plt.show()