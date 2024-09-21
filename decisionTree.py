import numpy as np
import pandas as pd

class DecisionTreeC45:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    # Función para calcular la entropía
    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return -np.sum(prob * np.log2(prob))

    # Función para calcular la ganancia de información
    def information_gain(self, y, split):
        ent_total = self.entropy(y)
        subsets = [y[split == val] for val in np.unique(split)]
        weighted_entropy = sum((len(subset) / len(y)) * self.entropy(subset) for subset in subsets)
        return ent_total - weighted_entropy

    # Selección del mejor atributo
    def best_attribute(self, X, y):
        gains = [self.information_gain(y, X[:, i]) for i in range(X.shape[1])]
        return np.argmax(gains)

    # Función recursiva para construir el árbol
    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return np.bincount(y).argmax()

        best_attr = self.best_attribute(X, y)
        tree = {best_attr: {}}

        for val in np.unique(X[:, best_attr]):
            X_subset = X[X[:, best_attr] == val]
            y_subset = y[X[:, best_attr] == val]
            tree[best_attr][val] = self.build_tree(X_subset, y_subset, depth + 1)

        return tree

    # Función para entrenar el modelo
    def fit(self, X, y):
        self.tree = self.build_tree(np.array(X), np.array(y))

    # Función para predecir nuevos valores
    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        attr = list(tree.keys())[0]
        subtree = tree[attr].get(x[attr], np.bincount(list(tree[attr].values())).argmax())
        return self.predict_one(x, subtree)

    def predict(self, X):
        return [self.predict_one(x, self.tree) for x in np.array(X)]


# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo (atributos y etiquetas)
    X = [[0, 1], [1, 0], [0, 1], [1, 1]]
    y = [0, 1, 0, 1]

    # Crear modelo y entrenar
    model = DecisionTreeC45()
    model.fit(X, y)

    # Predecir
    X_test = [[1, 0], [0, 1]]
    print(model.predict(X_test))
