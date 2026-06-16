import numpy as np
from sklearn.model_selection import train_test_split


class GDRegressor:
    def __init__(self, alpha=0.001, n_iter=100, progress=True):
        self.alpha = alpha
        self.n_iter = n_iter
        self.progress = progress
        self.degree = 5
        self.weights = None
        self.mean_ = None
        self.std_ = None
        self.poly_mean_ = None
        self.poly_std_ = None

    def fit(self, X_train, y_train):
        X = self._prepare_X(X_train, fit=True)
        y = self._prepare_y(y_train)

        self.weights = np.zeros((X.shape[1], 1))

        for i in range(self.n_iter):
            y_hat = X @ self.weights
            gradient = (2 / X.shape[0]) * (X.T @ (y_hat - y))
            self.weights -= self.alpha * gradient

            if self.progress and (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}, RMSE: {rmse(y, y_hat):.4f}")

        return self

    def predict(self, X_test):
        if self.weights is None:
            raise ValueError("Model is not fitted yet")

        X = self._prepare_X(X_test, fit=False)
        return X @ self.weights

    def _prepare_X(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1

        X_scaled = (X - self.mean_) / self.std_

        features = []
        for power in range(1, self.degree + 1):
            features.append(X_scaled**power)

        X_poly = np.hstack(features)

        if fit:
            self.poly_mean_ = X_poly.mean(axis=0)
            self.poly_std_ = X_poly.std(axis=0)
            self.poly_std_[self.poly_std_ == 0] = 1

        X_poly = (X_poly - self.poly_mean_) / self.poly_std_
        return np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])

    @staticmethod
    def _prepare_y(y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y


def z_scaler(feature):
    feature = np.asarray(feature, dtype=float)
    std = feature.std(axis=0)
    std = np.where(std == 0, 1, std)
    return (feature - feature.mean(axis=0)) / std


def min_max(feature):
    feature = np.asarray(feature, dtype=float)
    min_value = feature.min(axis=0)
    max_value = feature.max(axis=0)
    denominator = np.where(max_value - min_value == 0, 1, max_value - min_value)
    return (feature - min_value) / denominator


def rmse(y_hat, y):
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sqrt(np.mean((y_hat - y) ** 2))


def r_squared(y_hat, y):
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float)

    ss_res = np.sum((y_hat - y) ** 2)
    ss_tot = np.sum((y_hat - y_hat.mean()) ** 2)
    return 1 - ss_res / ss_tot


def find_optimal_params(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=18)

    candidates = [
        (10000, 0.0001),
        (20000, 0.0001),
        (30000, 0.0001),
        (5000, 0.0003),
        (10000, 0.0003),
        (20000, 0.0003),
        (5000, 0.001),
        (10000, 0.001),
    ]

    best_params = candidates[0]
    best_score = -np.inf

    for max_iter, alpha in candidates:
        model = GDRegressor(alpha=alpha, n_iter=max_iter, progress=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)

        score = r_squared(y_train, y_pred)
        error = rmse(y_train, y_pred)

        if score >= 0.49 and error <= 6.45 and score > best_score:
            best_score = score
            best_params = (max_iter, alpha)

    return best_params
