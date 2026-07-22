import numpy as np


class GDRegressor:
    """
    Скопируйте или наследуйте класс GDRegressor из файла linreg_housing/linreg.py,
    адаптировав обучение под многомерную регрессию с нефиксированным числом признаков.
    """

    def __init__(self, alpha: float = 0.01, n_iter: int = 3000, progress: bool = False):
        self.alpha = alpha
        self.n_iter = n_iter
        self.progress = progress
        self.weights = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X_train, y_train):
        X = self._prepare_X(X_train, fit=True)
        y = self._prepare_y(y_train)
        self.weights = np.zeros((X.shape[1], 1))

        for i in range(self.n_iter):
            y_pred = X @ self.weights
            gradient = (2 / X.shape[0]) * (X.T @ (y_pred - y))
            self.weights -= self.alpha * gradient

            if self.progress and (i + 1) % 500 == 0:
                loss = np.sqrt(np.mean((y_pred - y) ** 2))
                print(f"Iteration {i + 1}: RMSE = {loss:.4f}")

        return self

    def predict(self, X_test):
        if self.weights is None:
            raise ValueError("Model is not fitted yet")
        X = self._prepare_X(X_test, fit=False)
        return X @ self.weights

    def get_weights(self, feature_names=None) -> dict[str, float]:
        if self.weights is None:
            return {}

        scaled_coefficients = self.weights[1:, 0]
        coefficients = scaled_coefficients / self.std_
        intercept = self.weights[0, 0] - np.sum(scaled_coefficients * self.mean_ / self.std_)

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(coefficients))]

        result = {"intercept": float(intercept)}
        result.update({name: float(value) for name, value in zip(feature_names, coefficients)})
        return result

    def _prepare_X(self, X, fit: bool = False):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1

        X_scaled = (X - self.mean_) / self.std_
        bias = np.ones((X_scaled.shape[0], 1))
        return np.hstack([bias, X_scaled])

    @staticmethod
    def _prepare_y(y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y