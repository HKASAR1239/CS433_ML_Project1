from __future__ import annotations
import numpy as np


# ---------- helpers (MSE) ----------


def _as_1d(a) -> np.ndarray:
    """Return a as a contiguous 1-D float64 array."""
    return np.asarray(a, dtype=np.float64).ravel()


def _mse_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Mean squared error with a 0.5 factor,
        L(w) = (1/(2N)) * || y - Xw ||^2
    """
    e = y - tx @ w
    return 0.5 * np.mean(e * e)


def _mse_grad(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Gradient of the average MSE:
        ∇L(w) = -(1/N) X^T (y - Xw)
    """
    e = y - tx @ w
    return -(tx.T @ e) / y.size


# ---------- helpers (logistic) ----------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid function.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    Average negative log-likelihood for labels y ∈ {0,1}:
        L(w) = (1/N) * sum_i [ log(1 + exp(z_i)) - y_i * z_i ],
    where z = Xw. Uses logaddexp for stability.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0.0, z) - y * z)


def _logistic_grad(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Gradient of the average NLL:
        ∇L(w) = (1/N) X^T (σ(Xw) - y)
    """
    p = _sigmoid(tx @ w)
    return (tx.T @ (p - y)) / y.size


# ===========================================================
#                       STEP 1 : ML METHODS
# ===========================================================


def mean_squared_error_gd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
):
    """
    Linear regression using gradient descent on MSE.

    Returns:
        (w, loss) where loss is the final 0.5*mean squared error.
    """
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=np.float64)
    w = _as_1d(initial_w).copy()

    for _ in range(int(max_iters)):
        grad = _mse_grad(y, tx, w)
        w -= gamma * grad

    return w, _mse_loss(y, tx, w)


def mean_squared_error_sgd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
):
    """
    Linear regression using stochastic gradient descent.

    At each step t we pick a single random datapoint (x_i, y_i) and take one
    step with the per-sample gradient of 0.5*(y_i - x_i·w)^2,
        g_t = -(y_i - x_i·w) * x_i.

    Returns:
        (w, loss) with the final MSE (without any regularization term).
    """
    rng = np.random.default_rng()
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=np.float64)
    w = _as_1d(initial_w).copy()

    # n = y.size
    n = tx.shape[0]
    for _ in range(int(max_iters)):
        i = int(rng.integers(0, n))
        xi = tx[i]  # shape (D,)
        yi = y[i]
        err = yi - xi @ w
        grad_i = -err * xi  # per-sample gradient of 0.5*(err^2)
        w -= gamma * grad_i

    return w, _mse_loss(y, tx, w)


def least_squares(y: np.ndarray, tx: np.ndarray):
    """
    Least squares via the normal equations:
        w = (X^T X)^{-1} X^T y
    Uses np.linalg.solve when possible.

    Returns:
        (w, loss) with standard MSE loss (0.5 factor).
    """
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=np.float64)

    A = tx.T @ tx
    b = tx.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(tx) @ y

    return w, _mse_loss(y, tx, w)


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float):
    """
    Ridge regression via normal equations.

    We minimize:
        (1/(2N)) * ||y - Xw||^2 + λ ||w||^2

    Normal equation (with course scaling):
        (X^T X + 2Nλ I) w = X^T y

    Returned loss is the *plain* MSE (no penalty term), per the instructions.
    """
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=np.float64)

    n, d = tx.shape
    A = tx.T @ tx + (2.0 * n * lambda_) * np.eye(d, dtype=np.float64)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    return w, _mse_loss(y, tx, w)


def logistic_regression(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
):
    """
    Logistic regression (y ∈ {0,1}) using gradient descent on the average NLL.

    Returns:
        (w, loss) where loss is the final average negative log-likelihood.
    """
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=np.float64)
    w = _as_1d(initial_w).copy()

    for _ in range(int(max_iters)):
        grad = _logistic_grad(y, tx, w)
        w -= gamma * grad

    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
):
    """
    Regularized logistic regression with L2 penalty λ||w||^2.

    We optimize:
        average_NLL(w) + λ ||w||^2
    with gradient:
        ∇ = (1/N) X^T (σ(Xw) - y) + 2λ w
    Returns:
        (w, loss) where loss is the final average NLL.
    """
    y = _as_1d(y)
    tx = np.asarray(tx, dtype=np.float64)
    w = _as_1d(initial_w).copy()

    for _ in range(int(max_iters)):
        grad = _logistic_grad(y, tx, w) + 2.0 * lambda_ * w
        w -= gamma * grad

    return w, _logistic_loss(y, tx, w)
