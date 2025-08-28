import numpy as np
from scipy.stats import chi2

class QStatFaultDetector:
    """Generic Q-statistic based fault detector.

    Parameters
    ----------
    residual_dim : int
        Dimension of the residual vector.
    alpha : float
        Confidence level for the chi-square threshold (default 0.99).
    window : int
        Number of recent residuals used to estimate the covariance.
    """
    def __init__(self, residual_dim, alpha=0.99, window=50):
        self.residual_dim = residual_dim
        self.alpha = alpha
        self.window = window
        self.residuals = []
        self.threshold = chi2.ppf(alpha, residual_dim)

    def update(self, residual):
        """Update detector with new residual.

        Parameters
        ----------
        residual : array_like
            Prediction residual (measurement - prediction).

        Returns
        -------
        fault : bool
            True if Q-statistic exceeds the threshold.
        q_value : float
            Current value of the Q-statistic.
        """
        residual = np.asarray(residual).reshape(1, -1)
        if residual.shape[1] != self.residual_dim:
            raise ValueError("residual has wrong dimension")

        self.residuals.append(residual)
        if len(self.residuals) > self.window:
            self.residuals = self.residuals[-self.window:]

        if len(self.residuals) < 2:
            return False, 0.0

        R = np.concatenate(self.residuals, axis=0)
        cov = np.cov(R, rowvar=False)
        cov = np.atleast_2d(cov)
        inv_cov = np.linalg.pinv(cov)
        q_value = float(residual @ inv_cov @ residual.T)
        fault = q_value > self.threshold
        return fault, q_value
