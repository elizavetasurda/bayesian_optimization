"""
Gaussian Process surrogate model for Bayesian optimization.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from typing import Tuple, Optional


class SurrogateModel:
    """
    Gaussian Process surrogate for objective function.
    
    Provides posterior mean and variance predictions.
    """
    
    def __init__(
        self,
        length_scale: float = 1.0,
        noise: float = 1e-6,
        kernel: str = "matern"
    ):
        """
        Parameters
        ----------
        length_scale : float
            Kernel length scale.
        noise : float
            Noise level (WhiteKernel).
        kernel : str
            Kernel type: 'rbf' or 'matern'.
        """
        if kernel == "rbf":
            base_kernel = RBF(length_scale=length_scale)
        else:
            base_kernel = Matern(length_scale=length_scale, nu=2.5)
        
        self.kernel = base_kernel + WhiteKernel(noise_level=noise)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        self.X_train = None
        self.y_train = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit GP to training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_points, n_dim)
            Input points.
        y : np.ndarray, shape (n_points,)
            Objective values.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.gp.fit(X, y)
        self._is_fitted = True
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at new points.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_points, n_dim)
            Points to predict.
            
        Returns
        -------
        tuple
            (mean, variance)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        mean, std = self.gp.predict(X, return_std=True)
        return mean, std ** 2
    
    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """
        Update model with new data points.
        
        Parameters
        ----------
        X_new : np.ndarray
            New input points.
        y_new : np.ndarray
            New objective values.
        """
        if self.X_train is None:
            X_combined = X_new
            y_combined = y_new
        else:
            X_combined = np.vstack([self.X_train, X_new])
            y_combined = np.hstack([self.y_train, y_new])
        
        self.fit(X_combined, y_combined)
    
    def sample_from_posterior(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Sample from posterior distribution.
        
        Parameters
        ----------
        X : np.ndarray
            Input points.
        n_samples : int
            Number of samples.
            
        Returns
        -------
        np.ndarray
            Samples from posterior.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        return self.gp.sample_y(X, n_samples=n_samples, random_state=42)
