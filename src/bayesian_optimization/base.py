import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from typing import Callable, Optional, Tuple, Dict
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class BaseBayesianOptimization(ABC):
    def __init__(
        self,
        objective: Callable,
        constraint: Callable,
        bounds: np.ndarray,
        n_init: int = 10,
        n_iter: int = 30,
        random_state: Optional[int] = None
    ):
        self.objective = objective
        self.constraint = constraint
        self.bounds = bounds
        self.n_init = n_init
        self.n_iter = n_iter
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.n_dims = bounds.shape[0]
        self.X = None
        self.y = None
        self.c = None
        self.models = {}
        self.history = {'X': [], 'y': [], 'c': [], 'best_x': [], 'best_y': [], 'feasible': []}
    
    def _initialize(self):
        from src.experimental_design import lhs_sample
        
        self.X = lhs_sample(self.bounds, self.n_init, self.random_state)
        self.y = np.array([self.objective(x) for x in self.X])
        self.c = np.array([self.constraint(x) for x in self.X])
        self._train_models()
        self._update_history()
    
    def _train_models(self):
        # Улучшенные настройки керна для лучшей сходимости
        kernel_f = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.01)
        self.models['f'] = GaussianProcessRegressor(
            kernel=kernel_f, 
            n_restarts_optimizer=3,  # Уменьшено для скорости
            random_state=self.random_state,
            alpha=1e-6,
            normalize_y=True
        )
        self.models['f'].fit(self.X, self.y)
        
        kernel_g = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.01)
        self.models['g'] = GaussianProcessRegressor(
            kernel=kernel_g, 
            n_restarts_optimizer=3,
            random_state=self.random_state,
            alpha=1e-6,
            normalize_y=True
        )
        self.models['g'].fit(self.X, self.c)
    
    def _predict(self, X_new: np.ndarray):
        mu_f, sigma_f = self.models['f'].predict(X_new, return_std=True)
        mu_g, sigma_g = self.models['g'].predict(X_new, return_std=True)
        sigma_f = np.maximum(sigma_f, 1e-6)
        sigma_g = np.maximum(sigma_g, 1e-6)
        return mu_f, sigma_f, mu_g, sigma_g
    
    @abstractmethod
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def _optimize_acquisition(self):
        best_x = None
        best_acq = -np.inf
        
        # Многостартовый поиск
        for _ in range(10):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1)), 
                    x0, 
                    bounds=self.bounds, 
                    method='L-BFGS-B',
                    options={'maxiter': 100}
                )
                if result.success:
                    acq_value = self._acquisition_function(result.x.reshape(1, -1))[0]
                    if acq_value > best_acq:
                        best_acq = acq_value
                        best_x = result.x
            except:
                continue
        
        return best_x if best_x is not None else np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
    
    def _update_history(self):
        self.history['X'].append(self.X.copy())
        self.history['y'].append(self.y.copy())
        self.history['c'].append(self.c.copy())
        
        feasible_mask = self.c <= 0
        if np.any(feasible_mask):
            best_idx = np.argmin(self.y[feasible_mask])
            self.history['best_x'].append(self.X[feasible_mask][best_idx])
            self.history['best_y'].append(self.y[feasible_mask][best_idx])
        else:
            self.history['best_x'].append(self.X[np.argmin(self.c)])
            self.history['best_y'].append(np.inf)
        
        self.history['feasible'].append(np.sum(feasible_mask))
    
    def optimize(self):
        print(f"Initialization with {self.n_init} points...")
        self._initialize()
        
        for iteration in range(self.n_iter):
            print(f"Iteration {iteration + 1}/{self.n_iter}")
            x_next = self._optimize_acquisition()
            y_next = self.objective(x_next)
            c_next = self.constraint(x_next)
            
            self.X = np.vstack([self.X, x_next])
            self.y = np.append(self.y, y_next)
            self.c = np.append(self.c, c_next)
            self._train_models()
            self._update_history()
            
            # Показываем прогресс
            best_val = self.history['best_y'][-1]
            if np.isfinite(best_val):
                print(f"  Best feasible value: {best_val:.6f}")
        
        feasible_mask = self.c <= 0
        if np.any(feasible_mask):
            best_idx = np.argmin(self.y[feasible_mask])
            best_solution = {
                'x': self.X[feasible_mask][best_idx],
                'f': self.y[feasible_mask][best_idx],
                'g': self.c[feasible_mask][best_idx]
            }
        else:
            best_solution = {
                'x': self.X[np.argmin(self.c)],
                'f': np.inf,
                'g': np.min(self.c)
            }
        
        return {'best_solution': best_solution, 'history': self.history, 'n_evaluations': len(self.X)}
