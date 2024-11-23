"""
Defect Discerner - Inverse Parameters Module

Author: Michael C.M. Varney
Version: 1.0.1

This module defines the InverseParameters class for configuring inverse problem parameters in the Defect Discerner tool.
"""

from dataclasses import dataclass, field, asdict
import yaml
from typing import Optional, Dict, Any

@dataclass
class InverseParameters:
    """
    Encapsulates all parameters for inverse problem solving.

    General Parameters:
        prior_type (str): Type of prior to use ('random', 'load', 'disk').
        prior_filepath (Optional[str]): Filepath to load prior mask if prior_type is 'load'.
        solver_type (str): Solver to use ('FISTA', 'ADMM', 'Nesterov', 'PGD').
        max_iter (int): Maximum number of iterations for the solver.
        learning_rate (float): Learning rate for the solver.
        convergence_threshold (float): Threshold for convergence criteria.
        # update_method (str): Method to update the mask ('set_pixels', 'gradient', etc.).
        # epochs (int): Number of epochs for iterative solvers.
        save_interval (int): Interval for saving mask evolution.

    Regularization Parameters:
        tv_weight (float): Weight for total variation regularization.
        shape_weight (float): Weight for shape bias regularization.
        ellipse_center_x (Optional[int]): Center X of shape bias ellipse (defaults to canvas center).
        ellipse_center_y (Optional[int]): Center Y of shape bias ellipse (defaults to canvas center).
        ellipse_a (int): Major axis of shape bias ellipse.
        ellipse_b (int): Minor axis of shape bias ellipse.
        ellipse_theta (float): Rotation angle of shape bias ellipse.

    Solver-Specific Parameters:
        admm_rho (float): ADMM penalty parameter.
        nesterov_momentum (float): Nesterov momentum coefficient.
        pgd_momentum (float): PGD momentum (if used).
    """

    # General Parameters
    prior_type: str = 'random'               # 'random', 'load', 'disk'
    prior_filepath: Optional[str] = None     # Filepath to load prior mask
    solver_type: str = 'ADMM'               # 'FISTA', 'ADMM', 'Nesterov', 'PGD'
    max_iter: int = 50                      # Maximum number of iterations
    learning_rate: float = 1e-3              # Learning rate for the solver
    convergence_threshold: float = 1e-7      # Convergence threshold
    
    # update_method: str = 'set_pixels'        # 'set_pixels', 'gradient', etc.
    # epochs: int = 10                         # Number of epochs for iterative solvers
    save_interval: int = 1                  # Interval for saving mask evolution

    # Regularization Parameters
    tv_weight: float = 0.1                   # Weight for total variation regularization
    shape_weight: float = 0.001              # Weight for shape bias regularization
    ellipse_center_x: Optional[int] = None   # Center X of shape bias ellipse (defaults to canvas center)
    ellipse_center_y: Optional[int] = None   # Center Y of shape bias ellipse (defaults to canvas center)
    ellipse_a: int = 50                      # Major axis of shape bias ellipse
    ellipse_b: int = 30                      # Minor axis of shape bias ellipse
    ellipse_theta: float = 0.0               # Rotation angle of shape bias ellipse

    # Solver-Specific Parameters
    admm_rho: float = 8.0                    # ADMM penalty parameter
    nesterov_momentum: float = 0.6           # Nesterov momentum coefficient
    pgd_momentum: float = 0.3                # PGD momentum (if used)

    def to_yaml(self, filepath: str):
        """Save inverse parameters to a YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def from_yaml(cls, filepath: str):
        """Load inverse parameters from a YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def validate(self):
        """Validate the parameters to ensure they meet expected criteria."""
        valid_prior_types = ['random', 'load', 'disk']
        if self.prior_type not in valid_prior_types:
            raise ValueError(f"Invalid prior_type '{self.prior_type}'. Valid options are {valid_prior_types}.")

        valid_solver_types = ['FISTA', 'ADMM', 'Nesterov', 'PGD']
        if self.solver_type not in valid_solver_types:
            raise ValueError(f"Invalid solver_type '{self.solver_type}'. Valid options are {valid_solver_types}.")

        # valid_update_methods = ['set_pixels', 'gradient']
        # if self.update_method not in valid_update_methods:
        #     raise ValueError(f"Invalid update_method '{self.update_method}'. Valid options are {valid_update_methods}.")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        if not (0 < self.learning_rate <= 1):
            raise ValueError("learning_rate must be between 0 and 1.")

        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive.")

    def display_parameters(self):
        """Print all parameters to the console."""
        params = asdict(self)
        print("Inverse Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

    def get_ellipse_params(self, canvas_size):
        """Get ellipse parameters, using canvas center if not specified"""
        center_x = self.ellipse_center_x if self.ellipse_center_x is not None else canvas_size // 2
        center_y = self.ellipse_center_y if self.ellipse_center_y is not None else canvas_size // 2
        return (center_x, center_y, self.ellipse_a, self.ellipse_b, self.ellipse_theta)