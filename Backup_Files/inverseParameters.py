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
        max_iter (int): Maximum number of iterations for the solver.
        convergence_threshold (float): Threshold for convergence criteria.
        save_interval (int): Interval for saving mask evolution.

    Regularization Parameters:
        tv_weight (float): Weight for total variation regularization.

    Solver-Specific Parameters:
        admm_rho (float): ADMM penalty parameter.
    """

    # General Parameters
    prior_type: str = 'random'               # 'random', 'load', 'disk'
    prior_filepath: Optional[str] = None     # Filepath to load prior mask
    max_iter: int = 500                      # Maximum number of iterations
    convergence_threshold: float = 1e-7      # Convergence threshold
    save_interval: int = 10                  # Interval for saving mask evolution

    # Regularization Parameters
    tv_weight: float = 0.1                   # Weight for total variation regularization

    # Solver-Specific Parameters
    admm_rho: float = 12.0                   # ADMM penalty parameter

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

        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive.")

    def display_parameters(self):
        """Print all parameters to the console."""
        params = asdict(self)
        print("Inverse Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")