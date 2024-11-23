import numpy as np

class SolverBase:
    """
    Base class for optimization solvers.
    
    Parameters:
        forward_operator (callable): Function that applies the forward model.
        loss_function (callable): Loss function for data fidelity.
        regularizers (dict): Dictionary of regularization functions.
        constraints (dict): Dictionary of constraint functions.
    """
    def __init__(self, forward_operator, loss_function, regularizers, constraints):
        self.forward_operator = forward_operator
        self.loss_function = loss_function
        self.regularizers = regularizers
        self.constraints = constraints

    def solve(self, I_observed, M_init, **kwargs):
        """Abstract method to perform optimization."""
        raise NotImplementedError("Solve method must be implemented by subclasses.")

class NesterovSolver(SolverBase):
    """Solver using Nesterov Momentum."""
    def solve(self, I_observed, M_init, max_iter=100, learning_rate=1e-3, momentum=0.9):
        M = M_init.copy()
        y = M.copy()
        v = np.zeros_like(M)
        
        for i in range(max_iter):
            grad = self._compute_gradient(I_observed, y)
            v_prev = v.copy()
            v = momentum * v - learning_rate * grad
            M_prev = M.copy()
            M = y + v
            # Apply constraints
            for constraint in self.constraints.values():
                M = constraint(M)
            y = M + momentum * (M - M_prev)
        return M
    
    def _compute_gradient(self, I_observed, M):
        I_estimated = self.forward_operator(M)
        difference = I_estimated - I_observed
        # Gradient w.r.t. M (simplified approximation)
        grad_M = 2 * self._backpropagate(difference, M)
        return grad_M
    
    def _backpropagate(self, diff, M):
        # Placeholder for backpropagation through the forward model
        # In practice, this should be implemented using automatic differentiation
        return diff

class FISTASolver(SolverBase):
    """Solver using Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)."""
    def __init__(self, forward_operator, loss_function, regularizers, constraints, callback=None):
        super().__init__(forward_operator, loss_function, regularizers, constraints)
        self.callback = callback  # Add callback to capture iterations

    def solve(self, I_observed, M_init, max_iter=100, learning_rate=1e-3):
        M = M_init.astype(np.float32)
        Y = M.copy()
        t = 1

        for i in range(1, max_iter + 1):
            grad = self._compute_gradient(I_observed, Y)
            M_prev = M.copy()
            M = Y - learning_rate * grad
            # Apply proximal operators for constraints
            for constraint in self.constraints.values():
                M = constraint(M)
            t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
            Y = M + ((t - 1) / t_next) * (M - M_prev)
            t = t_next

            # Invoke the callback function if provided
            if self.callback:
                self.callback(M, i)

        return M

    def _compute_gradient(self, I_observed, M):
        I_estimated = self.forward_operator(M)
        difference = I_estimated - I_observed
        # Gradient w.r.t. M (simplified approximation)
        grad_M = 2 * self._backpropagate(difference, M)
        # Add regularization gradients
        for reg_name, reg_func in self.regularizers.items():
            reg_grad = self._compute_regularization_gradient(M, reg_name)
            grad_M = grad_M + reg_grad
        return grad_M.astype(np.float32)

    def _backpropagate(self, diff, M):
        # Ensure float32 dtype
        return diff.astype(np.float32)

    def _compute_regularization_gradient(self, M, reg_name):
        if reg_name == 'tv':
            grad = self._tv_gradient(M)
        elif reg_name == 'shape':
            grad = self._shape_gradient(M)
        else:
            grad = np.zeros_like(M)
        return grad.astype(np.float32)  # Ensure float32 dtype

    def _tv_gradient(self, M):
        # Ensure float32 dtype for TV gradient
        grad = np.zeros_like(M, dtype=np.float32)
        grad_x_forward = np.roll(M, -1, axis=1) - M
        grad_x_backward = M - np.roll(M, 1, axis=1)
        grad_y_forward = np.roll(M, -1, axis=0) - M
        grad_y_backward = M - np.roll(M, 1, axis=0)
        grad += grad_x_forward - grad_x_backward + grad_y_forward - grad_y_backward
        return grad

    def _shape_gradient(self, M):
        # Ensure float32 dtype for shape gradient
        grad = 2 * (M - self.regularizers['shape'](M))
        return grad.astype(np.float32)

class ADMMSolver(SolverBase):
    """Solver using Alternating Direction Method of Multipliers (ADMM)."""
    def solve(self, I_observed, M_init, max_iter=100, rho=1.0):
        M = M_init.copy()
        Z = M.copy()
        U = np.zeros_like(M)
        
        for i in range(max_iter):
            # Update M by solving the proximal operator of the loss function
            grad = self._compute_gradient(I_observed, M)
            M = Z - U - grad / rho
            # Apply constraints
            for constraint in self.constraints.values():
                M = constraint(M)
            # Update Z by solving the proximal operator of the regularization
            Z = M + U
            Z = self._proximal_operator(Z, 1 / rho)
            # Update dual variable U
            U += M - Z
        return M

    def _compute_gradient(self, I_observed, M):
        I_estimated = self.forward_operator(M)
        difference = I_estimated - I_observed
        # Gradient w.r.t. M (simplified approximation)
        grad_M = 2 * self._backpropagate(difference, M)
        return grad_M

    def _backpropagate(self, diff, M):
        # Placeholder for backpropagation through the forward model
        return diff

    def _proximal_operator(self, Z, alpha):
        # Proximal operator for regularizations
        # Placeholder implementation
        return Z

class PGDSolver(SolverBase):
    """Solver using Proximal Gradient Descent (PGD)."""
    def solve(self, I_observed, M_init, max_iter=100, learning_rate=1e-3):
        M = M_init.copy()
        
        for i in range(max_iter):
            grad = self._compute_gradient(I_observed, M)
            M = M - learning_rate * grad
            # Apply proximal operators for constraints
            for constraint in self.constraints.values():
                M = constraint(M)
        return M

    def _compute_gradient(self, I_observed, M):
        I_estimated = self.forward_operator(M)
        difference = I_estimated - I_observed
        # Gradient w.r.t. M (simplified approximation)
        grad_M = 2 * self._backpropagate(difference, M)
        # Add regularization gradients
        for reg_name, reg_func in self.regularizers.items():
            grad_M += self._compute_regularization_gradient(M, reg_name)
        return grad_M

    def _backpropagate(self, diff, M):
        # Placeholder for backpropagation through the forward model
        return diff

    def _compute_regularization_gradient(self, M, reg_name):
        if reg_name == 'tv':
            grad = self._tv_gradient(M)
        elif reg_name == 'shape':
            grad = self._shape_gradient(M)
        else:
            grad = np.zeros_like(M)
        return grad

    def _tv_gradient(self, M):
        grad = np.zeros_like(M)
        grad_x_forward = np.roll(M, -1, axis=1) - M
        grad_x_backward = M - np.roll(M, 1, axis=1)
        grad_y_forward = np.roll(M, -1, axis=0) - M
        grad_y_backward = M - np.roll(M, 1, axis=0)
        grad += grad_x_forward - grad_x_backward + grad_y_forward - grad_y_backward
        return grad

    def _shape_gradient(self, M):
        grad = 2 * (M - self.regularizers['shape'](M))
        return grad