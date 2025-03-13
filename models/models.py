import numpy as np
import cvxopt
from scipy.optimize import fmin_l_bfgs_b

class SVC:
    def __init__(self, kernel='precomputed', C=1.0, random_state=None):
        if kernel != 'precomputed':
            raise ValueError("This implementation only supports precomputed kernels.")
        self.kernel = kernel
        self.C = C
        self.random_state = random_state

    def fit(self, K, y):
        # Convert y to a 1D numpy array (works if y is a pandas DataFrame/Series)
        if hasattr(y, 'values'):
            y = y.values
        y = np.ravel(y)
        
        # For binary classification, map the two classes to -1 and 1.
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("This SVC implementation supports only binary classification.")
        self.classes_ = classes
        # If labels are not -1 and 1, map the first unique label to -1 and the other to 1.
        if set(classes) == set([-1, 1]):
            y_mapped = y.astype(float)
            self.label_mapping = {-1: -1, 1: 1}
        else:
            y_mapped = np.where(y == classes[0], -1.0, 1.0)
            self.label_mapping = {-1: classes[0], 1: classes[1]}
        
        self.y = y_mapped
        n_samples = K.shape[0]
        
        # Build the QP problem.
        # Dual form: minimize (1/2) α^T Q α - 1^T α, where Q = (y_i*y_j)*K_ij
        Q = np.outer(self.y, self.y) * K
        P = cvxopt.matrix(Q)
        q = cvxopt.matrix(-np.ones(n_samples))
        
        # Equality constraint: sum_i (α_i * y_i) = 0.
        A = cvxopt.matrix(self.y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        
        # Inequality constraints: 0 <= α_i <= C.
        G_std = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h_std = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))
        G = cvxopt.matrix(G_std)
        h = cvxopt.matrix(h_std)
        
        # Turn off solver output.
        cvxopt.solvers.options['show_progress'] = False
        
        # Solve QP problem.
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])
        
        # Select support vectors where alpha > threshold.
        sv = alphas > 1e-5
        self.alpha = alphas[sv]
        self.support_indices = np.arange(n_samples)[sv]
        self.y_sv = self.y[sv]
        # (Optionally, one might store the kernel submatrix of support vectors.)
        self.support_vectors_ = K[sv][:, sv]
        
        # Compute the intercept (bias). Use only support vectors with 0 < alpha < C.
        sv_non_bound = (alphas > 1e-5) & (alphas < self.C - 1e-5)
        if np.any(sv_non_bound):
            b_vals = []
            for i in np.arange(n_samples)[sv_non_bound]:
                # For each such support vector: b = y_i - sum_j (α_j * y_j * K[i,j])
                b_val = self.y[i] - np.sum(alphas[sv] * self.y[sv] * K[i, self.support_indices])
                b_vals.append(b_val)
            self.b = np.mean(b_vals)
        else:
            # Fall back to using the first support vector if none lie exactly on the margin.
            i = self.support_indices[0]
            self.b = self.y[i] - np.sum(alphas[sv] * self.y[sv] * K[i, self.support_indices])
        
        return self

    def predict(self, K):
        """
        K: Precomputed kernel matrix between test samples and the training samples.
           Shape should be (n_test_samples, n_train_samples)
        """
        # Compute decision function: f(x) = sum_{i in SV} (α_i * y_i * K(x, x_i)) + b.
        decision = np.dot(K[:, self.support_indices], (self.alpha * self.y_sv)) + self.b
        # Classify based on the sign of the decision function.
        y_pred = np.where(decision >= 0, 1.0, -1.0)
        # Map the predictions back to the original labels.
        y_pred_mapped = np.vectorize(self.label_mapping.get)(y_pred)
        return y_pred_mapped


class LogisticRegression:
    def __init__(self, max_iter=3000, random_state=None, C=1.0, tol=1e-5):
        """
        Parameters:
          max_iter: Maximum number of iterations for LBFGS.
          random_state: Not used in this implementation (for compatibility).
          C: Inverse regularization strength.
          tol: Tolerance for the stopping criterion (passed as pgtol to LBFGS).
        """
        self.max_iter = max_iter
        self.random_state = random_state
        self.C = C
        self.tol = tol
        self.coef_ = None  # Dual coefficients (one per training sample)
        self.intercept_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, K, y):
        """
        Fit kernel logistic regression given a precomputed kernel matrix K and labels y.

        Parameters:
          K: Precomputed kernel matrix of shape (n_train, n_train).
          y: Binary target labels of shape (n_train,). Labels are expected to be 0 or 1.
        
        Returns:
          self: The fitted model.
        """
        n_samples = K.shape[0]
        y = y.astype(np.float64)

        # The model: f = K.dot(alpha) + b, with p = sigmoid(f).
        # We optimize the regularized negative log-likelihood:
        #   L(alpha, b) = sum_i [ log(1+exp(f_i)) - y_i*f_i ] + 0.5/C * ||alpha||^2,
        # where f_i = (K.dot(alpha))[i] + b.
        def objective(theta):
            alpha = theta[:-1]  # Dual coefficients
            b = theta[-1]        # Intercept
            f = K.dot(alpha) + b
            # Use logaddexp for numerical stability
            loss = np.sum(np.logaddexp(0, f) - y * f) + 0.5 / self.C * np.dot(alpha, alpha)
            # Compute gradient
            p = self._sigmoid(f)
            error = p - y
            grad_alpha = K.T.dot(error) + (1 / self.C) * alpha
            grad_b = np.sum(error)
            grad = np.concatenate([grad_alpha, [grad_b]])
            return loss, grad

        # Initialize parameters: one coefficient per training sample plus the intercept
        theta0 = np.zeros(n_samples + 1)
        # Use LBFGS with pgtol instead of tol
        theta_opt, _, _ = fmin_l_bfgs_b(objective, theta0, maxiter=self.max_iter, pgtol=self.tol)
        self.coef_ = theta_opt[:-1]
        self.intercept_ = theta_opt[-1]
        return self

    def predict_proba(self, K):
        """
        Predict probability estimates for samples using a precomputed kernel matrix.

        Parameters:
          K: Precomputed kernel matrix between test samples and training samples of shape (n_test, n_train).

        Returns:
          A numpy array of shape (n_test, 2) with probabilities for class 0 and class 1.
        """
        f = K.dot(self.coef_) + self.intercept_
        p = self._sigmoid(f)
        return np.vstack([1 - p, p]).T

    def predict(self, K):
        """
        Predict binary class labels for samples using a precomputed kernel matrix.

        Parameters:
          K: Precomputed kernel matrix between test samples and training samples of shape (n_test, n_train).

        Returns:
          A numpy array of predicted class labels (0 or 1).
        """
        proba = self.predict_proba(K)[:, 1]
        return (proba >= 0.5).astype(int)
