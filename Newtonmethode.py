import sympy as sp
import numpy as np
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd


class NewtonMethod:
    """
    Implémentation flexible de la méthode de Newton pour l'optimisation
    de fonctions à 1 à n variables.
    """
    
    def __init__(self, f_expr, variables=None):
        
        t0 = time.time()  # <-- chronométrage ici seulement !

        if variables is None:
            variables = sorted([str(var) for var in f_expr.free_symbols])

        self.variables = [sp.Symbol(var) for var in variables]
        self.f_expr = f_expr
        self.dimension = len(self.variables)

        # Gradient + Hessienne
        self.grad_expr = sp.Matrix([sp.diff(f_expr, var) for var in self.variables])

        if self.dimension == 1:
            self.hess_expr = sp.diff(self.grad_expr[0], self.variables[0])
        else:
            self.hess_expr = sp.Matrix([
                [sp.diff(g, var) for var in self.variables] for g in self.grad_expr
            ])

        # Conversion en fonctions numériques
        self.f = sp.lambdify(self.variables, f_expr, 'numpy')
        self.grad = sp.lambdify(self.variables, self.grad_expr, 'numpy')
        self.hess = sp.lambdify(self.variables, self.hess_expr, 'numpy')

        self.init_time = time.time() - t0  # <-- correct maintenant

    def minimize(self, x0, tol=1e-6, max_iter=100, verbose=False):

        t0 = time.time()  # <-- temps d'optimisation

        x = np.array(x0, dtype=float)
        iterations = 0

        trajectory = [x.copy()]
        f_values = [self.f(*x)]
        history_loss = []
        history_loss = [self.f(*x)] 
        for _ in range(max_iter):
            if self.dimension == 1:
                g = np.array([self.grad(x[0])])
                H = np.array([[self.hess(x[0])]])
            else:
                g = np.array(self.grad(*x)).flatten()
                H = np.array(self.hess(*x))

            # Régularisation de la Hessienne
            if self.dimension == 1:
                if abs(H[0, 0]) < 1e-12:
                    H[0, 0] = 1e-6
            else:
                if np.linalg.det(H) == 0:
                    H += 1e-6 * np.eye(self.dimension)

            try:
                h = np.linalg.solve(H, -g)
            except:
                h = -np.linalg.pinv(H) @ g

            x_new = x + h
            iterations += 1
            trajectory.append(x_new.copy())
            f_values.append(self.f(*x_new))
            f_current = self.f(*x_new)
            #f_values.append(f_current)
            history_loss.append(f_current)

            if np.linalg.norm(x_new - x) < tol:
                break

            x = x_new
            
        optim_time = time.time() - t0
        #history_loss.append(self.f(*x_new))
        return {
            'solution': x,
            'f_min': self.f(*x),
            'iterations': iterations,
            'trajectory': np.array(trajectory),
            'f_values': f_values,
            'optim_time': optim_time,
            'init_time': self.init_time,
            'total_time': self.init_time + optim_time,
            'history_loss': history_loss,
            'success': iterations < max_iter
        }

    #  TRAJECTOIRE POUR RAPPORT

    def plot_trajectory(self, result, levels=50):
        """
        Trace la trajectoire sur les lignes de niveaux d'une fonction à 2 variables.
        Parfait pour Rosenbrock, Himmelblau, etc.
        """

        if self.dimension != 2:
            raise ValueError("La trajectoire ne peut être tracée que pour une fonction 2D.")

        traj = result["trajectory"]
        x_vals = traj[:, 0]
        y_vals = traj[:, 1]

        # grille
        X = np.linspace(min(x_vals)-1, max(x_vals)+1, 400)
        Y = np.linspace(min(y_vals)-1, max(y_vals)+1, 400)
        XX, YY = np.meshgrid(X, Y)

        Z = self.f(XX, YY)

        plt.figure(figsize=(8,6))
        plt.contour(XX, YY, Z, levels=levels, linewidths=0.8)
        plt.plot(x_vals, y_vals, color='red', marker="o", markersize=3)
        plt.title("Trajectoire de la méthode de Newton")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
