import decimal as dc
import random
import math
import matplotlib.pyplot as plt
import numpy as np

class ChaoticMap:
    def __init__(self, steps, trans, x0=None, prec=50, seed=None):
        dc.getcontext().prec = prec
        self.steps = steps
        self.trans = trans
        self.prec = prec

        if seed is not None:
            random.seed(seed)

        self.x0 = dc.Decimal(str(x0)) if x0 is not None else self.__get_initial_condition()

    def __repr__(self):
        return f"<{self.__class__.__name__}: x0={self.x0:.4f}, steps={self.steps}, trans={self.trans}>"

    # Renomeado para __ para ocultar do usuário
    def __get_initial_condition(self):
        a, b = self.domain
        return dc.Decimal(str(random.uniform(a, b)))

    def estimate_lyapunov(self, dec=False):

        x = self.x0
        soma = dc.Decimal(0)

        # Transient: estabiliza a órbita no atrator
        for _ in range(self.trans):
            x = self.f(x)

        # Summation: média logarítmica das derivadas
        for _ in range(self.steps):
            x = self.f(x)
            deriv = abs(self.df(x))
            # Proteção contra log(0) em pontos críticos
            if deriv > 0:
                soma += deriv.ln()
            else:
                soma += dc.Decimal("-1e10") # Penalidade para singularidades

        lambda_est = soma / dc.Decimal(self.steps)
        return lambda_est if dec else float(lambda_est)

    def lyapunov_convergence(self, plot=False):
        """Retorna a evolução do expoente ao longo das iterações."""
        x = self.x0
        for _ in range(self.trans):
            x = self.f(x)

        soma = dc.Decimal(0)
        evolution = []

        for i in range(1, self.steps + 1):
            x = self.f(x)
            soma += abs(self.df(x)).ln()
            evolution.append(soma / dc.Decimal(i))

        if plot:
            self._plot_convergence(evolution)

        return np.array([float(v) for v in evolution])

    def _plot_convergence(self, data):
        plt.figure(figsize=(8, 4))
        plt.plot(data, label="Estimated $\lambda$")
        plt.axhline(float(self.theoretical_lyapunov()), color='r', ls='--', label="Theoretical")
        plt.title(f"Lyapunov Convergence - {self.__class__.__name__}")
        plt.xlabel("Iterations")
        plt.ylabel("$\lambda$")
        plt.legend()
        plt.grid(True)
        plt.show()

    def time_series(self, dec=False, plot=False):

        x = self.x0
        # 1. Transient
        for _ in range(self.trans):
            x = self.f(x)

        # 2. Time Series
        orbit = []
        for _ in range(self.steps):
            x = self.f(x)
            orbit.append(x)


        if plot:
            self._plot_series(orbit)

        return orbit if dec else np.array(orbit, dtype=float)

    def _plot_series(self, data):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(data, lw=0.5, color='#2c3e50')
        plt.title(f"Time Series: {self.__class__.__name__} (x0={float(self.x0):.4f})")
        plt.xlabel("n (iterations)")
        plt.ylabel("$x_n$")
        plt.grid(True, alpha=0.3)
        plt.show()


class LogisticMap(ChaoticMap):
    domain = (0, 1)

    def f(self, x): return dc.Decimal('4') * x * (1 - x)
    def df(self, x): return dc.Decimal('4') * (1 - 2 * x)

    def theoretical_lyapunov(self):
        return dc.Decimal('2').ln()

class UlamMap(ChaoticMap):
    domain = (-1, 1)
    def f(self, x): return dc.Decimal('1') - dc.Decimal('2') * x**2
    def df(self, x): return dc.Decimal('-4') * x
    def theoretical_lyapunov(self): return dc.Decimal('2').ln()


class BernoulliMap(ChaoticMap):
    domain = (0, 1)

    def f(self, x): return (2 * x) % dc.Decimal(1)
    def df(self, x): return dc.Decimal(2)

    def theoretical_lyapunov(self):
        return dc.Decimal('2').ln()


class GaussMap(ChaoticMap):
    domain = (1e-12, 0.999999999999)

    def f(self, x):
        if x == 0: return dc.Decimal(0)
        inv_x = 1 / x
        # floor(1/x)
        return inv_x - inv_x.to_integral_value(rounding=dc.ROUND_FLOOR)

    def df(self, x):
        if x == 0: return dc.Decimal(0)
        return -1 / (x**2)

    def theoretical_lyapunov(self):
        ctx = dc.getcontext()
        pi = ctx.create_decimal_from_float(math.pi)
        return (pi**2) / (dc.Decimal('6') * dc.Decimal('2').ln())

class TentMap(ChaoticMap):
    domain = (0, 1)

    def f(self, x): return dc.Decimal('2') * min(x, 1 - x)

    def df(self, x): return dc.Decimal('2') if x < 0.5 else -dc.Decimal('2')

    def theoretical_lyapunov(self):
        return dc.Decimal('2').ln()


class AsymetricMap(ChaoticMap):
    domain = (0, 1)

    def f(self, x): return (x / dc.Decimal('0.4')) if x < dc.Decimal('0.4') else ((1 - x) / (1 - dc.Decimal('0.4')))
    def df(self, x): return (1 / dc.Decimal('0.4')) if x < dc.Decimal('0.4') else (-1 / (1 - dc.Decimal('0.4')))

    def theoretical_lyapunov(self):
        return -(dc.Decimal('0.4') * dc.Decimal('0.4').ln()) - ((1 - dc.Decimal('0.4')) * (1 - dc.Decimal('0.4')).ln())


class ChebyshevMap(ChaoticMap):
    domain = (-1, 1)
    def f(self, x): return dc.Decimal('2') * x**2 - dc.Decimal('1')
    def df(self, x): return dc.Decimal('4') * x
    def theoretical_lyapunov(self): return dc.Decimal('2').ln()
