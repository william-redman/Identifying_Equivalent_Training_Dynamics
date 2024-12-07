"""Implementation of the second example (log barrier with exponential
reparameterization) apperaing in [1].

[1] U. Ghai, Z. Lu, and E. Hazan, “Non-convex online learning via algorithmic
equivalence.” arXiv, May 30, 2022. doi: 10.48550/arXiv.2205.15235.

"""


try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-fs')
except ImportError:
    pass


from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import tqdm
from dynamic_mode_decomposition import time_delay, data_matrices, dmd

eta: float = 1e-2               # Step length.
T: int = 100                    # Number of steps.
d: int = 2                      # Dimension of convex body.

# Bounds defining the convex domain (hypercube). See also bounds_prime below.
bounds: List[Tuple[float, float]] = [[1.0 / T, 1.0] for _ in range(d)]

@jax.jit
def f(x: jnp.ndarray) -> float:
    """Loss function."""
    # return jnp.dot(x, x)
    #return (x**4).sum()
    return jnp.tan(x).sum()

@jax.jit
def ftilde(u: jnp.ndarray) -> float:
    return f(q(u))


@jax.jit
def R(x: jnp.ndarray) -> float:
    """Regularizer."""
    # Log-barrier regularization.
    return -jnp.log(x).sum()


@jax.jit
def DR(x: jnp.ndarray) -> jnp.ndarray:
    return -1.0 / x


@jax.jit
def DRinv(y: jnp.ndarray) -> jnp.ndarray:
    return -1.0 / y


@jax.jit
def q(u: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(u)


@jax.jit
def qinv(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(x)


# Compute the bounds for the inverse convex body.
bounds_prime = [tuple(b) for b
                in jnp.sort(jax.vmap(qinv)(jnp.array(bounds))).tolist()]


def is_in_convex_body(x: jnp.ndarray,
                      bounds: List[Tuple[float, float]]) -> bool:
    assert x.shape[0] == len(bounds)
    for value, (lb, ub) in zip(x, bounds):
        if not (lb <= value <= ub):
            return False
    return True


def bregman_divergence(x: jnp.ndarray, y: jnp.ndarray) -> float:
    assert x.shape == y.shape
    return R(x) - R(y) - jnp.dot(DR(y), x - y)


def euclidean_distance2(x: jnp.ndarray, y: jnp.ndarray) -> float:
    assert x.shape == y.shape
    return jnp.square(x - y).sum()


def project(y: jnp.ndarray, distance: Callable[[jnp.ndarray,
                                                jnp.ndarray], float],
            bounds: List[Tuple[float, float]]) -> jnp.ndarray:
    """Project point onto hypercube according to specified distance."""
    res = scipy.optimize.minimize(distance, x0=y, args=(y,), bounds=bounds)
    assert res.success
    return jnp.array(res.x)


def sanity_check():
    x = jnp.array([1e-1, 1e0, 1e1, 1e2])
    u = qinv(x)
    assert jnp.allclose(jax.jacobian(R)(x), DR(x))
    assert jnp.allclose(DRinv(DR(x)), x)
    assert jnp.allclose(x, q(u))
    assert jnp.allclose(qinv(x), u)
    assert jnp.allclose(jax.jacobian(q)(u) @ jax.jacobian(q)(u).T,
                        jnp.linalg.pinv(jax.hessian(R)(x)))


def online_mirror_descent(x0: jnp.ndarray) -> jnp.ndarray:
    X = np.zeros((T, d))
    X[0, :] = x0
    assert is_in_convex_body(X[0, :], bounds)

    for t in tqdm.tqdm(range(1, T)):
        x = X[t-1, :]
        nabla_f_x = jax.grad(f)(x)
        ynext = DRinv(DR(x) - eta * nabla_f_x)
        xnext = project(ynext, bregman_divergence, bounds)
        assert is_in_convex_body(xnext, bounds)
        X[t, :] = xnext

    return X


def online_gradient_descent(u0: jnp.ndarray) -> jnp.ndarray:
    U = np.zeros((T, d))
    U[0, :] = u0

    assert is_in_convex_body(U[0, :], bounds_prime)

    for t in tqdm.tqdm(range(1, T)):
        u = U[t-1, :]
        nabla_ftilde_u = jax.grad(ftilde)(u)
        vnext = u - eta * nabla_ftilde_u
        unext = project(vnext, euclidean_distance2, bounds_prime)
        # assert is_in_convex_body(xnext)
        U[t, :] = unext

    return U

def online_gradient_descent(u0: jnp.ndarray) -> jnp.ndarray:
    U = np.zeros((T, d))
    U[0, :] = u0

    assert is_in_convex_body(U[0, :], bounds_prime)

    for t in tqdm.tqdm(range(1, T)):
        u = U[t-1, :]
        nabla_ftilde_u = jax.grad(ftilde)(u)
        vnext = u - eta * nabla_ftilde_u
        unext = project(vnext, euclidean_distance2, bounds_prime)
        # assert is_in_convex_body(xnext)
        U[t, :] = unext

    return U

def bisection_method(a0: jnp.ndarray, b0: jnp.ndarray) -> jnp.ndarray:
    Z = np.zeros((T, d))
    z0 = (a0 + b0) / 2
    Z[0, :] = z0
    a = a0
    b = b0
    
    assert f(a) < 0
    assert f(b) > 0
        
    for t in tqdm.tqdm(range(1, T)):
        z = Z[t - 1, :]
        if f(z) < 0:
            a = z
        elif f(z) > 0:
            b = z
        else:
            return Z
        
        znext = (a + b) / 2
        Z[t, :] = znext
    
    return Z

def plot_loss():
    x, y = np.meshgrid(np.linspace(-0.125, 1.125, 20),
                       np.linspace(-0.125, 1.125, 20))

    def surface_function(x, y):
        return f(jnp.array([x, y]))

    plt.contourf(x, y, np.vectorize(surface_function)(x, y),
                 levels=20, cmap='BuPu')
    plt.colorbar(label='Loss')
    plt.xlim(-0.125, 1.125)
    plt.ylim(-0.125, 1.125)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.gca().set_aspect(1)

sanity_check()

grid_res = 5
x0 = np.reshape(jnp.meshgrid(np.linspace(0.1, 0.9, grid_res), np.linspace(0.1, 0.9, grid_res)), [2, grid_res**2])
u0 = np.reshape(jnp.meshgrid(np.linspace(-2.3, -0.1, grid_res), np.linspace(-2.3, -0.1, grid_res)), [2, grid_res**2])
a0 = np.reshape(jnp.meshgrid(np.linspace(-4/3, -1/3, grid_res), np.linspace(-4/3, -1/3, grid_res)), [2, grid_res**2])
b0 = np.reshape(jnp.meshgrid(np.linspace(1/7, 8/7, grid_res), jnp.linspace(1/7, 8/7, grid_res)), [2, grid_res**2])

X = np.zeros([T, 2,  grid_res**2])
U = np.zeros([T, 2, grid_res**2])
Y = np.zeros([T, 2, grid_res**2])
Z = np.zeros([T, 2, grid_res**2])
x0_plot = np.array([0.5, 0.7])
eps = 1e-8
k = -1

save_flag = True

for nn in range(grid_res**2):
    if nn == 17: 
        print("sanity checking")
        
    X[:, :, nn] = online_mirror_descent(x0[:, nn])
    U[:, :, nn] = online_gradient_descent(u0[:, nn])
    Y[:, :, nn] = jax.vmap(q)(U[:, :, nn])
    Z[:, :, nn] = bisection_method(a0[:, nn], b0[:, nn])
    
    if jnp.sum(jnp.abs(x0[:, nn] - x0_plot)) < eps:
        fig, ax = plt.subplots()
        plt.plot(X[:, 0, nn], 'k-', label = 'x_1')
        plt.plot(X[:, 1, nn], 'k--', label = 'x_2')
        plt.plot(U[:, 0, nn], 'r-', label='u_1')
        plt.plot(U[:, 1, nn], 'r--', label = 'u_2')
        plt.plot(Z[:, 0, nn], 'b-', label='z_1')
        plt.plot(Z[:, 1, nn], 'b--', label='z_2')
        plt.xlabel('Iterations')
        plt.ylabel('State variable')
        plt.legend()
        if save_flag:
            fig.savefig('Figures/OMD_OGD_BM_tan_trajectory.png')        
        plt.show()
        
        L_X = np.zeros(T)
        L_U = np.zeros(T)
        L_Z = np.zeros(T)
        
        for tt in range(T):
            L_X[tt] = f(X[tt, :, nn])
            L_U[tt] = f(q(U[tt, :, nn]))
            L_Z[tt] = np.abs(f(Z[tt, :, nn]))
        
        fig, ax = plt.subplots()
        plt.plot(L_X, 'k-', label = 'OMD')
        plt.plot(L_U, 'r-', label = 'OGD')
        plt.plot(L_Z, 'b-', label = 'BM')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        if save_flag:
            fig.savefig('Figures/OMD_OGD_BM_tan_loss.png')        
        plt.show()

# Time delaying observables
n_delays = 4
X = time_delay(X, n_delays)
U = time_delay(U, n_delays)
Z = time_delay(Z, n_delays)

# Constructing data matrices
X, X_prime = data_matrices(X)
U, U_prime = data_matrices(U)
Z, Z_prime = data_matrices(Z)

mirrorDescentEigs_all, _ = dmd(X.T, X_prime.T, k)
gradientDescentEigs_all, _ = dmd(U.T, U_prime.T, k)
bisectionMethodEigs_all, _ = dmd(Z.T, Z_prime.T, k)

if save_flag:
    np.save('Results/mirror_descent_eigs.npy', mirrorDescentEigs_all)
    np.save('Results/gradient_descent_eigs.npy', gradientDescentEigs_all)
    np.save('Results/bisection_method_eigs.npy', bisectionMethodEigs_all)

# Printing eigenvalues
print(np.sort(mirrorDescentEigs_all))
print(np.sort(gradientDescentEigs_all))

# Plotting
unit_circle_x = np.sin(np.arange(0, 2 * np.pi, 0.01))
unit_circle_y = np.cos(np.arange(0, 2 * np.pi, 0.01))

fig, ax = plt.subplots()
plt.plot(unit_circle_x, unit_circle_y, 'k--')
plt.plot(np.real(mirrorDescentEigs_all), np.imag(mirrorDescentEigs_all), 'ko', label='OMD')
plt.plot(np.real(gradientDescentEigs_all), np.imag(gradientDescentEigs_all), 'r^', label='OGD')
plt.plot(np.real(bisectionMethodEigs_all), np.imag(bisectionMethodEigs_all), 'b*', label='BM')
plt.xlabel('Real($\lambda$)')
plt.ylabel('Imag($\lambda$)')
plt.xlim([-1.25, 1.25])
plt.ylim([-1.25, 1.25])
ax.set_aspect('equal', 'box')
plt.legend()
if save_flag:
    fig.savefig('Figures/OMD_OGD_BM_tan_spectra.png') 
      
fig, ax = plt.subplots()
plt.plot(np.real(mirrorDescentEigs_all), np.imag(mirrorDescentEigs_all), 'ko', label='OMD')
plt.plot(np.real(gradientDescentEigs_all), np.imag(gradientDescentEigs_all), 'r^', label='OGD')
plt.plot(np.real(bisectionMethodEigs_all), np.imag(bisectionMethodEigs_all), 'b*', label='BM')
plt.xlabel('Real($\lambda$)')
plt.ylabel('Imag($\lambda$)')
plt.xlim([0.5, 1.1])
plt.ylim([-0.05, 0.05])
#ax.set_aspect('equal', 'box')
plt.legend()
if save_flag:
    fig.savefig('Figures/OMD_OGD_BM_tan_spectra_zoomed.png') 
    



