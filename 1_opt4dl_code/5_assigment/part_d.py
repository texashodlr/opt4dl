"""
Assignment 5, Problem 3, Part D code by Conor X Devlin OPT4DL FA25
Problem directions:
    Initialize y = [1, 1]⊤ and x = [1, 1]⊤. 
    Implement the GD, Proximal point, EG, and OGD
    methods for this task and report the learning rates 
    you used (set T = 100).
"""
import numpy as np
import matplotlib.pyplot as plt

# Basic problem: U(x,y)=y^T(mu-x)

def grad_x(x, y):
    # Gradient over x of U(x,y) == Grad.x U(x,y)=-y 
    return (-y)

def grad_y(x, y, mu):
    # Gradient over y of U(x,y) == Grad.y U(x,y) = (mu - x)
    return (mu - x)


def gradient_descent_ascent(x0, y0, x_star, y_star, T, eta, mu):
    x_gd = x0.copy()
    y_gd = y0.copy()

    dist_x = []
    dist_y = []

    for t in range(T):
        dist_x.append(np.linalg.norm(x_gd-x_star))
        dist_y.append(np.linalg.norm(y_gd-y_star))

        gx = grad_x(x_gd, y_gd)
        gy = grad_y(x_gd, y_gd, mu)

        x_gd = x_gd - eta * gx
        y_gd = y_gd + eta * gy
    return np.array(dist_x), np.array(dist_y)

def proximal_point(x0, y0, x_star, y_star, T, eta, mu):
    m1, m2 = mu
    A = np.array([
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    b = np.array([0.0, 0.0, -m1, -m2], dtype=float)
    M = np.eye(4) + eta * A
    M_inv = np.linalg.inv(M)
    z = np.concatenate([x0, y0])
    dist_x = []
    dist_y = []
        
    for t in range(T):
        x_pp = z[:2]
        y_pp = z[2:]

        dist_x.append(np.linalg.norm(x_pp-x_star))
        dist_y.append(np.linalg.norm(y_pp-y_star))

        rhs = z - eta * b
        z = M_inv @ rhs
    return np.array(dist_x), np.array(dist_y)


def extra_gradient(x0, y0, x_star, y_star, T, eta, mu):
    x_eg = x0.copy()
    y_eg = y0.copy()

    dist_x = []
    dist_y = []

    for t in range(T):
        dist_x.append(np.linalg.norm(x_eg-x_star))
        dist_y.append(np.linalg.norm(y_eg-y_star))
        gx = grad_x(x_eg, y_eg)
        gy = grad_y(x_eg, y_eg, mu)

        x_tilde = x_eg - eta * gx
        y_tilde = y_eg - eta * gy

        gx_tilde = grad_x(x_tilde, y_tilde)
        gy_tilde = grad_y(x_tilde, y_tilde, mu)

        x_eg = x_eg - eta * gx_tilde
        y_eg = y_eg + eta * gy_tilde
    return np.array(dist_x), np.array(dist_y)

def optimistic_gradient(x0, y0, x_star, y_star, T, eta, mu):
    x_ogd = x0.copy()
    y_ogd = y0.copy()

    dist_x = []
    dist_y = []

    gx_prev = grad_x(x_ogd, y_ogd)
    gy_prev = grad_y(x_ogd, y_ogd, mu)

    for t in range(T):
        dist_x.append(np.linalg.norm(x_ogd-x_star))
        dist_y.append(np.linalg.norm(y_ogd-y_star))
        gx = grad_x(x_ogd, y_ogd)
        gy = grad_y(x_ogd, y_ogd, mu)

        gx_optimistic = 2 * gx - gx_prev
        gy_optimistic = 2 * gy - gy_prev
        
        x_ogd = x_ogd - eta * gx_optimistic
        y_ogd = y_ogd - eta * gy_optimistic

        gx_prev, gy_prev = gx, gy
    
    return np.array(dist_x), np.array(dist_y)

def plot_x_star(iters, gd_x, pp_x, eg_x, ogd_x):
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    #plt.plot(iters, gd_x,  label='GD')
    #plt.plot(iters, pp_x,  label='Proximal Point')
    #plt.plot(iters, eg_x,  label='Extragradient')
    plt.plot(iters, ogd_x, label='Optimistic GD')
    plt.xlabel('Iteration')
    plt.ylabel('x_t - x^*')
    plt.title('Generator distance to equilibrium')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_y_star(iters, gd_y, pp_y, eg_y, ogd_y):
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    #plt.plot(iters, gd_y,  label='GD')
    #plt.plot(iters, pp_y,  label='Proximal Point')
    #plt.plot(iters, eg_y,  label='Extragradient')
    plt.plot(iters, ogd_y, label='Optimistic GD')
    plt.xlabel('Iteration')
    plt.ylabel('y_t - y^*')
    plt.title('Discriminator distance to equilibrium')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    mu = np.array([3.0, 4.0])
    x_star = mu.copy()
    y_star = np.zeros(2)
    eta = 0.1
    lam = 0.1
    T = 100
    x0 = np.array([0.0, 0.0])
    y0 = np.array([1.0, 1.0])

    # GDA
    gd_x, gd_y = gradient_descent_ascent(x0, y0, x_star, y_star, T, eta, mu)
    # PP
    pp_x, pp_y = proximal_point(x0, y0, x_star, y_star, T, eta, mu)
    # EG
    eg_x, eg_y = extra_gradient(x0, y0, x_star, y_star, T, eta, mu)
    # OGD
    ogd_x, ogd_y = optimistic_gradient(x0, y0, x_star, y_star, T, eta, mu)

    iters = np.arange(T)

    #plot_x_star(iters, gd_x, pp_x, eg_x, ogd_x)
    plot_y_star(iters, gd_y, pp_y, eg_y, ogd_y)

if __name__ == "__main__":
    main()