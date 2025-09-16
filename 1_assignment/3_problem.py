# HW 1 | Question 3 | Part C | Conor X Devlin

import math
import matplotlib.pyplot as plt

root_3 = math.sqrt(3.0)

def Loss(w):
    L = (1.0/3.0)*(w**3) - w
    return L

def gradient_Loss(w):
    g_L = w**2 - 1.0
    return g_L

def product_operator(w_t, low_bound=-root_3, high_bound=root_3):
    prod_op = min(high_bound, max(low_bound, w_t))
    #print(f"prod_op = {prod_op}\n")
    return prod_op

def proj_gd_step(w, eta):
    w_t_1 = w - (eta * gradient_Loss(w))
    return product_operator(w_t_1)

def run_proj_gd(w_init, eta=(1.0/(2.0*root_3)),eps=1e-10, eps_grad=1e-10, max_iter=100000):
    #print(f"W_init = {w_init}")
    #print(f"eta = {eta}")
    ws = [w_init]
    Ls = [Loss(w_init)]
    iters = 0
    while iters < max_iter:
        w_next = proj_gd_step(ws[-1], eta)
        ws.append(w_next)
        Ls.append(Loss(w_next))
        iters += 1

        dw = abs(ws[-1] - ws[-2])
        g = abs(gradient_Loss(ws[-1]))
        #print(f"dw = {dw}\n")
        #print(f"g  = {g}\n")
        x = (-root_3 < ws[-1] < root_3)
        if dw <= eps and (not x or g <= eps_grad):
            break
    
    wT = ws[-1]
    out = {
        "w0": w_init,
        "eta": eta,
        "iterations": iters,
        "w_final": wT,
        "L_final": Loss(wT),
        "grad_final": gradient_Loss(wT),
        "hit_boundary": (abs(wT) >= root_3 - 1e-15),
        "converged": (iters < max_iter)
    }

    print(f"\n=== PGD run (w0={w_init:+.6f}, eta={eta:.6f}) ===")
    print(f"Converged: {out['converged']}  in {out['iterations']} iters")
    print(f"w_T = {out['w_final']:+.12f}")
    print(f"L(w_T) = {out['L_final']:+.12f}")
    print(f"|gradL(w_T)| = {abs(out['grad_final']):.6e}")
    print(f"Hit boundary? {out['hit_boundary']}")
    return out, ws, Ls

def main():
    eta_prime = float(1.0 / (2.0*root_3))
    for eta in [0.01, 0.05, 0.1, 0.2, 0.3]:
        print(f"\n### PGD RUN ETA: {eta}###")
        for w_init in [-0.9, -1.1, -1.0]:
            run_proj_gd(w_init, eta=eta, eps=1e-10, eps_grad=1e-10, max_iter=100000)

main()

# Output with given eta (1.0/(2.0*root_3)): 
"""
=== PGD run (w0=-0.900000, eta=0.288675) ===
Converged: True  in 36 iters
w_T = +0.999999999975
L(w_T) = -0.666666666667
|gradL(w_T)| = 4.954170e-11
Hit boundary? False

=== PGD run (w0=-1.100000, eta=0.288675) ===
Converged: True  in 5 iters
w_T = -1.732050807569
L(w_T) = +0.000000000000
|gradL(w_T)| = 2.000000e+00
Hit boundary? True

=== PGD run (w0=-1.000000, eta=0.288675) ===
Converged: True  in 1 iters
w_T = -1.000000000000
L(w_T) = +0.666666666667
|gradL(w_T)| = 0.000000e+00
Hit boundary? False
"""

# Output with varied eta [0.01, 0.05, 0.1, 0.2, 0.3]:
"""

### PGD RUN ETA: 0.01###

=== PGD run (w0=-0.900000, eta=0.010000) ===
Converged: True  in 1358 iters
w_T = +0.999999999951
L(w_T) = -0.666666666667
|gradL(w_T)| = 9.821854e-11
Hit boundary? False

=== PGD run (w0=-1.100000, eta=0.010000) ===
Converged: True  in 89 iters
w_T = -1.732050807569
L(w_T) = +0.000000000000
|gradL(w_T)| = 2.000000e+00
Hit boundary? True

=== PGD run (w0=-1.000000, eta=0.010000) ===
Converged: True  in 1 iters
w_T = -1.000000000000
L(w_T) = +0.666666666667
|gradL(w_T)| = 0.000000e+00
Hit boundary? False

### PGD RUN ETA: 0.05###

=== PGD run (w0=-0.900000, eta=0.050000) ===
Converged: True  in 263 iters
w_T = +0.999999999952
L(w_T) = -0.666666666667
|gradL(w_T)| = 9.682188e-11
Hit boundary? False

=== PGD run (w0=-1.100000, eta=0.050000) ===
Converged: True  in 20 iters
w_T = -1.732050807569
L(w_T) = +0.000000000000
|gradL(w_T)| = 2.000000e+00
Hit boundary? True

=== PGD run (w0=-1.000000, eta=0.050000) ===
Converged: True  in 1 iters
w_T = -1.000000000000
L(w_T) = +0.666666666667
|gradL(w_T)| = 0.000000e+00
Hit boundary? False

### PGD RUN ETA: 0.1###

=== PGD run (w0=-0.900000, eta=0.100000) ===
Converged: True  in 126 iters
w_T = +0.999999999953
L(w_T) = -0.666666666667
|gradL(w_T)| = 9.351142e-11
Hit boundary? False

=== PGD run (w0=-1.100000, eta=0.100000) ===
Converged: True  in 11 iters
w_T = -1.732050807569
L(w_T) = +0.000000000000
|gradL(w_T)| = 2.000000e+00
Hit boundary? True

=== PGD run (w0=-1.000000, eta=0.100000) ===
Converged: True  in 1 iters
w_T = -1.000000000000
L(w_T) = +0.666666666667
|gradL(w_T)| = 0.000000e+00
Hit boundary? False

### PGD RUN ETA: 0.2###

=== PGD run (w0=-0.900000, eta=0.200000) ===
Converged: True  in 57 iters
w_T = +0.999999999955
L(w_T) = -0.666666666667
|gradL(w_T)| = 9.019607e-11
Hit boundary? False

=== PGD run (w0=-1.100000, eta=0.200000) ===
Converged: True  in 7 iters
w_T = -1.732050807569
L(w_T) = +0.000000000000
|gradL(w_T)| = 2.000000e+00
Hit boundary? True

=== PGD run (w0=-1.000000, eta=0.200000) ===
Converged: True  in 1 iters
w_T = -1.000000000000
L(w_T) = +0.666666666667
|gradL(w_T)| = 0.000000e+00
Hit boundary? False

### PGD RUN ETA: 0.3###

=== PGD run (w0=-0.900000, eta=0.300000) ===
Converged: True  in 34 iters
w_T = +0.999999999974
L(w_T) = -0.666666666667
|gradL(w_T)| = 5.278311e-11
Hit boundary? False

=== PGD run (w0=-1.100000, eta=0.300000) ===
Converged: True  in 5 iters
w_T = -1.732050807569
L(w_T) = +0.000000000000
|gradL(w_T)| = 2.000000e+00
Hit boundary? True

=== PGD run (w0=-1.000000, eta=0.300000) ===
Converged: True  in 1 iters
w_T = -1.000000000000
L(w_T) = +0.666666666667
|gradL(w_T)| = 0.000000e+00
Hit boundary? False

"""