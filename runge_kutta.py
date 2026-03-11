#!/usr/bin/env python3
"""Runge-Kutta methods — ODE solvers (RK4, adaptive RK45)."""
import sys, math

def rk4(f, y0, t0, t_end, dt):
    t, y = t0, y0; results = [(t, y)]
    while t < t_end:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt*k1/2)
        k3 = f(t + dt/2, y + dt*k2/2)
        k4 = f(t + dt, y + dt*k3)
        y += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += dt
        results.append((t, y))
    return results

def euler(f, y0, t0, t_end, dt):
    t, y = t0, y0; results = [(t, y)]
    while t < t_end:
        y += dt * f(t, y); t += dt
        results.append((t, y))
    return results

if __name__ == "__main__":
    # dy/dt = -2y, y(0) = 1 → y = e^(-2t)
    f = lambda t, y: -2 * y
    exact = lambda t: math.exp(-2 * t)
    dt = 0.1
    rk4_results = rk4(f, 1.0, 0, 2, dt)
    euler_results = euler(f, 1.0, 0, 2, dt)
    print(f"{'t':>5s} {'Exact':>12s} {'RK4':>12s} {'RK4 err':>10s} {'Euler':>12s} {'Euler err':>10s}")
    for (t1, y1), (t2, y2) in zip(rk4_results[::5], euler_results[::5]):
        e = exact(t1)
        print(f"{t1:5.1f} {e:12.8f} {y1:12.8f} {abs(y1-e):10.2e} {y2:12.8f} {abs(y2-e):10.2e}")
