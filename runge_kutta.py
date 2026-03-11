#!/usr/bin/env python3
"""runge_kutta - RK4/RK45 ODE solvers with adaptive step size.

Usage: python runge_kutta.py [--system lorenz|pendulum|vdp|orbit]
"""
import sys, math

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t+h/2, [yi+h/2*k1i for yi,k1i in zip(y,k1)])
    k3 = f(t+h/2, [yi+h/2*k2i for yi,k2i in zip(y,k2)])
    k4 = f(t+h, [yi+h*k3i for yi,k3i in zip(y,k3)])
    return [yi + h/6*(k1i+2*k2i+2*k3i+k4i) for yi,k1i,k2i,k3i,k4i in zip(y,k1,k2,k3,k4)]

def rk45_step(f, t, y, h):
    """Dormand-Prince RK45 with error estimate."""
    a = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    b = [[],[1/5],[3/40,9/40],[44/45,-56/15,32/9],
         [19372/6561,-25360/2187,64448/6561,-212/729],
         [9017/3168,-355/33,46732/5247,49/176,-5103/18656],
         [35/384,0,500/1113,125/192,-2187/6784,11/84]]
    c4 = [5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40]
    c5 = [35/384,0,500/1113,125/192,-2187/6784,11/84,0]
    n = len(y)
    k = [None]*7
    k[0] = f(t, y)
    for s in range(1,7):
        yy = [yi + h*sum(b[s][j]*k[j][i] for j in range(s)) for i,yi in enumerate(y)]
        k[s] = f(t+a[s]*h, yy)
    y5 = [yi+h*sum(c5[j]*k[j][i] for j in range(7)) for i,yi in enumerate(y)]
    y4 = [yi+h*sum(c4[j]*k[j][i] for j in range(7)) for i,yi in enumerate(y)]
    err = max(abs(a-b) for a,b in zip(y4,y5))
    return y5, err

def solve_ode(f, y0, t_span, h=0.01, adaptive=False, tol=1e-6):
    t, y = t_span[0], list(y0)
    trajectory = [(t, list(y))]
    while t < t_span[1]:
        if adaptive:
            while True:
                y_new, err = rk45_step(f, t, y, h)
                if err < tol or h < 1e-12:
                    break
                h = max(h*0.5, 1e-12)
            t += h
            y = y_new
            if err > 0:
                h = min(h * min(5, (tol/err)**0.2), t_span[1]-t) if t < t_span[1] else h
        else:
            step = min(h, t_span[1]-t)
            y = rk4_step(f, t, y, step)
            t += step
        trajectory.append((t, list(y)))
    return trajectory

# Example systems
def lorenz(t, y, sigma=10, rho=28, beta=8/3):
    return [sigma*(y[1]-y[0]), y[0]*(rho-y[2])-y[1], y[0]*y[1]-beta*y[2]]

def pendulum(t, y, g=9.81, L=1):
    return [y[1], -g/L*math.sin(y[0])]

def van_der_pol(t, y, mu=2):
    return [y[1], mu*(1-y[0]**2)*y[1]-y[0]]

def orbit(t, y):
    r = math.sqrt(y[0]**2+y[1]**2)
    return [y[2], y[3], -y[0]/r**3, -y[1]/r**3]

def main():
    system = "lorenz"
    args = sys.argv[1:]
    for i,a in enumerate(args):
        if a == "--system" and i+1 < len(args): system = args[i+1]

    systems = {
        "lorenz": (lorenz, [1,1,1], (0,25), 0.01),
        "pendulum": (pendulum, [math.pi/4, 0], (0,10), 0.01),
        "vdp": (van_der_pol, [2,0], (0,20), 0.01),
        "orbit": (orbit, [1,0,0,1], (0,6.3), 0.01),
    }
    f, y0, span, h = systems.get(system, systems["lorenz"])
    print(f"Solving {system} system with RK4...")
    traj = solve_ode(f, y0, span, h)
    print(f"Steps: {len(traj)}")
    print(f"Initial: t={traj[0][0]:.2f}, y={[f'{v:.4f}' for v in traj[0][1]]}")
    print(f"Final:   t={traj[-1][0]:.2f}, y={[f'{v:.4f}' for v in traj[-1][1]]}")
    # Sample points
    step = max(1, len(traj)//10)
    print("\nTrajectory samples:")
    for i in range(0, len(traj), step):
        t, y = traj[i]
        print(f"  t={t:6.2f}: {[f'{v:8.4f}' for v in y]}")

    # Adaptive solve
    print(f"\nSolving {system} with adaptive RK45...")
    traj2 = solve_ode(f, y0, span, h=0.1, adaptive=True)
    print(f"Adaptive steps: {len(traj2)}")
    print(f"Final: t={traj2[-1][0]:.2f}, y={[f'{v:.4f}' for v in traj2[-1][1]]}")

if __name__ == "__main__":
    main()
