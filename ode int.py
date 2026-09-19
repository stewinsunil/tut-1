import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ODE: y' = -k * y, with y(0) = 1
def rhs(t, y, k):
    return -k * y

k = 0.7
t_span = (0, 10)
y0 = [1.0]

# Solve the ODE
sol = solve_ivp(
    rhs,
    t_span,
    y0,
    args=(k,),
    t_eval=np.linspace(0, 10, 500),
    dense_output=True
)

# Plot the result
plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], "b-", linewidth=2, label="y(t)")
plt.scatter([0], [1.0], color="red", label="initial condition")
plt.xlabel("t")
plt.ylabel("y")
plt.title("Single ODE: y' = -k y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()