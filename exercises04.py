import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


# -----------------------------------------------
#   Exercise 1: Numerov Algorithm for TISE
# -----------------------------------------------

def numerov(psi0, psi1, eps, N, kfun):
    """ Numerov Integration of TISE
    Starting conditions:
      psi0: psi(x=0)
      psi1: psi(x=h)
    eps: normalized energy
    N: number of bins in x direction. Assume start at x=0, go to x = n
    kfun: The function k(x), as in y''(x) + k(x) y(x) = 0
    """

    # The equation has been normalized such that x spans from 0 to 1
    xrange = np.linspace(0, 1, N)

    # Initialize
    psi = np.zeros(N)
    psi[0] = psi0
    psi[1] = psi1

    # Discretize k based on function k(x)
    # Only need to calculate once! Neat.
    kval = kfun(xrange, eps)

    # Step size
    h = 1./N

    # Integrate using the Numerov Algorithm, to O(h^6)
    for i in range(2, N):
        rhs = 2*(1 - (5./12.)*(h**2)*kval[i-1])*psi[i-1] \
              - (1 + (1./12.)*(h**2)*kval[i-2])*psi[i-2]
        lhs = (1 + (1./12.)*(h**2)*kval[i])
        psi[i] = rhs / lhs

    return psi

# kfunc's
def harmonic_oscillator_k(x, eps):
    """" k(x) function as described in Exercise Sheet 4(normalized):
         y''(x) + k(x) y(x) = 0
         x: np.array 1xn
         eps: normalized energy
         This specific function k(x) for harmonic oscillator """
    return 2*eps - x**2

# kfunc's
def eps_mins_x_k(x, eps):
    """" psi''(x) + k(x) psi(x) = 0
         k(x) = (eps-x)
         x: np.array 1xn
         eps: normalized energy """
    #print(np.count_nonzero(x[x<0]), np.max(x), np.min(x))
    ## Never zero...
    x[x<0] = np.inf  # mirror, since x prop to V(z), as V(z)->inf, x->inf
    return eps - x

# analytic tools
def hermite_generator(x, n):
    """ Calculate the nth Hermite polynomial recursively.
        Formula from Exercise Sheet 4. """
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        return 2*x*hermite_generator(x, n-1) \
               -2*(n-1)*hermite_generator(x, n-2)

# analytic tools
def analytic_wavefunction_harmonic(x, n):
    """ Calculate the analytic solution to the TISE
        given that the energy is the nth eigenvalue
        eps = n+1/2 where n is given as an argument.
        x is just the range over which to calculate.
        Formula from Exercise Sheet 4 """
    hermiten = hermite_generator(x, n)
    return hermiten/(2**n * np.math.factorial(n) * np.pi**0.5) * np.exp(-x**2/2.)

def analytic_wavefunction_neutron(x, n):
    """  """
    return 


# integral normalization
def normalized_function(psi):
    """ Normalize a function such that the integral of it squared is one.
        Returns the normalized function. """
    psi2 = psi**2
    p2norm = simps(psi2)  # integral over the x direction
    return psi/np.sqrt(p2norm)



# Test odd solution (n = 1, 3, ...)
N = 100  # Steps
xr = np.linspace(0, 1, N)  # Normalize range
n = 1  # degree of Hermite
eps = 1.5
# First values
psi0 = 0
psi1 = 1  # a = 1
# Numeric Solution (normalized)
psi = numerov(psi0, psi1, eps, N, harmonic_oscillator_k)
# Analytic Solution (normalized)
analytic_soln = analytic_wavefunction_harmonic(xr, n)
# Normalize: x, numeric, analytic
normed_psi = normalized_function(psi)
analytic_soln = normalized_function(analytic_soln)
# Plot 1
fig, axarr = plt.subplots(2,1)
# Solutions
axarr[0].plot(xr, normed_psi, label="Numeric")
axarr[0].plot(xr, analytic_soln, 
         linestyle=":", label="Analytic")
axarr[0].legend()
axarr[0].set_xlabel("x")
axarr[0].set_ylabel("wavefunction psi")
axarr[0].set_title("Sample Antisymmetric function: Energy Eigenvalue {}".format(n))
# Remainder
axarr[1].plot((analytic_soln-normed_psi)/analytic_soln, label="Percentage Error (Analytic-Numeric)/Analytic")
axarr[1].legend()
axarr[1].set_xlabel("Step Count N")
axarr[1].set_ylabel("Percentage Error")
plt.tight_layout()
fig.savefig("exercise4_problem1_antisymEx.pdf")
# plt.show()

# Test even solution (n = 0, 2, ...)
# Note the starting conditions and energy are different
N = 100
xr = np.linspace(0, 1, N)  # Normalize range
n = 0  # degree of hermite
eps = 0.5
# First Values
psi0 = 1
psi1 = psi0 - (1./N)**2*psi0/2*harmonic_oscillator_k(0, eps)
# Numeric Solution
psi = numerov(psi0, psi1, eps, N, harmonic_oscillator_k)
# Analytic Solution
analytic_soln = analytic_wavefunction_harmonic(xr, n)
# Normalize: x, numeric, analytic
normed_psi = normalized_function(psi)**2  #
analytic_soln = normalized_function(analytic_soln)**2
# Plot 2
fig, axarr = plt.subplots(2,1)
# Plot 2: Solutions
axarr[0].plot(xr, normed_psi, label="Numeric")
axarr[0].plot(xr, analytic_soln, 
         linestyle=":", label="Analytic")
axarr[0].legend()
axarr[0].set_xlabel("x")
axarr[0].set_ylabel("wavefunction psi")
axarr[0].set_title("Sample Antisymmetric function: Energy Eigenvalue {}".format(n))
# Plot 2: Remainder
axarr[1].plot((analytic_soln-normed_psi)/analytic_soln, label="Percentage Error (Analytic-Numeric)/Analytic")
axarr[1].legend()
axarr[1].set_xlabel("Step Count N")
axarr[1].set_ylabel("Percentage Error")
plt.tight_layout()
fig.savefig("exercise4_problem1_symEx.pdf")






# -----------------------------------------------
#   Exercise 2: Neutrons in the gravitational field
# -----------------------------------------------
# Finding stationary states in the gravitational field of Earth
# https://www.physi.uni-heidelberg.de/Publications/dipl_krantz.pdf


## Part 1:

# a) Solve using Numerov
# Specify length & energy units:
#   eps = E*2m/hbar
#   x = (2m^2 g/hbar)*z = (2m/hbar) V(x)
# Or:
#   R = (hbar^2/(sm^2 g)^1/3
#   x = z/R
#   eps = E/(mgR)
# General
N = 10000
## Numeric Solutions
eps = 20
x_last = 1000  # x_last >> eps
psi0 = 0  # trivial first solution
psi1 = 1  # 
psi = numerov(psi0, psi1, eps, N, eps_mins_x_k)
## Analytic
n = 2
xr = np.linspace(0, x_last, N)
#analytic_soln = analytic_wavefunction_neutron(xr, n)
# Normalize 
normed_psi = normalized_function(psi)
analytic_soln = normalized_function(analytic_soln)
# b) Plot solution well into classically forbidden zone
#    that is:  from x=0 to x>>eps  for some values for eps
# Plot 3
fig, axarr = plt.subplots(2,1)
# Plot 3: Solutions
axarr[0].plot(xr, normed_psi, label="Numeric")
#axarr[0].plot(xr, analytic_soln, 
#         linestyle=":", label="Analytic")
axarr[0].legend()
axarr[0].set_xlabel("x")
axarr[0].set_ylabel("wavefunction psi")
axarr[0].set_title("Neutron in Gravitational Field: $\epsilon$={}".format(eps))
# Plot 3: Remainder
#axarr[1].plot(analytic_soln-normed_psi, label="Remainder (Analytic-Numeric)")
#axarr[1].legend()
axarr[1].set_xlabel("Step Count N")
axarr[1].set_ylabel("Difference")
plt.tight_layout()
fig.savefig("exercise4_problem2_numIntegration.pdf")


# c) for large x: does it approach +/- inf?


# d) Plot two solutions (for two values of eps), one of each

# plotting a bunch, all with the same shape
eps_list = [-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 200.0]
fig, axarr = plt.subplots(1,1, figsize=(12,7))
for eps in eps_list:
    normed_psi = normalized_function(numerov(psi0, psi1, eps, N, eps_mins_x_k))
    axarr.plot(xr, normed_psi, label="Numeric $\epsilon$={}".format(eps))
axarr.legend()
axarr.set_xlabel("x")
axarr.set_ylabel("wavefunction psi")
axarr.set_title("Neutron in Gravitational Field: $\epsilon$={}".format(eps_list))
plt.tight_layout()
fig.savefig("exercise4_problem2_varyEps.pdf")

## Part 2:
# eigenvalues, eps_n, belong to normalizable eigenfunctions
# with psi(x)->0 for x->inf
# therefore, increasing eps_n => psi(x) changes sign for x->inf
# a) use this property and eps_n of the first 3 bound states to 2 after comma decimals


plt.show()