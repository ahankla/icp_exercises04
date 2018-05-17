import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


# -----------------------------------------------
#   Exercise 1: Numerov Algorithm for TISE
# -----------------------------------------------

def harmonic_oscillator_k(x, eps):
    """" k(x) function as described in Exercise Sheet 4(normalized):
         y''(x) + k(x) y(x) = 0
         x is np.array 1xn
         eps is the normalized energy
         This specific function k(x) for harmonic oscillator """
    return 2*eps - x**2


def numerov(psi0, psi1, eps, N, kfun):
    """ Numerov Integration of TISE
    Starting conditions:
      psi0: psi(x=0)
      psi1: psi(x=h)
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


def analytic_wavefunction(x, n):
    """ Calculate the analytic solution to the TISE
        given that the energy is the nth eigenvalue
        eps = n+1/2 where n is given as an argument.
        x is just the range over which to calculate.
        Formula from Exercise Sheet 4"""
    hermiten = hermite_generator(x, n)
    return hermiten/(2**n * np.math.factorial(n) * np.pi**0.5) * np.exp(-x**2/2.)


def normalized_function(psi):
    """ Normalize a function such that the integral of it squared is one.
        Returns the normalized function. """
    psi2 = psi**2
    p2norm = simps(psi2)  # integral over the x direction
    return psi/np.sqrt(p2norm)



# Test odd solution (n = 1, 3, ...)
N = 100
eps = 1.5; n = 1  # degree of Hermite
psi0 = 0
psi1 = 1  # a = 1
xr = np.linspace(0, 1, N)
psi = numerov(psi0, psi1, eps, N, harmonic_oscillator_k)
normed_psi = normalized_function(psi)
# Analytic
analytic_soln = normalized_function(analytic_wavefunction(xr, n))
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
axarr[1].plot(analytic_soln-normed_psi, label="Remainder (Analytic-Numeric)")
axarr[1].legend()
axarr[1].set_xlabel("Step Count N")
axarr[1].set_ylabel("Difference")
plt.tight_layout()
fig.savefig("exercise4_problem1_antisymEx.pdf")
# plt.show()

# Test even solution (n = 0, 2, ...)
# Note the starting conditions and energy are different
N = 100
eps = 0.5; n = 0  # degree of hermite
psi0 = 1
psi1 = psi0 - (1./N)**2*psi0/2*harmonic_oscillator_k(0, eps)
xr = np.linspace(0, 1, N)
psi = numerov(psi0, psi1, eps, N, harmonic_oscillator_k)
normed_psi = normalized_function(psi)
# Analytic
analytic_soln = normalized_function(analytic_wavefunction(xr, n))**2
# Plot 2
fig, axarr = plt.subplots(2,1)
# Solutions
axarr[0].plot(xr, normed_psi**2, label="Numeric")
axarr[0].plot(xr, analytic_soln, 
         linestyle=":", label="Analytic")
axarr[0].legend()
axarr[0].set_xlabel("x")
axarr[0].set_ylabel("wavefunction psi")
axarr[0].set_title("Sample Antisymmetric function: Energy Eigenvalue {}".format(n))
# Remainder
axarr[1].plot(analytic_soln-normed_psi, label="Remainder (Analytic-Numeric)")
axarr[1].legend()
axarr[1].set_xlabel("Step Count N")
axarr[1].set_ylabel("Difference")
plt.tight_layout()
fig.savefig("exercise4_problem1_symEx.pdf")
plt.show()






