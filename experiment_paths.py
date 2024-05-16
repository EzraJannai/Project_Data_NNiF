import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import enum
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# Sample is an array of the form [m, tau, r, rho, kappa, gamma, vbar, v0]
# Results is an array of the form [m, tau, rho, kappa, gamma, vbar, v0, r, option price, implied volatility]

# Define the option type as an enum for clarity
class OptionType(enum.Enum):
    CALL = 1
    PUT = -1

# Define parameter ranges
param_ranges = {
    'm': (0.8, 1.2),
    'tau': (0.4, 2.4),
    'rho': (-0.9, 0.0),
    'kappa': (0, 3.0),
    'gamma': (0.01, 0.8),
    'vbar': (0.01, 0.5),
    'v0': (0.05, 0.5)
}


# Characteristic function for the Heston model
def ChFHestonModel(r, tau, kappa, gamma, vbar, v0, rho):
    i = 1j
    D1 = lambda u: np.sqrt((kappa - gamma * rho * i * u)**2 + (u**2 + i * u) * gamma**2)
    g = lambda u: (kappa - gamma * rho * i * u - D1(u)) / (kappa - gamma * rho * i * u + D1(u))
    C = lambda u: (1.0 - np.exp(-D1(u) * tau)) / (gamma**2 * (1 - g(u) * np.exp(-D1(u) * tau))) * (kappa - gamma * rho * i * u - D1(u))
    A = lambda u: r * i * u * tau + kappa * vbar * tau / gamma**2 * (kappa - gamma * rho * i * u - D1(u)) \
        - 2 * kappa * vbar / gamma**2 * np.log((1 - g(u) * np.exp(-D1(u) * tau)) / (1 - g(u)))
    return lambda u: np.exp(A(u) + C(u) * v0)

# COS method for option pricing
def CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L):
    i = 1j
    K = np.array(K).reshape([-1, 1])
    x0 = np.log(S0 / K)
    a = 0 - L * np.sqrt(tau)
    b = 0 + L * np.sqrt(tau)
    k = np.linspace(0, N-1, N).reshape([N, 1])
    u = k * np.pi / (b - a)
    mat = np.exp(i * np.outer((x0 - a), u))
    H_k = CallPutCoefficients(CP, a, b, k)
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = np.exp(-r * tau) * K * np.real(mat @ temp)
    return value

# Determine coefficients for put and call prices
def CallPutCoefficients(CP, a, b, k):
    if CP == OptionType.CALL:
        c = 0.0
        d = b
    elif CP == OptionType.PUT:
        c = a
        d = 0.0
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    chi = 1.0 / (1.0 + (k * np.pi / (b - a))**2)
    expr1 = np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * (np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c))
    chi *= (expr1 + expr2)
    H_k = 2.0 / (b - a) * (chi - psi if CP == OptionType.CALL else -chi + psi)
    return H_k

# Generate Latin Hypercube Samples
def generate_lhs_samples(param_ranges, num_samples):
    sampler = qmc.LatinHypercube(d=len(param_ranges), seed=2)
    sample = sampler.random(n=num_samples)
    bounds = np.array(list(param_ranges.values()))
    lhs_samples = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    return lhs_samples

# Pricing using the COS method with parameter samples
def cos_method_option_pricing(sample, S0=1):
    m, tau, rho, kappa, gamma, vbar, v0 = sample
    r = 0.03  # Set r to 0.03
    K = S0 / m  # Calculate strike price based on moneyness and stock price
    K_range = np.array([K])  # Use a single strike price derived from moneyness
    N = 4096*2
    L = 1
    cf = ChFHestonModel(r, tau, kappa, gamma, vbar, v0, rho)
    CP = OptionType.CALL
    prices = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K_range, N, L)
    return prices.mean()

def find_implied_volatility(S, K, T, r, option_price):
    func = lambda sigma: black_scholes_call(S, K, T, r, sigma) - option_price
    try:
        return brentq(func, 1e-6, 3.0)
    except ValueError:
        return np.nan

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

num_samples = 100000
samples = generate_lhs_samples(param_ranges, num_samples)
results = []
counter = 0
for sample in samples:
    option_price = cos_method_option_pricing(sample)
    implied_volatility = find_implied_volatility(1, 1/sample[0], sample[1], 0.03, option_price)  # Set r to 0.03 here too
    results.append(np.concatenate((sample, [0.03, option_price, implied_volatility])))  # Add r to results
    counter += 1
    if (counter+1) % 1000 == 0:
        print(f'sample [{counter}]')

results = np.array(results)
np.random.shuffle(results)
train_data, validation_data, test_data = np.split(results, [int(.8 * len(results)), int(.9 * len(results))])

import os

# Save results to CSV
script_dir = os.getcwd()  # Get the current working directory
np.savetxt(os.path.join(script_dir, "train_data_10.csv"), train_data, delimiter=",")
np.savetxt(os.path.join(script_dir, "validation_data_10.csv"), validation_data, delimiter=",")
np.savetxt(os.path.join(script_dir, "test_data_10.csv"), test_data, delimiter=",")

print("Data generation and saving complete.")
