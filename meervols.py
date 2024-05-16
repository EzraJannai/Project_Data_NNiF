import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import enum
from scipy.stats import norm
from scipy.optimize import brentq

# Define the option type as an enum for clarity
class OptionType(enum.Enum):
    CALL = 1
    PUT = -1

# Define parameter ranges
param_ranges = {
    'm': (0.6, 1.4),
    'tau': (0.05, 3.0),
    'r': (0.029, 0.03),
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

# Find the implied volatility using the Brent method
def find_implied_volatility(S, K, T, r, option_price):
    func = lambda sigma: black_scholes_call(S, K, T, r, sigma) - option_price
    try:
        return brentq(func, 1e-6, 3.0)
    except ValueError:
        return np.nan

# Black-Scholes call price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Heston parameters provided
param_sets = [
    [-0.13435391, 1.46158615, 0.69262351, 0.16237212, 0.12121352],
    [-0.7002343, 0.5990794, 0.11039746, 0.13329075, 0.13318902],
    [-0.49653327, 1.1996015, 0.28261122, 0.1453941, 0.13109724],
    [-0.3083942, 1.8036171, 0.4750225, 0.14860386, 0.12878878],
    [-0.14203967, 2.4040155, 0.6511036, 0.14658237, 0.14038663],
    [-0.07714149, 3.0, 0.77619934, 0.14705893, 0.14382745],
    [-0.6335662, 2.4000106, 0.3810037, 0.1359483, 0.13842705],
    [-0.38410205, 0.619107, 0.5746883, 0.19965382, 0.13472086],
    [-0.34344074, 1.1877161, 0.1574534, 0.13194533, 0.13628517],
    [-0.1799222, 1.8103846, 0.34285152, 0.14274177, 0.12650329]
]

# New part for plotting volatility smiles
tau_values = [0.5, 1.0]  # Time to maturity values
m_values = np.linspace(0.85, 1.15, 50)  # Different moneyness values


# Set up the figure
fig, axs = plt.subplots(len(tau_values), figsize=(10, 15))
fig.suptitle('Volatility Smiles for Different Time to Maturity (tau)')

for idx, tau in enumerate(tau_values):
    for i, param_set in enumerate(param_sets):
        rho, kappa, gamma, vbar, v0 = param_set
        implied_vols = []
        for m in m_values:
            adjusted_sample = np.array([m, tau, 0.03, rho, kappa, gamma, vbar, v0])
            S0 = 1  # Set the initial stock price
            K = S0 / m  # Calculate strike price based on moneyness
            N = 4096*4
            L = 2
            cf = ChFHestonModel(0.03, tau, kappa, gamma, vbar, v0, rho)
            CP = OptionType.CALL    
            option_price = CallPutOptionPriceCOSMthd(cf, CP, S0, 0.03, tau, [K], N, L).mean()
            implied_volatility = find_implied_volatility(S0, K, tau, 0.03, option_price)
            implied_vols.append(implied_volatility)
        
        axs[idx].plot(1 / m_values, implied_vols, label=f'Params {i+1}')
    
    axs[idx].set_xlabel('Strike Price')
    axs[idx].set_ylabel('Implied Volatility')
    axs[idx].legend()
    axs[idx].grid()

plt.tight_layout()
plt.show()

# Save results to CSV
np.savetxt("volatility_smiles.csv", np.column_stack([m_values, implied_vols]), delimiter=",")
print("Data generation and saving complete.")