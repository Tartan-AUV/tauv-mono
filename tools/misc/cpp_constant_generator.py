import math


# Helper for printing constexpr definitions
def constexpr(name, value):
    print(f'constexpr double {name} = {value:.17g};')


# PI and fractions
constexpr("PI", math.pi)
for denom in range(1, 9):
    frac = math.pi / denom
    name = f"FRAC_PI_{denom}"
    constexpr(name, frac)

for numer in range(1, 3):
    frac = math.pi * numer / 3
    name = f"FRAC_PI_{numer}_3"
    constexpr(name, frac)

for numer in range(1, 5):
    frac = math.pi * numer / 5
    name = f"FRAC_PI_{numer}_5"
    constexpr(name, frac)

for numer in range(1, 7):
    frac = math.pi * numer / 7
    name = f"FRAC_PI_{numer}_7"
    constexpr(name, frac)

# PI fractions of eigths (pi * n / 8)
for numer in range(1, 8):
    frac = math.pi * numer / 8
    name = f"FRAC_PI_{numer}_8"
    # Skip duplicates
    if math.gcd(numer, 8) != 1:
        continue
    constexpr(name, frac)

# Square roots
constexpr("SQRT_2", math.sqrt(2))
constexpr("SQRT_3", math.sqrt(3))
constexpr("SQRT_5", math.sqrt(5))
constexpr("FRAC_1_SQRT_2", 1 / math.sqrt(2))
constexpr("FRAC_1_SQRT_3", 1 / math.sqrt(3))
constexpr("FRAC_1_SQRT_5", 1 / math.sqrt(5))

# Euler number
constexpr("E", math.e)

# Physical constants
constexpr("G_EARTH_EQ", 9.81)  # m/s^2

# Fresh water density at 25C
constexpr("RHO_FRESH_WATER", 997.0)  # kg/m^3

constexpr("GAS_CONSTANT_R", 8.314462618)  # J/(mol·K)
constexpr("STANDARD_ATMOSPHERE", 101325.0)  # Pa
constexpr("SPEED_OF_SOUND_WATER", 1481.0)  # m/s, in seawater at 20 °C, 1 atm

constexpr("DEG_TO_RAD", math.pi / 180.0)
constexpr("RAD_TO_DEG", 180.0 / math.pi)
