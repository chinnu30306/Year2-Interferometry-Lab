import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.special import erf
from scipy.stats import median_abs_deviation
from scipy import stats

wavelengths, intensity = np.loadtxt('/Users/chinmayib/Downloads/diff_grat_final.csv', delimiter=',', unpack=True, skiprows=53)

plt.plot(wavelengths, intensity, label='Intensity Spectrum')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.title('Diffraction Grating Spectrum')
plt.legend()
plt.show()

# Alternative: Directly find the 2 highest local maxima
# Find all local maxima
local_maxima = argrelextrema(intensity, np.greater, order=10)[0]  # order=10 ignores small fluctuations
local_minima = argrelextrema(intensity, np.less, order=10)[0]
#print(np.max(intensity[local_minima]))
#print(np.where(intensity == 241046151.92712557)[0])

if len(local_maxima) >= 2:
    # Sort by intensity (highest first)
    sorted_maxima = local_maxima[np.argsort(intensity[local_maxima])[::-1]]
    peaks = sorted_maxima[:2]  # Take 2 highest
    peaks = peaks[np.argsort(wavelengths[peaks])]  # Sort left to right
    
    print(f"Two highest local maxima found at: {wavelengths[peaks]} nm")


# Plot to verify peak detection
plt.figure(figsize=(12, 4))
plt.plot(wavelengths, intensity, 'b-', label='Data')
plt.plot(wavelengths[peaks], intensity[peaks], 'rx', label='Detected peaks')
plt.xlim(300, 800)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.title('Peak Detection')
plt.show()


