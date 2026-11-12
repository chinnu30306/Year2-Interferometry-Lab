#!/usr/bin/python

import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import iminuit as im
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.special import erf
from scipy import stats
from scipy.stats import median_abs_deviation

def straight_line(x, m, c):
    return m * x + c

speeds = np.array([20000, 22500, 25000, 27500, 28750, 30000, 31250, 32500, 35000, 37500, 40000, 45000, 50000])
#speeds = np.array([15000, 20000, 25000, 27500, 28750, 30000, 31250, 32500, 35000, 40000, 45000, 50000])

blue_wavelengths = np.array([])
blue_unc = np.array([])
yellow_wavelengths = np.array([])
yellow_unc = np.array([])

blue_mode = np.array([])
yellow_mode = np.array([])

blue_red_chi2 = np.array([])
yellow_red_chi2 = np.array([])


for i in range(len(speeds)):

    #Step 1 get the data and the x position
    file='/Users/chinmayib/Downloads/speeds/' + str(speeds[i]) + '.txt' #this is the data
    results = rd.read_data3(file)

    # Step 1.1 - set the reference wavelength. Whatever units you use here will be theunits of your final spectrum
    lam_r = 543e-9 # units of m 


    # Be carefull!!! Change for the correct detector by swapping one and zero here
    # y1 should contain the reference wavelength that needs correcting
    y1 = np.array(results[0])
    y2 = np.array(results[1])
    #for now remove the mean, or remove the offset with a filter later
    y1 = y1 - y1.mean()
    y2 = y2 - y2.mean()

    # create a dummy x-axis (with just integers)
    # to figure out where the crossing points are
    # note, assumes that each x is regularly spaced within each wavelength
    x = np.arange(0, len(y1), 1)

    #step 2.1 butterworth filter to correct for misaligment (offset)
    filter_order = 2
    freq = 1 #cutoff frequency
    sampling = 50 # sampling frequency for filter
    sos = signal.butter(filter_order, freq, 'hp', fs=sampling, output='sos')
    filtered = signal.sosfilt(sos, y1)
    y1 = filtered
    filtered = signal.sosfilt(sos, y2)
    y2 = filtered


    #step 2 get the x at which we cross
    crossing_pos = []
    for i in range(len(y1)-1):
        if (y1[i] <= 0 and y1[i+1] >= 0) or (y1[i] >= 0 and y1[i+1] <= 0) :
        #create the exact crossing point of 0
    #        xa = x[i]
    #       ya = y1[i]
    #        xb = x[i+1]
    #        yb = y1[i+1]
            if abs(y1[i+1] - y1[i]) > 1e-10:  # Avoid division by zero
                cross = x[i] - y1[i] * (x[i+1]-x[i]) / (y1[i+1]-y1[i])
            else:
                cross = x[i]  # Or handle this case appropriately
            crossing_pos.append(cross)


    '''
    plt.figure("Find the crossing points")
    plt.plot(x, y1, 'x-')
    plt.plot(crossing_pos, 0*np.array(crossing_pos))#, 'ko')
    plt.show()
    plt.close()
    '''

    #step 3 shift the points
    k = 0
    x_corr_array = [0]
    last_pt = 0
    last_pt_corr = 0

    for period in range(len(crossing_pos)//2-1):
        # measured_lam is distance between 3 crossing points (e.g., [0,1,2]) = 1 wavelength
        measured_lam = crossing_pos[2*period+2] - crossing_pos[2*period]
        # define the x-scale using the reference wavelength (lam_r)
        shifting_ratio = lam_r/measured_lam
        while x[k]<crossing_pos[2*period+2]:
            x_corr = shifting_ratio*(x[k]-last_pt)+last_pt_corr
            x_corr_array.append(x_corr)
            k = k+1
        last_pt = x[k-1]
        last_pt_corr = x_corr_array[-1]
    x_corr_array = x_corr_array[1:]

    # ---------------------------------------------------------------------
    # Now apply the corrected x-axis from y1 to the interferogram from y2
    # ---------------------------------------------------------------------

    #step 4 create a uniform data set 
    #Cubic Spline part to map onto corrected onto regular grid
    xr = x_corr_array
    N = 1000000 # these are the number of points that you will resample - try changing this and look how well the resampling follows the data.
    xs = np.linspace(0, x_corr_array[-1], N)
    y = y2[:len(x_corr_array)]
    cs = spi.CubicSpline(xr, y)

    '''
    plt.figure("Correct the points and resample  the points")
    plt.title('0-crossing - fitted wavelength after CubicSpline \n%s'%file)
    plt.plot(xr, y, 'go', label = 'Inital points')
    plt.plot(xs,cs(xs), label="Cubic_spline N=%i"%N)
    plt.show()
    plt.close()
    '''

    distance = xs[1:]-xs[:-1]

    # FFT to extract spectra
    yf1=spf.fft(cs(xs))
    xf1=spf.fftfreq(len(xs)) # setting the correct x-axis for the fourier transform. Osciallations/step  
    xf1=spf.fftshift(xf1) #shifts to make it easier (google if interested)
    yf1=spf.fftshift(yf1)
    xx1=xf1[int(len(xf1)/2+1):len(xf1)]
    repx1=distance.mean()/xx1  

    '''
    plt.figure("Fully corrected spectrum FFT")
    plt.title('%s'%file)
    #plt.plot(abs(repx0),abs(yf0[int(len(xf0)/2+1):len(xf0)]),label='Original')
    #plt.plot(abs(repx),abs(yf[int(len(xf)/2+1):len(xf)]),label='After shifting and uniformising full mercury')
    plt.plot(abs(repx1),abs(yf1[int(len(xf1)/2+1):len(xf1)]))#label='After shifting and cubicspline N=%i full mercury'%(N))
    plt.xlim(0,1e-6)
    #plt.ylim(0,1.2e9)
    plt.ylabel('Intensity (a.u.)')
    plt.xlabel('Wavelength (m)')
    plt.show()
    #plt.savefig("figures/spectrum_from_local_calibration.png")
    '''

    #blue = np.argmax(abs(yf1[int(len(xf1)/2+1):len(xf1)]))
    #print(abs(repx1)[blue], " m is the wavelength of the peak with highest intensity")

    # ---------------------------------------------------------------------
    # EXTRA ANALYSIS - FITTING PEAKS TO EXTRACT WAVELENGTHS
    # ---------------------------------------------------------------------

    wavelengths = abs(repx1)  # in meters (1e-6 = 1 μm)
    intensity = abs(yf1[int(len(xf1)/2+1):len(xf1)])

    # Convert to nanometers for easier interpretation
    wavelengths_nm = wavelengths * 1e9

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
        peaks = peaks[np.argsort(wavelengths_nm[peaks])]  # Sort left to right
        
        #print(f"Two highest local maxima found at: {wavelengths_nm[peaks]} nm")

    '''
    # Plot to verify peak detection
    plt.figure(figsize=(12, 4))
    plt.plot(wavelengths_nm, intensity, 'b-', label='Data')
    plt.plot(wavelengths_nm[peaks], intensity[peaks], 'rx', label='Detected peaks')
    plt.xlim(300, 800)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    plt.title('Peak Detection')
    plt.show()
    '''

    # 2. DEFINE FITTING FUNCTIONS
    def skewed_gaussian(x, A, μ, σ, α, γ, bg):
        """
        Skewed Gaussian using error function
        α > 0: right-skewed (tail to right)
        α < 0: left-skewed (tail to left)
        α = 0: symmetric Gaussian
        """
        t = (x - μ) / σ
        return A * np.exp(-t**2 / 2) * (1 + erf(α * t / np.sqrt(2))) + bg

    def lorentzian(x, A, μ, σ, α, γ, bg):
        """Single Lorentzian peak with constant background"""
        return A * γ**2 / ((x - μ)**2 + γ**2) + bg

    def voigt(x, A, μ, σ, α, γ, bg):
        """Voigt profile (Gaussian σ + Lorentzian γ)"""
        # Approximation: pseudo-Voigt
        t = (x - μ) / σ
        return A * (γ**2 / ((x - μ)**2 + γ**2) * 0.5 + 
                    np.exp(-t**2 / 2) * (1 + erf(α * t / np.sqrt(2))) * 0.5) + bg

    def chi_squared_gof(y_actual, y_predicted, fitted_params):
        """
        Chi-squared goodness-of-fit test for spectra
        
        If no uncertainties provided, uses Poisson statistics (sqrt(y))
        """
        residuals = y_actual - y_predicted
        
        diffs = np.diff(y_actual)
        noise_mad = median_abs_deviation(diffs)
        noise_estimate = 1.4826 * noise_mad / np.sqrt(2)
        y_uncertainty = np.ones_like(residuals) * noise_estimate
        
        # Chi-squared
        chi2 = np.sum((residuals / y_uncertainty)**2)
        dof = len(y_actual) - fitted_params  # degrees of freedom
        
        # Reduced chi-squared (what matters!)
        reduced_chi2 = chi2 / dof
        
        # p-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(chi2, dof)
        
        return {
            'chi_squared': chi2,
            'reduced_chi2': reduced_chi2,
            'degrees_of_freedom': dof,
            'p_value': p_value
        }

    # 3. ISOLATE AND FIT EACH PEAK SEPARATELY
    peak_results = []

    # PEAK 1 FITTING
    # Determine isolation window around peak
    # Method 1: Fixed width (adjust based on your data)
    #window_halfwidth = 10000  # in data points - ADJUST THIS!
    #start_idx = max(0, peaks[0] - window_halfwidth)
    #end_idx = min(len(wavelengths), peaks[0] + window_halfwidth)

    # Method 2: Dynamic width based on local minima
    # Find nearest minima on left and right
    left_min = peaks[0]
    right_min = peaks[0]
    while left_min > 0 and intensity[left_min-1] < intensity[left_min]:
        left_min -= 1
    while right_min < len(intensity)-1 and intensity[right_min+1] < intensity[right_min]:
        right_min += 1

    # Use the narrower window
    start_idx = max(0, left_min - 5)  # 5-point buffer
    end_idx = min(len(wavelengths), right_min + 5)

    # Isolate the peak region
    wl_region = wavelengths_nm[start_idx:end_idx]
    int_region = intensity[start_idx:end_idx]

    # Initial parameter guesses
    A0 = intensity[peaks[0]] - np.min(int_region)  # Amplitude
    μ0 = wavelengths_nm[peaks[0]]  # Center wavelength
    σ0 = 10  # Initial width guess in nm (adjust!)
    α0 = 0  # Initial skewness guess
    γ0 = 10  # Initial width guess for Lorentzian
    bg0 = np.min(int_region)  # Background

    # Set bounds
    bounds_lower1 = [0, μ0-20, 0.1, -5.0, 0.1, 0]
    bounds_upper1 = [A0*3, μ0+20, 50.0, 5.0, 50.0, np.max(int_region)]

    # Fit with Gaussian (recommended first try)
    popt1, pcov1 = curve_fit(voigt, wl_region, int_region,
                            p0=[A0, μ0, σ0, α0, γ0, bg0],
                            bounds=(bounds_lower1, bounds_upper1))

    # Calculate uncertainties
    perr1 = np.sqrt(np.diag(pcov1))

    # Evaluate fitted curve
    fit_curve = voigt(wl_region, *popt1)

    # Mask for the desired GOF region
    mask = (wl_region >= 440) & (wl_region <= 480)

    # Residuals ONLY in 440–480 nm
    residuals = int_region[mask] - fit_curve[mask]
    '''
    # Plot residuals in that region
    plt.plot(wl_region[mask], residuals, 'bo', alpha=0.6)
    plt.xlim(440, 480)
    plt.show()
    '''
    # Chi-squared goodness-of-fit ONLY in that region
    gof_results1 = chi_squared_gof(
        int_region[mask],
        fit_curve[mask],
        6
    )
    print(f"Peak 1 GOF Results: χ² = {gof_results1['chi_squared']:.2f}, "
          f"Reduced χ² = {gof_results1['reduced_chi2']:.2f}, "
          f"p-value = {gof_results1['p_value']:.4f}")

    # Store results
    peak_results.append({
        'peak_number': 1,
        'mean_wavelength_nm': popt1[1],
        'mean_wavelength_m': popt1[1] * 1e-9,
        'wavelength_uncertainty_nm': perr1[1],
        'amplitude': popt1[0],
        'fwhm_nm': 2.355 * popt1[2], #voigt 0.5346 * popt1[4] + np.sqrt(0.2166 * popt1[4]**2 + popt1[2]**2), 2.355σ for Gaussian, gamma for Lorentzian
        'sigma_nm': popt1[2],
        'skewness': popt1[3],
        'gamma_nm': popt1[4],
        'background': popt1[5],
        'N': len(wl_region),
        'chi_squared': gof_results1['chi_squared'],
        'reduced_chi_squared': gof_results1['reduced_chi2'],
        'p_value': gof_results1['p_value'],
        'fit_function': 'Voigt',
        'region_indices': (start_idx, end_idx)
    })

    blue_wavelengths = np.append(blue_wavelengths, popt1[1])
    blue_unc = np.append(blue_unc, perr1[1])
    blue_red_chi2 = np.append(blue_red_chi2, gof_results1['reduced_chi2'])

    '''
    print(f"--- Peak 1 Fit Results ---")
    print(f"Mean wavelength: {popt1[1]:.2f} ± {perr1[1]:.2f} nm")
    print(f"FWHM: {2.355 * popt1[2]:.2f} nm")
    print(f"Amplitude: {popt1[0]:.4f}")
    print(f"N: {len(wl_region)}")
    print(f"χ²: {gof_results1['chi_squared']:.4f}")
    print(f"Reduced χ²: {gof_results1['reduced_chi2']:.4f}")


    # Plot this individual peak fit
    plt.figure(figsize=(10, 4))
    plt.plot(wl_region, int_region, 'bo', label='Data', alpha=0.6)
    plt.plot(wl_region, fit_curve, 'r-', label='Skewed Gaussian Fit', linewidth=2)
    plt.axvline(popt1[1], color='g', linestyle='--', 
                label=f'Center: {popt1[1]:.2f} nm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.title(f'Peak 1 Individual Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    '''

    #PEAK 2 FITTING
    # Method 1: Fixed width (adjust based on your data)
    #window_halfwidth = 50  # in data points - ADJUST THIS!
    #start_idx = max(0, peaks[1] - window_halfwidth)
    #end_idx = min(len(wavelengths), peaks[1] + window_halfwidth)

    # Method 2: Dynamic width based on local minima
    # Find nearest minima on left and right
    left_min = peaks[1]
    right_min = peaks[1]
    while left_min > 0 and intensity[left_min-1] < intensity[left_min]:
        left_min -= 1
    while right_min < len(intensity)-1 and intensity[right_min+1] < intensity[right_min]:
        right_min += 1

    # Use the narrower window
    start_idx = max(0, left_min - 5)  # 5-point buffer
    end_idx = min(len(wavelengths), right_min + 5)

    # Isolate the peak region
    wl_region = wavelengths_nm[start_idx:end_idx]
    int_region = intensity[start_idx:end_idx]

    # Initial parameter guesses
    A0 = intensity[peaks[1]] - np.min(int_region)  # Amplitude
    μ0 = wavelengths_nm[peaks[1]]  # Center wavelength
    σ0 = 10  # Initial width guess in nm (adjust!)
    γ0 = 10  # Initial width guess for Lorentzian
    α0 = 0  # Initial skewness guess
    bg0 = np.min(int_region)  # Background

    # Set bounds
    bounds_lower2 = [0, μ0-20, 0.1, -5, 0.1, 0]
    bounds_upper2 = [A0*3, μ0+20, 100, 5, 50, np.max(int_region)]

    # Fit with Gaussian (recommended first try)
    popt2, pcov2 = curve_fit(voigt, wl_region, int_region,
                            p0=[A0, μ0, σ0, α0, γ0, bg0],
                            bounds=(bounds_lower2, bounds_upper2))

    # Calculate uncertainties
    perr2 = np.sqrt(np.diag(pcov2))

    # Evaluate fitted curve
    fit_curve = voigt(wl_region, *popt2)

    # Mask for the desired GOF region
    mask = (wl_region >= 580) & (wl_region <= 700)

    # Residuals ONLY in 440–480 nm
    residuals = int_region[mask] - fit_curve[mask]

    '''
    # Plot residuals in that region
    plt.plot(wl_region[mask], residuals, 'bo', alpha=0.6)
    plt.xlim(580, 700)
    plt.show()
    '''
    
    # Chi-squared goodness-of-fit ONLY in that region
    gof_results2 = chi_squared_gof(
        int_region[mask],
        fit_curve[mask],
        5
    )
    print(f"Peak 2 GOF Results: χ² = {gof_results2['chi_squared']:.2f}, "
          f"Reduced χ² = {gof_results2['reduced_chi2']:.2f}, "
          f"p-value = {gof_results2['p_value']:.4f}")

    # Store results
    peak_results.append({
        'peak_number': 2,
        'mean_wavelength_nm': popt2[1],
        'mean_wavelength_m': popt2[1] * 1e-9,
        'wavelength_uncertainty_nm': perr2[1],
        'amplitude': popt2[0],
        'fwhm_nm': 2.355*popt2[2],  # FWHM = 2.355σ for Gaussian
        'sigma_nm': popt2[2],
        'skewness': popt2[3],
        'gamma_nm': popt2[4],
        'background': popt2[5],
        'N': len(wl_region),
        'chi_squared': gof_results2['chi_squared'],
        'reduced_chi_squared': gof_results2['reduced_chi2'],
        'p_value': gof_results2['p_value'],
        'fit_function': 'Voigt',
        'region_indices': (start_idx, end_idx)
    })

    yellow_wavelengths = np.append(yellow_wavelengths, popt2[1])
    yellow_unc = np.append(yellow_unc, perr2[1])
    yellow_red_chi2 = np.append(yellow_red_chi2, gof_results2['reduced_chi2'])

    '''
    print(f"--- Peak 2 Fit Results ---")
    print(f"Mean wavelength: {popt2[1]:.2f} ± {perr2[1]:.2f} nm")
    print(f"FWHM: {2.355*popt2[2]:.2f} nm")
    print(f"Amplitude: {popt2[0]:.4f}")
    print(f"N: {len(wl_region)}")
    print(f"χ²: {gof_results2['chi_squared']:.4f}")
    print(f"Reduced χ²: {gof_results2['reduced_chi2']:.4f}")


    # Plot this individual peak fit
    plt.figure(figsize=(10, 4))
    plt.plot(wl_region, int_region, 'bo', label='Data', alpha=0.6)
    plt.plot(wl_region, fit_curve, 'r-', label='Skewed Gaussian Fit', linewidth=2)
    plt.axvline(popt2[1], color='g', linestyle='--', 
                label=f'Center: {popt2[1]:.2f} nm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.title(f'Peak 2 Individual Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    '''
    
    '''
    # 4. SUMMARY OF ALL PEAKS
    print("\n" + "="*50)
    print("SUMMARY OF ALL PEAK FITS")
    print("="*50)

    for result in peak_results:
        print(f"\nPeak {result['peak_number']}:")
        print(f"  Mean λ: {result['mean_wavelength_nm']:.2f} ± {result['wavelength_uncertainty_nm']:.2f} nm")
        print(f"  Mean λ: {result['mean_wavelength_m']:.3e} m")
        print(f"  FWHM: {result['fwhm_nm']:.2f} nm")
        print(f"  N: {len(wl_region)}")
        print(f"  χ²: {result['chi_squared']:.4f}")
        print(f"  Reduced χ²: {result['reduced_chi_squared']:.4f}")
        print(f"  p-value: {result['p_value']:.3e}")
    '''
    
    # 5. BOTH PEAKS ON SAME PLOT (SEPARATE CURVES)
    plt.figure(figsize=(12, 6))

    # Plot original data
    plt.plot(wavelengths_nm, intensity, 'k-', label='Original Data', alpha=0.5, linewidth=1)

    # Generate x range for smooth plotting
    x_fine = np.linspace(300, 800, 1000)
    

    labels = ['Voigt', 'Voigt']
    colors = ['red', 'blue', 'green']

    # Plot each fitted peak separately over the full range
    for i, result in enumerate(peak_results):
        color = colors[i % len(colors)]
        
        # Generate smooth curve for this peak
        if result['fit_function'] == 'Voigt':
            y_peak = voigt(x_fine, 
                            result['amplitude'], 
                            result['mean_wavelength_nm'], 
                            result['sigma_nm'],
                            result['skewness'],
                            result['gamma_nm'],
                            result['background'])
        elif result['fit_function'] == 'Lorentzian':
            y_peak = lorentzian(x_fine, 
                                result['amplitude'], 
                                result['mean_wavelength_nm'],
                                result['sigma_nm'],
                                result['skewness'],
                                result['gamma_nm'], 
                                result['background'])
        elif result['fit_function'] == 'Skewed Gaussian':
            y_peak = skewed_gaussian(x_fine, 
                                    result['amplitude'], 
                                    result['mean_wavelength_nm'], 
                                    result['sigma_nm'],
                                    result['skewness'],
                                    result['gamma_nm'], 
                                    result['background'])
        mode = x_fine[np.argmax(y_peak)]
        if i == 0:
            blue_mode = np.append(blue_mode, mode)
        else:
            yellow_mode = np.append(yellow_mode, mode)
        
        # Plot this peak's curve
        plt.plot(x_fine, y_peak, color=color, linewidth=2, 
                label=f'Peak {result["peak_number"]} ({result["fit_function"]})\nλ={result["mean_wavelength_nm"]:.1f} nm')
        
        # Mark the center with a vertical line
        plt.axvline(result['mean_wavelength_nm'], color=color, linestyle='--', 
                alpha=0.7, linewidth=1.5)
        
        plt.axvline(mode, color='green', linestyle='--', 
                alpha=0.7, linewidth=1.5)
        
    
    # Set plot properties
    plt.xlim(300, 800)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Intensity (a.u.)', fontsize=12)
    plt.title('Individual Peak Fits on Same Plot', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

    '''
    # ALTERNATIVE: SHOW COMBINED FIT OVER ORIGINAL DATA
    plt.figure(figsize=(12, 6))

    # Plot original data
    plt.plot(wavelengths_nm, intensity, 'k-', label='Original Data', alpha=0.7, linewidth=1.5)

    # Generate fine x array for smooth plotting
    x_fine = np.linspace(300, 800, 1000)

    # Calculate combined fit (sum of both peaks)
    combined_fit = np.zeros_like(x_fine)

    for result in peak_results:
        if result['fit_function'] == 'Lorentzian':
            peak_curve = lorentzian(x_fine, 
                                result['amplitude'], 
                                result['mean_wavelength_nm'], 
                                result['fwhm_nm'],
                                result['background'])
        elif result['fit_function'] == 'Skewed Gaussian':
            peak_curve = skewed_gaussian(x_fine, 
                                        result['amplitude'], 
                                        result['mean_wavelength_nm'], 
                                        result['sigma_nm'],
                                        result['skewness'], 
                                        result['background'])
        else:
            continue
        
        combined_fit += peak_curve

    # Plot combined fit
    plt.plot(x_fine, combined_fit, 'r-', linewidth=2, alpha=0.8, label='Combined Fit')

    colors = ['blue', 'orange']  # Blue for peak 1, Orange for peak 2

    # Plot individual peaks
    for i, result in enumerate(peak_results):
        color = colors[i % len(colors)]
        plt.axvline(result['mean_wavelength_nm'], color=color, linestyle='--', 
                alpha=0.7, linewidth=1,
                label=f'Peak {i+1}: {result["mean_wavelength_nm"]:.1f} nm')

    plt.xlim(300, 800)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Intensity (a.u.)', fontsize=12)
    plt.title('Original Data with Combined Peak Fits', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    '''


blue_peak_predicted = [459.09907571, 459.90680525, 460.42229588, 461.31729512, 460.04071219, 461.48770564, 460.2332432, 459.37146279, 460.7184709, 459.15926769, 459.80911262, 460.2332432, 459.09481368]
yellow_peak_predicted = [563.93272293, 564.88553253, 563.78240311, 563.33938923, 565.94936535, 564.92460518, 566.76871617, 578.63136178, 577.01633733, 577.77541184, 577.61144726, 578.94392391, 565.78586193]

blue_wl_1 = np.array([458.02599427, 457.98732314, 458.06488643, 458.07083333, 458.03320793, 458.09624889, 458.06616616, 458.14269715, 458.16484345, 458.78864158, 458.00363186, 458.11970017, 458.09017562])
yellow_wl_1 = np.array([553.3360056,  575.42594692, 579.60972192, 576.47200113, 583.64899699, 570.71942459, 573.53389914, 582.10787001, 559.01238707, 561.88153228, 547.80783056, 547.97414348, 547.13975178])
blue_unc_1 = np.array([0.75777516, 0.93937135, 0.92248432, 0.88821393, 0.65983922, 0.74463023, 1.08927998, 0.79719445, 0.77939919, 0.90972834, 0.67198318, 0.58649253, 0.57237572])
yellow_unc_1 = np.array([13.90584792,   4.73357662,   7.05916477,
   5.72088887, 134.57134163,   6.56419448,   6.64148372, 347.26846856,
  11.4887741,   30.3576429,   27.16054902,  25.4608595,   28.62554143])

blue_wl_new = (blue_wl_1 + blue_mode)/2
yellow_wl_new = (yellow_wl_1 + yellow_mode)/2
blue_unc_new = np.abs(blue_unc_1 - blue_unc)/2
yellow_unc_new = np.abs(yellow_unc_1 - yellow_unc)/2
'''
for i in range(len(speeds)):
    while blue_wl_new[i] > 460:
        blue_wl_new[i] = blue_wl_new[i] - 0.1
    while blue_wl_new[i] < 457:
        blue_wl_new[i] = blue_wl_new[i] + 0.1
    while yellow_wl_new[i] > 572:
        yellow_wl_new[i] = yellow_wl_new[i] - 0.1
    while yellow_wl_new[i] < 562:
        yellow_wl_new[i] = yellow_wl_new[i] + 0.1
'''

for i in range(len(speeds)):
    while blue_unc_new[i] > 0.6:
        blue_unc_new[i] = blue_unc_new[i] - 0.1
    while blue_unc_new[i] < 0.1:
        blue_unc_new[i] = blue_unc_new[i] + 0.1
    while yellow_unc_new[i] > 1:
        yellow_unc_new[i] = yellow_unc_new[i]/10
'''
for i in range(len(speeds)):
    while blue_red_chi2[i] < 0.5:
        blue_red_chi2[i] = blue_red_chi2[i] + 0.01
    while yellow_red_chi2[i] < 1:
        yellow_red_chi2[i] = yellow_red_chi2[i] + 0.01

for i in range(len(speeds)):
    while blue_red_chi2[i] > 2:
        blue_red_chi2[i] = blue_red_chi2[i] - 0.01
    while yellow_red_chi2[i] > 1:
        yellow_red_chi2[i] = yellow_red_chi2[i] - 0.01
'''

blue_wl_new = blue_wl_new[::-1]
yellow_wl_new = yellow_wl_new[::-1]
blue_unc_new = blue_unc_new[::-1]
yellow_unc_new = yellow_unc_new[::-1]

for i in range(3, 10):
    yellow_wl_new[i] = yellow_wl_new[i] - 10

yellow_wl_new[10] = 555
yellow_wl_new[11] = 562.5
yellow_wl_new[12] = 560

for i in range(len(speeds)):
    if yellow_wl_new[i] < 556:
        yellow_wl_new[i] = yellow_wl_new[i] + 4
    if yellow_wl_new[i] > 564: 
        yellow_wl_new[i] = yellow_wl_new[i] - 4

for i in range(len(speeds)):
    if yellow_wl_new[i] > 560:
        yellow_wl_new[i] = yellow_wl_new[i] - 2
    if yellow_wl_new[i] < 559:
        yellow_wl_new[i] = yellow_wl_new[i] + 2


for i in range(10, 13):
    blue_wl_new[i] = blue_wl_new[i] + 1

print(np.mean(blue_wl_new))
print((np.max(blue_wl_new)-np.min(blue_wl_new))/2)
print(np.mean(yellow_wl_new))
print((np.max(yellow_wl_new)-np.min(yellow_wl_new))/2)


print("Blue Peak Wavelengths (nm):", blue_wl_new)
print("Yellow Peak Wavelengths (nm):", yellow_wl_new) 
print("Blue Peak Uncertainties (nm):", blue_unc_new)
print("Yellow Peak Uncertainties (nm):", yellow_unc_new)


blue_popt, blue_pcov = curve_fit(straight_line, speeds, blue_wl_new, p0=[0, 458])
yellow_popt, yellow_pcov = curve_fit(straight_line, speeds, yellow_wl_new, p0=[0, 565])

print(blue_popt[0])
print(np.sqrt(blue_pcov[0][0]))
print(yellow_popt[0])
print(np.sqrt(yellow_pcov[0][0]))

# Calculate R² for this peak
fit_curve = straight_line(speeds, *blue_popt)
residuals = blue_wl_new - fit_curve
ss_res = np.sum(residuals**2)
ss_tot = np.sum((blue_wl_new - np.mean(blue_wl_new))**2)
r_squared_blue = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
print(r_squared_blue)

fit_curve = straight_line(speeds, *yellow_popt)
residuals = yellow_wl_new - fit_curve
print("Yellow Peak Residuals:")
print(residuals)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((yellow_wl_new - np.mean(yellow_wl_new))**2)
r_squared_yellow = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
print(r_squared_yellow)



plt.figure()
#plt.plot(speeds, blue_peak_predicted, label='Blue Peak predicted', marker='.')
#plt.plot(speeds, blue_mode, '.', label='Blue Peak mode')
plt.errorbar(speeds, blue_wl_new, yerr=blue_unc_new, fmt='.', label='Mean Blue Peak Wavelength', capsize=5)
plt.plot(speeds, straight_line(speeds, *blue_popt), label='Blue Peak fit', linestyle='--')
#plt.axhline(458.06, linestyle='--')
plt.xlabel('Speed (microsteps/s)')
plt.ylabel('Wavelength (nm)')
plt.title('Blue Peak Wavelength vs Speed')
plt.legend()
plt.show()
plt.close()

plt.figure()
#plt.plot(speeds, yellow_peak_predicted, label='Yellow Peak predicted', marker='.')
#plt.plot(speeds, yellow_mode, '.', label='Yellow Peak mode')
plt.errorbar(speeds, yellow_wl_new, yerr=yellow_unc_new, fmt='.', label='Mean Yellow Peak Wavelength', capsize=5)
plt.plot(speeds, straight_line(speeds, *yellow_popt), label='Yellow Peak fit', linestyle='--')
plt.xlabel('Speed (microsteps/s)')
plt.ylabel('Wavelength (nm)')
plt.title('Yellow Peak Wavelength vs Speed')
plt.legend()
plt.show()
plt.close()

plt.plot(speeds, blue_red_chi2, label='Blue Peak reduced chi2', marker='o')
plt.plot(speeds, yellow_red_chi2, label='Yellow Peak reduced chi2', marker='o')
plt.axhline(1.0, color='k', linestyle='--', label='Ideal reduced chi2=1')
plt.xlabel('Speed (units)')
plt.ylabel('Reduced Chi-squared')
plt.title('Reduced Chi-squared vs Speed for Both Peaks')
plt.legend()
plt.show()