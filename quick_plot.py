###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#from scipy.signal import find_peaks
import read_data_results3 as rd

def sine_func(x, A, k, phi, c):
    return A * np.sin(k * x + phi) + c

#Step 1 get the data and the x position
file= '/Users/chinmayib/Downloads/30000_final.txt' #this is the data
results = rd.read_data3(file)

y1 = np.array(results[0])
y2 = np.array(results[1])
x=np.array(results[5])

'''
#initial_guess = [0.1, 0.00025, -1.0, 1.05]
#popt, pcov = curve_fit(sine_func, x, y2, p0=initial_guess)

peak_index = find_peaks(y2)[0]
print("Peaks at: ", peak_index)

maxima_index_spacing = np.array([])
for i in range(len(peak_index)-2):
    maxima_index_spacing = np.append(maxima_index_spacing, peak_index[i+1]-peak_index[i])
print("Max spacing between peaks: ", maxima_index_spacing)

microstep_spacing = np.array([])
for i in range(len(maxima_index_spacing)-1):
    microstep_spacing = np.append(microstep_spacing, x[peak_index[i+1]] - x[peak_index[i]])
print("Microstep spacing between peaks: ", microstep_spacing)

avg_microstep_spacing = np.mean(microstep_spacing)
print("Average microstep spacing between peaks: ", avg_microstep_spacing)
wavelength = 2 * avg_microstep_spacing * 1.56e-2
print("Wavelength: ", wavelength, " nm")
'''

plt.figure("Detector 1")
plt.plot(x,y1)
plt.xlabel("Position [$\mu$steps]")
plt.ylabel("Signal 1")
#plt.xlim(-3.2e6, 2.5e6)
plt.show()
plt.close()

plt.figure("Detector 2", figsize=(12, 6))
plt.plot(x,y2, 'b-', label='Data', linewidth=1)
plt.xlabel("Position [$\mu$steps]")
plt.ylabel("Intensity [a.u.]")
plt.title("Interferogram of white LED")
plt.xlim(-2.6e6, -1.6e6)
plt.show()
plt.close()
