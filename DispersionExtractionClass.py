import numpy as np
import matplotlib.pyplot as plt
import cmath
import sympy as sp
from sympy import diff

import os
import sys

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
add_to_path = os.path.join(parent_directory, "Modules/Refractive_Indices")
os.listdir(parent_directory)
# print(add_to_path)
sys.path.append(add_to_path)
import RefractiveIndexClass as RI

class DispersionExtraction():
    def __init__(self, c = 3e17): # Default is in nm/s, assuming that the class is used with wavelengths in nm.
        self.c = c

    def TraceFFT(x, y, normalise, hanning):
        # Construct the Fourier Domain
        N = len(x)                      # Number of data points
        T = (max(x) - min(x)) / N       # Sample spacing
        xf = np.fft.fftfreq(N, T)       # Create the Fourier domain
        xf = np.fft.fftshift(xf)        # Shift the domain to be centered around 0

        if normalise == True:
            # Normalise and ground the data:
            y = self._groundAndNormalise(y)

        if hanning == True:
            # Apply Hanning window to compensate for end discontinuity
            window = np.hanning(len(y))
            y = y * window
        # Perform the FFT
        yf = np.fft.fft(y)
        yf = np.fft.fftshift(yf)
        return [xf, yf]
    
    def TraceFFT_Zero_Filling(x, y, zero_filling_factor = 2):
        # Construct the Fourier Domain
        N = zero_filling_factor * len(y)                      # Number of data points
        T = (max(x) - min(x)) / N       # Sample spacing
        xf = np.fft.fftfreq(N, T)       # Create the Fourier domain
        xf = np.fft.fftshift(xf)        # Shift the domain to be centered around 0

        # Perform the FFT
        padded_array = np.zeros(N)
        padded_array[:len(y)] = y
        y = padded_array
        yf = np.fft.fft(y)
        yf = np.fft.fftshift(yf)
        print(len(yf))
        print(len(xf))
        return [xf, yf]

    def FilterIndicesFFT(xf, yf, side, keep_min_freq, keep_max_freq):
        # Filter the FFT to keep only the desired frequencies
        if side == "both":
            if keep_max_freq == -1: # Go to max
                idx_left = np.array(np.where(xf < -keep_min_freq)).flatten()                                # left of DC
                idx_right = np.array(np.where(keep_min_freq < xf)).flatten()                                # right of DC
            else:
                idx_left = np.array(np.where((-keep_max_freq < xf) & (xf < -keep_min_freq))).flatten()      # left of DC
                idx_right = np.array(np.where((keep_min_freq < xf) & (xf < keep_max_freq))).flatten()       # right of DC
            idx = (np.concatenate((idx_left, idx_right)))
        elif side == "right":
            if keep_max_freq == -1: # Go to max
                idx = np.array(np.where(keep_min_freq < xf)).flatten()                                # right of DC
            else:
                idx = np.array(np.where((keep_min_freq < xf) & (xf < keep_max_freq))).flatten() 
        elif side == "left":
            if keep_max_freq == -1: # Go to max
                idx = np.array(np.where(xf < -keep_min_freq)).flatten()                                # left of DC                           # right of DC
            else:
                idx = np.array(np.where((-keep_max_freq < xf) & (xf < -keep_min_freq))).flatten()      # left of DC
        else:
            print("'side' is not a valid argument. It should be 'left', 'right', or 'both'.")
            return
        return idx
        
    def BoxFilter(yf, idx):
        # Define the box filter
        box_filter = np.zeros(len(yf), dtype=complex)
        box_filter[idx] = 1
        filtered_fourier_data = yf * box_filter
        return filtered_fourier_data
    
    def InverseFFT(filtered_fourier_data):
        # Perform the inverse FFT
        shifted_filtered_fourier_data = np.fft.ifftshift(filtered_fourier_data)
        index_max_before_shift = np.argmax(np.abs(filtered_fourier_data))                               # Find the index of the maximum value before and after the shift
        index_max_after_shift = np.argmax(np.abs(shifted_filtered_fourier_data))
        shift_amount = index_max_after_shift - index_max_before_shift                                   # Calculate the shift amount
        # print("Shift amount: ", shift_amount)
        filtered_y = np.fft.ifft(shifted_filtered_fourier_data)
        return filtered_y
    
    def ExtractAndUnwrap(filtered_y):
        # Extract phase and unwrap
        final_ys = np.zeros(len(filtered_y))
        for i in range(len(filtered_y)):
            final_ys[i] = cmath.phase((filtered_y[i]))
        final_ys = np.unwrap(final_ys)
        return final_ys

    def DeltaPhiRetrievalProcedure(self, x, y, keep_min_freq = 0.08, keep_max_freq = -1, side = "left", show_plots = True, fft_x_lim = [-1e-12, 1e-12], fft_y_lim = None, hanning = False, normalise = False):
        '''
        Retrieves the spectral phase difference from spectral interference fringes, with flat oscillations, approx. between -1 and +1.
         

        Parameters
        -------
        x ([float]): Array of the wavelengths in nm.
        y ([float]): Intensity of the SI.
        keep_min_freq (float): The minimum frequency in the fourier domain to keep. Setting to -1 takes the first array entry.
        keep_max_freq (float): The maximum frequency in the foutier domain to keep. Setting to -1 takes the last array entry.
        side ("left" or "right" or "both"): Determines the side of the fourier transform to analyse.
        show_plots (bool): Show or hide plots.
        fft_x_lim ([float, float]): The limits of the fourier transform if plots are shown. Can be None for auto-limits. 
        fft_y_lim ([float, float] or None): The limits of the fourier transform if plots are shown. Can be None for auto-limits.
        hanning (bool): Applies a hanning window to mitigate effects of finite edges in data.
         

        Returns
        -------
        [x, coefficients].
        '''
        [xf, yf] = DispersionExtraction.TraceFFT(x, y, normalise, hanning)

        idx = DispersionExtraction.FilterIndicesFFT(xf, yf, side, keep_min_freq, keep_max_freq)

        filtered_fourier_data = DispersionExtraction.BoxFilter(yf, idx)

        filtered_y = DispersionExtraction.InverseFFT(filtered_fourier_data)
        
        final_ys = DispersionExtraction.ExtractAndUnwrap(filtered_y)
        return final_ys
        

        

        # # Plot the FFT results
        # if show_plots == True:
        #     plt.figure(figsize=(8, 6))
        #     plt.subplot(2, 1, 1)
        #     plt.plot(x, y)
        #     plt.title("Signal")
        #     plt.subplot(2, 1, 2)
        #     plt.plot(xf, yf, label = "Full fft") # Normalised
        #     plt.title("FFT")
        #     if fft_x_lim != None:
        #         try:
        #             plt.xlim(fft_x_lim)  # Limit the x-axis
        #         except:
        #             print("Not valid fft_x_lim.")
        #     if fft_y_lim != None:
        #         try:
        #             plt.ylim(fft_y_lim)  # Limit the y-axis
        #         except:
        #             print("Not valid fft_y_lim.")
        #     plt.xlabel("Fourier Domain")
        #     plt.tight_layout()
        #     plt.subplot(2, 1, 2)
        #     if side == "both":
        #         plt.plot(xf[idx_left], yf[idx_left], color='r', label = "Selected region")
        #         plt.plot(xf[idx_right], yf[idx_right], color='r')   
        #     elif side == "right":
        #         plt.plot(xf[idx], yf[idx], color='r', label = "Selected region")
        #     elif side == "left":
        #         plt.plot(xf[idx], yf[idx], color='r', label = "Selected region")
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()


            # # Plot the original data and the filtered data in the original domain
            # plt.figure(figsize=(8, 6))
            # plt.subplot(2, 1, 1)
            # plt.plot(x, y)
            # plt.title("Original Signal")
            # plt.subplot(2, 1, 2)
            # plt.plot(x, np.abs(filtered_y)**2, label="filtered_y")
            # # plt.xlim([np.real(x[np.nonzero(filtered_y)[0][0]]), np.real(x[np.nonzero(filtered_y)[0][-1]])])
            # plt.title("Filtered Signal (ifft of selected region)")
            # plt.tight_layout()
            # plt.legend()
            # plt.show()
        


        
    
    def ObtainBetaFromPhi(phi, length):
        '''
        Obtains beta as a function of wavelength from delta phi.
        

        Parameters
        -------
        phi ([float]): Lambda function for phi.
        length (float): Length of the fibre.
        

        Returns
        -------
        Beta as a function of wavelength, as a lambda function. 
        '''
        return phi / length
    
    def ObtainRefractiveIndex(beta_lambda, wavelengths):
        return beta_lambda * wavelengths / (2 * np.pi)
    
    def CDA2(func_vals, step_size):
        '''
        Performs the second order derivative using centered difference approximation. FDA and BDA (Ord(h^2)) obtained at https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
        
        ! Step size must be the same as the grid step. !
        '''
        second_derivative = []
        last_point = len(func_vals) - 1
        second_derivative.append((1 / (step_size**2)) * (2 * func_vals[0] - 5 * func_vals[1] + 4 * func_vals[2] - func_vals[3]))
        # second_derivative.append((1 / (step_size**2)) * (func_vals[2] - 2 * func_vals[1] + func_vals[0])) # If the FDA ever fails use these.
        for i in range(1, last_point):
            second_derivative.append((1 / (step_size**2)) * (func_vals[i + 1] + func_vals[i - 1] - 2 * func_vals[i]))
        # second_derivative.append((1 / (step_size**2)) * (func_vals[last_point] - 2 * func_vals[last_point - 1] + func_vals[last_point - 2])) # If the BDA ever fails use these.
        second_derivative.append((1 / (step_size**2)) * (2 * func_vals[last_point] - 5 * func_vals[last_point - 1] + 4 * func_vals[last_point - 2] - func_vals[last_point - 3]))
        return second_derivative

    def CDA1(func_vals, step_size):
        '''
        Performs the second order derivative using centered difference approximation. FDA and BDA (Ord(h^2)) obtained at https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

        ! Step size must be the same as the grid step. !
        '''
        first_derivative = []
        last_point = len(func_vals) - 1
        first_derivative.append((1 / (2 * step_size)) * (-3 * func_vals[0] + 4 * func_vals[1] - func_vals[2] - func_vals[3]))
        for i in range(1, last_point):
            first_derivative.append((1 / (2 * step_size)) * (func_vals[i + 1] - func_vals[i - 1]))
        # first_derivative.append((1 / (step_size)) * (func_vals[last_point] - func_vals[last_point - 1]))
        first_derivative.append((1 / (2 * step_size)) * (3 * func_vals[last_point] - 4 * func_vals[last_point - 1] + func_vals[last_point - 2]))
        return first_derivative
    
    def _groundAndNormalise(self, y_data):
        y_data = y_data - min(y_data)
        return (y_data - min(y_data))/ max(y_data - min(y_data))
    
    def beta_lambda(refractive_index, wavelengths):
        beta = []
        for i in range(len(refractive_index)):
            beta.append(2 * np.pi * refractive_index[i] / wavelengths[i])
        return beta

    def GVD_lambda(beta, wavelengths, method = "Fit", order = 3, show_plots = False, output_ps_nm_km = True):
        '''
        GVD which is expressed as beta_2 * (-2 pi c / lambda**2). Sometimes denoted D.

        Parameters
        -------
        beta ([float]]): Array of beta values in nm^-1
        wavelengths ([float]): Array of wavelengths corresponding to the beta array in nm        
        method (string): One of "CDA" for CDA method or "Fit" for polynomial fit method
        order (int): Default order 3 for polynomial fit in "Fit" method
        show_plots (bool): Either true or false to show plots within function
        output_ps_nm_km (bool): Output can be given in expected units from input (s / nm*nm) [False] or in conventional (ps / nm*km) [True - Default].

        Returns
        -------
        GVD as an array.
        '''
        c0 = 3e17                                                           # Speed of light in vacuum in nm / s
        if method.lower() == "fit":
            # Perform the fit
            coefficients = np.polyfit(wavelengths, beta, int(order))
            beta_fit = np.polyval(coefficients, wavelengths)
            if show_plots == True:
                plt.plot(wavelengths, beta, color = 'k', linewidth = 0.8, label="Input phase")
                plt.plot(wavelengths, beta_fit, color='r', linestyle='--', label = f"Fit (ord: {order})")
                plt.title("Phase fit")
                plt.xlabel("$Wavelengths [nm]$")
                plt.ylabel(r"$\beta(\lambda)$")
                plt.grid()
                plt.legend()
                plt.show()
            beta = beta_fit
        first_derivative = DispersionExtraction.CDA1(beta, wavelengths[1] - wavelengths[0])
        second_derivative = DispersionExtraction.CDA2(beta, wavelengths[1] - wavelengths[0])
        GVD = []
        for i in range(len(beta)):
            # GVD.append(second_derivative[i])
            GVD.append((2 * wavelengths[i]**3) / ((2 * np.pi * c0)**2) * first_derivative[i] + (wavelengths[i]**4) / ((2 * np.pi * c0)**2) * second_derivative[i])
            # GVD.append(-1 * ((2 * np.pi * c0) / (wavelengths[i]**2)) * ( ( (wavelengths[i]**3) / (2 * np.pi**2 * c0**2) ) * first_derivative[i] + ( (wavelengths[i]**4) / ((2 * np.pi * c0)**2) ) * second_derivative[i] ) )
        if output_ps_nm_km:
            GVD = np.array(GVD) * 1e24 #15?                                      # Converts from s / nm*nm to ps / nm*km (conventional).             
        return GVD

    def Vg_lambda(beta, wavelengths):
        first_derivative = DispersionExtraction.CDA1(beta, wavelengths[1] - wavelengths[0])
        Vg = []
        for i in range(len(beta)):
            Vg.append( - (2 * np.pi * 3e17 / (wavelengths[i]**2)) * (1 / (first_derivative[i])))
        return Vg