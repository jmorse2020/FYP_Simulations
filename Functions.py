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

class SI_Functions_Lambda():
    '''
    This class contains functions for spectral interference project work.
    The wavelengths are assumed to be in nm (as c defaults to 3e17 nm/s), unless you manually specify c in the init to be c = 3e8 m/s.
    '''
    def __init__(self, c = 3e17): # Default is in nm/s, assuming that the class is used with wavelengths in nm.
        self.c = c

    def _groundAndNormalise(self, y_data):
        y_data = y_data - min(y_data)
        return (y_data - min(y_data))/ max(y_data - min(y_data))

    def DeltaPhiRetrievalProcedure(self, x, y, order = 2, keep_min_freq = 0.08, keep_max_freq = -1, side = "left", show_plots = True, fft_x_lim = [-1e-12, 1e-12], fft_y_lim = None, hanning = False, normalise = False):
        '''
        Retrieves the spectral phase difference from spectral interference fringes, with flat oscillations, approx. between -1 and +1.
         

        Parameters
        -------
        x ([float]): Array of the wavelengths.
        y ([float]): Intensity of the SI.
        order (int): Order to approximate the phase.
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
        
        # Define the box filter
        box_filter = np.zeros(len(yf), dtype=complex)
        box_filter[idx] = 1
        filtered_fourier_data = yf * box_filter

        # Perform the inverse FFT
        shifted_filtered_fourier_data = np.fft.ifftshift(filtered_fourier_data)
        index_max_before_shift = np.argmax(np.abs(filtered_fourier_data))                               # Find the index of the maximum value before and after the shift
        index_max_after_shift = np.argmax(np.abs(shifted_filtered_fourier_data))
        shift_amount = index_max_after_shift - index_max_before_shift                                   # Calculate the shift amount
        print("Shift amount: ", shift_amount)
        filtered_y = np.fft.ifft(shifted_filtered_fourier_data)

        # Plot the FFT results
        if show_plots == True:
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.plot(x, y)
            plt.title("Signal")
            plt.subplot(2, 1, 2)
            plt.plot(xf, yf, label = "Full fft") # Normalised
            plt.title("FFT")
            if fft_x_lim != None:
                try:
                    plt.xlim(fft_x_lim)  # Limit the x-axis
                except:
                    print("Not valid fft_x_lim.")
            if fft_y_lim != None:
                try:
                    plt.ylim(fft_y_lim)  # Limit the y-axis
                except:
                    print("Not valid fft_y_lim.")
            plt.xlabel("Fourier Domain")
            plt.tight_layout()
            plt.subplot(2, 1, 2)
            if side == "both":
                plt.plot(xf[idx_left], yf[idx_left], color='r', label = "Selected region")
                plt.plot(xf[idx_right], yf[idx_right], color='r')   
            elif side == "right":
                plt.plot(xf[idx], yf[idx], color='r', label = "Selected region")
            elif side == "left":
                plt.plot(xf[idx], yf[idx], color='r', label = "Selected region")
            plt.legend()
            plt.tight_layout()
            plt.show()


            # Plot the original data and the filtered data in the original domain
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.plot(x, y)
            plt.title("Original Signal")
            plt.subplot(2, 1, 2)
            plt.plot(x, np.abs(filtered_y)**2, label="filtered_y")
            # plt.xlim([np.real(x[np.nonzero(filtered_y)[0][0]]), np.real(x[np.nonzero(filtered_y)[0][-1]])])
            plt.title("Filtered Signal (ifft of selected region)")
            plt.tight_layout()
            plt.legend()
            plt.show()
        


        # Extract phase and unwrap
        final_ys = np.zeros(len(filtered_y))
        for i in range(len(filtered_y)):
            final_ys[i] = cmath.phase((filtered_y[i]))

        final_ys = np.unwrap(final_ys)
        # final_ys = final_ys - final_ys[0]
        print(final_ys)
        print("MIN: ", min(final_ys))
        print("MAX: ", max(final_ys))
       
        print("Final ys:")
        print(final_ys)
        
        # Perform the fit
        coefficients = np.polyfit(x, final_ys, order)

        if show_plots == True:
            plt.plot(x, final_ys)
            plt.title("$\Phi(\omega)$")
            plt.xlabel("$\omega$")
            plt.ylabel("Intensity")
            plt.plot(x, np.polyval(coefficients, x), color='r', linestyle='--', label="Extracted phase")
            # plt.plot(x, -1.95698265e-05*x**3 +   6.79351537e-02*x**2 -7.89563528e+01*x +3.04750604e+04,label="Original simulated phase", color='orange', linestyle = '-.')
            
            plt.legend()
            plt.show()
            # plt.xlim([np.real(x[np.nonzero(filtered_y)[0][0]]), np.real(x[np.nonzero(filtered_y)[0][-1]])])
        return [final_ys, coefficients]
    
    def MakeDeltaPhiLambda(coefficients):
        '''
        Makes the phase a lambda function so that it can be used in subsequent functions in this class.
        

        Parameters
        -------
        coefficients ([float]): Coefficients from the output of the 'DeltaPhiRetrievalProcedure'. 
        

        Returns
        -------
        Lambda function for delta phi.
        '''
        import sympy as sp
        return lambda var: np.poly1d(coefficients)(var)
    
    # GIVEN x IS IN WAVELENGTHS
    
    def ObtainBetaFromPhi(self, phi, length):
        '''
        Obtains beta as a function of wavelength from delta phi.
        

        Parameters
        -------
        phi (lambda): Lambda function for phi.
        length (float): Length of the fibre.
        

        Returns
        -------
        Beta as a function of wavelength, as a lambda function. 
        '''
        return lambda var: phi(var) / length

        
    def V_g(self, beta):                                      # Input beta as a function of wavelength
        '''
        Obtains the group velocity, v_g, as a function of wavelength from beta (function of wavelength).
        v_g := (dBeta / dOmega) ^ (-1)
        

        Parameters
        -------
        beta (lambda): Lambda function for beta, in terms of wavelength.
        

        Returns
        -------
        Lambda function for group velocity as a function of wavelength, as a lambda function. 
        ''' 
        x = sp.symbols('x')
        expr = beta(x)
        beta_1 = diff(expr, x)                                          # Perform differentiation with respect to wavelength
        dLambda_dOmega = - x**2 / (2 * np.pi * self.c)                       # Obtain the multiplication constant
        beta_1 = beta_1 * dLambda_dOmega                                # Multiply by the chain to make the diff. wrt omega
        return lambda vars: [1 / beta_1.subs(x, var) for var in vars]   # Invert and return

    def DBeta_dOmega(self, beta):
        '''
        Performs the first derivative of beta (inputted as a lambda funciton of wavelengths) with respect to omega.
        This is otherwise known as beta_1.

        Parameters
        -------
        beta (lambda): Lambda function for beta, in terms of wavelength.
        

        Returns
        -------
        Lambda function for first derivative of beta wrt wavelength, otherwise known as beta_1.
        '''
        x = sp.symbols('x')
        expr = beta(x)
        beta_1 = diff(expr, x)                          # Perform differentiation with respect to wavelength
        dLambda_dOmega = - x**2 / (2 * np.pi * self.c)  # Obtain the multiplication constant
        beta_1 = beta_1 * dLambda_dOmega                # Multiply by the chain to make the diff. wrt omega
        return lambda var: beta_1.subs(x, var)          

    def GVD(self, beta):                          # Input beta as a function on wavelength
        '''
        Calculates the Group Velocity Dispersion (GVD) otherwise known as beta_2.
        GVD := d^2 Beta / d Omega^2

        Parameters
        -------
        beta (lambda): Lambda function for beta, in terms of wavelength.
        

        Returns
        -------
        A lambda function for the group velocity dispersion.
        '''
        # Need to compute first and second derivative of beta wrt lambda:
        from sympy import diff
        import sympy as sp    
        x = sp.symbols('x')
        expr = beta(x)
        dBeta_dLambda = diff(expr, x)                # First derivative wrt lambda
        d2Beta_dLambda2 = diff(expr, x, 2)           # Second derivative wrt lambda

        # Also need first and second derivative of lambda wrt omega:
        dLambda_dOmega = - x**2 / (2 * np.pi * self.c)
        d2Lambda_dOmega2 = 2 * x**3 / (2 * np.pi * self.c)**2

        # The final expression d2Beta / dOmega2 is
        d2Beta_dOmega2 = d2Beta_dLambda2 * dLambda_dOmega**2 + dBeta_dLambda * d2Lambda_dOmega2 # Expansion by chain rule
        return lambda vars: [d2Beta_dOmega2.subs(x, var) for var in vars]

    def Big_D(self, beta): # Input beta as a function of wavelength
        '''
        Performs the first derivative of beta (inputted as a lambda function) wrt omega followed by the first derivative with respect to lambda.
        D := d2Beta / (dLambda * dOmega)

        Parameters
        -------
        beta (lambda): Lambda function for beta, in terms of wavelength.
        

        Returns
        -------
        D as a lambda function.
        '''
        # Need to compute first and second derivative of beta wrt lambda:
        x = sp.symbols('x')
        expr = beta(x)
        beta_1 = diff(expr, x)              # First derivative wrt lambda
        beta_2 = diff(expr, x, 2)           # Second derivative wrt lambda

        # Need to compute dLambda_dOmega and d2Lambda_dLambdadOmega:
        dLambda_dOmega = - x**2 / (2 * np.pi * self.c)
        d2Lambda_dLambdadOmega = - x / (np.pi * self.c)

        D = d2Lambda_dLambdadOmega * beta_1 + dLambda_dOmega * beta_2 # Obtained using the chain rule
        return lambda vars: [D.subs(x, var) for var in vars]

    def Obtain_n(self, beta):
        '''
        Obtains the effective refractive index from beta, inputted as a lambda function.
        n := beta * lambda / (2 * pi))
        
        Parameters
        -------
        beta (lambda): Lambda function for beta, in terms of wavelength.
        
        Returns
        -------
        The refractive index as a lambda function.
        '''
        return lambda vars: [beta(var) * var / (2 * np.pi) for var in vars]
        # return lambda vars: [beta(var) * self.c / var for var in vars]

class SpectralInterferometry:
    def __init__(self):
            pass
        
    def m2nm(self, x):
        return x * 1e9

    def SimulateInterference(self, wavelengths, zero_delay_wavelength, central_wavelength, L_f, bandwidth = float('inf'), fibre_refractive_index_function = RI.RefractiveIndex.n_fs, delta = 0, delta_phi_fit_order = 3, show_plots = False):
        '''
        Returns the spectral interference data for the interference of two similar pulses, with a phase difference of delta phi, as a result of 
        dispersion accumulated in fibre of length L_f with refractive index function entered, and coefficients for the fit to delta phi.

        Parameters
        -------
        wavelengths ([float]): Wavelength data
        zero_delay_wavelength (float): Wavelength where the delay between interfered pulses is zero.
        central_wavelength (float): Central wavelength of the Gaussian envelope of the spectrum.
        L_f (float): The length of the fibre in the interferometer arm.
        bandwidth (float): Wavelength bandwidth of the pulses.
        fibre_refractive_index_function (lambda): A lambda function for the refractive index of the fibre.
         

        Returns
        -------
        [wavelengths, spectral_interference_data_points, delta_phi_fit_coefficients]
        '''
        
        L_f = self.m2nm(L_f)                     # Fibre length [nm]
        L_air = - L_f * RI.RefractiveIndex.n_group(fibre_refractive_index_function, zero_delay_wavelength) + delta                               # Required air difference for ZDW

        # Interfernce
        deltaPhi = 2 * np.pi / wavelengths * (L_air  + L_f * fibre_refractive_index_function(wavelengths))     # Spectral phase difference
        if bandwidth == float('inf'):
            Gaussian = np.ones(len(wavelengths))
        elif isinstance(bandwidth, float):
            Gaussian = np.exp(- ((wavelengths - central_wavelength)/ bandwidth)**2)            # Modelled as a Gaussian spectrum
        else:
            raise NotImplementedError("Not valid argument for bandwidth. Should be of type float")

        # Plots
        if show_plots:
            plt.plot(wavelengths, deltaPhi) 
            delta_phi_coefficients = np.polyfit(wavelengths, deltaPhi, delta_phi_fit_order)
            plt.plot(wavelengths, np.polyval(delta_phi_coefficients, wavelengths), color='r', linestyle='--', label="Fit")
            print("Delta phi coefficients: ",delta_phi_coefficients)
            plt.show()
            plt.plot(wavelengths, Gaussian * np.cos(deltaPhi / 2)**2)
            plt.show()

        return [wavelengths, Gaussian * np.cos(deltaPhi / 2)**2, delta_phi_coefficients]
      
class SI_Functions_Omega():
    '''
    This class contains functions for spectral interference project work.
    The omegas are assumed to be in nm (as c defaults to 3e17 nm/s), unless you manually specify c in the init to be c = 3e8 m/s.
    '''
    def __init__(self, c = 3e17): # Default is in nm/s, assuming that the class is used with wavelengths in nm.
        self.c = c

    def ConvertWavelengthSpectraToOmega(self, wavelengths, intensity):
        import scipy.interpolate as interpolate
        omegas = 2 * np.pi * self.c / wavelengths                                          # Frequency in rad/s
        x_grid = np.linspace(min(omegas), max(omegas), len(omegas))                    # Adjust the number of points as needed        
        linear_interp = interpolate.interp1d(omegas, intensity, kind='linear')       # Perform linear interpolation
        y_interp = linear_interp(x_grid) 
        return [x_grid, y_interp] 
    
    def ConvertOmegaSpectraToWavelength(self, omegas, intensity):
        import scipy.interpolate as interpolate
        wavelengths = 2 * np.pi * self.c / omegas                                          # Frequency in rad/s
        x_grid = np.linspace(min(wavelengths), max(wavelengths), len(wavelengths))                    # Adjust the number of points as needed        
        linear_interp = interpolate.interp1d(wavelengths, intensity, kind='linear')       # Perform linear interpolation
        y_interp = linear_interp(x_grid) 
        return [x_grid, y_interp] 
    
    def _groundAndNormalise(self, y_data):
        y_data = y_data - min(y_data)
        return (y_data - min(y_data))/ max(y_data - min(y_data))

    def DeltaPhiRetrievalProcedure(self, x, y, order = 2, keep_min_freq = 0.08, keep_max_freq = -1, side = "left", show_plots = True, fft_x_lim = [-1e-12, 1e-12], fft_y_lim = None, hanning = False, normalise = False):
        '''
        Retrieves the spectral phase difference from spectral interference fringes, with flat oscillations, approx. between -1 and +1.
         

        Parameters
        -------
        x ([float]): Array of the omegas.
        y ([float]): Intensity of the SI.
        order (int): Order to approximate the phase.
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
        
        # Define the box filter
        box_filter = np.zeros(len(yf), dtype=complex)
        box_filter[idx] = 1
        filtered_fourier_data = yf * box_filter

        # Perform the inverse FFT
        shifted_filtered_fourier_data = np.fft.ifftshift(filtered_fourier_data)
        index_max_before_shift = np.argmax(np.abs(filtered_fourier_data))                               # Find the index of the maximum value before and after the shift
        index_max_after_shift = np.argmax(np.abs(shifted_filtered_fourier_data))
        shift_amount = index_max_after_shift - index_max_before_shift                                   # Calculate the shift amount
        print("Shift amount: ", shift_amount)
        filtered_y = np.fft.ifft(shifted_filtered_fourier_data)

        # Plot the FFT results
        if show_plots == True:
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.plot(x, y)
            plt.title("Signal")
            plt.subplot(2, 1, 2)
            plt.plot(xf, yf, label = "Full fft") # Normalised
            plt.title("FFT")
            if fft_x_lim != None:
                try:
                    plt.xlim(fft_x_lim)  # Limit the x-axis
                except:
                    print("Not valid fft_x_lim.")
            if fft_y_lim != None:
                try:
                    plt.ylim(fft_y_lim)  # Limit the y-axis
                except:
                    print("Not valid fft_y_lim.")
            plt.xlabel("Fourier Domain")
            plt.tight_layout()
            plt.subplot(2, 1, 2)
            if side == "both":
                plt.plot(xf[idx_left], yf[idx_left], color='r', label = "Selected region")
                plt.plot(xf[idx_right], yf[idx_right], color='r')   
            elif side == "right":
                plt.plot(xf[idx], yf[idx], color='r', label = "Selected region")
            elif side == "left":
                plt.plot(xf[idx], yf[idx], color='r', label = "Selected region")
            plt.legend()
            plt.tight_layout()
            plt.show()


            # Plot the original data and the filtered data in the original domain
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.plot(x, y)
            plt.title("Original Signal")
            plt.subplot(2, 1, 2)
            plt.plot(x, np.abs(filtered_y)**2, label="filtered_y")
            # plt.xlim([np.real(x[np.nonzero(filtered_y)[0][0]]), np.real(x[np.nonzero(filtered_y)[0][-1]])])
            plt.title("Filtered Signal (ifft of selected region)")
            plt.tight_layout()
            plt.legend()
            plt.show()
        


        # Extract phase and unwrap
        final_ys = np.zeros(len(filtered_y))
        for i in range(len(filtered_y)):
            final_ys[i] = cmath.phase((filtered_y[i]))

        final_ys = np.unwrap(final_ys)
        # final_ys = final_ys - final_ys[0]
        print(final_ys)
        print("MIN: ", min(final_ys))
        print("MAX: ", max(final_ys))
       
        print("Final ys:")
        print(final_ys)
        
        # Perform the fit
        coefficients = np.polyfit(x, final_ys, order)

        if show_plots == True:
            plt.plot(x, final_ys)
            plt.title("$\Phi(\omega)$")
            plt.xlabel("$\omega$")
            plt.ylabel("Intensity")
            plt.plot(x, np.polyval(coefficients, x), color='r', linestyle='--', label="Extracted phase")
            # plt.plot(x, -1.95698265e-05*x**3 +   6.79351537e-02*x**2 -7.89563528e+01*x +3.04750604e+04,label="Original simulated phase", color='orange', linestyle = '-.')
            
            plt.legend()
            plt.show()
            # plt.xlim([np.real(x[np.nonzero(filtered_y)[0][0]]), np.real(x[np.nonzero(filtered_y)[0][-1]])])
        return [final_ys, coefficients]
    
    def MakeDeltaPhiLambda(coefficients):
        '''
        Makes the phase a lambda function so that it can be used in subsequent functions in this class.
        

        Parameters
        -------
        coefficients ([float]): Coefficients from the output of the 'DeltaPhiRetrievalProcedure'. 
        

        Returns
        -------
        Lambda function for delta phi.
        '''
        import sympy as sp
        return lambda var: np.poly1d(coefficients)(var)
    
    # GIVEN x IS IN WAVELENGTHS
    
    def ObtainBetaFromPhi(self, phi, length):
        '''
        Obtains beta as a function of omega from delta phi.
        

        Parameters
        -------
        phi (lambda): Lambda function for phi.
        length (float): Length of the fibre.
        

        Returns
        -------
        Beta as a function of omega, as a lambda function. 
        '''
        return lambda var: phi(var) / length

        
    def V_g(self, beta):                                      # Input beta as a function of wavelength
        '''
        Obtains the group velocity, v_g, as a function of wavelength from beta (function of omega).
        v_g := (dBeta / dOmega) ^ (-1)
        

        Parameters
        -------
        beta (omega): Lambda function for beta, in terms of omega.
        

        Returns
        -------
        Lambda function for group velocity as a function of omega, as a lambda function. 
        ''' 
        x = sp.symbols('x')
        expr = beta(x)
        beta_1 = diff(expr, x)                                          # Perform differentiation with respect to omega
        return lambda vars: [1 / beta_1.subs(x, var) for var in vars]   # Invert and return

    def DBeta_dOmega(self, beta):
        '''
        Performs the first derivative of beta (inputted as a lambda funciton of omegas) with respect to omega.
        This is otherwise known as beta_1.

        Parameters
        -------
        beta (lambda): Lambda function for beta, in terms of omega.
        

        Returns
        -------
        Lambda function for first derivative of beta wrt omega, otherwise known as beta_1.
        '''
        x = sp.symbols('x')
        expr = beta(x)
        beta_1 = diff(expr, x)                          # Perform differentiation with respect to wavelength
        return lambda var: beta_1.subs(x, var)          

    def GVD(self, beta):                          # Input beta as a function on wavelength
        '''
        Calculates the Group Velocity Dispersion (GVD) otherwise known as beta_2.
        GVD := d^2 Beta / d Omega^2

        Parameters
        -------
        beta (omega): Lambda function for beta, in terms of omega.
        

        Returns
        -------
        A lambda function for the group velocity dispersion.
        '''
        # Need to compute first and second derivative of beta wrt omega:
        from sympy import diff
        import sympy as sp    
        x = sp.symbols('x')
        expr = beta(x)
        d2Beta_dOmega2 = diff(expr, x, 2)           # Second derivative wrt omega

        return lambda vars: [d2Beta_dOmega2.subs(x, var) for var in vars]

    def Big_D(self, beta): # Input beta as a function of omega
        '''
        Performs the second derivative of beta (inputted as a lambda function) wrt omega and then multiplied by scaling factor to get D.
        D := d2Beta / (dLambda * dOmega)
        D := - w**2 / (2pic) * GVD

        Parameters
        -------
        beta (omega): Lambda function for beta, in terms of omega.
        

        Returns
        -------
        D as a lambda function.
        '''
        # Need to compute second derivative of beta wrt omega:
        x = sp.symbols('x')
        expr = beta(x)
        beta_2 = diff(expr, x, 2)           # Second derivative wrt omega

        D = - (x**2) / (2 * np.pi * self.c) * beta_2
        return lambda vars: [D.subs(x, var) for var in vars]

    def Obtain_n(self, beta):
        '''
        Obtains the effective refractive index from beta, inputted as a lambda function.
        n := beta1 * c
        
        Parameters
        -------
        beta (lambda): Lambda function for beta, in terms of omega.
        
        Returns
        -------
        The refractive index as a lambda function.
        '''
        return lambda vars: [beta(var) * self.c for var in vars]
        # return lambda vars: [beta(var) * self.c / var for var in vars]

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
            GVD = np.array(GVD) * 1e24                                      # Converts from s / nm*nm to ps / nm*km (conventional).             
        return GVD

    def Vg_lambda(beta, wavelengths):
        first_derivative = DispersionExtraction.CDA1(beta, wavelengths[1] - wavelengths[0])
        Vg = []
        for i in range(len(beta)):
            Vg.append( - (2 * np.pi * 3e17 / (wavelengths[i]**2)) * (1 / (first_derivative[i])))
        return Vg
            

