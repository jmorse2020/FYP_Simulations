import numpy as np
import matplotlib.pyplot as plt
class PhaseMatching():
    def __init__(self) -> None:
        pass
    
    def BuildMatrixFromArray(array, n=-1, repeat_array_as = "columns"):
        if n == -1: 
            n = len(array)
        if repeat_array_as.lower() == "columns":
            constructed_matrix = np.zeros((len(array), n))
            for i in range(n):
                constructed_matrix[:, i] = np.transpose(array)
            return constructed_matrix
        elif repeat_array_as.lower() == "rows":
            constructed_matrix = np.zeros((n, len(array)))
            for i in range(n):
                constructed_matrix[i, :] = array
            return constructed_matrix
        else:
            return ValueError("Argument given for 'repeat_array_as' is unrecognised. It takes arguments 'rows' or 'columns'")
        
    # Function to convert wavelength to angular frequency
    def lambda_to_ang_freq(wavelength_matrix):
        #return (2*np.pi*3e8*RefractiveIndexClass.RefractiveIndex.n_fs(wavelength_matrix*1.0e-9,parameter="wavelength")) / wavelength_matrix
        #return (2*np.pi*3e8) / wavelength_matrix
        return (2*np.pi*3e8) / wavelength_matrix
    
    def phase_matching(pump_wavelengths, signal_wavelengths, refractive_index, gamma = 2e-4, P = 100, show_plots = True):
        # *** Construct the pump and signal matrices *** #
        pump_matrix_wavelengths = PhaseMatching.BuildMatrixFromArray(pump_wavelengths, len(signal_wavelengths), repeat_array_as = "rows")
        signal_matrix_wavelengths = PhaseMatching.BuildMatrixFromArray(signal_wavelengths, len(pump_wavelengths), repeat_array_as = "columns")

        # *** Convert to frequency *** #
        pump_matrix_omegas = PhaseMatching.lambda_to_ang_freq(pump_matrix_wavelengths)
        signal_matrix_omegas = PhaseMatching.lambda_to_ang_freq(signal_matrix_wavelengths)
        
        # *** Conservation of energy to create idler matrix *** #
        idler_matrix_omegas = 2 * pump_matrix_omegas - signal_matrix_omegas
        
        
        # *** Slightly different conversion needed for wavelength version *** #
        idler_matrix_wavelengths = 1 / (2/(pump_matrix_wavelengths) - 1/(signal_matrix_wavelengths))
        
        #idler_matrix_wavelengths = 2 * pump_matrix_wavelengths - signal_matrix_wavelengths

        
        # *** Find the betas *** #
        pump_betas = refractive_index(pump_matrix_wavelengths*1e9) * pump_matrix_omegas / 3e8
        # print("pump betas")
        # print(pump_betas)
        signal_betas = refractive_index(signal_matrix_wavelengths*1e9) * signal_matrix_omegas / 3e8
        # print("signal betas")
        # print(signal_betas)
        idler_betas = refractive_index(idler_matrix_wavelengths*1e9) * idler_matrix_omegas / 3e8
        # print("wavelength idler matrix")
        # print(idler_betas)

        # *** Phase matching: Momentum conservation *** #
        delta_beta = 2 * pump_betas - signal_betas - idler_betas - 2 * gamma * P
        # plt.imshow(delta_beta, cmap='jet', interpolation='nearest')
        # plt.colorbar()  # Add colorbar to show the scale
        # plt.show()
    
        
        # plt.plot(contour_data)
        if show_plots == True:
            phase_matching_contour = plt.contour(pump_wavelengths, signal_wavelengths, delta_beta, levels=[0], color='k')
            contour_data = phase_matching_contour.collections[0].get_paths()[0].vertices
            plt.xlabel('Pump [m]')
            plt.ylabel('Signal, Idler [m]')
            plt.title('Phase Matching Plot')  
            plt.show() 
            plt.imshow(delta_beta, cmap='jet', interpolation='nearest')
            plt.colorbar()  # Add colorbar to show the scale
            plt.show()
        return delta_beta
    
    def find_loss_wavelengths(w, n_gas, n_wall, wavelengths, ms):
        lambda_ms = []
        for m in ms:
            # print(n_wall(wavelengths*1e9))
            m_val = (2 * n_gas(wavelengths*1e9) * w / wavelengths) * ( (n_wall(wavelengths * 1e9) / n_gas(wavelengths * 1e9))**2 - 1 )**(1/2)
            # print("M_val")
            # print(m_val)
            differences = np.abs(m - m_val)
            # print(differences)
            idx = np.where(min(differences) == differences)[0]
            lambda_ms.append((wavelengths[idx], m))
        return lambda_ms
    
    def find_contour_intersections(contours, x_intercept):
        intersections = []
        for collection in contours.collections:
            paths = collection.get_paths()
            for path in paths:
                vertices = path.vertices
                for i in range(len(vertices) - 1):
                    x1, y1 = vertices[i]
                    x2, y2 = vertices[i + 1]
                    if (x1 <= x_intercept <= x2) or (x2 <= x_intercept <= x1):
                        slope = (y2 - y1) / (x2 - x1)
                        y_intercept = y1 + slope * (x_intercept - x1)
                        intersections.append((x_intercept, y_intercept))
        return intersections