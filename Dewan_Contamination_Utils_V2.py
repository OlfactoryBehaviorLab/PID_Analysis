import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def get_on_off_times(trial_data):
    passivation_on_times = []
    passivation_off_times = []

    for time in range(1, 31):
        column_name = str(time)
        passivation_on_times.append(trial_data[f'{column_name}PassOnTime'])
        passivation_off_times.append(trial_data[f'{column_name}PassOffTime'])

    passivation_on_times = np.transpose(passivation_on_times)
    passivation_off_times = np.transpose(passivation_off_times)

    return passivation_on_times, passivation_off_times


def plot_pulse_differences(differences, trial_duration, coefficients):
    fig, ax = plt.subplots()
    ax.set_title('Contamination v. deltaT')
    ax.set_xlabel('deltaT (sec)')
    ax.set_ylabel('Contamination Level')

    max_time = 30 * (trial_duration / 1000)
    x_values = np.arange(0, max_time)

    a, b, c = coefficients[0], coefficients[1], coefficients[2]
    new_y_values = np.apply_along_axis(lambda x: exponential(x, a, b, c), x_values)

    ax.plot(x_values, differences, color='b')
    ax.plot(x_values, new_y_values, color='r')



def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def fit_function(differences):
    x_vals = np.linspace(5, 150, 5, num=30)

    parameters, covariance = curve_fit(exponential, x_vals, differences)

    return parameters, covariance

