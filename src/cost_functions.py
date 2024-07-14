import numpy as np


def mse(setpoint, measurements):
    """
    Mean Squared Error (MSE) cost function
    """
    error = np.array(setpoint) - np.array(measurements)
    return np.mean(error ** 2)


def ise(setpoint, measurements, dt=0.01):
    """
    Integral of Squared Error (ISE) cost function
    """
    error = np.array(setpoint) - np.array(measurements)
    return np.sum(error ** 2) * dt


def iae(setpoint, measurements, dt=0.01):
    """
    Integral of Absolute Error (IAE) cost function
    """
    error = np.array(setpoint) - np.array(measurements)
    return np.sum(np.abs(error)) * dt


def itse(setpoint, measurements, dt=0.01):
    """
    Integral of Time-weighted Squared Error (ITSE) cost function
    """
    error = np.array(setpoint) - np.array(measurements)
    time = np.arange(len(error)) * dt
    return np.sum(time * error ** 2) * dt


def itae(setpoint, measurements, dt=0.01):
    """
    Integral of Time-weighted Absolute Error (ITAE) cost function
    """
    error = np.array(setpoint) - np.array(measurements)
    time = np.arange(len(error)) * dt
    return np.sum(time * np.abs(error)) * dt


def lqr(setpoint, measurements, Q=1, R=1):
    """
    Linear Quadratic Regulator (LQR) cost function
    Q: state cost
    R: control effort cost
    """
    error = np.array(setpoint) - np.array(measurements)
    control_effort = np.diff(measurements)  # Example of control effort
    return Q * np.sum(error ** 2) + R * np.sum(control_effort ** 2)
