from collections import deque

import numpy as np

from scipy.special import comb
from scipy import interpolate

def ls_circle(points):
    '''
    Input: Nx2 points
    Output: cx, cy, r
    '''
    xs = points[:,0]
    ys = points[:,1]

    us = xs - np.mean(xs)
    vs = ys - np.mean(ys)

    Suu = np.sum(us**2)
    Suv = np.sum(us*vs)
    Svv = np.sum(vs**2)
    Suuu = np.sum(us**3)
    Suvv = np.sum(us*vs*vs)
    Svvv = np.sum(vs**3)
    Svuu = np.sum(vs*us*us)

    A = np.array([
        [Suu, Suv],
        [Suv, Svv]
    ])

    b = np.array([1/2.*Suuu+1/2.*Suvv, 1/2.*Svvv+1/2.*Svuu])

    cx, cy = np.linalg.solve(A, b)
    r = np.sqrt(cx*cx+cy*cy+(Suu+Svv)/len(xs))

    cx += np.mean(xs)
    cy += np.mean(ys)

    return np.array([cx, cy]), r


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, fps=10, n=30, **kwargs):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._dt = 1.0 / fps
        self._n = n
        self._window = deque(maxlen=self._n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = sum(self._window) * self._dt
            derivative = (self._window[-1] - self._window[-2]) / self._dt
        else:
            integral = 0.0
            derivative = 0.0

        control = 0.0
        control += self._K_P * error
        control += self._K_I * integral
        control += self._K_D * derivative

        return control


class CustomController():
    def __init__(self, controller_args, k=0.5, n=2, wheelbase=2.89, dt=0.1):
        self._wheelbase = wheelbase
        self._k = k

        self._n = n
        self._t = 0

        self._dt = dt
        self._controller_args = controller_args

        self._e_buffer = deque(maxlen=10)
        

    def run_step(self, alpha, cmd):
        self._e_buffer.append(alpha)
        
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        Kp = self._controller_args[str(cmd)]["Kp"]
        Ki = self._controller_args[str(cmd)]["Ki"]
        Kd = self._controller_args[str(cmd)]["Kd"]

        return (Kp * alpha) + (Kd * _de) + (Ki * _ie)
