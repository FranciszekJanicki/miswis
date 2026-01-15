import numpy as np
from scipy.integrate import odeint
import scipy.linalg
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from scipy.linalg import solve_continuous_are
from plotly.subplots import make_subplots
import dash
from dash import dcc, html

class PID:
    def __init__(self, Kp, Ki, Kd, windup = 100000000, offset = 0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.windup = windup
        self.offset = offset
        
        self.integral = 0
        self.prev_t = 0
        self.prev_e = 0
    def step(self, curr_t, curr_e):
        delta_t = curr_t-self.prev_t
        delta_e = curr_e-self.prev_e

        proportion = self.Kp * curr_e
        self.integral = self.integral + self.Ki*curr_e*delta_t
        if(self.integral > self.windup):
            self.integral = self.windup + self.offset
        elif(self.integral < -self.windup):
            self.integral = -self.windup - self.offset
        if(abs(delta_t) < 0.001):
            derivative = 0
        else:
            derivative = self.Kd*delta_e/delta_t

        u_signal = proportion + self.integral + derivative + self.offset

        self.prev_e = curr_e
        self.prev_t = curr_t
        return u_signal
    
class LQR:
    def __init__(self, Ki_theta):
        self.Ki_theta = Ki_theta
        self.integral_theta = 0
        self.prev_t = 0
    def step(self, curr_t, theta, dtheta, phi, dphi, x, dx):
        # integral action
        delta_t = curr_t-self.prev_t
        self.prev_t = curr_t
        self.integral_theta = 0 # self.integral_theta + self.Ki_theta*(-theta)*delta_t
        u1 = -(-7.1409*(theta-self.integral_theta) -1.9000*dtheta + 0.0007*phi + 0.0015*dphi -0.7071*x -0.8803*dx)*1000
        u2 = -(-7.1409*(theta-self.integral_theta) -1.9000*dtheta - 0.0007*phi - 0.0015*dphi -0.7071*x -0.8803*dx)*1000
        
        return (u1, u2)

pid = PID(200, 400, 3, 1000)
lqr = LQR(1)

def equations(state, t, params):
    theta, dtheta, phi, dphi, x, dx, px, py = state
    IB, IV, R, b, LT, mB, mW, g = params
    
    # CONTROLLER LQR
    (u1, u2) = lqr.step(t,theta, dtheta, phi, dphi, x, dx)
    uL = u1
    uR = u2

    # WHEELS
    dpsiR = (dx - (b * dphi)/2)/R
    dpsiL = (dx + (b * dphi)/2)/R

    TL = (uL-dpsiL)*10000
    TR = (uR-dpsiR)*10000
    # TL = u1
    # TR = u2

    # SEGWAY
    ddtheta = (-(4*LT**3*R**5*TL*b*mB**2*np.cos(theta)+4*LT**3*R**5*TR*b*mB**2*np.cos(theta)-4*LT**3*R**4*b*g*mB**3*np.sin(theta)-4*LT**3*R**5*TL*b*mB**2*np.cos(theta)**3-4*LT**3*R**5*TR*b*mB**2*np.cos(theta)**3-4*IW**2*LT*b**3*g*mB*np.sin(theta)+2*IW**2*LT**2*b**3*dphi**2*mB*np.sin(2*theta)-2*IW**2*LT**2*b**3*dtheta**2*mB*np.sin(2*theta)+2*LT**4*R**4*b*dphi**2*mB**3*np.sin(2*theta)-2*LT**4*R**4*b*dtheta**2*mB**3*np.sin(2*theta)+2*IW*LT*R**3*TL*b**3*mB*np.cos(theta)+2*IW*LT*R**3*TR*b**3*mB*np.cos(theta)-4*IV*LT*R**4*b*g*mB**2*np.sin(theta)+2*LT*R**5*TL*b**3*mB*mW*np.cos(theta)+2*LT*R**5*TR*b**3*mB*mW*np.cos(theta)-2*LT**4*R**4*b*dphi**2*mB**3*np.sin(2*theta)*np.cos(theta)**2+2*LT**4*R**4*b*dtheta**2*mB**3*np.sin(2*theta)*np.cos(theta)**2+2*IV*LT**2*R**4*b*dphi**2*mB**2*np.sin(2*theta)+4*IW*LT**4*R**2*b*dphi**2*mB**2*np.sin(2*theta)-2*IV*LT**2*R**4*b*dtheta**2*mB**2*np.sin(2*theta)-4*IW*LT**4*R**2*b*dtheta**2*mB**2*np.sin(2*theta)+4*LT**4*R**4*b*dphi**2*mB**2*mW*np.sin(2*theta)-4*LT**4*R**4*b*dtheta**2*mB**2*mW*np.sin(2*theta)-8*LT**4*R**4*b*dphi**2*mB**3*np.cos(theta)*np.sin(theta)+4*LT**4*R**4*b*dtheta**2*mB**3*np.cos(theta)*np.sin(theta)+4*LT**3*R**4*b*g*mB**3*np.cos(theta)**2*np.sin(theta)-2*IW*LT*R**2*b**3*g*mB**2*np.sin(theta)-8*IW*LT**3*R**2*b*g*mB**2*np.sin(theta)+IW*LT**2*R**2*b**3*dphi**2*mB**2*np.sin(2*theta)-IW*LT**2*R**2*b**3*dtheta**2*mB**2*np.sin(2*theta)+4*IV*LT*R**5*TL*b*mB*np.cos(theta)+4*IV*LT*R**5*TR*b*mB*np.cos(theta)-4*LT*R**4*b**3*g*mB*mW**2*np.sin(theta)-2*LT*R**4*b**3*g*mB**2*mW*np.sin(theta)-8*LT**3*R**4*b*g*mB**2*mW*np.sin(theta)+2*LT**2*R**4*b**3*dphi**2*mB*mW**2*np.sin(2*theta)+LT**2*R**4*b**3*dphi**2*mB**2*mW*np.sin(2*theta)-2*LT**2*R**4*b**3*dtheta**2*mB*mW**2*np.sin(2*theta)-LT**2*R**4*b**3*dtheta**2*mB**2*mW*np.sin(2*theta)+8*LT**4*R**4*b*dphi**2*mB**3*np.cos(theta)**3*np.sin(theta)-4*LT**4*R**4*b*dtheta**2*mB**3*np.cos(theta)**3*np.sin(theta)-8*IV*IW*LT*R**2*b*g*mB*np.sin(theta)+16*LT**3*R**4*b*g*mB**3*np.cos(phi)**2*np.sin(phi)**2*np.sin(theta)**3-4*IW*LT**2*R**2*b**3*dphi**2*mB**2*np.cos(theta)*np.sin(theta)+2*IW*LT**2*R**2*b**3*dtheta**2*mB**2*np.cos(theta)*np.sin(theta)+4*IV*IW*LT**2*R**2*b*dphi**2*mB*np.sin(2*theta)-4*IV*IW*LT**2*R**2*b*dtheta**2*mB*np.sin(2*theta)-8*IV*LT*R**4*b*g*mB*mW*np.sin(theta)-4*LT**2*R**4*b**3*dphi**2*mB**2*mW*np.cos(theta)*np.sin(theta)+2*LT**2*R**4*b**3*dtheta**2*mB**2*mW*np.cos(theta)*np.sin(theta)+4*IV*LT**2*R**4*b*dphi**2*mB*mW*np.sin(2*theta)-4*IV*LT**2*R**4*b*dtheta**2*mB*mW*np.sin(2*theta)-4*IW*LT**4*R**2*b*dphi**2*mB**2*np.sin(2*theta)*np.cos(theta)**2+4*IW*LT**4*R**2*b*dtheta**2*mB**2*np.sin(2*theta)*np.cos(theta)**2-8*IW*LT*R**2*b**3*g*mB*mW*np.sin(theta)-16*IW**2*LT**2*b**3*dphi**2*mB*np.cos(phi)**2*np.cos(theta)*np.sin(theta)-4*LT**4*R**4*b*dphi**2*mB**2*mW*np.sin(2*theta)*np.cos(theta)**2+4*LT**4*R**4*b*dtheta**2*mB**2*mW*np.sin(2*theta)*np.cos(theta)**2+4*IW*LT**2*R**2*b**3*dphi**2*mB*mW*np.sin(2*theta)-4*IW*LT**2*R**2*b**3*dtheta**2*mB*mW*np.sin(2*theta)-8*IV*LT**2*R**4*b*dphi**2*mB**2*np.cos(theta)*np.sin(theta)+4*IV*LT**2*R**4*b*dtheta**2*mB**2*np.cos(theta)*np.sin(theta)+8*IW*LT**3*R**2*b*g*mB**2*np.cos(theta)**2*np.sin(theta)+8*LT**3*R**4*b*g*mB**2*mW*np.cos(theta)**2*np.sin(theta)-32*LT**4*R**4*b*dphi**2*mB**2*mW*np.cos(phi)**2*np.cos(theta)*np.sin(theta)-16*IW*LT**2*R**3*TL*mB*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)+16*IW*LT**2*R**3*TR*mB*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)+8*LT**3*R**4*b*dphi*dx*mB**3*np.cos(phi)*np.cos(theta)*np.sin(phi)+32*LT**4*R**4*b*dphi**2*mB**3*np.cos(phi)**2*np.cos(theta)*np.sin(phi)**2*np.sin(theta)**3-16*LT**4*R**4*b*dtheta**2*mB**3*np.cos(phi)**2*np.cos(theta)*np.sin(phi)**2*np.sin(theta)**3+32*IW*LT**4*R**2*b*dphi**2*mB**2*np.cos(phi)**2*np.cos(theta)**3*np.sin(theta)-16*LT**2*R**5*TL*mB*mW*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)+16*LT**2*R**5*TR*mB*mW*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)-16*LT**2*R**4*b**3*dphi**2*mB*mW**2*np.cos(phi)**2*np.cos(theta)*np.sin(theta)+32*LT**4*R**4*b*dphi**2*mB**2*mW*np.cos(phi)**2*np.cos(theta)**3*np.sin(theta)+8*LT**4*R**4*b*dphi*dtheta*mB**3*np.cos(phi)*np.cos(theta)**2*np.sin(phi)-8*LT**4*R**4*b*dphi*dtheta*mB**3*np.cos(phi)*np.cos(theta)**4*np.sin(phi)-8*LT**3*R**4*b*dphi*dx*mB**3*np.cos(phi)*np.cos(theta)**3*np.sin(phi)-8*LT**4*R**4*b*dphi**2*mB**3*np.sin(2*theta)*np.cos(phi)**2*np.sin(phi)**2*np.sin(theta)**2+8*LT**4*R**4*b*dtheta**2*mB**3*np.sin(2*theta)*np.cos(phi)**2*np.sin(phi)**2*np.sin(theta)**2-16*LT**3*R**5*TL*b*mB**2*np.cos(phi)**2*np.cos(theta)*np.sin(phi)**2*np.sin(theta)**2-16*LT**3*R**5*TR*b*mB**2*np.cos(phi)**2*np.cos(theta)*np.sin(phi)**2*np.sin(theta)**2+8*IW**2*LT*b**3*dphi*dx*mB*np.cos(phi)*np.cos(theta)*np.sin(phi)-32*IW*LT**4*R**2*b*dphi**2*mB**2*np.cos(phi)**2*np.cos(theta)*np.sin(theta)-16*LT**3*R**4*b*dphi*dx*mB**2*mW*np.cos(phi)*np.cos(theta)**3*np.sin(phi)+8*IV*LT*R**4*b*dphi*dx*mB**2*np.cos(phi)*np.cos(theta)*np.sin(phi)-32*LT**3*R**4*b*dphi*dx*mB**3*np.cos(phi)**3*np.cos(theta)*np.sin(phi)**3*np.sin(theta)**2+4*IW*LT**2*R**2*b**3*dphi*dtheta*mB**2*np.cos(phi)*np.cos(theta)**2*np.sin(phi)-32*IW*LT**2*R**2*b**3*dphi**2*mB*mW*np.cos(phi)**2*np.cos(theta)*np.sin(theta)+4*LT**2*R**4*b**3*dphi*dtheta*mB**2*mW*np.cos(phi)*np.cos(theta)**2*np.sin(phi)+4*IW*LT*R**2*b**3*dphi*dx*mB**2*np.cos(phi)*np.cos(theta)*np.sin(phi)+16*IW*LT**3*R**2*b*dphi*dx*mB**2*np.cos(phi)*np.cos(theta)*np.sin(phi)-64*IW*LT**4*R**2*b*dtheta**2*mB**2*np.cos(phi)**2*np.cos(theta)**3*np.sin(phi)**2*np.sin(theta)-32*LT**4*R**4*b*dphi*dtheta*mB**3*np.cos(phi)**3*np.cos(theta)**2*np.sin(phi)**3*np.sin(theta)**2+8*LT*R**4*b**3*dphi*dx*mB*mW**2*np.cos(phi)*np.cos(theta)*np.sin(phi)+4*LT*R**4*b**3*dphi*dx*mB**2*mW*np.cos(phi)*np.cos(theta)*np.sin(phi)+16*LT**3*R**4*b*dphi*dx*mB**2*mW*np.cos(phi)*np.cos(theta)*np.sin(phi)-64*LT**4*R**4*b*dtheta**2*mB**2*mW*np.cos(phi)**2*np.cos(theta)**3*np.sin(phi)**2*np.sin(theta)-32*IV*IW*LT**2*R**2*b*dphi**2*mB*np.cos(phi)**2*np.cos(theta)*np.sin(theta)+8*IV*LT**2*R**4*b*dphi*dtheta*mB**2*np.cos(phi)*np.cos(theta)**2*np.sin(phi)-16*IW*LT**3*R**2*b*dphi*dx*mB**2*np.cos(phi)*np.cos(theta)**3*np.sin(phi)-32*IV*LT**2*R**4*b*dphi**2*mB*mW*np.cos(phi)**2*np.cos(theta)*np.sin(theta)+16*IV*IW*LT*R**2*b*dphi*dx*mB*np.cos(phi)*np.cos(theta)*np.sin(phi)+16*IV*LT*R**4*b*dphi*dx*mB*mW*np.cos(phi)*np.cos(theta)*np.sin(phi)-32*IW*LT**3*R**2*b*dtheta*dx*mB**2*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)+16*IW*LT*R**2*b**3*dphi*dx*mB*mW*np.cos(phi)*np.cos(theta)*np.sin(phi)+16*IW*LT**4*R**2*b*dtheta**2*mB**2*np.sin(2*phi)*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)-32*LT**3*R**4*b*dtheta*dx*mB**2*mW*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)+16*LT**4*R**4*b*dtheta**2*mB**2*mW*np.sin(2*phi)*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)+16*IW*LT**4*R**2*b*dphi*dtheta*mB**2*np.sin(2*theta)*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)+16*LT**4*R**4*b*dphi*dtheta*mB**2*mW*np.sin(2*theta)*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta))/(2*b*(2*IB*IW**2*b**2+2*IB*IV*R**4*mB+4*IB*IV*R**4*mW+2*IB*LT**2*R**4*mB**2+2*IB*R**4*b**2*mW**2+4*IB*IV*IW*R**2-2*IB*LT**2*R**4*mB**2*np.cos(theta)**2+4*IW*LT**4*R**2*mB**2*np.cos(theta)**2-4*IW*LT**4*R**2*mB**2*np.cos(theta)**4+2*IW**2*LT**2*b**2*mB*np.cos(theta)**2+4*LT**4*R**4*mB**2*mW*np.cos(theta)**2-4*LT**4*R**4*mB**2*mW*np.cos(theta)**4+4*IB*IW*LT**2*R**2*mB+IB*IW*R**2*b**2*mB+4*IB*IW*R**2*b**2*mW+4*IB*LT**2*R**4*mB*mW+IB*R**4*b**2*mB*mW-4*IB*LT**2*R**4*mB*mW*np.cos(theta)**2+4*IV*LT**2*R**4*mB*mW*np.cos(theta)**2+2*LT**2*R**4*b**2*mB*mW**2*np.cos(theta)**2-4*IB*IW*LT**2*R**2*mB*np.cos(theta)**2+4*IV*IW*LT**2*R**2*mB*np.cos(theta)**2-8*IB*LT**2*R**4*mB**2*np.cos(phi)**2*np.sin(phi)**2*np.sin(theta)**2+4*IW*LT**2*R**2*b**2*mB*mW*np.cos(theta)**2-16*IW*LT**4*R**2*mB**2*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)**2-16*LT**4*R**4*mB**2*mW*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)**2)))

    ddphi = ((2*(IB*R**5*TL*mB-IB*R**5*TR*mB+2*IB*R**5*TL*mW-2*IB*R**5*TR*mW+2*IB*IW*R**3*TL-2*IB*IW*R**3*TR+2*LT**2*R**5*TL*mB*mW*np.cos(theta)**2-2*LT**2*R**5*TR*mB*mW*np.cos(theta)**2+2*IW*LT**2*R**3*TL*mB*np.cos(theta)**2-2*IW*LT**2*R**3*TR*mB*np.cos(theta)**2-IB*LT**2*R**4*b*dtheta**2*mB**2*np.sin(2*phi)-2*IB*IW*LT**2*R**2*b*dtheta**2*mB*np.sin(2*phi)-IB*LT**2*R**4*b*dphi*dtheta*mB**2*np.sin(2*theta)-2*IB*LT**2*R**4*b*dtheta**2*mB*mW*np.sin(2*phi)-2*IW*LT**4*R**2*b*dtheta**2*mB**2*np.sin(2*phi)*np.cos(theta)**2-2*LT**4*R**4*b*dtheta**2*mB**2*mW*np.sin(2*phi)*np.cos(theta)**2+8*LT**4*R**4*b*dtheta**2*mB**2*mW*np.cos(phi)*np.cos(theta)**4*np.sin(phi)-2*IB*IW*LT**2*R**2*b*dphi*dtheta*mB*np.sin(2*theta)-2*IB*LT**2*R**4*b*dphi*dtheta*mB*mW*np.sin(2*theta)+8*IB*LT**2*R**4*b*dphi**2*mB**2*np.cos(phi)**3*np.sin(phi)*np.sin(theta)**2+2*IB*LT*R**5*TL*b*mB*np.cos(phi)*np.sin(phi)*np.sin(theta)+2*IB*LT*R**5*TR*b*mB*np.cos(phi)*np.sin(phi)*np.sin(theta)-2*IW*LT**4*R**2*b*dphi*dtheta*mB**2*np.sin(2*theta)*np.cos(theta)**2-2*LT**4*R**4*b*dphi*dtheta*mB**2*mW*np.sin(2*theta)*np.cos(theta)**2+4*IB*LT**2*R**4*b*dtheta**2*mB**2*np.cos(phi)*np.cos(theta)**2*np.sin(phi)+8*IW*LT**4*R**2*b*dtheta**2*mB**2*np.cos(phi)*np.cos(theta)**4*np.sin(phi)-4*IB*LT**2*R**4*b*dphi**2*mB**2*np.cos(phi)*np.sin(phi)*np.sin(theta)**2+2*IB*LT**2*R**4*b*dtheta**2*mB**2*np.cos(phi)*np.sin(phi)*np.sin(theta)**2+4*LT**3*R**4*b*dtheta*dx*mB**2*mW*np.cos(phi)*np.cos(theta)**3*np.sin(phi)+2*IB*LT*R**4*b*dtheta*dx*mB**2*np.cos(phi)*np.cos(theta)*np.sin(phi)+4*IW*LT**3*R**2*b*g*mB**2*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)**2+16*IW*LT**4*R**2*b*dphi**2*mB**2*np.cos(phi)**3*np.cos(theta)**2*np.sin(phi)*np.sin(theta)**2+4*LT**3*R**4*b*g*mB**2*mW*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)**2+16*LT**4*R**4*b*dphi**2*mB**2*mW*np.cos(phi)**3*np.cos(theta)**2*np.sin(phi)*np.sin(theta)**2+8*IB*IW*LT**2*R**2*b*dtheta**2*mB*np.cos(phi)*np.cos(theta)**2*np.sin(phi)+4*IW*LT**3*R**2*b*dtheta*dx*mB**2*np.cos(phi)*np.cos(theta)**3*np.sin(phi)+8*IB*LT**2*R**4*b*dtheta**2*mB*mW*np.cos(phi)*np.cos(theta)**2*np.sin(phi)+4*IB*IW*LT*R**2*b*dtheta*dx*mB*np.cos(phi)*np.cos(theta)*np.sin(phi)+4*IB*LT**2*R**4*b*dphi*dtheta*mB**2*np.cos(phi)**2*np.cos(theta)*np.sin(phi)**2*np.sin(theta)+4*IB*LT*R**4*b*dtheta*dx*mB*mW*np.cos(phi)*np.cos(theta)*np.sin(phi)-8*IW*LT**3*R**2*b*dphi*dx*mB**2*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)-2*IW*LT**4*R**2*b*dphi**2*mB**2*np.sin(2*theta)*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)+2*IW*LT**4*R**2*b*dtheta**2*mB**2*np.sin(2*theta)*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)-8*LT**3*R**4*b*dphi*dx*mB**2*mW*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)-2*LT**4*R**4*b*dphi**2*mB**2*mW*np.sin(2*theta)*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)+2*LT**4*R**4*b*dtheta**2*mB**2*mW*np.sin(2*theta)*np.cos(phi)*np.cos(theta)*np.sin(phi)*np.sin(theta)))/(b*(2*IB*IW**2*b**2+2*IB*IV*R**4*mB+4*IB*IV*R**4*mW+2*IB*LT**2*R**4*mB**2+2*IB*R**4*b**2*mW**2+4*IB*IV*IW*R**2-2*IB*LT**2*R**4*mB**2*np.cos(theta)**2+4*IW*LT**4*R**2*mB**2*np.cos(theta)**2-4*IW*LT**4*R**2*mB**2*np.cos(theta)**4+2*IW**2*LT**2*b**2*mB*np.cos(theta)**2+4*LT**4*R**4*mB**2*mW*np.cos(theta)**2-4*LT**4*R**4*mB**2*mW*np.cos(theta)**4+4*IB*IW*LT**2*R**2*mB+IB*IW*R**2*b**2*mB+4*IB*IW*R**2*b**2*mW+4*IB*LT**2*R**4*mB*mW+IB*R**4*b**2*mB*mW-4*IB*LT**2*R**4*mB*mW*np.cos(theta)**2+4*IV*LT**2*R**4*mB*mW*np.cos(theta)**2+2*LT**2*R**4*b**2*mB*mW**2*np.cos(theta)**2-4*IB*IW*LT**2*R**2*mB*np.cos(theta)**2+4*IV*IW*LT**2*R**2*mB*np.cos(theta)**2-8*IB*LT**2*R**4*mB**2*np.cos(phi)**2*np.sin(phi)**2*np.sin(theta)**2+4*IW*LT**2*R**2*b**2*mB*mW*np.cos(theta)**2-16*IW*LT**4*R**2*mB**2*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)**2-16*LT**4*R**4*mB**2*mW*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)**2)))

    ddx = ((4*IB*IV*R**5*TL*b+4*IB*IV*R**5*TR*b+2*IB*IW*R**3*TL*b**3+2*IB*IW*R**3*TR*b**3+2*IB*R**5*TL*b**3*mW+2*IB*R**5*TR*b**3*mW+4*IB*LT**2*R**5*TL*b*mB+4*IB*LT**2*R**5*TR*b*mB+4*LT**4*R**5*TL*b*mB**2*np.cos(theta)**2-4*LT**4*R**5*TL*b*mB**2*np.cos(theta)**4+4*LT**4*R**5*TR*b*mB**2*np.cos(theta)**2-4*LT**4*R**5*TR*b*mB**2*np.cos(theta)**4-4*LT**4*R**4*b*g*mB**3*np.cos(theta)*np.sin(theta)-2*LT**5*R**4*b*dphi**2*mB**3*np.sin(2*theta)*np.cos(theta)**3+2*LT**5*R**4*b*dtheta**2*mB**3*np.sin(2*theta)*np.cos(theta)**3-4*IB*LT**2*R**5*TL*b*mB*np.cos(theta)**2-4*IB*LT**2*R**5*TR*b*mB*np.cos(theta)**2+4*IV*LT**2*R**5*TL*b*mB*np.cos(theta)**2+4*IV*LT**2*R**5*TR*b*mB*np.cos(theta)**2+4*LT**4*R**4*b*g*mB**3*np.cos(theta)**3*np.sin(theta)+2*IW*LT**2*R**3*TL*b**3*mB*np.cos(theta)**2+2*IW*LT**2*R**3*TR*b**3*mB*np.cos(theta)**2+2*LT**5*R**4*b*dphi**2*mB**3*np.sin(2*theta)*np.cos(theta)-8*LT**5*R**4*b*dphi**2*mB**3*np.cos(theta)**2*np.sin(theta)+8*LT**5*R**4*b*dphi**2*mB**3*np.cos(theta)**4*np.sin(theta)-2*LT**5*R**4*b*dtheta**2*mB**3*np.sin(2*theta)*np.cos(theta)+4*LT**5*R**4*b*dtheta**2*mB**3*np.cos(theta)**2*np.sin(theta)-4*LT**5*R**4*b*dtheta**2*mB**3*np.cos(theta)**4*np.sin(theta)-8*IB*LT**3*R**4*b*dphi**2*mB**2*np.sin(theta)+4*IB*LT**3*R**4*b*dtheta**2*mB**2*np.sin(theta)+2*LT**2*R**5*TL*b**3*mB*mW*np.cos(theta)**2+2*LT**2*R**5*TR*b**3*mB*mW*np.cos(theta)**2+8*IB*LT*R**5*TL*mB*np.cos(phi)*np.sin(phi)*np.sin(theta)-8*IB*LT*R**5*TR*mB*np.cos(phi)*np.sin(phi)*np.sin(theta)+16*IB*LT**3*R**4*b*dphi**2*mB**2*np.cos(phi)**2*np.sin(theta)+8*IB*LT**3*R**4*b*dphi**2*mB**2*np.cos(theta)**2*np.sin(theta)-4*IB*LT**3*R**4*b*dtheta**2*mB**2*np.cos(theta)**2*np.sin(theta)+2*IV*LT**3*R**4*b*dphi**2*mB**2*np.sin(2*theta)*np.cos(theta)-8*IV*LT**3*R**4*b*dphi**2*mB**2*np.cos(theta)**2*np.sin(theta)-2*IV*LT**3*R**4*b*dtheta**2*mB**2*np.sin(2*theta)*np.cos(theta)+4*IV*LT**3*R**4*b*dtheta**2*mB**2*np.cos(theta)**2*np.sin(theta)-4*IV*LT**2*R**4*b*g*mB**2*np.cos(theta)*np.sin(theta)-8*IB*IV*LT*R**4*b*dphi**2*mB*np.sin(theta)+4*IB*IV*LT*R**4*b*dtheta**2*mB*np.sin(theta)+IW*LT**3*R**2*b**3*dphi**2*mB**2*np.sin(2*theta)*np.cos(theta)-4*IW*LT**3*R**2*b**3*dphi**2*mB**2*np.cos(theta)**2*np.sin(theta)-IW*LT**3*R**2*b**3*dtheta**2*mB**2*np.sin(2*theta)*np.cos(theta)+2*IW*LT**3*R**2*b**3*dtheta**2*mB**2*np.cos(theta)**2*np.sin(theta)+LT**3*R**4*b**3*dphi**2*mB**2*mW*np.sin(2*theta)*np.cos(theta)-4*LT**3*R**4*b**3*dphi**2*mB**2*mW*np.cos(theta)**2*np.sin(theta)-LT**3*R**4*b**3*dtheta**2*mB**2*mW*np.sin(2*theta)*np.cos(theta)+2*LT**3*R**4*b**3*dtheta**2*mB**2*mW*np.cos(theta)**2*np.sin(theta)-2*IW*LT**2*R**2*b**3*g*mB**2*np.cos(theta)*np.sin(theta)-4*IB*IW*LT*R**2*b**3*dphi**2*mB*np.sin(theta)+2*IB*IW*LT*R**2*b**3*dtheta**2*mB*np.sin(theta)-2*LT**2*R**4*b**3*g*mB**2*mW*np.cos(theta)*np.sin(theta)-4*IB*LT*R**4*b**3*dphi**2*mB*mW*np.sin(theta)+2*IB*LT*R**4*b**3*dtheta**2*mB*mW*np.sin(theta)+8*IB*LT*R**4*b**3*dphi**2*mB*mW*np.cos(phi)**2*np.sin(theta)-16*LT**4*R**5*TL*b*mB**2*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)**2-16*LT**4*R**5*TR*b*mB**2*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)**2-16*IB*LT**3*R**4*b*dphi**2*mB**2*np.cos(phi)**2*np.cos(theta)**2*np.sin(theta)+16*IB*IV*LT*R**4*b*dphi**2*mB*np.cos(phi)**2*np.sin(theta)+8*LT**5*R**4*b*dphi*dtheta*mB**3*np.cos(phi)*np.cos(theta)**3*np.sin(phi)-8*LT**5*R**4*b*dphi*dtheta*mB**3*np.cos(phi)*np.cos(theta)**5*np.sin(phi)+8*LT**4*R**4*b*dphi*dx*mB**3*np.cos(phi)*np.cos(theta)**2*np.sin(phi)-8*LT**4*R**4*b*dphi*dx*mB**3*np.cos(phi)*np.cos(theta)**4*np.sin(phi)+32*LT**5*R**4*b*dphi**2*mB**3*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)**3-16*LT**5*R**4*b*dtheta**2*mB**3*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)**3+16*LT**4*R**4*b*g*mB**3*np.cos(phi)**2*np.cos(theta)*np.sin(phi)**2*np.sin(theta)**3+8*IB*IW*LT*R**2*b**3*dphi**2*mB*np.cos(phi)**2*np.sin(theta)+4*IW*LT**3*R**2*b**3*dphi*dtheta*mB**2*np.cos(phi)*np.cos(theta)**3*np.sin(phi)+4*IW*LT**2*R**2*b**3*dphi*dx*mB**2*np.cos(phi)*np.cos(theta)**2*np.sin(phi)+4*LT**3*R**4*b**3*dphi*dtheta*mB**2*mW*np.cos(phi)*np.cos(theta)**3*np.sin(phi)+4*LT**2*R**4*b**3*dphi*dx*mB**2*mW*np.cos(phi)*np.cos(theta)**2*np.sin(phi)-8*LT**5*R**4*b*dphi**2*mB**3*np.sin(2*theta)*np.cos(phi)**2*np.cos(theta)*np.sin(phi)**2*np.sin(theta)**2+8*LT**5*R**4*b*dtheta**2*mB**3*np.sin(2*theta)*np.cos(phi)**2*np.cos(theta)*np.sin(phi)**2*np.sin(theta)**2+8*IB*LT**3*R**4*b*dphi*dtheta*mB**2*np.cos(phi)*np.cos(theta)*np.sin(phi)+32*IB*LT**3*R**4*b*dtheta**2*mB**2*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)-32*LT**5*R**4*b*dphi*dtheta*mB**3*np.cos(phi)**3*np.cos(theta)**3*np.sin(phi)**3*np.sin(theta)**2-32*LT**4*R**4*b*dphi*dx*mB**3*np.cos(phi)**3*np.cos(theta)**2*np.sin(phi)**3*np.sin(theta)**2-8*IB*LT**3*R**4*b*dtheta**2*mB**2*np.sin(2*phi)*np.cos(phi)*np.sin(phi)*np.sin(theta)-8*IB*LT**3*R**4*b*dphi*dtheta*mB**2*np.cos(phi)*np.cos(theta)**3*np.sin(phi)+8*IV*LT**3*R**4*b*dphi*dtheta*mB**2*np.cos(phi)*np.cos(theta)**3*np.sin(phi)+8*IV*LT**2*R**4*b*dphi*dx*mB**2*np.cos(phi)*np.cos(theta)**2*np.sin(phi)+8*IB*IV*LT*R**4*b*dphi*dtheta*mB*np.cos(phi)*np.cos(theta)*np.sin(phi)+16*IB*LT**2*R**4*b*dtheta*dx*mB**2*np.cos(phi)**2*np.cos(theta)*np.sin(phi)**2*np.sin(theta)+4*IB*IW*LT*R**2*b**3*dphi*dtheta*mB*np.cos(phi)*np.cos(theta)*np.sin(phi)+4*IB*LT*R**4*b**3*dphi*dtheta*mB*mW*np.cos(phi)*np.cos(theta)*np.sin(phi)-8*IB*LT**3*R**4*b*dphi*dtheta*mB**2*np.sin(2*theta)*np.cos(phi)*np.sin(phi)*np.sin(theta))/(2*b*(2*IB*IW**2*b**2+2*IB*IV*R**4*mB+4*IB*IV*R**4*mW+2*IB*LT**2*R**4*mB**2+2*IB*R**4*b**2*mW**2+4*IB*IV*IW*R**2-2*IB*LT**2*R**4*mB**2*np.cos(theta)**2+4*IW*LT**4*R**2*mB**2*np.cos(theta)**2-4*IW*LT**4*R**2*mB**2*np.cos(theta)**4+2*IW**2*LT**2*b**2*mB*np.cos(theta)**2+4*LT**4*R**4*mB**2*mW*np.cos(theta)**2-4*LT**4*R**4*mB**2*mW*np.cos(theta)**4+4*IB*IW*LT**2*R**2*mB+IB*IW*R**2*b**2*mB+4*IB*IW*R**2*b**2*mW+4*IB*LT**2*R**4*mB*mW+IB*R**4*b**2*mB*mW-4*IB*LT**2*R**4*mB*mW*np.cos(theta)**2+4*IV*LT**2*R**4*mB*mW*np.cos(theta)**2+2*LT**2*R**4*b**2*mB*mW**2*np.cos(theta)**2-4*IB*IW*LT**2*R**2*mB*np.cos(theta)**2+4*IV*IW*LT**2*R**2*mB*np.cos(theta)**2-8*IB*LT**2*R**4*mB**2*np.cos(phi)**2*np.sin(phi)**2*np.sin(theta)**2+4*IW*LT**2*R**2*b**2*mB*mW*np.cos(theta)**2-16*IW*LT**4*R**2*mB**2*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)**2-16*LT**4*R**4*mB**2*mW*np.cos(phi)**2*np.cos(theta)**2*np.sin(phi)**2*np.sin(theta)**2)))

    # POSITION
    dpx = np.cos(phi) * dx
    dpy = np.sin(phi) * dx

    return [dtheta, ddtheta, dphi, ddphi, dx, ddx, dpx, dpy]

IB, IW,  IV, R, b, LT, mB, mW, g = 0.1, 0.02, 0.05, 0.05, 0.3, 0.2, 1, 0.5, 9.81
params = (IB, IV, R, b, LT, mB, mW, g)

sim_time = 20
t = np.linspace(0, sim_time, 1000)
initial_state = [0.3, 0, 3.14/4, 0, 0, 0, 0, 0]
solution = odeint(equations, initial_state, t, args=(params,), rtol=1e-4, atol=1e-6, mxstep=50000)

theta = solution[:, 0]
phi   = solution[:, 2]
x     = solution[:, 4]
px    = solution[:, 6]
py    = solution[:, 7]

fig = make_subplots(
    rows=2, cols=4,
    specs=[[{"colspan": 4}, None, None, None],
           [{}, {}, {}, {}]],
    subplot_titles=[
        "Self Balancing Robot (Animation)",
        "Trajectory",
        "Theta",
        "Phi",
        "X Position"
    ]
)

fig.add_trace(go.Scatter(x=px, y=py, name="Trajectory"), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=theta, name="θ"), row=2, col=2)
fig.add_trace(go.Scatter(x=t, y=phi, name="φ"), row=2, col=3)
fig.add_trace(go.Scatter(x=t, y=x, name="x"), row=2, col=4)

frames = []
for k in range(len(t)):
    x1, y1 = x[k], R
    x2 = x1 + LT*np.sin(theta[k])
    y2 = y1 + LT*np.cos(theta[k])

    frames.append(go.Frame(
        data=[
            go.Scatter(x=[x1, x2], y=[y1, y2],
                       mode="lines", line=dict(width=6)),
        ],
        name=str(k)
    ))

fig.add_trace(go.Scatter(), row=1, col=1)
fig.frames = frames

fig.update_layout(
    height=700,
    showlegend=False,
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 30, "redraw": True}}]
        }]
    }]
)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Self Balancing Robot – Plotly/Dash"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)