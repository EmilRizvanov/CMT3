import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sympy import nsolve
import pandas as pd


GHz = 1e+9
uA = 1e-6
mm = 1e-3
fF = 1e-15
Phi0 = 2e-15


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 30
rc_params = {'text.usetex': 'True', 'svg.fonttype': 'none'}
plt.figure(figsize=(8, 6))    # 8, 6
size = 24
linewidth = 2.0

plt.rcParams.update(rc_params)
plt.rc('font', size=  size)         
plt.rc('axes', titlesize= size)  
plt.rc('axes', labelsize= size)    
plt.rc('xtick', labelsize= size)   
plt.rc('ytick', labelsize= size)    
plt.tick_params(axis="x",which="both",direction="in",width=1.,length=5,labelsize= size)
plt.tick_params(axis="y",which="major",direction="in",width=1.,length=5,labelsize= size)

Harmonics = {
            "0" : ["$f_{p}$","Blue"],
            "1" : ["$f_{s}$","Green"], 
            "2" : ["$f_{p-s}$","Orange"],
            "3" : ["$f_{2p-s}$","Red"],
            "4" : ["$f_{p+s}$","Purple"],
            "5" : ["$f_{2p}$","Brown"],
            "6" : ["$f_{3p-s}$","Pink"],
            "7" : ["$f_{2p+s}$","Gray"],
            "8" : ["$f_{3p}$","black"],
}


class CME_TWPA:
    def __init__(self, params):
        self.Ic,self.Cjj,self.Cp,self.Cg,self.Ip0,self.Id,self.Is0,self.N,self.n,self.l  = list(params.values())
        I_norm = np.sqrt(2)*self.Ic  
        self.xi =(1/24)*(1/(I_norm**2 + self.Id**2))
        self.epsilon =(1/8)*(2*self.Id / (I_norm**2 + self.Id**2))
        self.L0= Phi0 / (2*np.pi*self.Ic)
        self.L =self.L0*(1 + self.Id**2 /  I_norm **2 ) 
        self.plasma_freq = 1 /(2*np.pi*np.sqrt((self.L*(self.Cjj+self.Cp))))
        self.w_plasma_0 = 1 /(2*np.pi*np.sqrt((self.L0*(self.Cjj+self.Cp))))*2*np.pi
        self.w_plasma = self.plasma_freq*2*np.pi
        self.dZ = self.l / self.N
        self.dX = self.l / (self.N*(self.n+1))
        self.vp = self.dX / np.sqrt(self.L*self.Cg)
        self.Z = np.sqrt(self.L/self.Cg)
        self.Dk3p = 0
        self.Dks = 0


    def ABCD_matrix(self,w):
        Cp = self.Cp
        Cjj = self.Cjj
        Cg = self.Cg
        L = self.L
        w = w + 0j
        n = self.n
        A = 1 - (w**2*L*Cg) / (1 -w**2*L*Cjj)
        B = 1j*w*L / (1 -w**2*L*Cjj)
        C = 1j*w*Cg
        D = 1
        M1 = np.array([[A, B], [C, D]])
        A = 1 - (w**2*L*Cg) / (1 -w**2*L*(Cjj+Cp))
        B = 1j*w*L / (1 -w**2*L*(Cjj+Cp))
        C = 1j*w*Cg
        D = 1
        M2 = np.array([[A, B], [C, D]])
        M = np.linalg.matrix_power(M1, n) @ M2
        return M 

        
    
    def compute_k(self,omega, N):
        matrix = self.ABCD_matrix(omega)
        result = np.eye(2, dtype=matrix.dtype)
        result =np.linalg.matrix_power(matrix, N) 
        K = (1/(1*(self.n+1)*N))*np.arccos((result[0][0]+result[1][1])/2)  # (self.n+1)
        if np.imag(K) < 0:
           K = np.real(K) - 1j*np.imag(K)
        return K

    
    def k_w(self,range,approach, order ):
        omega = np.linspace(range[0]*2*np.pi,range[1]*2*np.pi,1000)
        f =  np.linspace(range[0], range[1], 1000)
        if approach == 'Numeric':
            k = np.array([ self.compute_k(w,order)  for w in  omega ])
        if approach == 'linear':
            k = np.array([ w / self.vp *self.dX for w in  omega ])
        return f, k 

    def equations_CME3(self,x, y, w):
        Ip, Is, Ii, Iip, Ips, I2p, I2ip, I2ps, I3p = y 
        wp, ws = w
        wi = wp - ws
        wip = 2*wp - ws
        wps = wp + ws
        w2p = 2*wp 
        w3p = 3*wp
        w2ip = 3*wp - ws
        w2ps = 2*wp + ws
        kp =  self.compute_k(wp,1) / (self.dZ)
        ks =  self.compute_k(ws,1) / (self.dZ)
        ki =  self.compute_k(wi,1) / (self.dZ)
        kip =  self.compute_k(wip,1) / (self.dZ)
        kps  =  self.compute_k(wps,1) / (self.dZ)
        k2p =  self.compute_k(w2p,1) / (self.dZ)
        k3p =  self.compute_k(w3p,1) / (self.dZ)
        k2ip  = self.compute_k(w2ip,1) / (self.dZ)
        k2ps =  self.compute_k(w2ps,1) / (self.dZ)
        xi = self.xi  
        epsilon = self.epsilon
        self.Dk3p = np.round((-k3p + 3 * kp)*self.dZ,10)
        self.Dks = np.round((kp -ks - ki)*self.dZ,4)
       
        dIp_dx=  (
    6j * I2ip * Ip * kp * xi * np.conj(I2ip)
    + 6j * I2ip * Ips * kp * xi * np.exp(1j * x *  np.real(k2ip - k3p - kp + kps)) * np.conj(I3p)
    + 6j * I2ip * Is * kp * xi * np.exp(1j * x *  np.real(k2ip - k2p - kp + ks)) * np.conj(I2p)
    + 2j * I2ip * epsilon * kp * np.exp(1j * x *  np.real(k2ip - kip - kp)) * np.conj(Iip)
    + 6j * I2ip * kp * xi * np.exp(1j * x *  np.real(k2ip - ki - 2 * kp)) * np.conj(Ii) * np.conj(Ip)
    + 3j * I2p**2 * kp * xi * np.exp(1j * x *  np.real(2 * k2p - k3p - kp)) * np.conj(I3p)
    + 6j * I2p * Ii * kp * xi * np.exp(1j * x *  np.real(k2p + ki - kip - kp)) * np.conj(Iip)
    + 6j * I2p * Iip * kp * xi * np.exp(1j * x *  np.real(-k2ip + k2p + kip - kp)) * np.conj(I2ip)
    + 6j * I2p * Ip * kp * xi * np.conj(I2p)
    + 6j * I2p * Ips * kp * xi * np.exp(1j * x *  np.real(k2p - k2ps - kp + kps)) * np.conj(I2ps)
    + 6j * I2p * Is * kp * xi * np.exp(1j * x *  np.real(k2p - kp - kps + ks)) * np.conj(Ips)
    + 2j * I2p * epsilon * kp * np.exp(1j * x *  np.real(k2p - 2 * kp)) * np.conj(Ip)
    + 6j * I2p * kp * xi * np.exp(1j * x *  np.real(k2p - ki - kp - ks)) * np.conj(Ii) * np.conj(Is)
    + 6j * I2ps * Ii * kp * xi * np.exp(1j * x *  np.real(-k2p + k2ps + ki - kp)) * np.conj(I2p)
    + 6j * I2ps * Iip * kp * xi * np.exp(1j * x *  np.real(k2ps - k3p + kip - kp)) * np.conj(I3p)
    + 6j * I2ps * Ip * kp * xi * np.conj(I2ps)
    + 2j * I2ps * epsilon * kp * np.exp(1j * x *  np.real(k2ps - kp - kps)) * np.conj(Ips)
    + 6j * I2ps * kp * xi * np.exp(1j * x *  np.real(k2ps - 2 * kp - ks)) * np.conj(Ip) * np.conj(Is)
    + 6j * I3p * Ii * kp * xi * np.exp(1j * x *  np.real(-k2ip + k3p + ki - kp)) * np.conj(I2ip)
    + 6j * I3p * Ip * kp * xi * np.conj(I3p)
    + 6j * I3p * Is * kp * xi * np.exp(1j * x *  np.real(-k2ps + k3p - kp + ks)) * np.conj(I2ps)
    + 2j * I3p * epsilon * kp * np.exp(1j * x *  np.real(-k2p + k3p - kp)) * np.conj(I2p)
    + 3j * I3p * kp * xi * np.exp(1j * x *  np.real(k3p - 3 * kp)) * np.conj(Ip)**2
    + 6j * I3p * kp * xi * np.exp(1j * x *  np.real(k3p - ki - kp - kps)) * np.conj(Ii) * np.conj(Ips)
    + 6j * I3p * kp * xi * np.exp(1j * x *  np.real(k3p - kip - kp - ks)) * np.conj(Iip) * np.conj(Is)
    + 6j * Ii * Ip * kp * xi * np.conj(Ii)
    + 6j * Ii * Ips * kp * xi * np.exp(1j * x *  np.real(ki - 2 * kp + kps)) * np.conj(Ip)
    + 2j * Ii * Is * epsilon * kp * np.exp(1j * x *  np.real(ki - kp + ks))
    + 6j * Iip * Ip * kp * xi * np.conj(Iip)
    + 6j * Iip * Ips * kp * xi * np.exp(1j * x *  np.real(-k2p + kip - kp + kps)) * np.conj(I2p)
    + 6j * Iip * Is * kp * xi * np.exp(1j * x *  np.real(kip - 2 * kp + ks)) * np.conj(Ip)
    + 2j * Iip * epsilon * kp * np.exp(1j * x *  np.real(-ki + kip - kp)) * np.conj(Ii)
    + 3j * Ip**2 * kp * xi * np.conj(Ip)
    + 6j * Ip * Ips * kp * xi * np.conj(Ips)
    + 6j * Ip * Is * kp * xi * np.conj(Is)
    + 2j * Ips * epsilon * kp * np.exp(1j * x *  np.real(-kp + kps - ks)) * np.conj(Is)
)
        
        
        dIps_dx =  (
    6j * I2ip * Ips * np.real(kps) * xi * np.conj(I2ip)
    + 6j * I2ip * Is *np.real(kps) * xi * np.exp(1j * x *  np.real(k2ip - kip - kps + ks)) * np.conj(Iip)
    + 3j * I2ip *np.real(kps) * xi * np.exp(1j * x *  np.real(k2ip - 2 * ki - kps)) * np.conj(Ii)**2
    + 3j * I2p**2 * np.real(kps) * xi * np.exp(1j * x *  np.real(-k2ip + 2 * k2p - kps)) * np.conj(I2ip)
    + 6j * I2p * I2ps * np.real(kps) * xi * np.exp(1j * x *  np.real(k2p + k2ps - k3p - kps)) * np.conj(I3p)
    + 6j * I2p * Ip * np.real(kps) * xi * np.exp(1j * x *  np.real(k2p - kip + kp - kps)) * np.conj(Iip)
    + 6j * I2p * Ips * np.real(kps) * xi * np.conj(I2p)
    + 6j * I2p * Is * np.real(kps) * xi * np.exp(1j * x *  np.real(k2p - kp - kps + ks)) * np.conj(Ip)
    + 2j * I2p * epsilon * np.real(kps) * np.exp(1j * x *  np.real(k2p - ki - kps)) * np.conj(Ii)
    + 6j * I2ps * Ii *np.real(kps) * xi * np.exp(1j * x *  np.real(k2ps + ki - kip - kps)) * np.conj(Iip)
    + 6j * I2ps * Iip * np.real(kps) * xi * np.exp(1j * x *  np.real(-k2ip + k2ps + kip - kps)) * np.conj(I2ip)
    + 6j * I2ps * Ip * np.real(kps) * xi * np.exp(1j * x *  np.real(-k2p + k2ps + kp - kps)) * np.conj(I2p)
    + 6j * I2ps * Ips * np.real(kps) * xi * np.conj(I2ps)
    + 6j * I2ps * Is * np.real(kps) * xi * np.exp(1j * x *  np.real(k2ps - 2 * kps + ks)) * np.conj(Ips)
    + 2j * I2ps * epsilon * np.real(kps) * np.exp(1j * x *  np.real(k2ps - kp - kps)) * np.conj(Ip)
    + 6j * I2ps * np.real(kps) * xi * np.exp(1j * x *  np.real(k2ps - ki - kps - ks)) * np.conj(Ii) * np.conj(Is)
    + 6j * I3p * Ip * np.real(kps) * xi * np.exp(1j * x *  np.real(-k2ip + k3p + kp - kps)) * np.conj(I2ip)
    + 6j * I3p * Ips * np.real(kps) * xi * np.conj(I3p)
    + 6j * I3p * Is *np.real(kps) * xi * np.exp(1j * x *  np.real(-k2p + k3p - kps + ks)) * np.conj(I2p)
    + 2j * I3p * epsilon * np.real(kps) * np.exp(1j * x *  np.real(k3p - kip - kps)) * np.conj(Iip)
    + 6j * I3p * np.real(kps) * xi * np.exp(1j * x *  np.real(k3p - ki - kp - kps)) * np.conj(Ii) * np.conj(Ip)
    + 6j * Ii * Ips *np.real(kps) * xi * np.conj(Ii)
    + 3j * Ii * Is**2 * np.real(kps) * xi * np.exp(1j * x *  np.real(ki - kps + 2 * ks))
    + 6j * Iip * Ips * np.real(kps) * xi * np.conj(Iip)
    + 6j * Iip * Is * np.real(kps) * xi * np.exp(1j * x *  np.real(-ki + kip - kps + ks)) * np.conj(Ii)
    + 3j * Ip**2 *np.real(kps) * xi * np.exp(1j * x *  np.real(-ki + 2 * kp - kps)) * np.conj(Ii)
    + 6j * Ip * Ips * np.real(kps) * xi * np.conj(Ip)
    + 2j * Ip * Is * epsilon * np.real(kps) * np.exp(1j * x *  np.real(kp - kps + ks))
    + 3j * Ips**2 * np.real(kps) * xi * np.conj(Ips)
    + 6j * Ips * Is * np.real(kps) * xi * np.conj(Is)
)
        
        dIip_dx = (
    6j * I2ip * I2p * np.real(kip) * xi * np.exp(1j * x *  np.real(k2ip + k2p - k3p - kip)) * np.conj(I3p)
    + 6j * I2ip * Ii * np.real(kip) * xi * np.exp(1j * x *  np.real(k2ip + ki - 2 * kip)) * np.conj(Iip)
    + 6j * I2ip * Iip * np.real(kip)* xi * np.conj(I2ip)
    + 6j * I2ip * Ip * np.real(kip)* xi * np.exp(1j * x *  np.real(k2ip - k2p - kip + kp)) * np.conj(I2p)
    + 6j * I2ip * Ips * np.real(kip) * xi * np.exp(1j * x *  np.real(k2ip - k2ps - kip + kps)) * np.conj(I2ps)
    + 6j * I2ip * Is * np.real(kip)* xi * np.exp(1j * x *  np.real(k2ip - kip - kps + ks)) * np.conj(Ips)
    + 2j * I2ip * epsilon * np.real(kip) * np.exp(1j * x *  np.real(k2ip - kip - kp)) * np.conj(Ip)
    + 6j * I2ip * np.real(kip) * xi * np.exp(1j * x *  np.real(k2ip - ki - kip - ks)) * np.conj(Ii) * np.conj(Is)
    + 3j * I2p**2 * np.real(kip) * xi * np.exp(1j * x *  np.real(2 * k2p - k2ps - kip)) * np.conj(I2ps)
    + 6j * I2p * Ii * np.real(kip) * xi * np.exp(1j * x *  np.real(k2p + ki - kip - kp)) * np.conj(Ip)
    + 6j * I2p * Iip * np.real(kip) * xi * np.conj(I2p)
    + 6j * I2p * Ip * np.real(kip) * xi * np.exp(1j * x *  np.real(k2p - kip + kp - kps)) * np.conj(Ips)
    + 2j * I2p * epsilon * np.real(kip) * np.exp(1j * x *  np.real(k2p - kip - ks)) * np.conj(Is)
    + 6j * I2ps * Ii * np.real(kip) * xi * np.exp(1j * x *  np.real(k2ps + ki - kip - kps)) * np.conj(Ips)
    + 6j * I2ps * Iip * np.real(kip) * xi * np.conj(I2ps)
    + 3j * I2ps * np.real(kip)* xi * np.exp(1j * x *  np.real(k2ps - kip - 2 * ks)) * np.conj(Is)**2
    + 6j * I3p * Ii * np.real(kip) * xi * np.exp(1j * x *  np.real(-k2p + k3p + ki - kip)) * np.conj(I2p)
    + 6j * I3p * Iip * np.real(kip) * xi * np.conj(I3p)
    + 6j * I3p * Ip *np.real(kip) * xi * np.exp(1j * x *  np.real(-k2ps + k3p - kip + kp)) * np.conj(I2ps)
    + 2j * I3p * epsilon * np.real(kip)* np.exp(1j * x *  np.real(k3p - kip - kps)) * np.conj(Ips)
    + 6j * I3p * np.real(kip) * xi * np.exp(1j * x *  np.real(k3p - kip - kp - ks)) * np.conj(Ip) * np.conj(Is)
    + 3j * Ii**2 * Is * np.real(kip) * xi * np.exp(1j * x *  np.real(2 * ki - kip + ks))
    + 6j * Ii * Iip * np.real(kip) * xi * np.conj(Ii)
    + 2j * Ii * Ip * epsilon * np.real(kip) * np.exp(1j * x *  np.real(ki - kip + kp))
    + 6j * Ii * Ips * np.real(kip) * xi * np.exp(1j * x *  np.real(ki - kip + kps - ks)) * np.conj(Is)
    + 3j * Iip**2 * np.real(kip) * xi * np.conj(Iip)
    + 6j * Iip * Ip * np.real(kip) * xi * np.conj(Ip)
    + 6j * Iip * Ips * np.real(kip) * xi * np.conj(Ips)
    + 6j * Iip * Is * np.real(kip) * xi * np.conj(Is)
    + 3j * Ip**2 * np.real(kip) * xi * np.exp(1j * x *  np.real(-kip + 2 * kp - ks)) * np.conj(Is)
)

                   
        
        dIi_dx = (
    6j * I2ip * Ii * np.real(ki) * xi * np.conj(I2ip)
    + 6j * I2ip * Ip * np.real(ki) * xi * np.exp(1j * x *  np.real(k2ip - k3p - ki + kp)) * np.conj(I3p)
    + 6j * I2ip * Is *np.real(ki) * xi * np.exp(1j * x *  np.real(k2ip - k2ps - ki + ks)) * np.conj(I2ps)
    + 2j * I2ip * epsilon * np.real(ki) * np.exp(1j * x *  np.real(k2ip - k2p - ki)) * np.conj(I2p)
    + 6j * I2ip * np.real(ki) * xi * np.exp(1j * x *  np.real(k2ip - 2 * ki - kps)) * np.conj(Ii) * np.conj(Ips)
    + 3j * I2ip * np.real(ki) * xi * np.exp(1j * x *  np.real(k2ip - ki - 2 * kp)) * np.conj(Ip)**2
    + 6j * I2ip * np.real(ki) * xi * np.exp(1j * x *  np.real(k2ip - ki - kip - ks)) * np.conj(Iip) * np.conj(Is)
    + 6j * I2p * Ii * np.real(ki) * xi * np.conj(I2p)
    + 6j * I2p * Iip * np.real(ki) * xi * np.exp(1j * x *  np.real(k2p - k3p - ki + kip)) * np.conj(I3p)
    + 6j * I2p * Ip * np.real(ki)* xi * np.exp(1j * x *  np.real(k2p - k2ps - ki + kp)) * np.conj(I2ps)
    + 2j * I2p * epsilon *np.real(ki) * np.exp(1j * x *  np.real(k2p - ki - kps)) * np.conj(Ips)
    + 6j * I2p * np.real(ki) * xi * np.exp(1j * x *  np.real(k2p - ki - kp - ks)) * np.conj(Ip) * np.conj(Is)
    + 6j * I2ps * Ii * np.real(ki) * xi * np.conj(I2ps)
    + 6j * I2ps * np.real(ki) * xi * np.exp(1j * x *  np.real(k2ps - ki - kps - ks)) * np.conj(Ips) * np.conj(Is)
    + 6j * I3p * Ii * np.real(ki) * xi * np.conj(I3p)
    + 2j * I3p * epsilon * np.real(ki) * np.exp(1j * x *  np.real(-k2ps + k3p - ki)) * np.conj(I2ps)
    + 6j * I3p * np.real(ki) * xi * np.exp(1j * x *  np.real(-k2p + k3p - ki - ks)) * np.conj(I2p) * np.conj(Is)
    + 6j * I3p * np.real(ki) * xi * np.exp(1j * x *  np.real(k3p - ki - kp - kps)) * np.conj(Ip) * np.conj(Ips)
    + 3j * Ii**2 * np.real(ki)* xi * np.conj(Ii)
    + 6j * Ii * Iip * np.real(ki) * xi * np.conj(Iip)
    + 6j * Ii * Ip * np.real(ki) * xi * np.conj(Ip)
    + 6j * Ii * Ips * np.real(ki) * xi * np.conj(Ips)
    + 6j * Ii * Is * np.real(ki) * xi * np.conj(Is)
    + 3j * Iip**2 * np.real(ki) * xi * np.exp(1j * x *  np.real(-k2ip - ki + 2 * kip)) * np.conj(I2ip)
    + 6j * Iip * Ip * np.real(ki) * xi * np.exp(1j * x *  np.real(-k2p - ki + kip + kp)) * np.conj(I2p)
    + 6j * Iip * Ips * np.real(ki) * xi * np.exp(1j * x *  np.real(-k2ps - ki + kip + kps)) * np.conj(I2ps)
    + 6j * Iip * Is * np.real(ki) * xi * np.exp(1j * x *  np.real(-ki + kip - kps + ks)) * np.conj(Ips)
    + 2j * Iip * epsilon * np.real(ki) * np.exp(1j * x *  np.real(-ki + kip - kp)) * np.conj(Ip)
    + 6j * Iip * np.real(ki) * xi * np.exp(1j * x *  np.real(-2 * ki + kip - ks)) * np.conj(Ii) * np.conj(Is)
    + 3j * Ip**2 * np.real(ki) * xi * np.exp(1j * x *  np.real(-ki + 2 * kp - kps)) * np.conj(Ips)
    + 2j * Ip * epsilon * np.real(ki) * np.exp(1j * x *  np.real(-ki + kp - ks)) * np.conj(Is)
    + 3j * Ips * np.real(ki) * xi * np.exp(1j * x *  np.real(-ki + kps - 2 * ks)) * np.conj(Is)**2
)
        

        dI2ip_dx =2*(
                    1j * I2p * Ii * epsilon * np.real(k2ip) * np.exp(1j * x * np.real(-k2ip + k2p + ki)) +
                    1j * I3p * epsilon * np.real(k2ip) * np.exp(1j * x * np.real(-k2ip + k3p - ks)) * np.conj(Is)+
                    1j * Iip * Ip * epsilon * np.real(k2ip) * np.exp(1j * x * np.real(-k2ip + kip + kp))) +\
         + 2* ((3j * I2ip**2 * np.real(k2ip) * xi * np.conj(I2ip)) / 2 \
         + 3j * I2ip * I2p * np.real(k2ip) * xi * np.conj(I2p) \
         + 3j * I2ip * I2ps * np.real(k2ip) * xi * np.conj(I2ps) \
         + 3j * I2ip * I3p * np.real(k2ip) * xi * np.conj(I3p) \
         + 3j * I2ip * Ii * np.real(k2ip) * xi * np.conj(Ii) \
         + 3j * I2ip * Iip * np.real(k2ip) * xi * np.conj(Iip) \
         + 3j * I2ip * Ip * np.real(k2ip) * xi * np.conj(Ip) \
        + 3j * I2ip * Ips * np.real(k2ip) * xi * np.conj(Ips) \
         + 3j * I2ip * Is * np.real(k2ip) * xi * np.conj(Is) \
         + (3j * I2p**2 * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + 2*k2p - kps)) * np.conj(Ips)) / 2 \
         + 3j * I2p * I3p * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + k2p - k2ps + k3p)) * np.conj(I2ps) \
         + 3j * I2p * Iip * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + k2p + kip - kp)) * np.conj(Ip) \
         + 3j * I2p * Ip * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + k2p + kp - ks)) * np.conj(Is) \
         + 3j * I2ps * Ii * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + k2ps + ki - ks)) * np.conj(Is) \
         + 3j * I2ps * Iip * np.real(k2ip)* xi * np.exp(1j * x * np.real(-k2ip + k2ps + kip - kps)) * np.conj(Ips) \
         + 3j * I3p * Ii * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + k3p + ki - kp)) * np.conj(Ip) \
         + 3j * I3p * Iip * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip - k2p + k3p + kip)) * np.conj(I2p) \
         + 3j * I3p * Ip *np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + k3p + kp - kps)) * np.conj(Ips) \
         + (3j * Ii**2 * Ips * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + 2*ki + kps))) / 2 \
         + 3j * Ii * Iip * Is * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + ki + kip + ks)) \
         + (3j * Ii * Ip**2 * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + ki + 2*kp))) / 2 \
         + (3j * Iip**2 * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip - ki + 2*kip)) * np.conj(Ii)) / 2 \
         + 3j * Iip * Ips * np.real(k2ip) * xi * np.exp(1j * x * np.real(-k2ip + kip + kps - ks)) * np.conj(Is))
        

        dI2ps_dx = (
    6j * I2ip * I2ps *  np.real(k2ps)* xi * np.conj(I2ip)
    + 6j * I2ip * Ips * np.real(k2ps) * xi * np.exp(1j * x *  np.real(k2ip - k2ps - kip + kps)) * np.conj(Iip)
    + 6j * I2ip * Is *  np.real(k2ps) * xi * np.exp(1j * x *  np.real(k2ip - k2ps - ki + ks)) * np.conj(Ii)
    + 3j * I2p**2 *  np.real(k2ps)* xi * np.exp(1j * x *  np.real(2 * k2p - k2ps - kip)) * np.conj(Iip)
    + 6j * I2p * I2ps *  np.real(k2ps) * xi * np.conj(I2p)
    + 6j * I2p * I3p *  np.real(k2ps) * xi * np.exp(1j * x * np.real(-k2ip + k2p - k2ps + k3p)) * np.conj(I2ip)
    + 6j * I2p * Ip * np.real(k2ps) * xi * np.exp(1j * x *  np.real(k2p - k2ps - ki + kp)) * np.conj(Ii)
    + 6j * I2p * Ips *  np.real(k2ps) * xi * np.exp(1j * x *  np.real(k2p - k2ps - kp + kps)) * np.conj(Ip)
    + 2j * I2p * Is * epsilon *  np.real(k2ps) * np.exp(1j * x *  np.real(k2p - k2ps + ks))
    + 3j * I2ps**2 *  np.real(k2ps) * xi * np.conj(I2ps)
    + 6j * I2ps * I3p *  np.real(k2ps) * xi * np.conj(I3p)
    + 6j * I2ps * Ii *  np.real(k2ps) * xi * np.conj(Ii)
    + 6j * I2ps * Iip *  np.real(k2ps) * xi * np.conj(Iip)
    + 6j * I2ps * Ip *  np.real(k2ps) * xi * np.conj(Ip)
    + 6j * I2ps * Ips *  np.real(k2ps) * xi * np.conj(Ips)
    + 6j * I2ps * Is *  np.real(k2ps) * xi * np.conj(Is)
    + 6j * I3p * Ip *  np.real(k2ps) * xi * np.exp(1j * x *  np.real(-k2ps + k3p - kip + kp)) * np.conj(Iip)
    + 6j * I3p * Ips *  np.real(k2ps) * xi * np.exp(1j * x *  np.real(-k2p - k2ps + k3p + kps)) * np.conj(I2p)
    + 6j * I3p * Is *  np.real(k2ps) * xi * np.exp(1j * x *  np.real(-k2ps + k3p - kp + ks)) * np.conj(Ip)
    + 2j * I3p * epsilon *  np.real(k2ps) * np.exp(1j * x *  np.real(-k2ps + k3p - ki)) * np.conj(Ii)
    + 6j * Ii * Ips * Is *  np.real(k2ps) * xi * np.exp(1j * x *  np.real(-k2ps + ki + kps + ks))
    + 6j * Iip * Ips *  np.real(k2ps) * xi * np.exp(1j * x *  np.real(-k2ps - ki + kip + kps)) * np.conj(Ii)
    + 3j * Iip * Is**2 *  np.real(k2ps) * xi * np.exp(1j * x *  np.real(-k2ps + kip + 2 * ks))
    + 3j * Ip**2 * Is *  np.real(k2ps) * xi * np.exp(1j * x *  np.real(-k2ps + 2 * kp + ks))
    + 2j * Ip * Ips * epsilon *  np.real(k2ps) * np.exp(1j * x *  np.real(-k2ps + kp + kps))
    + 3j * Ips**2 * np.real(k2ps) * xi * np.exp(1j * x *  np.real(-k2ps + 2 * kps - ks)) * np.conj(Is)
)




        dIs_dx =(
    6j * I2ip * Is * ks * xi * np.conj(I2ip)
    + 6j * I2ip * ks * xi * np.exp(1j * x *  np.real(k2ip - ki - kip - ks)) * np.conj(Ii) * np.conj(Iip)
    + 6j * I2p * Ip * ks * xi * np.exp(1j * x *  np.real(-k2ip + k2p + kp - ks)) * np.conj(I2ip)
    + 6j * I2p * Ips * ks * xi * np.exp(1j * x *  np.real(k2p - k3p + kps - ks)) * np.conj(I3p)
    + 6j * I2p * Is * ks * xi * np.conj(I2p)
    + 2j * I2p * epsilon * ks * np.exp(1j * x *  np.real(k2p - kip - ks)) * np.conj(Iip)
    + 6j * I2p * ks * xi * np.exp(1j * x *  np.real(k2p - ki - kp - ks)) * np.conj(Ii) * np.conj(Ip)
    + 6j * I2ps * Ii * ks * xi * np.exp(1j * x *  np.real(-k2ip + k2ps + ki - ks)) * np.conj(I2ip)
    + 6j * I2ps * Ip * ks * xi * np.exp(1j * x *  np.real(k2ps - k3p + kp - ks)) * np.conj(I3p)
    + 6j * I2ps * Is * ks * xi * np.conj(I2ps)
    + 2j * I2ps * epsilon * ks * np.exp(1j * x *  np.real(-k2p + k2ps - ks)) * np.conj(I2p)
    + 6j * I2ps * ks * xi * np.exp(1j * x * np.real(k2ps - kip - 2 * ks)) * np.conj(Iip) * np.conj(Is)
    + 3j * I2ps * ks * xi * np.exp(1j * x *  np.real(k2ps - 2 * kp - ks)) * np.conj(Ip)**2
    + 6j * I2ps * ks * xi * np.exp(1j * x *  np.real(k2ps - ki - kps - ks)) * np.conj(Ii) * np.conj(Ips)
    + 6j * I3p * Is * ks * xi * np.conj(I3p)
    + 2j * I3p * epsilon * ks * np.exp(1j * x * np.real(-k2ip + k3p - ks)) * np.conj(I2ip)
    + 6j * I3p * ks * xi * np.exp(1j * x *  np.real(-k2p + k3p - ki - ks)) * np.conj(I2p) * np.conj(Ii)
    + 6j * I3p * ks * xi * np.exp(1j * x *  np.real(k3p - kip - kp - ks)) * np.conj(Iip) * np.conj(Ip)
    + 6j * Ii * Ips * ks * xi * np.exp(1j * x *  np.real(ki - kip + kps - ks)) * np.conj(Iip)
    + 6j * Ii * Is * ks * xi * np.conj(Ii)
    + 6j * Iip * Ips * ks * xi * np.exp(1j * x *  np.real(-k2ip + kip + kps - ks)) * np.conj(I2ip)
    + 6j * Iip * Is * ks * xi * np.conj(Iip)
    + 3j * Iip * ks * xi * np.exp(1j * x *  np.real(-2 * ki + kip - ks)) * np.conj(Ii)**2
    + 3j * Ip**2 * ks * xi * np.exp(1j * x *  np.real(-kip + 2 * kp - ks)) * np.conj(Iip)
    + 6j * Ip * Ips * ks * xi * np.exp(1j * x *  np.real(-k2p + kp + kps - ks)) * np.conj(I2p)
    + 6j * Ip * Is * ks * xi * np.conj(Ip)
    + 2j * Ip * epsilon * ks * np.exp(1j * x *  np.real(-ki + kp - ks)) * np.conj(Ii)
    + 3j * Ips**2 * ks * xi * np.exp(1j * x *  np.real(-k2ps + 2 * kps - ks)) * np.conj(I2ps)
    + 6j * Ips * Is * ks * xi * np.conj(Ips)
    + 2j * Ips * epsilon * ks * np.exp(1j * x *  np.real(-kp + kps - ks)) * np.conj(Ip)
    + 6j * Ips * ks * xi * np.exp(1j * x *  np.real(-ki + kps - 2 * ks)) * np.conj(Ii) * np.conj(Is)
    + 3j * Is**2 * ks * xi * np.conj(Is)
)



        if np.imag(k3p) == 0:  
             dI3p_dx =(
    6j * I2ip * I2p * np.real(k3p)* xi * np.exp(1j * x *  np.real(k2ip + k2p - k3p - kip)) * np.conj(Iip)
    + 6j * I2ip * I2ps *np.real(k3p)* xi * np.exp(1j * x *  np.real(k2ip - k2p + k2ps - k3p)) * np.conj(I2p)
    + 6j * I2ip * I3p * np.real(k3p)* xi * np.conj(I2ip)
    + 6j * I2ip * Ip *np.real(k3p) * xi * np.exp(1j * x *  np.real(k2ip - k3p - ki + kp)) * np.conj(Ii)
    + 6j * I2ip * Ips * np.real(k3p)  * xi * np.exp(1j * x *  np.real(k2ip - k3p - kp + kps)) * np.conj(Ip)
    + 2j * I2ip * Is * epsilon * np.real(k3p) * np.exp(1j * x *  np.real(k2ip - k3p + ks))
    + 3j * I2p**2 * np.real(k3p) * xi * np.exp(1j * x *  np.real(2 * k2p - k3p - kp)) * np.conj(Ip)
    + 6j * I2p * I2ps * np.real(k3p) * xi * np.exp(1j * x *  np.real(k2p + k2ps - k3p - kps)) * np.conj(Ips)
    + 6j * I2p * I3p * np.real(k3p)  * xi * np.conj(I2p)
    + 6j * I2p * Ii * Is * np.real(k3p)  * xi * np.exp(1j * x *  np.real(k2p - k3p + ki + ks))
    + 6j * I2p * Iip * np.real(k3p)  * xi * np.exp(1j * x *  np.real(k2p - k3p - ki + kip)) * np.conj(Ii)
    + 2j * I2p * Ip * epsilon * np.real(k3p) * np.exp(1j * x *  np.real(k2p - k3p + kp))
    + 6j * I2p * Ips * np.real(k3p) * xi * np.exp(1j * x *  np.real(k2p - k3p + kps - ks)) * np.conj(Is)
    + 6j * I2ps * I3p * np.real(k3p) * xi * np.conj(I2ps)
    + 2j * I2ps * Ii * epsilon * np.real(k3p) * np.exp(1j * x *  np.real(k2ps - k3p + ki))
    + 6j * I2ps * Iip * np.real(k3p) * xi * np.exp(1j * x *  np.real(k2ps - k3p + kip - kp)) * np.conj(Ip)
    + 6j * I2ps * Ip * np.real(k3p) * xi * np.exp(1j * x *  np.real(k2ps - k3p + kp - ks)) * np.conj(Is)
    + 3j * I3p**2 * np.real(k3p) * xi * np.conj(I3p)
    + 6j * I3p * Ii * np.real(k3p) * xi * np.conj(Ii)
    + 6j * I3p * Iip * np.real(k3p) * xi * np.conj(Iip)
    + 6j * I3p * Ip * np.real(k3p) * xi * np.conj(Ip)
    + 6j * I3p * Ips * np.real(k3p) * xi * np.conj(Ips)
    + 6j * I3p * Is * np.real(k3p) * xi * np.conj(Is)
    + 6j * Ii * Ip * Ips * np.real(k3p) * xi * np.exp(1j * x *  np.real(-k3p + ki + kp + kps))
    + 6j * Iip * Ip * Is * np.real(k3p) * xi * np.exp(1j * x *  np.real(-k3p + kip + kp + ks))
    + 2j * Iip * Ips * epsilon * np.real(k3p) * np.exp(1j * x *  np.real(-k3p + kip + kps))
    + 1j * Ip**3 *  np.real(k3p) * xi * np.exp(1j * x *  np.real(-k3p + 3 * kp))
)

        else:
            delta = 0.01
            dI3p_dx = -1*np.imag(k3p)*delta*I3p

            
        if np.imag(k2p) == 0:
            dI2p_dx = (
     6*1j*I2ip*I2p*k2p**2*xi*np.conj(I2ip)
    + 6*1j*I2ip*I2ps*k2p**2*xi*np.exp(1j*x*(k2ip - k2p + k2ps - k3p))*np.conj(I3p)
    + 6*1j*I2ip*Ip*k2p**2*xi*np.exp(1j*x*(k2ip - k2p - kip + kp))*np.conj(Iip)
    + 6*1j*I2ip*Ips*k2p**2*xi*np.exp(1j*x*(k2ip - 2*k2p + kps))*np.conj(I2p)
    + 6*1j*I2ip*Is*k2p**2*xi*np.exp(1j*x*(k2ip - k2p - kp + ks))*np.conj(Ip)
    + 2*1j*I2ip*epsilon*k2p**2*np.exp(1j*x*(k2ip - k2p - ki))*np.conj(Ii)
    + 3*1j*I2p**2*k2p**2*xi*np.conj(I2p)
    + 6*1j*I2p*I2ps*k2p**2*xi*np.conj(I2ps)
    + 6*1j*I2p*I3p*k2p**2*xi*np.conj(I3p)
    + 6*1j*I2p*Ii*k2p**2*xi*np.conj(Ii)
    + 6*1j*I2p*Iip*k2p**2*xi*np.conj(Iip)
    + 6*1j*I2p*Ip*k2p**2*xi*np.conj(Ip)
    + 6*1j*I2p*Ips*k2p**2*xi*np.conj(Ips)
    + 6*1j*I2p*Is*k2p**2*xi*np.conj(Is)
    + 6*1j*I2ps*Ii*k2p**2*xi*np.exp(1j*x*(-k2p + k2ps + ki - kp))*np.conj(Ip)
    + 6*1j*I2ps*Iip*k2p**2*xi*np.exp(1j*x*(-2*k2p + k2ps + kip))*np.conj(I2p)
    + 6*1j*I2ps*Ip*k2p**2*xi*np.exp(1j*x*(-k2p + k2ps + kp - kps))*np.conj(Ips)
    + 2*1j*I2ps*epsilon*k2p**2*np.exp(1j*x*(-k2p + k2ps - ks))*np.conj(Is)
    + 6*1j*I3p*Ii*k2p**2*xi*np.exp(1j*x*(-k2p + k3p + ki - kip))*np.conj(Iip)
    + 6*1j*I3p*Iip*k2p**2*xi*np.exp(1j*x*(-k2ip - k2p + k3p + kip))*np.conj(I2ip)
    + 6*1j*I3p*Ip*k2p**2*xi*np.exp(1j*x*(-2*k2p + k3p + kp))*np.conj(I2p)
    + 6*1j*I3p*Ips*k2p**2*xi*np.exp(1j*x*(-k2p - k2ps + k3p + kps))*np.conj(I2ps)
    + 6*1j*I3p*Is*k2p**2*xi*np.exp(1j*x*(-k2p + k3p - kps + ks))*np.conj(Ips)
    + 2*1j*I3p*epsilon*k2p**2*np.exp(1j*x*(-k2p + k3p - kp))*np.conj(Ip)
    + 6*1j*I3p*k2p**2*xi*np.exp(1j*x*(-k2p + k3p - ki - ks))*np.conj(Ii)*np.conj(Is)
    + 6*1j*Ii*Ip*Is*k2p**2*xi*np.exp(1j*x*(-k2p + ki + kp + ks))
    + 2*1j*Ii*Ips*epsilon*k2p**2*np.exp(1j*x*(-k2p + ki + kps))
    + 6*1j*Iip*Ip*k2p**2*xi*np.exp(1j*x*(-k2p - ki + kip + kp))*np.conj(Ii)
    + 6*1j*Iip*Ips*k2p**2*xi*np.exp(1j*x*(-k2p + kip - kp + kps))*np.conj(Ip)
    + 2*1j*Iip*Is*epsilon*k2p**2*np.exp(1j*x*(-k2p + kip + ks))
    + 1j*Ip**2*epsilon*k2p**2*np.exp(1j*x*(-k2p + 2*kp))
    + 6*1j*Ip*Ips*k2p**2*xi*np.exp(1j*x*(-k2p + kp + kps - ks))*np.conj(Is)
)
        else:
            delta = 0.0003
            dI2p_dx= -1*np.imag(k2p)*delta*I2p
        
        return [dIp_dx, dIs_dx, dIi_dx, dIip_dx, dIps_dx,   dI2p_dx , dI2ip_dx, dI2ps_dx,dI3p_dx]
    


    def Power_node_CME3(self,w, Ip0, method):  
        self.Ip0 = Ip0
        y0 =np.complex128([self.Ip0 + 0.0j,self.Is0+ 0.0j,  0.0 + 0.0j,0.0 + 0.0j,0.0+ 0.0j,0.1*0.8*self.Ip0 + 0.0j,0.0 + 0.0j,0.0 + 0.0j,0.1*0.6*self.Ip0 + 0.0j])  
        x_span = (0, self.l)  
        x_eval = np.linspace(0,  self.l , int(self.N / 5) ) 
        if method == 'BDF':
             sol = solve_ivp(self.equations_CME3, x_span, y0, args=(list(w.values()),), t_eval=x_eval, rtol = 1e-6, atol = 1e-9,method = method)
             print('BDF')
        else:
            sol = solve_ivp(self.equations_CME3, x_span, y0, args=(list(w.values()),), t_eval=x_eval, method=method,rtol = 1e+90, atol = 1e+90,max_step =self.l/  int(self.N / 5),  first_step = self.l/   int(self.N / 5))
       
        solutions = [10*np.log10(0.5*np.abs((sol.y[x]))**2*self.Z / 0.001) for x in [0,1,2,3,4,5,6,7,8]]  
        return sol.t, solutions 
    

TWPA_Param = {
              "I_crit": 2*uA,
              "C_junction": 12*fF,
              "C_plasma":  396*fF,  #
              "C_ground": 71.5*fF,
              "I_pump": 0.8*uA, #  0.8*uA
              "I_dc":  0.8*uA, #
              "I_signal": 0.005*uA,
              "Number of JJ": 1200,
              "period of Plasma Capacitance": 4,
              "Length": 0.006,   #7.128 0.0042
            }


PTWPA = CME_TWPA(TWPA_Param)

ws = 4.8*GHz
wp1 =9.061*GHz #11.05*GHz

Ip = 1*uA
x,y = PTWPA.Power_node_CME3({'wp': wp1 *2*np.pi,'ws': ws*2*np.pi},Ip,  method = 'RK23') 

for i in [0,1,2,3,4,5,6,7,8]:
    plt.plot(x/ ((PTWPA.n+1)*PTWPA.dZ),y[i],label = Harmonics[str(i)][0], linestyle='-', color = Harmonics[str(i)][1])

plt.xlabel("Unit cell number")
plt.ylabel("Power [dBm]")

#omega,k = PTWPA.k_w([1*GHz,30*GHz],'Numeric',1)
#plt.xlim(1,30) 
#plt.plot(omega/1e+9,np.real(k)/ (2*np.pi),label = r'$\textrm{Re}(k)$',color = 'green',linewidth=linewidth )
#plt.plot(omega/1e+9, np.imag(k)/ (2*np.pi),label = r'$\textrm{Im}(k)$',color = 'red',linewidth=linewidth )
#omega,k = PTWPA.k_w([1*GHz,30*GHz],'linear',1)
#plt.plot(omega/1e+9, k/ (2*np.pi) , label =  r'$ k = \frac{2 \pi f }{v_{p}}$', color = 'orange',linewidth=linewidth )
#plt.xlabel(r"$\textrm{Signal frequency (GHz)}$") #"Unit cell number"
#plt.ylabel(r"$k [2\pi / dz]$")    #"Power [dBm]"

plt.legend(loc = "upper left", ncol = 1)
plt.show()