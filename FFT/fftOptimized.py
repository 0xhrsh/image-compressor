from cmath import sqrt,pi
import numpy as np
from pprint import pprint

def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * pi * 1j / N )
    W = np.power( omega, i * j )
    return W

#assuming n =512 to save time
# w = DFT_matrix(8)
# w_t = w.transpose()
# n2=np.power(8,2)
def preset(n):
    global w
    global w_t
    global n2
    n2=np.power(n,2)
    w = DFT_matrix(n)
    w_t = w.transpose()


def FFT2(X):
    return w.dot(X).dot(w_t)

def INV_FFT2(X):
    return np.rot90(w_t.dot(X).dot(w)/n2,2)

# a = [[0,1],[3,4]]
# x = DFT_matrix(len(a))
# x_t = x.transpose()
# y = x.dot(a).dot(x_t)
# z = np.fft.fft2(a)
# # pprint(z)
# # pprint((y//.001)*.001)

# p = x_t.dot(y).dot(x)/np.power(len(y),2)
# pprint(q)
# pprint(p)