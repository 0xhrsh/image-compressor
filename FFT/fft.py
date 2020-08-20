from cmath import sqrt,cos,exp,pi
import numpy as np 

def FFT(Y):
    X=np.array(Y,dtype=complex)
    n = len(X)
    w = exp(-2*pi*1j/n)
    if n > 1:
       X = np.append(FFT(X[0::2]),FFT(X[1::2]))
       for k in range(n//2):
            xk = X[k]
            X[k] = xk + w**k*X[k+n//2]
            X[k+n//2] = xk - w**k*X[k+n//2]   
    return X

def INV_FFT2(Y):
    X=np.array(Y,dtype=complex)
    n = len(X)
    w = exp(2*pi*1j/n)
    if n > 1:
       X = np.append(FFT(X[0::2]),FFT(X[1::2]))
       for k in range(n//2):
            xk = X[k]
            X[k] = xk + w**k*X[k+n//2]
            X[k+n//2] = xk - w**k*X[k+n//2]   
    return X/n

def INV_FFT(Y):
    X=Y[1:]
    ans = np.append(Y[0],np.flip(X,0))
    ans = FFT(ans)
    return ans/len(Y)
    


def FFT2(X):
       fft = np.array(X,dtype=complex)
       i=0
       for row in fft:
              fft[i]=FFT(row)
              i+=1
       fft = np.transpose(fft)
       i=0
       for row in fft:
              fft[i]=FFT(row)
              i+=1
       fft = np.transpose(fft)

       return fft

def INV_FFT2(ifft):
       i=0
       for row in ifft:
              ifft[i]=INV_FFT(row)
              i+=1
       ifft = np.transpose(ifft)
       i=0
       for row in ifft:
              ifft[i]=INV_FFT(row)
              i+=1
       ifft = np.transpose(ifft)
       return ifft

# a = [[0,1],[2,3]]
# fft = np.fft.fft2(a)
# ifft1 = INV_FFT2(fft)
# ifft2 = np.fft.ifft2(fft)
# ifft1 = ifft1
# print(ifft2)
# print(ifft1)
