# Generates npy files for K = 90, 100, 110 and alpha = -2, -1, 0 to .9


import numpy as np
import scipy.stats as stats
from scipy.special import gammainc
import concurrent.futures
from scipy.optimize import root


def sigma(sig_ln, F0, alpha):
    ''' Compute sigma'''
    return sig_ln * F0**(1-alpha)

def X(F, alpha, sig):
    ''' X transform variable '''
    return F**(2 * (1-alpha)) / (sig**2 * (1-alpha)**2)

def F_Xinv(X, alpha, sig):
    ''' F from X'''
    return (X * (sig**2 * (1-alpha)**2)) ** (1/(2*(1-alpha)))

def delta(alpha):
    ''' delta from alpha'''
    return (1 - 2*alpha) / (1 - alpha)

def analytical_C(F,K,T, alpha, sig_ln):
    sig = sigma(sig_ln, F, alpha)
    K_tilde = X(K,alpha, sig)
    X0 = X(F,alpha, sig)
    C = F * (1 - stats.ncx2.cdf(x = K_tilde/T, df = 4 - delta(alpha), nc = X0/T)) - K * stats.ncx2.cdf(x = X0/T, df = 2 - delta(alpha),nc = K_tilde/T)
    return C

def inversex2(F, T,K, alpha, sig_ln, M = 20):
    print("K:", K, "alpha:", alpha )
    # transform sig to correct dimension
    sig = sigma(sig_ln, F, alpha)
            
    # Sobol Sample
    s = stats.qmc.Sobol(1)
    N = 2**M
    U = s.random_base2(M).flatten()
    
    # Umax
    v = delta(alpha)/2 - 1
    X0 = X(F, alpha, sig)
    Umax = gammainc(-v, X0/(2*T))
    
    # compute root solve for X_T
    X_T = np.zeros(N)
    def func(x, u):
        return stats.ncx2.cdf(X0/T, df = 2 - delta(alpha), nc = x/T) - Umax + u
    for idx, u in enumerate(U):
        if u > Umax:
            continue
        else:
            X_T[idx] = root(func, X0, args=(u)).x
            
    # compute inverse to get F_T
    F_T = F_Xinv(X_T, alpha, sig)
    # compute C by subtracting K and taking max. Anywhere less than K is 0, otherwise F - K
    C = np.where(F_T<K, 0, F_T-K)
    P = C - (F - K)
    
    return C.mean(), P.mean(), C.std()/(N**(1/2)), P.std()/(N**(1/2))

if __name__ == '__main__':
    # parameters
    F = 100
    alphas = [-2,-1,0,0.1,0.2,.3,.4,.5,.6,.7,.8,.9]
    sig_ln = 0.5
    T = 4
    Ks = [90, 100, 110]
    # power of 2 to sobol sample by
    M = 20
    
    

    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        for K in Ks:
            for alpha in alphas:
                
                futures.append((executor.submit(inversex2,F, T, K, alpha, sig_ln, M), K, alpha))
                
    
    
    for f, K, alpha in futures:
        EC, EP, STEC, STEP = f.result()
        
        aC = analytical_C(F, K, T, alpha, sig_ln)
        aP = aC + K - F

    
        data = {'K': K, 'alpha': alpha,
                'F': F, 'T': T, 'sig_ln': sig_ln, 'sig': sigma(sig_ln, F, alpha),
                'E[C]':EC, 'STE[C]': STEC,
                'E[P]':EP, 'STE[P]': STEP,
                'A[C]':aC, 'A[P]': aP}
        
        np.save(f"data_x2/{K}_{alpha}.npy", data, allow_pickle=True)
        print(data)