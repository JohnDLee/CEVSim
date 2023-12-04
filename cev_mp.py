# Generates npy files for K = 90, 100, 110 and alpha = -2, -1, 0 to .9


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import gammainc
import tqdm
import concurrent.futures


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

def analytical_C(F,K,T, alpha, sig):
    K_tilde = X(K,alpha, sig)
    X0 = X(F,alpha, sig)
    C = F * (1 - stats.ncx2.cdf(x = K_tilde/T, df = 4 - delta(alpha), nc = X0/T)) - K * stats.ncx2.cdf(x = X0/T, df = 2 - delta(alpha),nc = K_tilde/T)
    return C

def em_CEV(T, dt, F0, alpha, sig):
    ''' em scheme simulation '''
    
    # set up Brownian motion and t
    t = np.arange(dt, T + dt, dt)
    N = len(t)
    dW=np.sqrt(dt)*np.random.randn(N)
    
    # init variables
    X0 = X(F0, alpha, sig)
    d = delta(alpha)
    X_emC, X_em = X0, []
    hit_zero = False
    
    # For all timesteps
    for j in range(N):
        # emstein 
        dX = d*dt + 2*np.sqrt(X_emC)*dW[j] #+ (dW[j]**2 - dt)
        X_emC += dX
        
        # if <= 0, presumed it hit boundary of 0
        if X_emC <= 0:
            # fill with 0's since it is absorbed
            X_em += [0 for i in range(N - len(X_em))]
            hit_zero = True
            break
        X_em.append(X_emC)
    
    return X_em, hit_zero

def em_CEV_wrapped(N, T, dt, F0, alpha, sig):
    ''' Em scheme simulation wrapped '''
    num_absorbed = 0
    X_T = []
    for i in range(N):
        em, hz = em_CEV(T, dt, F0, alpha, sig)
        X_T.append(em[-1])
        num_absorbed += int(hz)

    # only save 1,
    return em, np.array(X_T), num_absorbed

if __name__ == '__main__':
    # parameters
    dt = 10**-3
    F = 100
    alphas = [-2,-1,0,0.1,0.2,.3,.4,.5,.6,.7,.8,.9]
    sig_ln = 0.5
    T = 4
    Ks = [90, 100, 110]
    
    for K in Ks:
        for alpha in alphas:
            # transform sig to correct dimension
            sig = sigma(sig_ln, F, alpha)

            # Monte Carlo of CEV Process
            N = 10**(6); NPP = 1000
            t = np.arange(dt, T + dt, dt)
            X_T = []
            num_absorbed = 0
            ems = []


            futures = []
            with concurrent.futures.ProcessPoolExecutor(max_tasks_per_child=5) as executor:
                for i in tqdm.trange( 0, N, NPP):
                    
                    futures.append(executor.submit( em_CEV_wrapped, NPP, T, dt, F, alpha, sig))
                    
                    if i % (10*NPP) == 0:
                        for f in futures:
                            em, fT, hz = f.result()
                            num_absorbed += hz
                            # save final X values
                            X_T.append(fT)
                            ems.append(em)
                            del futures
                            futures = []

            X_T = np.concatenate(X_T)
            # compute inverse to get F
            F_T = F_Xinv(X_T, alpha, sig)
            # compute C by subtracting K and taking max. Anywhere less than K is 0, otherwise F - K
            C = np.where(F_T<K, 0, F_T-K)
            P = C - (F_T - K)
            # print(C)
            # print(C[np.argwhere(C).flatten()])
            aC = analytical_C(F, K, T, alpha, sig)
            aP = aC + K - F
            
            data = {'K': K, 'alpha': alpha,
                    'dt': dt, 'F': F, 'T': T, 'sig_ln': sig_ln, 'sig': sig,
                    'ems': ems,
                    'F_T': F_T,
                    'E[C]':C.mean(), 'STE[C]': C.std()/(N**(1/2)),
                    'E[P]':P.mean(), 'STE[P]': P.std()/(N**(1/2)),
                    'A[C]':aC, 'A[P]': aP,
                    'Sim Absorption Ratio':num_absorbed/N,
                    'Analytical Absorption Ratio': 1-gammainc(-(delta(alpha)/2 - 1), X(F, alpha, sig)/(2*T))}
            
            np.save(f"data/{K}_{alpha}.npy", data, allow_pickle=True)
            print(f"For K={K} and alpha={alpha}")
            print(f"E[C] = {C.mean()}, STD[C] = {C.std()/(N**(1/2))}")
            print(f"Analytical C = {aC}")
            print(f"E[P] = {P.mean()}, STD[P] = {P.std()/(N**(1/2))}")
            print(f"Analytical P = {aP}")
            print(f"Simulated Absorbed Ratio = {num_absorbed/N}")
            print(f"Analytical Absorbed Ratio = {1-gammainc(-(delta(alpha)/2 - 1), X(F, alpha, sig)/(2*T))}")
    