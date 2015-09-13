import numpy as np
import scipy as sp
import scipy.integrate as spint
from scipy.optimize import minimize
import warnings
import matplotlib.pyplot as plt
import tabular

def odeint(f,u0,t0,tb):
    r = spint.ode(f)
    r.set_integrator('dopri5',atol=1e-10,rtol=1e-10)
    r.set_initial_value(u0,t0)
    y = []
    t = []
    dt = 1e-2
    warnings.filterwarnings("ignore",category=UserWarning)
    while r.successful() and r.t < tb:
        r.integrate(min(r.t+dt,tb))
        y.append(r.y)
        t.append(r.t)
    return np.array(t),np.array(y)

# kernel(c) returns f(t,u)
def odefit(kernel,c_init,u0,t0,tb,u_star):
    F = lambda c: npla.norm(odeint(kernel(c),u0,t0,tb) - U_star)
    c_star = minimize(F,c_init)
    return c_star

def ebola_kernel(beta):
    br,mu,eps,gamma, ddr = .037/365, 0.012/365, 1./6, 0.1, 0.70/10
    def f(t,Y):
        S,E,I,R,D, I_c = tuple(Y)
        N = sum(Y[0:4])
        return np.array([
            br*N-beta*S*I/N-mu*S,
            beta*S*I/N - (eps+mu)*E,
            eps*E - (gamma+ddr)*I,
            (1./10 -ddr)*I- mu*R,
            ddr*I,
            eps*E
        ])
    return f


if __name__ == "__main__":
    rfields = ['t','C']
    realdata = np.array(tabular.tbarr('mcm2015files/betatest.csv',rfields,{field:'float' for field in rfields},{}))    
    initState = np.array([22e6,0,86,0,0,86])
    lobound,hibound = 0.0, 1.0
    guess = lobound*0.5+hibound*0.5
    T,Y = odeint(ebola_kernel(guess),initState,0,317)
    ret = Y[-1][5]
    print guess, ret
    while abs(ret-22460) > 0.1:
        if ret-22460 < 0:
            lobound = guess
        else:
            hibound = guess
        guess = lobound*0.5 + hibound*0.5
        T,Y = odeint(ebola_kernel(guess),initState,0,317)
        ret = Y[-1][5]
        print guess, ret
    #f = ebola_kernel(guess)
    #T,Y = odeint(ebola_kernel(guess),initState,0,365*12)
    fig,ax = plt.subplots()
    pltlist = ((1,'E'),(2,'I'),(4,'D'),(5,'C'))
    for k, s in pltlist[3:]:
        ax.plot(T,Y[:,k],label=s)    
    ax.plot(realdata[:,0],realdata[:,1],label='real C')
    legend = ax.legend(shadow=True,loc=2)
    print Y[-1]
    plt.show()
    
