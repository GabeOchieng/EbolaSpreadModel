from collections import deque, defaultdict
import numpy as np
import numpy.linalg as npla
from numpy.random import  multivariate_normal
import scipy as sp
import scipy.linalg as spla
import tabular
import plot
import cv2
import matplotlib.pyplot as plt

def odeint(f,u0,t0,tb):
    r = spint.ode(f)
    r.set_integrator('dopri5',atol=1e-10,rtol=1e-10)
    r.set_initial_value(u0,t0)
    y = []
    t = []
    dt = 1e-6
    warnings.filterwarnings("ignore",category=UserWarning)
    while r.successful() and r.t < tb:
        r.integrate(min(r.t+dt,tb))
        y.append(r.y)
        t.append(r.t)
    return np.array(t),np.array(y)


# Y = N x (N+8) matrix = [SEIR|DV|M]
# [
#   [S1 E1 I1 R1 rD1 rV1 M1_1 M2_1 ... Mn_1 ]
#   [S2 E2 I2 R2 rD2 rV2 M1_2 M2_2 ... Mn_2 ]
#   ...
# ]
nD = 50.
nV = 10000.
days = 7
optbeta = 0.193064332008
beta, br,mu,eps,gamma, ddr, drr = optbeta,.037/365, 0.012/365, 1./6, 0.1, 0.70/10, 0.30/10
eD = .5
def f(t,Y):
    n = Y.shape[0]
    S,E,I,R,rDV,M = Y[:,0:1],Y[:,1:2],Y[:,2:3],Y[:,3:4],Y[:,4:6],Y[:,6:]
    SEIR = Y[:,:4]
    N = SEIR.sum(axis=1)
    seir = (SEIR.T/N).T
    flow = np.array([M*x for x in seir.T])
    flow_out = flow.sum(axis=1).T
    flow_in = flow.sum(axis=2).T
    bSI = ((beta*S*I).T/N).T
    dDV = np.array([ [ min(max(0,d),nD/days), min(max(0,v),nV/days)] for d,v in rDV ])
    dD = dDV[:,0:1]
    dV = dDV[:,1:2]
    dSEIR = (np.hstack([
                (br-mu)*S-bSI-dV,
                bSI-(eps+mu)*E,
                eps*E-(gamma+ddr+mu)*I-eD*dD,
                drr*I-mu*R+dV+eD*dD]) - flow_out + flow_in
             )*np.array(SEIR>=0,dtype=float)# change for rD, rV to be done
    dM =np.zeros((n,n))
    dY = np.hstack([dSEIR,-dDV,dM])    
    return dY
scr_h,scr_w = 300,400
graph_buffer = np.zeros((scr_h,scr_w,3),np.uint8)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
def gray(x):
    return x,x,x
def display(name,t,Y,abbr):
    cv2.rectangle(graph_buffer,(0,0),(scr_w,scr_h),gray(255),thickness=-1)
    nbars = Y.shape[0]
    maxpop = 5000
    bar_width = (scr_w-40)/nbars
    colors = ((255,255,192),(64,255,255),(64,64,255),(64,192,54))
    for j in xrange(nbars):
        y = Y[j]
        baroffset = 0
        for i in [1,2]:
            x = y[i]
            lh = x*(scr_h-40.)/ maxpop
            cv2.rectangle(graph_buffer,(int(20+j*bar_width),int(scr_h-20 + baroffset-lh)),(int(20+(j+1)*bar_width),int(scr_h-20+baroffset)),colors[i],thickness=-1)
            baroffset = baroffset - lh
        cv2.putText(graph_buffer,'%.3g'%(max(y[1]+y[2],0)),(int(20+j*bar_width),int(scr_h-20 +baroffset-4)), font, .5,gray(0))
        cv2.putText(graph_buffer,abbr[j],(int(20+(j+.35)*bar_width),scr_h-8), font, .5,gray(0))
        cv2.rectangle(graph_buffer,(int(20+j*bar_width),int(scr_h-20 + baroffset)),(int(20+(j+1)*bar_width),int(scr_h-20)),gray(0))
    cv2.rectangle(graph_buffer,(20,20),(scr_w-20,scr_h-20),gray(0),thickness=1)
    cv2.putText(graph_buffer,'t=%s'%t,(20,10), font, .5,gray(0))
    cv2.putText(graph_buffer,name+'_EI',(170,10), font, .5,gray(0))
    cv2.imshow(name+'-EI',graph_buffer)
    
    cv2.rectangle(graph_buffer,(0,0),(scr_w,scr_h),gray(255),thickness=-1)    
    maxpop = 2000000    
    for j in xrange(nbars):
        y = Y[j]
        baroffset = 0
        for i in xrange(4):
            x = y[i]
            lh = x*(scr_h-40.)/ maxpop
            cv2.rectangle(graph_buffer,(int(20+j*bar_width),int(scr_h-20 + baroffset-lh)),(int(20+(j+1)*bar_width),int(scr_h-20+baroffset)),colors[i],thickness=-1)
            baroffset = baroffset - lh
        cv2.putText(graph_buffer,'%.3g'%(max(y[:4].sum(),0)),(int(20+j*bar_width),int(scr_h-20 +baroffset-4)), font, .5,gray(0))
        cv2.putText(graph_buffer,abbr[j],(int(20+(j+.35)*bar_width),scr_h-8), font, .5,gray(0))
        cv2.rectangle(graph_buffer,(int(20+j*bar_width),int(scr_h-20 + baroffset)),(int(20+(j+1)*bar_width),int(scr_h-20)),gray(0))
    cv2.rectangle(graph_buffer,(20,20),(scr_w-20,scr_h-20),gray(0),thickness=1)
    cv2.putText(graph_buffer,'t=%s'%t,(20,10), font, .5,gray(0))
    cv2.putText(graph_buffer,name+'_All',(170,10), font, .5,gray(0))
    cv2.imshow(name+'-All',graph_buffer)

def drug_distributor(YY,nD,nV,T,dt):
    _t = 0
    _T = T
    nnodes = YY[-1].shape[0]
    nDV  = np.array([[nD,nV]]*nnodes)/nnodes
    zero = nDV*0
    while True:        
        if _t >= _T:
            _T = _T + T            
            yield nDV
        else:
            yield zero
        _t = _t+dt

dist_hist = []
def drug_distributor_opt(YY,nD,nV,T,dt):
    _t = 0
    _T = T
    nDV  = np.zeros((YYd[-1].shape[0],2))
    zero = nDV*0
    dist_hist[:]=[]
    while True:        
        if _t >= _T:
            _T = _T + T
            # calculate new nDV            
            Y = YY[-1]
            S,E,I,R,rDV,M = Y[:,0:1],Y[:,1:2],Y[:,2:3],Y[:,3:4],Y[:,4:6],Y[:,6:]
            SEIR = Y[:,:4]
            N = SEIR.sum(axis=1)
            sD = E + I
            sV = ((S*I).T/N).T
            if sV.sum()!=0 and sD.sum()!=0:
                nDV = np.hstack([sD*nD/sD.sum(),sV*nV/sV.sum()])
            elif sV.sum()!=0:
                nDV = np.hstack([zero[:,1:2],sV*nV/sV.sum()])
            elif sD.sum()!=0:
                nDV = np.hstack([sD*nD/sD.sum(),zero[:,1:2]])
            else:
                nDV = zero
            dist_hist.append(nDV)
            yield nDV
        else:
            yield zero
        _t = _t+dt

def write_dist_hist():
    if len(dist_hist)>0:
        disth = np.hstack(dist_hist)
        N = disth.sum(axis=0)
        disth = disth/N
        np.savetxt("mcm2015files/output/dvdist.csv", disth, delimiter=",")

def wdh(name):
    if len(dist_hist)>0:
        disth = np.hstack(dist_hist)
        N = disth.sum(axis=0)
        disth = disth/N
        np.savetxt("mcm2015files/output/%s.csv"%name, disth, delimiter=",")
        print name
    

def savefig(name, t,Y,abbr):
    cv2.rectangle(graph_buffer,(0,0),(scr_w,scr_h),gray(255),thickness=-1)
    nbars = Y.shape[0]
    maxpop = 5000
    bar_width = (scr_w-40)/nbars
    colors = ((255,255,192),(64,255,255),(64,64,255),(64,192,54))
    for j in xrange(nbars):
        y = Y[j]
        baroffset = 0
        for i in [1,2]:
            x = y[i]
            lh = x*(scr_h-40.)/ maxpop
            cv2.rectangle(graph_buffer,(int(20+j*bar_width),int(scr_h-20 + baroffset-lh)),(int(20+(j+1)*bar_width),int(scr_h-20+baroffset)),colors[i],thickness=-1)            
            baroffset = baroffset - lh
        cv2.putText(graph_buffer,'%.3g'%(max(y[1]+y[2],0)),(int(20+j*bar_width),int(scr_h-20 +baroffset-4)), font, .5,gray(0))
        cv2.putText(graph_buffer,abbr[j],(int(20+(j+.35)*bar_width),scr_h-8), font, .5,gray(0))
        cv2.rectangle(graph_buffer,(int(20+j*bar_width),int(scr_h-20 + baroffset)),(int(20+(j+1)*bar_width),int(scr_h-20)),gray(0))
    cv2.rectangle(graph_buffer,(20,20),(scr_w-20,scr_h-20),gray(0),thickness=1)
    cv2.putText(graph_buffer,'t=%s'%t,(20,10), font, .5,gray(0))
    cv2.putText(graph_buffer,name+'-EI',(170,10), font, .5,gray(0))
    cv2.imwrite('mcm2015files/output/'+name+'_EI_%s'%int(t)+'.png',graph_buffer)
    
    cv2.rectangle(graph_buffer,(0,0),(scr_w,scr_h),gray(255),thickness=-1)    
    maxpop = 2000000    
    for j in xrange(nbars):
        y = Y[j]
        baroffset = 0
        for i in xrange(4):
            x = y[i]
            lh = x*(scr_h-40.)/ maxpop
            cv2.rectangle(graph_buffer,(int(20+j*bar_width),int(scr_h-20 + baroffset-lh)),(int(20+(j+1)*bar_width),int(scr_h-20+baroffset)),colors[i],thickness=-1)
            baroffset = baroffset - lh
        cv2.putText(graph_buffer,'%.3g'%(max(y[:4].sum(),0)),(int(20+j*bar_width),int(scr_h-20 +baroffset-4)), font, .5,gray(0))
        cv2.putText(graph_buffer,abbr[j],(int(20+(j+.35)*bar_width),scr_h-8), font, .5,gray(0))
        cv2.rectangle(graph_buffer,(int(20+j*bar_width),int(scr_h-20 + baroffset)),(int(20+(j+1)*bar_width),int(scr_h-20)),gray(0))
    cv2.rectangle(graph_buffer,(20,20),(scr_w-20,scr_h-20),gray(0),thickness=1)
    cv2.putText(graph_buffer,'t=%s'%t,(20,10), font, .5,gray(0))
    cv2.putText(graph_buffer,name+'-All',(170,10), font, .5,gray(0))
    cv2.imwrite('mcm2015files/output/'+name+'_All_%s'%int(t)+'.png',graph_buffer)

def saveIR(TT,YY,YYd,YYo):
    fig,ax = plt.subplots()
    II,RR,IId,RRd,IIo,RRo = YY[:,:,2].sum(axis=1),YY[:,:,3].sum(axis=1),YYd[:,:,2].sum(axis=1),YYd[:,:,3].sum(axis=1),YYo[:,:,2].sum(axis=1),YYo[:,:,3].sum(axis=1)
    ax.plot(TT,II,label='I')
    ax.plot(TT,IId,label='Id')
    ax.plot(TT,IIo,label='Iopt')
    ax.legend(shadow=True,loc=2)
    plt.xlabel('time (days)')
    plt.ylabel('infected count')
    plt.title('comparison between methods: number of infected people')
    plt.savefig('mcm2015files/output/I_cmp_%s.png'%TT[-1])
    fig,ax = plt.subplots()
    ax.plot(TT,RR,label='R')
    ax.plot(TT,RRd,label='Rd')
    ax.plot(TT,RRo,label='Ropt')
    ax.legend(shadow=True,loc=2)
    plt.xlabel('time (days)')
    plt.ylabel('recovered count')
    plt.title('comparison between methods: number of recovered people')
    plt.savefig('mcm2015files/output/R_cmp_%s.png'%TT[-1])


    
if __name__ == "__main__":
    fields = ['S','E','I','R','rD','rV'] + ['from%s'%i for i in xrange(1,11)]
    abbr = ['CO','FR','MO','LA','KA','GR','KI','MA','VA','KO']
    Y = np.array(tabular.tbarr('mcm2015files/Pop_Transfer.csv',fields,{field:'float' for field in fields},{}))
    t = 0
    dt = .1
    final_t = 200    
    idx = 0
    YY = [Y]
    YYd = [Y]
    YYo = [Y]
    TT = [t]
    aborted = False
    playing = True
    playdir = 1
    tlen = 1
    idx = 0
    distributor = drug_distributor(YYd,nD,nV,days,dt)
    distributor_opt = drug_distributor_opt(YYo,nD,nV,days,dt)
    while not aborted:
        display('SEIR',TT[idx],YY[idx],abbr)
        display('SEIRDV',TT[idx],YYd[idx],abbr)
        display('SEIRDVopt',TT[idx],YYo[idx],abbr)
        c = cv2.waitKey(1)
        if c == ord('x'): aborted = True
        if not playing:            
            if c==ord('a'):
                idx = max(idx - int(1/dt),0)
            elif c==ord('d'):
                idx = idx + int(1/dt)
            elif c==ord('s'):
                idx = max(idx - int(100/dt),0)
            elif c==ord('w'):
                idx = idx + int(100/dt)
            elif c==ord('z'):
                idx = 0
            elif c==ord('p'):
                playing = True
            elif c==ord('o'):
                savefig('SEIR',TT[idx],YY[idx],abbr)
                savefig('SEIRDV',TT[idx],YYd[idx],abbr)
                savefig('SEIRDVopt',TT[idx],YYo[idx],abbr)
                saveIR(TT,np.array(YY),np.array(YYd),np.array(YYo))
                write_dist_hist()
                cv2.waitKey(1000)
        else:
            if c==ord('i'):
                playdir = -playdir
            elif c==ord('p'):
                playing = False
            idx = max(idx+playdir,0)
        if idx >= tlen:
            while idx>=tlen:
                Y = YY[-1] + f(TT[-1],YY[-1])*dt
                Y[:,:4] = Y[:,:4] * np.array(Y[:,:4]>0,dtype=float)
                
                Yd = YYd[-1] + f(TT[-1],YYd[-1])*dt                
                delivered = distributor.next()
                Yd[:,4:6] = Yd[:,4:6] + delivered
                Yd[:,:4] = Yd[:,:4] * np.array(Yd[:,:4]>0,dtype=float)

                Yo = YYo[-1] + f(TT[-1],YYo[-1])*dt                
                delivered_opt = distributor_opt.next()
                Yo[:,4:6] = Yo[:,4:6] + delivered_opt
                Yo[:,:4] = Yo[:,:4] * np.array(Yo[:,:4]>0,dtype=float)

                t = t+dt
                
                YY.append(Y)                
                YYd.append(Yd)
                YYo.append(Yo)                
                TT.append(t)

                tlen = tlen+1
                
    cv2.destroyAllWindows()
    Y = np.array(tabular.tbarr('mcm2015files/Pop_Transfer.csv',fields,{field:'float' for field in fields},{}))
    ntest = 10
    tfields = ['S','E','I','R']
    testcases = [np.array(tabular.tbarr('mcm2015files/input/SA%s.csv'%i,tfields,{field:'float' for field in tfields},{})) for i in xrange(ntest)]
    testcases = [np.hstack([t,Y[:,4:]]) for t in testcases]
    testresults = []    
    for i in xrange(ntest):
        y = testcases[i]
        yy = [y]
        tt = [t]
        d_opt = drug_distributor_opt(yy,nD,nV,days,dt)
        t = 0
        while t <= final_t:
            y = yy[-1] + f(t,yy[-1])*dt
            dd = d_opt.next()
            y[:,4:6] = y[:,4:6] + dd
            y[:,:4] = y[:,:4]*np.array(y[:,:4]>0,dtype=float)
            t = t + dt
            yy.append(y)
            tt.append(t)            
        wdh('test%s'%i)
        disth = np.hstack(dist_hist)
        Npop = disth.sum(axis=0)
        disth = disth/Npop
        testresults.append(disth)
    di = []
    do = []
    count = 0
    relchanges = []    
    for i in xrange(ntest):
        for j in xrange(i):
            n_i = npla.norm(testcases[i][:,:4].sum(axis=0) - testcases[j][:,:4].sum(axis=0))            
            n_o = npla.norm(testresults[i] - testresults[j])
            di.append(n_i)
            do.append(n_o)
            if n_i!=0: relchanges.append(n_o/n_i)
            count = count + 1
    
    print min(di),max(di)
    print min(do),max(do)
    print min(relchanges),max(relchanges)
    
