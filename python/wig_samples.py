import numpy as np

# ===================================================
def gen_samps(n_samps, support, ordered):
    if(ordered==True):
        # Not working atm. 
        if(np.floor(np.sqrt(n_samps))**2!=n_samps):
            raise ValueError("For ordered sampling, n_samps must be a square integer")
        root_samps = np.floor(np.sqrt(n_samps))
        xx, uu = np.mgrid[support[0]:support[1]:root_samps*1j, support[2]:support[3]:root_samps*1j]
        samples = np.vstack([xx.ravel(), uu.ravel()])
    else:
        samples = np.random.rand(int(support.size/2), n_samps)
        for i in range(0,int(support.size/2)):
            samples[i,:] *= support[2*i+1]-support[2*i]
            samples[i,:] += support[2*i]
        
    return samples
# ----------------------------------------------------

# ===================================================
def gen_samps1(n_samps, support, ordered):
    if(ordered==True):
        if(np.floor(np.sqrt(n_samps))**2!=n_samps):
            raise ValueError("For ordered sampling, n_samps must be a square integer")
        root_samps = np.floor(np.sqrt(n_samps))
        xx, uu = np.mgrid[support[0]:support[1]:root_samps*1j, support[2]:support[3]:root_samps*1j]
        samples = np.vstack([xx.ravel(), uu.ravel()])
    else:
        samples = np.random.rand(int(support.size/2), n_samps)
        samples *= np.array([[support[1]-support[0]], [support[3]-support[2]]])
        samples += np.array([[support[0]], [support[2]]])
        
    return samples
# ----------------------------------------------------

# ===================================================
def gen_samps2(n_samps, support, ordered):
    if(ordered==True):
        if(np.floor((n_samps)**(1/4))**4!=n_samps):
            raise ValueError("For ordered sampling, n_samps must be a square integer")
        root_samps = np.floor((n_samps)**(1/4))
        xx, yy, uu, vv = np.mgrid[support[0]:support[1]:root_samps*1j, support[2]:support[3]:root_samps*1j, support[4]:support[5]:root_samps*1j, support[6]:support[7]:root_samps*1j]
        samples = np.vstack([xx.ravel(), yy.ravel(), uu.ravel(), vv.ravel()])
    else:
        samples = np.random.rand(int(support.size/2), n_samps)
        samples *= np.array([[support[1]-support[0]], [support[3]-support[2]], [support[5]-support[4]], [support[7]-support[6]]])
        samples += np.array([[support[0]], [support[2]], [support[4]], [support[6]]])
        
    return samples
# ----------------------------------------------------

# ===================================================
def gen_samps3(n_samps, support, ordered):
    if(ordered==True):
        if(np.floor((n_samps)**(1/4))**4!=n_samps):
            raise ValueError("For ordered sampling, n_samps must be a square integer")
        root_samps = np.floor((n_samps)**(1/4))
        xx, yy, uu, vv = np.mgrid[support[0]:support[1]:root_samps*1j, support[2]:support[3]:root_samps*1j, support[4]:support[5]:root_samps*1j, support[6]:support[7]:root_samps*1j]
        samples = np.vstack([xx.ravel(), yy.ravel(), uu.ravel(), vv.ravel()])
    else:
        samples = np.random.rand(int(support.size/2), n_samps)
        samples *= np.array([[support[1]-support[0]], [support[3]-support[2]], [support[5]-support[4]], [support[7]-support[6]], [support[9]-support[8]], [support[11]-support[10]]])
        samples += np.array([[support[0]], [support[2]], [support[4]], [support[6]], [support[8]], [support[10]]])
        
    return samples
# ----------------------------------------------------