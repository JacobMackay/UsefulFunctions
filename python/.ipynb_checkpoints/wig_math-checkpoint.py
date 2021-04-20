import numpy as np
import sys
import scipy as sp

# ===================================================
def rect_j(x):
    # /// Rectangular function, base length 1.
    
    mask = np.abs(x)<0.5
    result = np.zeros(x.shape)
    result[mask] = 1.0
        
    return result
# ----------------------------------------------------

# ===================================================
def circ_j(x):
    # /// Circular function, radius 1.
    
    mask = np.abs(x)<0.5
    result = np.zeros(x.shape)
    result[mask] = 1.0
        
    return result
# ----------------------------------------------------

# ===================================================
def tri_j(x):
    # /// Triangular function, base length 1.
    
    mask = np.abs(x)<0.5
    result = np.zeros(x.shape)
    result[mask] = 1.0 - 2.0*np.abs(x[mask]) 
        
    return result
# ----------------------------------------------------

# ===================================================
def sinc_j(x):
    # /// Sinc function basic. Sin(x)/x
    # Note python has pix / pix
    
    mask = np.abs(x) > sys.float_info.epsilon
    result = np.ones(x.shape)
    result[mask] = np.sin(x[mask])/x[mask]     
        
    return result
# ----------------------------------------------------

# ===================================================
def rect2_j(r):
    # /// Rectangular function, base length 1.
    
    mask = (np.abs(r[:,0])<0.5) & (np.abs(r[:,1])<0.5)
    result = np.zeros(r.shape)
    result[mask] = 1.0
    
    return result
# ----------------------------------------------------

# ===================================================
def tri2_j(p):
    # /// Triangular function, base length 1, square base
    
    result = 0.0
    
    if np.all(np.abs(p) < 0.5):
#         result = (1.0 - 2.0*np.abs(p[0])) * (1.0 - 2.0*np.abs(p[1]))
        result = np.prod(1.0 - 2.0*np.abs(p))
    
    return result
# ----------------------------------------------------

# ===================================================
def pyr2_j(p, w):
    # /// Triangular function, base length 1, square base
    
    mask = np.abs(p[:,0]*w[:,1]/(p[:,1]*w[:,0]))<1
    result = np.zeros(x.shape)
    result[mask] = tri_j(p[mask,0]/w[mask,0])
    result[~mask] = tri_j(p[~mask,1]/w[~mask,1])
    
    return result
# ----------------------------------------------------

# ===================================================
def cone2_j(r):
    # /// Cone function, base length 1, circular base
        
    mask = np.abs(r)<0.5
    result = np.zeros(r.shape)
#     result[mask] = (1.0 - 2.0*r)**2
    result[mask] = (1.0 - 2.0*np.abs(r[mask]))
    
    return result
# ----------------------------------------------------

# ===================================================
def sinc2_j(p):
    # /// Sinc function basic. Sin(x)/x
    result = 1.0
    
    if np.all(np.abs(p) > sys.float_info.epsilon):
        result = np.prod(np.sin(p)/p)
        
    return result
# ----------------------------------------------------

# ===================================================
def sinc2r_j(r):
    # /// Sinc function based on radius. Sin(x)/x
#     result = 1.0
    
#     if (r > sys.float_info.epsilon):
#         result = (np.sin(r)/r)**2
        
    mask = np.abs(r) > sys.float_info.epsilon
    result = np.ones(r.shape)
    result[mask] = np.sin(r[mask])/r[mask]  
        
    return result
# ----------------------------------------------------

# ===================================================
def bes_j(r):
    # /// Bessel function based on radius. Sin(x)/x    

    # sinc=besselsphere
#     result = sp.special.jv(0, r)
    result = sp.special.spherical_jn(0, r)
         
        
    return result
# ----------------------------------------------------

# ===================================================
def jinc_j(r):
    # /// Bessel function based on radius. Sin(x)/x    

    mask = np.abs(r) > 100*sys.float_info.epsilon
    result = np.ones(r.shape)
#     result[mask] = sp.special.jv(0, r[mask])/r[mask]
    result[mask] = sp.special.spherical_jn(0, r[mask])/r[mask]
         
        
    return result
# ----------------------------------------------------

# ====================================================
def W_rect_2D(r, nu, w, A, wlen):
    # 2D Rectangular Wigner function
    # Centered at 0, can be shifted by changing the argument r to be offset
    r_hat = r/w
#     result = 4*A**2*np.prod(w)*tri_j(r_hat[:,0])*tri_j(r_hat[:,1]) * sinc_j(2*np.pi*nu[0]*w[:,0]*tri_j(r_hat[:,0])) * sinc_j(2*np.pi*nu[1]*w[:,1]*tri_j(r_hat[:,1]))
#     result = 4**2*A**2*np.prod(w)*tri_j(r_hat[:,0])*tri_j(r_hat[:,1]) * sinc_j(2*np.pi*nu[0]*w[:,0]*tri_j(r_hat[:,0])) * sinc_j(2*np.pi*nu[1]*w[:,1]*tri_j(r_hat[:,1]))

#     result = (4*np.pi*A/wlen)**2*np.sum(np.prod(w,1))*tri_j(r_hat[:,0])*tri_j(r_hat[:,1]) * sinc_j(2*np.pi*nu[0]*w[:,0]*tri_j(r_hat[:,0])) * sinc_j(2*np.pi*nu[1]*w[:,1]*tri_j(r_hat[:,1]))

#     result = (4*np.pi*A/wlen)**2*w[:,0]*w[:,1]*tri_j(r_hat[:,0])*tri_j(r_hat[:,1]) * sinc_j(2*np.pi*nu[0]*w[:,0]*tri_j(r_hat[:,0])) * sinc_j(2*np.pi*nu[1]*w[:,1]*tri_j(r_hat[:,1]))

    result = 4*A**2*w[:,0]*w[:,1]*tri_j(r_hat[:,0])*tri_j(r_hat[:,1]) * sinc_j(2*np.pi*nu[0]*w[:,0]*tri_j(r_hat[:,0])) * sinc_j(2*np.pi*nu[1]*w[:,1]*tri_j(r_hat[:,1]))
    
#     result = 4*A**2*np.prod(w)*pyr_2j(r_hat, w) * sinc_j(2*np.pi*nu[0]*w[:,0]*tri_j(r_hat[:,0])) * sinc_j(2*np.pi*nu[1]*w[:,1]*tri_j(r_hat[:,1]))
    
    # I should be doing pyramid functions I think.

    return result
# ----------------------------------------------------

# ====================================================
def W_circ_2D(r, nu, w, A, wlen):
    # 2D Circular Wigner function
    # Centered at 0, can be shifted by changing the argument r to be offset
#     result = 4*w[0]*w[1] * cone2_j(np.linalg.norm((s[0:2]-s0)/w, ord=2)) * sinc2r_j(np.linalg.norm(s[2:4], ord=2)*np.sqrt(w[0]*w[1]*cone2_j(np.linalg.norm((s[0:2]-s0)/w, ord=2))))

    r_hat = np.sqrt(np.sum(r**2, 1))/w
    k_hat = np.sqrt(np.sum((2*np.pi*nu)**2, -1))
    
#     result = 4*A**2 * (np.pi*(w/2)**2) * cone2_j(r_hat) *sinc2r_j(k_hat* w/2 * cone2_j(r_hat))
#     result = 4*A**2 * (np.pi*(w/2)**2) * cone2_j(r_hat) *bes_j(k_hat* w/2 * cone2_j(r_hat))
    
    result = (4*np.pi*A/wlen)**2 * (np.pi*(w/2)**2) * cone2_j(r_hat) *bes_j(k_hat* w/2 * cone2_j(r_hat))

    
    return result
# ----------------------------------------------------

# ====================================================
def phase_front_3D(r, nu, r_, nu_):
    result = np.exp(2*np.pi*1j*(np.dot(r, nu_) + np.dot(nu, r_)))
    return result
# ----------------------------------------------------

# ====================================================
def phase_front_2D(nu, r_):
    result = np.exp(2*np.pi*1j*np.dot(nu, r_))
    return result
# ----------------------------------------------------

# ====================================================
def W_chirp_1D(t, f, w, a):
    # Wigner function for a time-windowed chirp
    # For different behaviour, offset t, or make f=f-fi(t)
    t_hat = t/w
    result = 2*a**2*w*tri_j(t_hat) * sinc_j(2*np.pi*f*w*tri_j(t_hat))
    
    return result
# ----------------------------------------------------