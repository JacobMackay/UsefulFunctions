from mitsuba.core import ScalarTransform4f
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def velocity_transform(dr, dq, dt):
    linvel = dr*dt

    rotvel_z = ScalarTransform4f.rotate([0,0,1], dq[2]*dt)
    rotvel_y = ScalarTransform4f.rotate([0,1,0], dq[1]*dt)
    rotvel_x = ScalarTransform4f.rotate([1,0,0], dq[0]*dt)
    rotvel = rotvel_x*rotvel_y*rotvel_z

    return ScalarTransform4f.translate(linvel)*rotvel

def position_transform(r, q):
    pos = r

    rot_z = ScalarTransform4f.rotate([0,0,1], q[2])
    rot_y = ScalarTransform4f.rotate([0,1,0], q[1])
    rot_x = ScalarTransform4f.rotate([1,0,0], q[0])
    rot = rot_x*rot_y*rot_z

    return ScalarTransform4f.translate(pos)*rot


def display_render_frame(points, support, power_min=np.finfo(float).eps, decibels=False):

    # Axis definitions -----------------
    ft_ax = 0
    st_ax = 1
    frame_ax = 2
    # ==================================
    
    # Extract the data -----------------
    n_t_bins = points.shape[st_ax]
    n_f_bins = points.shape[ft_ax]
    
    fmin = support[0]
    fmax = support[1]
    tmin = support[2]
    tmax = support[3]
    
    F = np.linspace(fmin, fmax, n_f_bins)
    T = np.linspace(tmin, tmax, n_t_bins)
    # ==================================
    
    # Fast time projection -------------
    ft_binned = np.sum(points, axis=st_ax)
    # ==================================
    
    # Slow time projection -------------
    st_binned = np.sum(points, axis=ft_ax)
    # ==================================
    
#     # Mixer 
#     t_scale = 1e6
#     f_scale = 1e-3
#     p_scale = 1
    
#     t_string = "Time $[\mu s]$"
#     f_string = "Beat Frequency $[kHz]$"
#     w_string_dB = "Power \n $[dBW]$"
#     w_string = "Power \n $[W]$"

#     # Mixer Multipath
#     t_scale = 1
#     f_scale = 1e-3
#     p_scale = 1
    
#     t_string = "Time $[s]$"
#     f_string = "Beat Frequency $[kHz]$"
#     w_string_dB = "Power \n $[dBW]$"
#     w_string = "Power \n $[W]$"

#     # Mixer Multipath Ultra
#     t_scale = 1
#     f_scale = 1
#     p_scale = 1
    
#     t_string = "Time $[s]$"
#     f_string = "Beat Frequency $[Hz]$"
#     w_string_dB = "Power \n $[dBW]$"
#     w_string = "Power \n $[W]$"

    # Mixer Multipath Ultra Dodgy
    t_scale = 1
#     f_scale = 1
#     f_scale = 1/((SIG_TX['freq_sweep']/SIG_TX['chirp_len']) / C_MED)
    f_scale = 1.0/29.41176470588235
    p_scale = 1
    
    t_string = "Time $[s]$"
    f_string = "Range $[m]$"
    w_string_dB = "Power \n $[dBW]$"
    w_string = "Power \n $[W]$"
    
#     # Raw
#     t_scale = 1e6
#     f_scale = 1e-9
#     p_scale = 1
    
#     t_string = "Time $[\mu s]$"
#     f_string = "Frequency $[GHz]$"
#     w_string_dB = "Power \n $[dBW]$"
#     w_string = "Power \n $[W]$"
    
    # Ensure none are below minimum ----
    points[points<power_min] = power_min
    ft_binned[ft_binned<power_min] = power_min
    st_binned[st_binned<power_min] = power_min
    # ==================================
    
    # Setup the colour axis ------------
    ftst_max = np.amax([np.amax(points), power_min])
    ftst_min = np.amax([np.amin(points), power_min])
    ft_max = np.amax([np.amax(ft_binned), power_min])*1.1
    ft_min = np.amax([np.amin(ft_binned), power_min])*1.1
    st_max = np.amax([np.amax(st_binned), power_min])*1.1
    st_min = np.amax([np.amin(st_binned), power_min])*1.1
    cmap = plt.cm.viridis
    # ==================================

    # Make the figure ------------------
    figx = 11
    figy = 8
    fig = plt.figure(figsize=(figx, figy))
#     fig = plt.figure()
    # ==================================
    
    # Axes spacing ---------------------
#     left, width = 0.1, 0.27
#     bottom, height = 0.1, 0.27
#     spacing = 0.01
    left, width = 0.1, 0.42
    bottom, height = 0.15, 0.6
    spacing = 0.005
    # ==================================
    
     # Axes definitions ----------------
    ft_proj = [left, bottom, width*0.25, height]
    ftst_proj = [left + width*0.25 + spacing, bottom, 2*width  - 15*spacing, height] 
    st_proj = [left + width*0.25 + spacing, bottom + height + spacing, 1.6*width - 12*spacing, height*0.3]
    # ==================================
    
    # Set up axes ----------------------
    ax_ft = plt.axes(ft_proj)
    ax_ft.tick_params(direction='in', top=True, right=True)
    ax_ftst = plt.axes(ftst_proj)
    ax_ftst.tick_params(direction='in', labelleft=False, top=True, right=True)
    ax_st = plt.axes(st_proj)
    ax_st.tick_params(direction='in', labelleft=False, labelright=True, labelbottom=False, top=True, right=True)
    # ==================================
    
    # FTST Plot --------------------------
    if(decibels):
        ax_ftst.imshow(10*np.log10(points), extent=[tmin*t_scale, tmax*t_scale, fmin*f_scale, fmax*f_scale], interpolation='none', aspect='auto', origin='lower', vmin=10*np.log10(ftst_min), vmax=10*np.log10(ftst_max), cmap=cmap)
    else:
        ax_ftst.imshow(points, extent=[tmin*t_scale, tmax*t_scale, fmin*f_scale, fmax*f_scale], interpolation='none', aspect='auto', origin='lower', vmin=ftst_min, vmax=ftst_max, cmap=cmap)
    ax_ftst.set_xlabel(t_string)
    # ==================================
    
    # FT Plot --------------------------
    if(decibels):
        ax_ft.step(10*np.log10(ft_binned), F*f_scale)
        ax_ft.set_xlabel(w_string_dB)
        ax_ft.set_xlim([10*np.log10(ft_min), 10*np.log10(ft_max)])
    else:
        ax_ft.step(ft_binned, F*f_scale)
        ax_ft.set_xlabel(w_string)
        ax_ft.set_xlim([ft_min, ft_max])
    ax_ft.set_ylabel(f_string)
    ax_ft.set_xlim(np.flip(ax_ft.get_xlim()))
    ax_ft.set_ylim(ax_ftst.get_ylim())
    # ==================================
    
    # ST Plot --------------------------
    if(decibels):
        ax_st.step(T*t_scale, 10*np.log10(st_binned))
        ax_st.set_ylabel(w_string_dB)
        ax_st.set_ylim([10*np.log10(st_min), 10*np.log10(st_max)])
    else:
        ax_st.step(T*t_scale, st_binned)
        ax_st.set_ylabel(w_string)
        ax_st.set_ylim([st_min, st_max])
    ax_st.yaxis.set_label_position('right')
    ax_st.set_xlim(ax_ftst.get_xlim())
    # ==================================
               
        
    # Colourbars -----------------------
    if(decibels):
        norm = matplotlib.colors.Normalize(10*np.log10(ftst_min), 10*np.log10(ftst_max))
    else:
        norm = matplotlib.colors.Normalize(ftst_min, ftst_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=[ax_ftst], location='right')
    if(decibels):
        cbar.set_label(w_string_dB)
    else:
        cbar.set_label(w_string)
    # ==================================
    
    return fig