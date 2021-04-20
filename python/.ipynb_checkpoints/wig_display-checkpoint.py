import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors
import sys

# ===================================================
def discretise2DWig(points, support, nbins, wlen, area, display, dbs, min_value):
    
    # Extract the data -----------------
    X = points[0]
    Y = points[1]
    U = points[2]
    V = points[3]
    Z = points[4]
    n_samps = points[0].shape[0]
    
    xmin = support[0]
    xmax = support[1]
    ymin = support[2]
    ymax = support[3]
    umin = support[4]
    umax = support[5]
    vmin = support[6]
    vmax = support[7]
    
    n_x_bins = nbins[0]
    n_y_bins = nbins[1]
    n_u_bins = nbins[2]
    n_v_bins = nbins[3]

    x_bins = np.linspace(xmin, xmax - (xmax-xmin)/n_x_bins, n_x_bins)
    x_centres = x_bins + (xmax-xmin)/(2*n_x_bins)
    x_bin_w = xmax-xmin
    
    y_bins = np.linspace(ymin, ymax - (ymax-ymin)/n_y_bins, n_y_bins)
    y_centres = y_bins + (ymax-xmin)/(2*n_y_bins)
    y_bin_w = ymax-ymin
    
    u_bins = np.linspace(umin, umax - (umax-umin)/n_u_bins, n_u_bins)
    u_centres = u_bins + (umax-umin)/(2*n_u_bins)
    u_bin_w = umax-umin
    
    v_bins = np.linspace(vmin, vmax - (vmax-vmin)/n_v_bins, n_v_bins)
    v_centres = v_bins + (vmax-vmin)/(2*n_v_bins)
    v_bin_w = vmax-vmin
    # ====================================
    
    # Discretise the wdf ------------------
    wdf_binned  = np.histogramdd([X, Y, U, V], [x_centres, y_centres, u_centres, v_centres], weights=Z)[0]
    wdf_bcount  = np.histogramdd([X, Y, U, V], [x_centres, y_centres, u_centres, v_centres])[0]
    wdf_bcount[wdf_bcount == 0] = 1
    wdf_binned = wdf_binned/wdf_bcount
    # =====================================

    # Compute the XU Projection ----------
    dvdy = (v_bin_w * y_bin_w) / (n_v_bins*n_y_bins)**0.5
    xu_binned  = np.histogramdd([X, U], [x_bins, u_bins], weights=Z)[0] * dvdy
    xu_bcount  = np.histogramdd([X, U], [x_bins, u_bins])[0]
    xu_bcount[xu_bcount == 0] = 1
    xu_binned = xu_binned.T/xu_bcount
    xu_v_max = np.amax(np.abs(xu_binned))
    xu_v_min = -xu_v_max
    # ====================================
    
    # Compute the XY projection ----------
    dudv = (u_bin_w * v_bin_w) / (n_u_bins*n_v_bins)**0.5 
    xy_binned  = np.histogramdd([X, Y], [x_bins, y_bins], weights=Z)[0] *  dudv
    xy_bcount  = np.histogramdd([X, Y], [x_bins, y_bins])[0]
    xy_bcount[xy_bcount == 0] = 1
    xy_binned = xy_binned.T/xy_bcount
    xy_v_max = np.amax(np.abs(xy_binned))
    xy_v_min = -xy_v_max
    # ====================================
    
    # Compute the UV projection ----------
    dxdy = (x_bin_w * y_bin_w) / (n_x_bins*n_y_bins)**0.5
    uv_binned  = np.histogramdd([U, V], [u_bins, v_bins], weights=Z)[0] * dxdy
    uv_bcount  = np.histogramdd([U, V], [u_bins, v_bins])[0]
    uv_bcount[uv_bcount == 0] = 1
    uv_binned = uv_binned.T/uv_bcount
    # Optionally Convert to RCS ----------
#     to_rcs = 1/(4*np.pi*wlen**2*area[0]*area[1])
    to_rcs = 1/(4*np.pi*wlen**2*area)
    uv_binned *= to_rcs
    # -----------------
    uv_v_max = np.amax(np.abs(uv_binned))
    uv_v_min = -uv_v_max
    # ====================================
    
    # Compute the YV Projection ----------
    dudx = (u_bin_w * x_bin_w) / (n_u_bins*n_x_bins)**0.5
    yv_binned  = np.histogramdd([Y, V], [y_bins, v_bins], weights=Z)[0] * dudx
    yv_bcount  = np.histogramdd([Y, V], [y_bins, v_bins])[0]
    yv_bcount[yv_bcount == 0] = 1
    yv_binned = yv_binned.T/yv_bcount
    yv_v_max = np.amax(np.abs(yv_binned))
    yv_v_min = -yv_v_max
    # ====================================
    
    # Compute the X projection -----------
    dudvdy = u_bin_w * v_bin_w * y_bin_w
    x_binned  = np.histogramdd([X], [x_bins], weights=Z)[0] * dudvdy
    x_bcount  = np.histogramdd([X], [x_bins])[0]
    x_bcount[x_bcount == 0] = 1
    x_binned = x_binned.T/x_bcount
    x_v_max = np.amax(x_binned)
    x_v_min = np.amin(x_binned)
    # ====================================
    
    # Compute the V projection -----------
    dxdydu = x_bin_w * y_bin_w * u_bin_w
    v_binned  = np.histogramdd([V], [v_bins], weights=Z)[0] * dxdydu
    v_bcount  = np.histogramdd([V], [v_bins])[0]
    v_bcount[v_bcount == 0] = 1
    v_binned = v_binned.T/v_bcount
    # Optionally Convert to RCS ----------
#     to_rcs = 1/(4*np.pi*wlen**2*area[0])
    to_rcs = 1/(4*np.pi*wlen**2*area)
#     to_rcs = 1/(4*np.pi*wlen**2)
    v_binned *= to_rcs
    v_v_max = np.amax(v_binned)
    v_v_min = np.amin(v_binned)
    # ====================================
    
    # Compute the U projection -----------
    dxdydv = x_bin_w * y_bin_w * v_bin_w
    u_binned  = np.histogramdd([U], [u_bins], weights=Z)[0] * dxdydv
    u_bcount  = np.histogramdd([U], [u_bins])[0]
    u_bcount[u_bcount == 0] = 1
    u_binned = u_binned.T/u_bcount
    # Optionally Convert to RCS ----------
#     to_rcs = 1/(4*np.pi*wlen**2*area[1])
    to_rcs = 1/(4*np.pi*wlen**2*area)
#     to_rcs = 1/(4*np.pi*wlen**2)
    u_binned *= to_rcs
    u_v_max = np.amax(u_binned)
    u_v_min = np.amin(u_binned)
    # ====================================
    
    # Compute the Y projection -----------
    dudvdx = u_bin_w * v_bin_w * x_bin_w
    y_binned  = np.histogramdd([Y], [y_bins], weights=Z)[0] * dudvdx
    y_bcount  = np.histogramdd([Y], [y_bins])[0]
    y_bcount[y_bcount == 0] = 1
    y_binned = y_binned.T/y_bcount
    y_v_max = np.amax(y_binned)
    y_v_min = np.amin(y_binned)
    # ====================================
    
    # Compute the moments (0, 1, 2) ------
    # Energy
    dxdydudv = x_bin_w * y_bin_w * u_bin_w * v_bin_w / n_samps
    m0 = np.sum(wdf_binned) * dxdydudv
    # Mean
    m1_xu = 1/m0 * (np.sum(x_centres[1:] * u_centres[1:] * np.sum(wdf_binned, axis=(0,2)))) * dxdydudv
    m1_xy = 1/m0 * (np.sum(x_centres[1:] * y_centres[1:] * np.sum(wdf_binned, axis=(0,1)))) * dxdydudv
    m1_uv = 1/m0 * (np.sum(u_centres[1:] * v_centres[1:] * np.sum(wdf_binned, axis=(2,3)))) * dxdydudv
    m1_vy = 1/m0 * (np.sum(v_centres[1:] * y_centres[1:] * np.sum(wdf_binned, axis=(3,1)))) * dxdydudv
    m1_x = 1/m0 * (np.sum(x_centres[1:] * np.sum(wdf_binned, 0))) * dxdydudv
    m1_v = 1/m0 * (np.sum(v_centres[1:] * np.sum(wdf_binned, 3))) * dxdydudv *wlen
    m1_u = 1/m0 * (np.sum(u_centres[1:] * np.sum(wdf_binned, 2))) * dxdydudv *wlen
    m1_y = 1/m0 * (np.sum(y_centres[1:] * np.sum(wdf_binned, 1))) * dxdydudv
    # Variance
    m2_xu = 1/m0 * (np.sum((x_centres[1:]+u_centres[1:]-m1_xu)**2 * np.sum(wdf_binned, axis=(0,2)))) * dxdydudv
    m2_xy = 1/m0 * (np.sum((x_centres[1:]+y_centres[1:]-m1_xy)**2 * np.sum(wdf_binned, axis=(0,1)))) * dxdydudv
    m2_uv = 1/m0 * (np.sum((u_centres[1:]+u_centres[1:]-m1_uv)**2 * np.sum(wdf_binned, axis=(2,3)))) * dxdydudv
    m2_vy = 1/m0 * (np.sum((v_centres[1:]+y_centres[1:]-m1_vy)**2 * np.sum(wdf_binned, axis=(3,1)))) * dxdydudv
    m2_xx = 1/m0 * (np.sum((x_centres[1:]-m1_x)**2 * np.sum(wdf_binned, 0))) * dxdydudv
    m2_vv = 1/m0 * (np.sum((v_centres[1:]-m1_v)**2 * np.sum(wdf_binned, 3))) * dxdydudv *wlen**2
    m2_uu = 1/m0 * (np.sum((u_centres[1:]-m1_u)**2 * np.sum(wdf_binned, 2))) * dxdydudv *wlen**2
    m2_yy = 1/m0 * (np.sum((y_centres[1:]-m1_y)**2 * np.sum(wdf_binned, 1))) * dxdydudv
    # Extent πσ and Beamwidth σ
#     m2_xx = 2.5*np.pi*np.sqrt(m2_xx)
    m2_xx = np.pi*np.sqrt(m2_xx)
    m2_vv = 0.8*np.pi*np.sqrt(m2_vv)
    # m2_vv = np.sqrt(m2_vv)
    m2_uu = 2*np.pi*np.sqrt(m2_uu)
    # m2_vv = np.sqrt(m2_vv)
#     m2_yy = 2.5*np.pi*np.sqrt(m2_yy)
    m2_yy = np.pi*np.sqrt(m2_yy)
    # =================================
    


    # Display ------------------------
    if(display==True):
        
#         # Element
#         x_scale = 1e3
#         y_scale = 1e3
#         xy_v_scale = 1e3
#         uv_v_scale = 1e3
#         x_v_scale = 1e3
#         y_v_scale = x_v_scale
#         u_v_scale = 1
#         v_v_scale = u_v_scale
#         e_scale = 1e6
        
#         x_string_dB = "Planar Radiant \n Flux Gain \n $\Phi^P_e ~ [dB]$"
#         x_string = "Planar Radiant \n Flux Gain \n $\Phi^P_e ~ [mW]$"
#         u_string_dB = "Planar RCS \n $\sigma ^P ~ [dBm]$"
#         u_string = "Planar RCS \n $\sigma ^P ~ [W\cdot m]$"
#         e_string_dB = "Total Radiation Gain :" + "\n\n" + f"{10*np.log10(m0*e_scale):0.2f}" + " $dBm^2$"
#         e_string = "Total Radiation Gain :" + "\n\n" + f"{m0*e_scale:0.2f}" + " $\mu Wm^2$"
#         xy_string = "Radiant Flux Gain \n $\Phi_e ~ [mW]$"
#         uv_string = "RCS \n $\sigma ~ [mW\cdot m^2]$"
        
        # Cascade
        x_scale = 1e3
        y_scale = 1e3
        xy_v_scale = 1e3
        uv_v_scale = 1e3
        x_v_scale = 1e3
        y_v_scale = x_v_scale
        u_v_scale = 1
        v_v_scale = u_v_scale
        e_scale = 1e6
        
        x_string_dB = "Planar Radiant \n Flux Gain \n $\Phi^P_e ~ [dB]$"
        x_string = "Planar Radiant \n Flux Gain \n $\Phi^P_e ~ [mW]$"
        u_string_dB = "Planar RCS \n $\sigma ^P ~ [dBm]$"
        u_string = "Planar RCS \n $\sigma ^P ~ [W\cdot m]$"
        e_string_dB = "Total Radiation Gain :" + "\n\n" + f"{(10*np.log10(m0*e_scale)-10*np.log10(e_scale)):0.2f}" + " $dBm^2$"
        e_string = "Total Radiation Gain :" + "\n\n" + f"{m0*e_scale:0.2f}" + " $\mu Wm^2$"
        xy_string = "Radiant Flux Gain \n $\Phi_e ~ [mW]$"
        uv_string = "RCS \n $\sigma ~ [mW\cdot m^2]$"
        
        # Centre the colourmap on 0 --
        binned_max = np.amax(np.abs([xu_binned, yv_binned, xy_binned, uv_binned]))
        binned_min = -binned_max
        proj_max = np.amax([x_binned, v_binned, u_binned, y_binned])*1.1
        proj_min = np.amin([x_binned, v_binned, u_binned, y_binned])*1.1
        cmap = plt.cm.seismic
        # =============================
        
        # Axes spacing ---------------
        left, width = 0.1, 0.27
        bottom, height = 0.1, 0.27
        spacing = 0.01
        # ============================

        # Axes definitions ----------
        xu_proj = [left, bottom, width, height]
        xy_proj = [left, bottom + height + spacing, width, height]
        uv_proj = [left + width + spacing, bottom, width, height]
        yv_proj = [left + width + spacing, bottom + height + spacing, width, height]
        x_proj = [left, bottom + 2*(height + spacing), width, height]
        v_proj = [(left + width + spacing), bottom + 2*(height + spacing), width, height]
        u_proj = [left + 2*(width + spacing), bottom, width, height]
        y_proj = [left + 2*(width + spacing), (bottom + height + spacing), width, height]
        rect_energy = [left + 2*(width + spacing), bottom + 2*(height + spacing), 0.8*width, height]
        # ===========================

        # Make the figure -----------
        fig = plt.figure(figsize=(14, 10))
        # ==========================

        # Set up axes ---------------
        ax_xu = plt.axes(xu_proj)
        ax_xu.tick_params(direction='in', top=True, right=True)
        ax_xy = plt.axes(xy_proj)
        ax_xy.tick_params(direction='in', labelbottom=False, top=True, right=True)
        ax_uv = plt.axes(uv_proj)
        ax_uv.tick_params(direction='in', labelleft=False, top=True, right=True)
        ax_yv = plt.axes(yv_proj)
        ax_yv.tick_params(direction='in', labelleft=False, labelbottom=False, top=True, right=True)
        
        ax_x = plt.axes(x_proj)
        ax_x.tick_params(direction='in', labelbottom=False)
        ax_v = plt.axes(v_proj)
        ax_v.tick_params(direction='in', labelbottom=False, labelleft=False, labelright=False, left=False, right=False)
        ax_u = plt.axes(u_proj)
        ax_u.tick_params(direction='in', labelleft=False)
        ax_y = plt.axes(y_proj)
        ax_y.tick_params(direction='in', labelleft=False, labelbottom=False, labeltop=False, top=False, bottom=False)
        
        ax_energy = plt.axes(rect_energy)
        ax_energy.tick_params(top=False, right=False, left=False, bottom=False, labelleft=False, labelbottom=False)
        # ==========================
        
        # Plots --------------------
        
        if dbs:
        
            # XU Plot ------------------
            p_xu = ax_xu.imshow(xu_binned, extent=[xmin*x_scale, xmax*x_scale, np.rad2deg(np.arcsin(umin*wlen)), np.rad2deg(np.arcsin(umax*wlen))], interpolation='none', aspect='auto', origin='lower', vmin=xu_v_min, vmax=xu_v_max, cmap=cmap)
            ax_xu.set_xlabel("X [mm]")
            ax_xu.set_ylabel("Azimuth [deg]")
            # ==========================

            # XY Plot ------------------
#             xy_binned[xy_binned < min_value] = min_value
#             xy_binned = 10*np.log10(xy_binned)
#             p_xy = ax_xy.imshow(xy_binned, extent=[xmin*x_scale, xmax*x_scale, ymin*y_scale, ymax*y_scale], interpolation='none', aspect='auto', origin='lower')
            p_xy = ax_xy.imshow(xy_binned, extent=[xmin*x_scale, xmax*x_scale, ymin*y_scale, ymax*y_scale], interpolation='none', aspect='auto', origin='lower', vmin=xy_v_min, vmax=xy_v_max, cmap=cmap)
            ax_xy.set_ylabel("Y [mm]")
            # ==========================

            # UV Plot ------------------
#             uv_binned[uv_binned < min_value] = min_value
#             uv_binned = 10*np.log10(uv_binned)
#             p_uv = ax_uv.imshow(uv_binned.T, extent=[np.rad2deg(np.arcsin(vmin*wlen)), np.rad2deg(np.arcsin(vmax*wlen)), np.rad2deg(np.arcsin(umin*wlen)), np.rad2deg(np.arcsin(umax*wlen))], interpolation='none', aspect='auto', origin='lower')
            p_uv = ax_uv.imshow(uv_binned.T, extent=[np.rad2deg(np.arcsin(vmin*wlen)), np.rad2deg(np.arcsin(vmax*wlen)), np.rad2deg(np.arcsin(umin*wlen)), np.rad2deg(np.arcsin(umax*wlen))], interpolation='none', aspect='auto', origin='lower', vmin=uv_v_min, vmax=uv_v_max, cmap=cmap)
            ax_uv.set_xlabel("Elevation [deg]")
            # ==========================

            # YV Plot ------------------
            p_yv = ax_yv.imshow(yv_binned.T, extent=[np.rad2deg(np.arcsin(vmin*wlen)), np.rad2deg(np.arcsin(vmax*wlen)), ymin*y_scale, ymax*y_scale], interpolation='none', aspect='auto', origin='lower', vmin=yv_v_min, vmax=yv_v_max, cmap=cmap)
            # ==========================
    
            # X Plot -------------------
            x_binned[x_binned < min_value] = min_value
            x_binned = 10*np.log10(x_binned*x_v_scale)
            ax_x.step(x_centres[1:]*x_scale, x_binned)
            ax_x.axvline(m1_x*x_scale - m2_xx*x_scale/2, color='tab:orange')
            ax_x.axvline(m1_x*x_scale + m2_xx*x_scale/2, color='tab:orange')
            ax_x.set_ylabel(x_string_dB)
            ax_x.set_xlim(ax_xy.get_xlim())
            # x_x.set_ylim([proj_min, proj_max])
            # ==========================

            # V Plot -------------------
            v_binned[v_binned < min_value] = min_value
            v_binned = 10*np.log10(v_binned*v_v_scale)
            ax_v.step(np.rad2deg(np.arcsin(v_centres[1:]*wlen)), v_binned)
            ax_v.axvline(np.rad2deg(np.arcsin(m1_v - m2_vv/2)), color='tab:orange')
            ax_v.axvline(np.rad2deg(np.arcsin(m1_v + m2_vv/2)), color='tab:orange')
            # ax_v.set_ylabel('Radiant Intensity [W/rad]')
            ax_v.yaxis.set_label_position('right')
            # ax_v.set_ylim([proj_min, proj_max])
            ax_v.set_xlim(ax_yv.get_xlim())

            # ==========================

            # U Plot -------------------
            u_binned[u_binned < min_value] = min_value
            u_binned = 10*np.log10(u_binned*u_v_scale)
            ax_u.step(u_binned, np.rad2deg(np.arcsin(u_centres[1:]*wlen)))
            ax_u.axhline(np.rad2deg(np.arcsin(m1_u - m2_uu/2)), color='tab:orange')
            ax_u.axhline(np.rad2deg(np.arcsin(m1_u + m2_uu/2)), color='tab:orange')
            ax_u.set_xlabel(u_string_dB)
            # ax_u.set_xlim([proj_min, proj_max])
            ax_u.set_ylim(ax_uv.get_ylim())
            # ==========================

            # Y Plot -------------------
            y_binned[y_binned < min_value] = min_value
            y_binned = 10*np.log10(y_binned*y_v_scale)
            ax_y.step(y_binned, y_centres[1:]*y_scale)
            ax_y.axhline(m1_y*y_scale - m2_yy*y_scale/2, color='tab:orange')
            ax_y.axhline(m1_y*y_scale + m2_yy*y_scale/2, color='tab:orange')
            # ax_y.set_xlabel('Radiosity [W/m]')
            ax_y.xaxis.set_label_position('top')
            # ax_y.set_xlim([proj_min, proj_max])
            ax_y.set_ylim(ax_yv.get_ylim())
            # ==========================
        
            # Energy box ---------------
            # energy_string = 'Total Radiated Power $P$:' + '\n\n' + f'{m0:0.2f}' + ' $W$'
            text = ax_energy.text(0.5, 0.5, e_string_dB,  ha='center', va='center')
            # ==========================
            
            # Colourbars ---------------
            norm_xy = matplotlib.colors.Normalize(vmin=xy_v_min*xy_v_scale, vmax=xy_v_max*xy_v_scale)
            sm_xy = plt.cm.ScalarMappable(cmap=cmap, norm=norm_xy)
            cbar_xy = plt.colorbar(sm_xy, ax=[ax_y], location='right')
            cbar_xy.set_label(xy_string)

            # norm_uv = matplotlib.colors.Normalize(vmin=uv_v_min*1e6, vmax=uv_v_max*1e6)
            norm_uv = matplotlib.colors.Normalize(vmin=uv_v_min*uv_v_scale, vmax=uv_v_max*uv_v_scale)
            sm_uv = plt.cm.ScalarMappable(cmap=cmap, norm=norm_uv)
            cbar_uv = plt.colorbar(sm_uv, ax=[ax_u], location='right')
            # cbar_uv.set_label('Radiant Intensity Gain \n $I_{e,\Omega} ~ [uW/sr]$')
            cbar_uv.set_label(uv_string)
            # =========================
            
        else:
            
            # XU Plot ------------------
            p_xu = ax_xu.imshow(xu_binned, extent=[xmin*x_scale, xmax*x_scale, np.rad2deg(np.arcsin(umin*wlen)), np.rad2deg(np.arcsin(umax*wlen))], interpolation='none', aspect='auto', origin='lower', vmin=xu_v_min, vmax=xu_v_max, cmap=cmap)
            ax_xu.set_xlabel("X [mm]")
            ax_xu.set_ylabel("Azimuth [deg]")
            # ==========================

            # XY Plot ------------------
            p_xy = ax_xy.imshow(xy_binned, extent=[xmin*x_scale, xmax*x_scale, ymin*y_scale, ymax*y_scale], interpolation='none', aspect='auto', origin='lower', vmin=xy_v_min, vmax=xy_v_max, cmap=cmap)
            ax_xy.set_ylabel("Y [mm]")
            # ==========================

            # UV Plot ------------------
            p_uv = ax_uv.imshow(uv_binned.T, extent=[np.rad2deg(np.arcsin(vmin*wlen)), np.rad2deg(np.arcsin(vmax*wlen)), np.rad2deg(np.arcsin(umin*wlen)), np.rad2deg(np.arcsin(umax*wlen))], interpolation='none', aspect='auto', origin='lower', vmin=uv_v_min, vmax=uv_v_max, cmap=cmap)
            ax_uv.set_xlabel("Elevation [deg]")
            # ==========================

            # YV Plot ------------------
            p_yv = ax_yv.imshow(yv_binned.T, extent=[np.rad2deg(np.arcsin(vmin*wlen)), np.rad2deg(np.arcsin(vmax*wlen)), ymin*y_scale, ymax*y_scale], interpolation='none', aspect='auto', origin='lower', vmin=yv_v_min, vmax=yv_v_max, cmap=cmap)
            # ==========================
        
            # X Plot -------------------
            ax_x.step(x_centres[1:]*x_scale, x_binned*x_v_scale)
            ax_x.axvline(m1_x*x_scale - m2_xx*x_scale/2, color='tab:orange')
            ax_x.axvline(m1_x*x_scale + m2_xx*x_scale/2, color='tab:orange')
            ax_x.set_ylabel(x_string)
            ax_x.set_xlim(ax_xy.get_xlim())
            # x_x.set_ylim([proj_min, proj_max])
            # ==========================

            # V Plot -------------------
            ax_v.step(np.rad2deg(np.arcsin(v_centres[1:]*wlen)), v_binned*v_v_scale)
            ax_v.axvline(np.rad2deg(np.arcsin(m1_v - m2_vv/2)), color='tab:orange')
            ax_v.axvline(np.rad2deg(np.arcsin(m1_v + m2_vv/2)), color='tab:orange')
            # ax_v.set_ylabel('Radiant Intensity [W/rad]')
            ax_v.yaxis.set_label_position('right')
            # ax_v.set_ylim([proj_min, proj_max])
            ax_v.set_xlim(ax_yv.get_xlim())

            # ==========================

            # U Plot -------------------
            ax_u.step(u_binned*u_v_scale, np.rad2deg(np.arcsin(u_centres[1:]*wlen)))
            ax_u.axhline(np.rad2deg(np.arcsin(m1_u - m2_uu/2)), color='tab:orange')
            ax_u.axhline(np.rad2deg(np.arcsin(m1_u + m2_uu/2)), color='tab:orange')
            ax_u.set_xlabel(u_string)
            # ax_u.set_xlim([proj_min, proj_max])
            ax_u.set_ylim(ax_uv.get_ylim())
            # ==========================

            # Y Plot -------------------
            ax_y.step(y_binned*y_v_scale, y_centres[1:]*y_scale)
            ax_y.axhline(m1_y*y_scale - m2_yy*y_scale/2, color='tab:orange')
            ax_y.axhline(m1_y*y_scale + m2_yy*y_scale/2, color='tab:orange')
            # ax_y.set_xlabel('Radiosity [W/m]')
            ax_y.xaxis.set_label_position('top')
            # ax_y.set_xlim([proj_min, proj_max])
            ax_y.set_ylim(ax_yv.get_ylim())
            # ==========================
        
            # Energy box ---------------
            text = ax_energy.text(0.5, 0.5, e_string,  ha='center', va='center')
            # ==========================
      
            # Colourbars ---------------
            norm_xy = matplotlib.colors.Normalize(vmin=xy_v_min*xy_v_scale, vmax=xy_v_max*xy_v_scale)
            sm_xy = plt.cm.ScalarMappable(cmap=cmap, norm=norm_xy)
            cbar_xy = plt.colorbar(sm_xy, ax=[ax_y], location='right')
            cbar_xy.set_label(xy_string)

            # norm_uv = matplotlib.colors.Normalize(vmin=uv_v_min*1e6, vmax=uv_v_max*1e6)
            norm_uv = matplotlib.colors.Normalize(vmin=uv_v_min*uv_v_scale, vmax=uv_v_max*uv_v_scale)
            sm_uv = plt.cm.ScalarMappable(cmap=cmap, norm=norm_uv)
            cbar_uv = plt.colorbar(sm_uv, ax=[ax_u], location='right')
            # cbar_uv.set_label('Radiant Intensity Gain \n $I_{e,\Omega} ~ [uW/sr]$')
            cbar_uv.set_label(uv_string)
            # =========================
    
        # Title -------------------
        # fig.suptitle('WDF Projections in Position and Angle')
        # =========================

        # =========================
        return fig
# ----------------------------------------------------

# ===================================================
def discretiseTFWig(points, support, nbins, display):
    
    # Extract the data -----------------
    T = points[0]
    F = points[1]
    W = points[2]
    n_samps = points[0].shape[0]
    
    tmin = support[0]
    tmax = support[1]
    fmin = support[2]
    fmax = support[3]
    
    ntimes = nbins[0]
    nfreqs = nbins[1]

    t_bins = np.linspace(tmin, tmax, ntimes)
    t_centres = t_bins + (tmax-tmin)/(2*ntimes)
    t_len = (tmax-tmin)
    
    f_bins = np.linspace(fmin, fmax, nfreqs)
    f_centres = f_bins + (fmax-fmin)/(2*nfreqs)
    f_len = (fmax-fmin)
    # ==================================
    
    # Discretise the wdf ---------------
    wdf_binned  = np.histogramdd([T, F], [t_centres, f_centres], weights=W)[0]
    wdf_bcount  = np.histogramdd([T, F], [t_centres, f_centres])[0]
    wdf_bcount[wdf_bcount == 0] = 1
    wdf_binned = (wdf_binned/wdf_bcount).T
    # ==================================
    
    # Compute projections --------------
    t_proj = np.histogramdd([T], [t_centres], weights=W)[0] * f_len
    t_bcount = np.histogramdd([T], [t_centres])[0]
    t_bcount[t_bcount == 0] = 1
    t_proj = t_proj/t_bcount

    f_proj = np.histogramdd([F], [f_centres], weights=W)[0] * t_len #/2
    f_bcount = np.histogramdd([F], [f_centres])[0]
    f_bcount[f_bcount == 0] = 1
    f_proj = f_proj/f_bcount
    # =================================
    
    # Compute the moments (0, 1, 2) ---
    # Energy
    e_const_disc = (t_len*f_len)/np.prod(nbins)
    m0 = np.sum(wdf_binned) * e_const_disc
    # Mean
    m1_t = 1/m0 * (np.sum(t_centres[1:] * np.sum(wdf_binned, 0))) * e_const_disc
    m1_f = 1/m0 * (np.sum(f_centres[1:] * np.sum(wdf_binned, 1))) * e_const_disc
    # Variance
    m2_tt = 1/m0 * (np.sum((t_centres[1:]-m1_t)**2 * np.sum(wdf_binned, 0))) * e_const_disc
    m2_ff = 1/m0 * (np.sum((f_centres[1:]-m1_f)**2 * np.sum(wdf_binned, 1))) * e_const_disc
    m2_tt = np.pi*np.sqrt(m2_tt) # πσ Width good for extent
    # m2_ff = np.sqrt(m2_ff) # σ Width good for HPBW
    m2_ff = np.pi*np.sqrt(m2_ff) # πσ Width
    # =================================
    
    # Display ------------------------
    if(display==True):    
        
#         # Pulse
#         f_scale = 1e-9
#         t_scale = 1e6
#         p_scale = 1
#         esd_scale = 1e12
#         w_scale = 1e6
#         e_scale = 1e6
        
#         t_string = "Time $t~[\mu s]$"
#         f_string = "Frequency $f~[GHz]$"
#         p_string = "Signal Power \n $P_x~[V^2]$"
#         esd_string = "Energy Spectral Density \n $\overline{S}_{xx} ~ [p V^2 s /Hz]$"
#         w_string = "Instantaneous Energy Density \n  $W_x~[\mu V^2/Hz]$"
#         e_string = "Signal Energy $E_x$:" + "\n\n" + f"{m0*e_scale:0.2f}" + " $ \mu V^2s$"
        
        # TXChirp
        f_scale = 1e-9
        t_scale = 1e6
        p_scale = 1
        esd_scale = 1e12
        w_scale = 1e6
        e_scale = 1e6
        
        t_string = "Time $t~[\mu s]$"
        f_string = "Frequency $f~[GHz]$"
        p_string = "Signal Power \n $P_x~[V^2]$"
        esd_string = "Energy Spectral Density \n $\overline{S}_{xx} ~ [p V^2 s /Hz]$"
        w_string = "Instantaneous Energy Density \n  $W_x~[\mu V^2/Hz]$"
        e_string = "Signal Energy $E_x$:" + "\n\n" + f"{m0*e_scale:0.2f}" + " $ \mu V^2s$"
        
        # Centre the colourmap on 0 --
        binned_max = np.amax(np.abs(wdf_binned*w_scale))
        binned_min = -binned_max
        proj_max = np.amax([np.amax(t_proj), np.amax(f_proj)])*1.1
        proj_min = np.amin([np.amin(t_proj), np.amin(f_proj)])*1.1
        cmap = plt.cm.seismic
        # =============================
        
        # Axes spacing ---------------
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005
        # ============================

        # Axes definitions ----------
        rect_wdf = [left, bottom, width, height]
        rect_projt = [left, bottom + height + spacing, width, 0.2]
        rect_projf = [left + width + spacing, bottom, 0.2, height]
        rect_energy = [left + width + spacing, bottom + height + spacing, 0.2, 0.2]
        # ===========================

        # Make the figure -----------
        fig = plt.figure(figsize=(10, 8))
        # ==========================

        # Set up axes ---------------
        ax_wdf = plt.axes(rect_wdf)
        ax_wdf.tick_params(direction='in', top=True, right=True)
        ax_projt = plt.axes(rect_projt)
        ax_projt.tick_params(direction='in', labelbottom=False)
        ax_projf = plt.axes(rect_projf)
        ax_projf.tick_params(direction='in', labelleft=False)
        ax_energy = plt.axes(rect_energy)
        ax_energy.tick_params(top=False, right=False, left=False, bottom=False, labelleft=False, labelbottom=False)
        # ==========================

        # WDF plot -----------------
#         p_wdf = ax_wdf.pcolormesh(t_centres*t_scale, f_centres*f_scale, wdf_binned*w_scale, vmin=binned_min, vmax=binned_max, cmap=cmap)
        p_wdf = ax_wdf.imshow(wdf_binned*w_scale, extent=[tmin*t_scale, tmax*t_scale, fmin*f_scale, fmax*f_scale], interpolation='none', aspect='auto', origin='lower', vmin=binned_min, vmax=binned_max, cmap=cmap)
        ax_wdf.set_xlabel(t_string)
        ax_wdf.set_ylabel(f_string)
        # ax_wdf.grid()
        ax_wdf.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        # =========================
        
        # Time plot ---------------
        ax_projt.step(t_centres[1:]*t_scale, t_proj*p_scale)
        ax_projt.axvline(m1_t*t_scale - m2_tt*t_scale/2, color='tab:orange')
        ax_projt.axvline(m1_t*t_scale + m2_tt*t_scale/2, color='tab:orange')
        ax_projt.set_xlim(ax_wdf.get_xlim())
        ax_projt.set_ylabel(p_string)
        ax_projt.grid()
        # =========================

        # Freq plot ---------------
        ax_projf.step(f_proj*esd_scale, (f_centres[1:]-(fmax-fmin)/(nfreqs))*f_scale)
        ax_projf.axhline(m1_f*f_scale - m2_ff*f_scale/2, color='tab:orange')
        ax_projf.axhline(m1_f*f_scale + m2_ff*f_scale/2, color='tab:orange')
        ax_projf.set_ylim(ax_wdf.get_ylim())
        ax_projf.set_xlabel(esd_string)
        ax_projf.grid()
        # =========================

        # Energy box --------------
        text = ax_energy.text(0.5, 0.5, e_string,  ha='center', va='center', size=13)
        # =========================

        # Colourbar ---------------
        norm = matplotlib.colors.Normalize(vmin=binned_min, vmax=binned_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=[ax_energy, ax_projf], location='right')
        cbar.set_label(w_string)
        # =========================
        
    return wdf_binned, t_proj, f_proj, m0, fig
# ----------------------------------------------------

# ===================================================
def discretiseTFCoherence2(points, support, nbins, display):
    
    TAU = points[0]
    ETA = points[1]
    MU = points[2]
    n_samps = points[0].shape[0]
    
    taumin = support[0]
    taumax = support[1]
    etamin = support[2]
    etamax = support[3]
    
    ntaus = nbins[0]
    netas = nbins[1]

    tau_bins = np.linspace(taumin, taumax, ntaus)
    tau_centres = tau_bins + (taumax-taumin)/(2*ntaus)
    tau_bw = (taumax-taumin)
    
    eta_bins = np.linspace(etamin, etamax, netas)
    eta_centres = eta_bins + (etamax-etamin)/(2*netas)
    eta_bw = (etamax-etamin)
    
    # Discretise the degree of coherence    
    mu_binned  = np.histogramdd([TAU, ETA], [tau_centres, eta_centres], weights=MU)[0]
    mu_bcount  = np.histogramdd([TAU, ETA], [tau_centres, eta_centres])[0]
    mu_bcount[mu_bcount == 0] = 1
    mu_binned = (mu_binned/mu_bcount).T
    
    # Compute projections
    tau_proj = np.histogramdd([TAU], [tau_centres], weights=MU)[0] * eta_bw/(2*np.pi)
    tau_bcount = np.histogramdd([TAU], [tau_centres])[0]
    tau_bcount[tau_bcount == 0] = 1
    tau_proj = tau_proj/tau_bcount

    eta_proj = np.histogramdd([ETA], [eta_centres], weights=MU)[0] * tau_bw/2
    eta_bcount = np.histogramdd([ETA], [eta_centres])[0]
    eta_bcount[eta_bcount == 0] = 1
    eta_proj = eta_proj/eta_bcount
    


    if(display==True): 
        
#         # Pulse
#         f_scale = 1e-6
#         t_scale = 1e6
#         p_scale = 1e-6
#         esd_scale = 1e6
#         coh_scale = 1
#         e_scale = 1e6
        
#         t_string = "Time Difference $ \\tau~[ms]$"
#         f_string = "Frequency Difference $\eta~[MHz]$"
#         p_string = "Time Coherence \n $\gamma_{\\tau}~\cdot 10^{-6}$"
#         esd_string = "Frequency Coherence \n $\gamma_{\eta}~\cdot 10^{6}$"
#         coh_string = "Degree of Self-Coherence $\gamma_S$"
#         e_string = "Overall Degree \n of Coherence $\Gamma$:" + "\n\n" + f"{m0*e_scale:0.2f}" +"$\cdot 10^{-6}$"
        
        # Chirp
        f_scale = 1e-3
        t_scale = 1e6
        p_scale = 1e-3
        esd_scale = 1e6
        coh_scale = 1
        e_scale = 1e6
        
        t_string = "Time Difference $ \\tau~[ms]$"
        f_string = "Frequency Difference $\eta~[kHz]$"
        p_string = "Time Coherence \n $\gamma_{\\tau}~\cdot 10^{-3}$"
        esd_string = "Frequency Coherence \n $\gamma_{\eta}~\cdot 10^{6}$"
        coh_string = "Degree of Self-Coherence $\gamma_S$"
        e_string = "Overall Degree \n of Coherence $\Gamma$:" + "\n\n" + f"{m0*e_scale:0.2f}" +"$\cdot 10^{-6}$"
        
        # Find Minima and Maxima for consistent plotting
        binned_max = np.amax(np.abs(mu_binned))
        binned_min = np.amin(mu_binned)
        proj_max = np.amax([np.amax(tau_proj), np.amax(eta_proj)])*1.1
        proj_min = np.amin([np.amin(tau_proj), np.amin(eta_proj)])*1.1
        cmap = plt.cm.viridis
        
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005

        rect_mu = [left, bottom, width, height]
        rect_projtau = [left, bottom + height + spacing, width, 0.2]
        rect_projeta = [left + width + spacing, bottom, 0.2, height]
        rect_energy = [left + width + spacing, bottom + height + spacing, 0.2, 0.2]

        # Make a rectangular figure
        fig = plt.figure(figsize=(10, 8))

        # Set up axes
        ax_mu = plt.axes(rect_mu)
        ax_mu.tick_params(direction='in', top=True, right=True)
        ax_projtau = plt.axes(rect_projtau)
        ax_projtau.tick_params(direction='in', labelbottom=False)
        ax_projeta = plt.axes(rect_projeta)
        ax_projeta.tick_params(direction='in', labelleft=False)
        ax_energy = plt.axes(rect_energy)
        ax_energy.tick_params(top=False, right=False, left=False, bottom=False, labelleft=False, labelbottom=False)

        # Centre axes
#         p_mu = ax_mu.pcolormesh(tau_centres*t_scale, eta_centres*f_scale, mu_binned*coh_scale, vmin=binned_min, vmax=binned_max, cmap=cmap)
        p_mu = ax_mu.imshow(mu_binned*coh_scale, extent=[taumin*t_scale, taumax*t_scale, etamin*f_scale, etamax*f_scale], interpolation='none', aspect='auto', origin='lower', vmin=binned_min, vmax=binned_max, cmap=cmap)
        
        ax_mu.set_xlabel(t_string)
        ax_mu.set_ylabel(f_string)

        #  Figure Garnishing
        norm = matplotlib.colors.Normalize(vmin=binned_min, vmax=binned_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=[ax_energy, ax_projeta], location='right')
        cbar.set_label(coh_string)

        # Projections
        ax_projtau.step(tau_centres[1:]*t_scale, tau_proj*p_scale)
        ax_projeta.step(eta_proj*esd_scale, (eta_centres[1:]-(etamax-etamin)/(netas))*f_scale)

        ax_projtau.set_xlim(ax_mu.get_xlim())
        ax_projeta.set_ylim(ax_mu.get_ylim())

        ax_projtau.set_ylabel(p_string)
        ax_projtau.grid()
        ax_projeta.set_xlabel(esd_string)
        ax_projeta.grid()
        
        # Total degree of coherence Box
#         energy_string = 'Signal Energy $E_x$:' + '\n\n' + f'{energy*1e3:0.2f}' + ' $mV^2s/Hz$'
        text = ax_energy.text(0.5, 0.5, e_string,  ha='center', va='center', size=13)


        # Note these labels are correct because we're describing the function by sampling and binning. More samples increases accuracy. More bins increases resolution. 
        # I also want to to retain this kind of functionality in my sim for checking.
        
    return mu_binned, tau_proj, eta_proj, m0, fig
# ----------------------------------------------------