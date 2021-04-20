import numpy as np
from scipy import signal

def CFAR2d(data, p_fa, n_guard, n_train):
    #     CFAR 2D: CFAR detector along two dimensions
    #     Copyright (C) 2019  Jacob Mackay
    #     Contact: j.mackay@acfr.usyd.edu.au

    #     This program is free software: you can redistribute it and/or modify
    #     it under the terms of the GNU General Public License as published by
    #     the Free Software Foundation, either version 3 of the License, or
    #     (at your option) any later version.

    #     This program is distributed in the hope that it will be useful,
    #     but WITHOUT ANY WARRANTY; without even the implied warranty of
    #     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #     GNU General Public License for more details.

    #     You should have received a copy of the GNU General Public License
    #     along with this program.  If not, see <https://www.gnu.org/licenses/>.

    # INPUTS
    # data, MxN vector containing returns
    # pFA, Probability of false alarm between 0 and 1
    # nGuard, number of guard cells before|after CUT
    # nTrain, number of training cells before|after CUT
    # RETURNS
    # hits, MxN logical vector specifying if a cell has a power above noise floor
    # thresholds, MxN vector containing the noise threshold at each cell

    # TDetect = thresFactor*Pnoise
    # PNoise = (1/numTraining)*sum(trainingValue(i))
    # thresFactor = numTraining*(PFalseAlarm^(-1/numTraining)-1)
    # ?value>TDetect:true

    cfar_kernel = np.zeros([(n_guard+n_train)*2 + 1, (n_guard+n_train)*2 + 1])
    cfar_kernel[:n_train, :] = 1.0
    cfar_kernel[-n_train:, :] = 1.0
    cfar_kernel[:, :n_train] = 1.0
    cfar_kernel[:, -n_train:] = 1.0

    n_train_total = np.sum(cfar_kernel)
    cfar_kernel /= n_train_total

    p_noise = signal.convolve2d(data, cfar_kernel, mode='same', boundary='fill', fillvalue=0)
    thres_factor = n_train_total*(p_fa**(-1/n_train_total) - 1)
    t_detect = thres_factor*p_noise

    hits = data>t_detect

    return hits, t_detect


def CFAR1D(data, pFA, nGuard, nTrain):
    ##    CFAR 1D: CFAR detector along one dimension
    #     Copyright (C) 2019  Jacob Mackay
    #     Contact: j.mackay@acfr.usyd.edu.au
    #
    #     This program is free software: you can redistribute it and/or modify
    #     it under the terms of the GNU General Public License as published by
    #     the Free Software Foundation, either version 3 of the License, or
    #     (at your option) any later version.
    #
    #     This program is distributed in the hope that it will be useful,
    #     but WITHOUT ANY WARRANTY; without even the implied warranty of
    #     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #     GNU General Public License for more details.
    #
    #     You should have received a copy of the GNU General Public License
    #     along with this program.  If not, see <https://www.gnu.org/licenses/>.
    #
    # INPUTS
    # data, Mx1 vector containing returns
    # pFA, Probability of false alarm between 0 and 1
    # nGuard, number of guard cells before|after CUT
    # nTrain, number of training cells before|after CUT
    # RETURNS
    # hits, Mx1 logical vector specifying if a cell has a power above noise floor
    # thresholds, Mx1 vector containing the noise threshold at each cell

    # TDetect = thresFactor*Pnoise
    # PNoise = (1/numTraining)*sum(trainingValue(i))
    # thresFactor = numTraining*(PFalseAlarm^(-1/numTraining)-1)
    # ?value>TDetect:true

    cfar_kernel = np.zeros([(n_guard+n_train)*2 + 1])
    cfar_kernel[:n_train] = 1.0
    cfar_kernel[-n_train:] = 1.0

    n_train_total = np.sum(cfar_kernel)
    cfar_kernel /= n_train_total

    p_noise = signal.convolve(data, cfar_kernel, mode='same')
    thres_factor = n_train_total*(p_fa**(-1/n_train_total) - 1)
    t_detect = thres_factor*p_noise

    hits = data>t_detect

    return hits, t_detect
