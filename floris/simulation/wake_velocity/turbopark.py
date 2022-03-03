# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from typing import Any, Dict

from attrs import define, field
import numpy as np
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator
import scipy.io
import os

from floris.simulation import BaseModel
from floris.simulation import FlowField
from floris.simulation import Grid


@define
class TurbOParkVelocityDeficit(BaseModel):
    A: float = field(default=0.04)
    sigma_max_rel: float = field(default=4.0)
    overlap_gauss_interp: RegularGridInterpolator = field(init=False)
    model_string = "turbopark"

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = dict(
            x=grid.x,
            y=grid.y,
            z=grid.z,
            u_initial=flow_field.u_initial,
        )

        file_name = os.path.dirname(os.path.realpath(__file__)) + '/gauss_lookup_table.mat'
        mat = scipy.io.loadmat(file_name)
        dist = mat['overlap_lookup_table'][0][0][0][0]
        radius_down = mat['overlap_lookup_table'][0][0][1][0]
        overlap_gauss = mat['overlap_lookup_table'][0][0][2]

        self.overlap_gauss_interp = RegularGridInterpolator((dist, radius_down), overlap_gauss, method='linear')
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        ambient_turbulence_intensity: np.ndarray,
        Cts: np.ndarray,
        rotor_diameter_i: np.ndarray,
        rotor_diameters: np.ndarray,
        i: int,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
    ) -> None:
        delta_total = np.zeros_like(u_initial)

        # Distances along x between the i'th turbine and all other turbines
        # normalized by the rotor diameters of turbine j
        downstream_mask = np.array(x_i - x >= 0.0)
        x_dist = (x_i - x) * downstream_mask / rotor_diameters # * downstream_mask

        # Radial distance between the rotor center of turbine i and the centerlines
        # of wakes from all turbines
        r_dist = np.sqrt((y_i - y) ** 2 + (z_i - z) ** 2)

        # Radial distance between the rotor center of turbine i and the centerlines
        # of wakes from all image turbines
        r_dist_image = np.sqrt((y_i - y) ** 2 + (z_i - (-z)) ** 2)

        Cts[:,:,i:,:,:] = 0.00001

        # Characteristic width of wakes from all turbines at the position of the i'th turbine
        dw = characteristic_wake_width(x_dist, ambient_turbulence_intensity, Cts, self.A)
        epsilon = 0.25 * np.sqrt(np.min(0.5 * (1 + np.sqrt(1 - Cts)) / np.sqrt(1 - Cts), 3, keepdims=True))
        sigma = rotor_diameters * (epsilon + dw)

        # Peak wake deficits
        val = 1 - Cts / (8 * (sigma / rotor_diameters) ** 2)
        C = 1 - np.sqrt(val)

        # Find upstream turbines with wakes overlapping the rotor of turbine i and calculate 
        # the deficit contribution from each. sigma_max_rel filters out turbines past a certain
        # spanwise distance.
        effective_width = self.sigma_max_rel * sigma
        is_overlapping = effective_width / 2 + rotor_diameter_i / 2 > r_dist

        wtg_overlapping = np.array(x_dist > 0) * is_overlapping
        wtg_overlapping_idx = np.argwhere(wtg_overlapping)

        if wtg_overlapping.any():
            delta_real = np.empty(np.shape(u_initial)) * np.nan
            delta_image = np.empty(np.shape(u_initial)) * np.nan
            for j in wtg_overlapping_idx:
                sigma_j = sigma[j[0], j[1], j[2], j[3], j[4]]
                r_dist_j = r_dist[j[0], j[1], j[2], j[3], j[4]]
                r_dist_image_j = r_dist_image[j[0], j[1], j[2], j[3], j[4]]
                C_j = C[j[0], j[1], j[2], j[3], j[4]] * is_overlapping[j[0], j[1], j[2], j[3], j[4]]

                delta_real[j[0], j[1], j[2], j[3], j[4]] = C_j * self.overlap_gauss_interp((r_dist_j / sigma_j, rotor_diameter_i / 2 / sigma_j))
                delta_image[j[0], j[1], j[2], j[3], j[4]] = C_j * self.overlap_gauss_interp((r_dist_image_j / sigma_j, rotor_diameter_i / 2 / sigma_j))

            delta = np.concatenate((delta_real, delta_image), axis=2)
    
            delta_total[:, :, i, :, :] = np.sqrt(np.sum(np.nan_to_num(delta)**2, axis=2))
        return delta_total          


def precalculate_overlap():
    dist = np.arange(0, 10, 1.0)
    radius_down = np.arange(0, 20, 1.0)
    overlap_gauss = np.zeros((len(dist), len(radius_down)))

    for i in range(len(dist)):
        for j in range(len(radius_down)):
            if radius_down[j] > 0:
                fun = lambda r, theta: np.exp(-(r ** 2 + dist[i] ** 2 - 2 * dist[i] * r * np.cos(theta))/2) * r
                out = integrate.dblquad(fun, 0, radius_down[j], lambda x: 0, lambda x: 2 * np.pi)[0]
                out = out / (np.pi * radius_down[j] ** 2)
            else:
                out = np.exp(-(dist[i] ** 2) / 1)
            overlap_gauss[i, j] = out

    return dist, radius_down, overlap_gauss


def characteristic_wake_width(x_dist, TI, Cts, A):
    # Parameter values taken from S. T. Frandsen, “Risø-R-1188(EN) Turbulence and turbulence generated structural
    # loading in wind turbine clusters” Risø, Roskilde, Denmark, 2007.
    c1 = 1.5
    c2 = 0.8

    alpha = TI * c1
    beta = c2 * TI / np.sqrt(Cts)
    # print(Cts)

    # Formula for characteristic wake width: sigma/rotor_diameter = epsilon + dw
    dw = A * TI / beta * (
        np.sqrt((alpha + beta * x_dist) ** 2 + 1) - np.sqrt(1 + alpha ** 2) - np.log(
            ((np.sqrt((alpha + beta * x_dist) ** 2 + 1) + 1) * alpha) / ((np.sqrt(1 + alpha ** 2) + 1) * (alpha + beta * x_dist))
        )
    )

    return dw
