# Copyright 2022 NREL

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
import numexpr as ne
import numpy as np

from floris.simulation import BaseModel, flow_field
from floris.simulation import Farm
from floris.simulation import FlowField
from floris.simulation import Grid
from floris.simulation import Turbine
from floris.utilities import cosd, sind, tand

from floris.simulation import Ct, axial_induction
from floris.simulation.turbine import average_velocity
from scipy.ndimage.filters import gaussian_filter


class CurledWakeVelocityDeficit(BaseModel):

    def __attrs_post_init__(self) -> None:
        # Reynolds number used for stability
        # Notice that the purpose of this number is to stabilize the solution
        # and does not provide any physical value.
        Re=10**4

        # Molecular viscosity based on Reynolds number
        # This viscosity can be adjusted to account for turbulence effects
        self.nu = self.Uh * self.h / Re

        # Identify the point in the x location where the wake is active
        self.activate = [
            np.argmin(np.abs(self.x - t.location[0])) for t in self.turbines
        ]


    def function(
        self,
        xi1: np.ndarray,
        xi: np.ndarray,
        grid: Grid,
        flow_field: FlowField,
        farm: Farm,
        i: int,
        uw_prev: np.ndarray,
        vw: np.ndarray,
        ww: np.ndarray,
        C: float = field(default=4.),
        cf: float = field(default=2.)
    ):
        # Write the advecting velocities
        U = flow_field.u_initial_sorted
        V = flow_field.v_initial_sorted
        W = flow_field.w_initial_sorted

        # uw_prev =  (flow_field.u_sorted[:,:,i-1,:,:] - flow_field.u_initial_sorted[:,:,i-1,:,:])[:,:,None,:,:]

        '''
        First compute the velocity gradients
        '''
        # Compute the derivatives in the plane (y and z)
        duwdy = (np.gradient(uw_prev, grid.dy, edge_order=2, axis=4))
        duwdz = (np.gradient(uw_prev, grid.dz, edge_order=2, axis=3))
        #~ duwdy = (np.gradient(self.U + self.uw[-1], self.dy, edge_order=2, axis=1))
        #~ duwdz = (np.gradient(self.U + self.uw[-1], self.dz, edge_order=2, axis=0))

        '''
        Now solve the marching problem for u'
        '''
        # Discrete equation
        # Notice laplacian term - this is a stabilizing term
        uwi = (
                # Term from previous x location
                uw_prev
                +
                # Coefficient from discretization
                #dx / (U) *
                grid.dx / (U + uw_prev) *
                    # V and W advection
                    ( - (V + vw) * duwdy - (W + ww) * duwdz
                    +
                    # Viscous term
                    C * self.nu * self.laplacian(uw_prev, self.dy, self.dz)
                    )
                )

        # Add the new added wake
        for j, n in enumerate(self.activate):
            if n == i:
                ct_i = Ct(
                    velocities=U + uwi,
                    yaw_angle=farm.yaw_angles_sorted,
                    fCt=farm.turbine_fCts,
                    turbine_type_map=farm.turbine_type_map_sorted,
                    ix_filter=[j],
                )

                print('Activating turbine', str(j))
                # Point to the turbine object
                t = self.turbines[j]

                # The wake deficit
                # TODO: Setup turbine grid at solver level 
                uwi += self.initial_condition(self.Y - t.location[1], self.Z - t.location[2], U + uwi, farm, j)

                # Condition to determine the incluence of the vortices
                # within a certain distance of the rotor
                # ~ cond = np.argwhere(np.abs(self.Y - t.location[1]) < t.D * 2)
                # ~ cond = np.where(np.abs(self.Y - t.location[1]) < t.D * 2)
                cond = np.asarray((np.abs(self.Y - t.location[1]) < (t.D * cf))).nonzero()

                # TODO: add back in
                # Add the effct of curl
                # if (t.alpha != 0): 
                    
                #     # Add the velocities from the curl
                #     vw[cond], ww[cond] = self.add_curl(
                #         self.Y[cond] - t.location[1], 
                #         self.Z[cond] - t.location[2], 
                #         V[cond], W[cond],
                #         ct_i
                #     )

                # # Add the wake rotation
                # t.add_rotation(
                #     self.Y - t.location[1], 
                #     self.Z - t.location[2], 
                #     V, W)
                # ~ t.add_rotation(self.Y - t.location[1], self.Z - t.location[2], V, W)

        # Adjust the boundary conditions to be zero at the edges
        uwi[ :,  0] *= 0
        uwi[ :, -1] *= 0
        uwi[ 0,  :] *= 0
        uwi[-1,  :] *= 0
        # ~ self.V[ :,  0] *= 0
        # ~ self.V[ :, -1] *= 0
        # ~ self.V[ 0,  :] *= 0
        # ~ self.V[-1,  :] *= 0

        #~ print(np.shape(uwi))
        # Add the new time
        self.uw.append(uwi)
        self.vw.append(V)
        self.ww.append(W)

        # Store the previous xi
        xi = xi1

        return xi


    def initial_condition(self, Y, Z, U, farm, j, sigma=2):
        '''
        Set the initial condition for the wake profile
        This is the first profile
        Y, Z - the mesh in the y-z plane
        U - the velocity at the plane
        '''
        # Factor to increase wake deficit (assuming the initial condition
        #   is after the rotor plane)
        # This is based on the induced velocity in the fully developed wake
        # to be U(1-2a) from actuator disk theory
        f = 2

        # Project the rotor onto the x plane
        # yp = Y * np.cos(self.alpha)
        # xp = Y * np.sin(self.alpha)

        # The z axis in reference rame where 0 is the center of the rotor
        Zt = Z #- self.th

        # The radial distance for these points
        r2 = np.sqrt(Y**2 + Zt**2) # + xp**2)

        # The values inside the rotor
        condition = np.where(r2 <= self.D/2)
        
        # The average velocity at the rotor plane used to compute the induced
        #   velocity in the wake
        self.Uh = average_velocity(turb_inflow_field)
        
        # Compute the mean velocity
        # Compute the ct and cp
        # self.ct_function()
        # self.cp_function()
        # ~ print('Velocity of turbine is ', self.Uh)
        # ~ print('Cp of turbine is ', self.cp_function())
        # ~ print('Power of turbine is ', self.power())
        # ~ print('Ct of turbine is ', self.ct_function())

        # The induction takes into account the yaw angle
        # self._compute_induction()
        axial_induction_i = axial_induction(
            velocities=U,
            yaw_angle=farm.yaw_angles_sorted,
            fCt=farm.turbine_fCts,
            turbine_type_map=farm.turbine_type_map_sorted,
            ix_filter=[j],
        )

        # Initial condition yawed
        uw_initial = - (U * f * axial_induction_i)
        ratio = np.sqrt((1-axial_induction_i) / (1-f * axial_induction_i))# The area ration of wind turbine wakes
        uw = gaussian_filter(uw_initial * (r2 <= ratio * self.D/2), sigma=sigma)

        return uw


    def add_curl(self, Y, Z, V, W, Ct, ):
        '''
        Add vortices due to the curled wake using an elliptical distribution

        Y - the spanwise location with 0 being the rotor center
        Z - the wall-normal location with 0 being the rotor center
        '''

        # The Ct changes according to sin * cos^2
        # The units for circulation are m^2/s
        # Circulation is per unit span (1/D)
        self.Gamma = -(np.pi * self.D / 4 * 1/2 * Ct * self.Uh *
                    np.sin(self.alpha) *  np.cos(self.alpha)**2 
                    )

        # ~ print('Circulation due to curl is:', self.Gamma, '[m^2/s]')

        # The widht of the vortex
        eps = 0.2 * self.D

        # The range of radii from the center of the rotor to the tip (0 to R)
        z_vector = np.linspace(0, self.D/2, 50)

        # The length of each section dz
        dz = z_vector[1] - z_vector[0]

        # Scale the circulation by the circulation at the center
        Gamma0 = 4 / np.pi * self.Gamma

        # Loop through all the vortices from an elliptic wing distribution
        # Skip the last point because it has zero circulation
        for z in z_vector[:-1]:

            # Compute the non-dimensional circulation
            Gamma = (- 4 * Gamma0 * z * dz /
                    (self.D**2 * np.sqrt(1 - (2 * z/self.D)**2)))

            # Locations of the tip vortices
            # Top
            yt1, zt1 = 0,  z
            # Bottom
            yt2, zt2 = 0, -z

            # Tip vortex velocities
            # Top
            vt1, wt1 = self.vortex(Y - yt1, Z - zt1,
                                    Gamma=Gamma, eps=eps,
                                  )

            # Bottom
            vt2, wt2 = self.vortex(Y - yt2, Z - zt2,
                                    Gamma=-Gamma, eps=eps
                                  )

            # Add the velocity components
            V += vt1 + vt2
            W += wt1 + wt2

            '''
            Add the ground effects my mirroring the vortices from the curl
            #~ '''
            if self.ground:
                # Tip vortex velocities
                # Top
                vt1, wt1 = self.vortex(Y - yt1, Z + zt1 + 2 * self.th,
                                        Gamma=-Gamma, eps=eps,
                                      )

                # Bottom
                vt2, wt2 = self.vortex(Y - yt2, Z + zt2 + 2 * self.th ,
                                        Gamma=Gamma, eps=eps
                                      )

                # Add the velocity components
                V += vt1 + vt2
                W += wt1 + wt2

        return V, W


    @staticmethod
    def laplacian(u, dy, dz):
        '''
        Compute the laplacian in 2D
        '''
        d2udy2 = np.gradient(np.gradient(u, dy, axis=1), dy, axis=1)
        #~ dy = np.gradient(y,axis=1)**2
        d2udz2 = np.gradient(np.gradient(u, dz, axis=0), dz, axis=0)
        #~ dz = np.gradient(z,axis=0)**2

        return d2udy2 + d2udz2