# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
import matplotlib.pyplot as plt
from ...utilities import Vec3
import copy

class Layout2Var():
    def __init__(self, fi, boundaries, n_wt, D, grid_spc,
                 wdir=None, wspd=None, wfreq=None):
        self.fi = fi
        self.boundaries = boundaries

        x_layout, y_layout = self.make_grid_layout(n_wt, D, grid_spc)
        fi.reinitialize_flow_field(layout_array=[x_layout, y_layout])

        self.coords = \
            copy.deepcopy(self.fi.floris.farm.flow_field.turbine_map.coords)

        x_center, y_center = self.find_center(x_layout, y_layout)
        self.center_of_rotation = Vec3(x_center, y_center, 0.0)

        self.x0 = self.fi.layout_x
        self.y0 = self.fi.layout_y
        self.x = self.fi.layout_x
        self.y = self.fi.layout_y

        self.thetamin = 0.
        self.thetamax = 359.9
        self.theta0 = 60.

        self.min_dist = 2*self.rotor_diameter
        self.max_dist = 15*self.rotor_diameter

        self.spacing0 = grid_spc

        if wdir is not None:
            self.wdir = wdir
        else:
            self.wdir = self.fi.floris.farm.flow_field.wind_direction
        if wspd is not None:
            self.wspd = wspd
        else:
            self.wspd = self.fi.floris.farm.flow_field.wind_speed
        if wfreq is not None:
            self.wfreq = wfreq
        else:
            self.wfreq = 1.

    def __str__(self):
        return 'layout2var'

    ###########################################################################
    # Required private optimization methods
    ###########################################################################

    def reinitialize(self):
        pass

    def obj_func(self, varDict):
        # Parse the variable dictionary
        self.parse_opt_vars(varDict)
        print('theta: ', self.theta)

        # coords = self.fi.floris.farm.flow_field.turbine_map.coords

        self.x, self.y = self.rotate_farm(self.coords, self.theta)

        # Update turbine map with turbince locations
        self.fi.reinitialize_flow_field(layout_array=[self.x, self.y])

        # Compute the objective function
        funcs = {}
        funcs['obj'] = -1*self.fi.get_farm_AEP(
            self.wdir,
            self.wspd,
            self.wfreq
        )*1e-9

        print('AEP: ', funcs['obj'])

        # Compute constraints, if any are defined for the optimization
        # funcs = self.compute_cons(funcs)

        fail = False
        return funcs, fail

    # Optionally, the user can supply the optimization with gradients
    # def _sens(self, varDict, funcs):
    #     funcsSens = {}
    #     fail = False
    #     return funcsSens, fail

    def parse_opt_vars(self, varDict):
        self.theta = varDict['theta']
        # self.spacing = varDict['spacing']

    def parse_sol_vars(self, sol):
        self.theta = list(sol.getDVs().values())[0]
        # self.spacing = list(sol.getDVs().values())[1]

    def add_var_group(self, optProb):
        optProb.addVar('theta', type='c',
                            lower=self.thetamin,
                            upper=self.thetamax,
                            value=self.theta0,
                            scale=1e-2)
        # optProb.addVarGroup('theta', 1, type='c',
        #                     lower=self.thetamin,
        #                     upper=self.thetamax,
        #                     value=self.theta0,
        #                     scale=1e-2)
        # optProb.addVarGroup('spacing', 1, type='c',
        #                     lower=self.min_dist,
        #                     upper=self.max_dist,
        #                     value=self.spacing0)

        return optProb

    def add_con_group(self, optProb):
        # optProb.addConGroup('boundary_con', self.nturbs, lower=0.0)
        # optProb.addConGroup('spacing_con', self.nturbs,
        #                     lower=self.min_dist)

        return optProb

    def compute_cons(self, funcs):
        # funcs['boundary_con'] = self.distance_from_boundaries()
        # funcs['spacing_con'] = self.space_constraint()

        return funcs

    ###########################################################################
    # User-defined methods
    ###########################################################################

    def rotate_farm(self, coords, theta):
        layout_x = np.zeros(len(coords))
        layout_y = np.zeros(len(coords))
        for i, coord in enumerate(coords):
            coord.rotate_on_x3(theta, self.center_of_rotation)
            layout_x[i] = coord.x1prime
            layout_y[i] = coord.x2prime
        return layout_x, layout_y

    def find_center(self, x_layout, y_layout):
        length = len(x_layout)
        sum_x = np.sum(x_layout)
        sum_y = np.sum(y_layout)
        return sum_x/length, sum_y/length

    def isPerfect(self, N):
        """Function to check if a number is perfect square or not

        taken from:
        https://www.geeksforgeeks.org/closest-perfect-square-and-its-distance/
        by sahishelangia
        """
        if (np.sqrt(N) - np.floor(np.sqrt(N)) != 0):
            return False
        return True

    def getClosestPerfectSquare(self, N):
        """Function to find the closest perfect square taking minimum steps to
            reach from a number

        taken from:
        https://www.geeksforgeeks.org/closest-perfect-square-and-its-distance/
        by sahishelangia
        """
        if (self.isPerfect(N)):
            distance = 0
            return N, distance

        # Variables to store first perfect square number above and below N
        aboveN = -1
        belowN = -1
        n1 = 0

        # Finding first perfect square number greater than N
        n1 = N + 1
        while (True):
            if (self.isPerfect(n1)):
                aboveN = n1
                break
            else:
                n1 += 1

        # Finding first perfect square number less than N
        n1 = N - 1
        while (True):
            if (self.isPerfect(n1)):
                belowN = n1
                break
            else:
                n1 -= 1

        # Variables to store the differences
        diff1 = aboveN - N
        diff2 = N - belowN

        if (diff1 > diff2):
            return belowN, -diff2
        else:
            return aboveN, diff1

    def make_grid_layout(self, n_wt, D, grid_spc):
        """Make a grid layout (close as possible to a square grid)

        Inputs:
        -------
            n_wt : float
                Number of wind turbines in the plant
            D : float (or might want array_like if diff wt models are used)
                Wind turbine rotor diameter(s) in meters
            grid_spc : float
                Spacing between rows and columns in number of rotor diams D
            plant_cap_MW : float
                Total wind plant capacity in MW

        Returns:
        --------
            layout_x : array_like
                X positions of the wind turbines in the plant
            layout_y : array_like
                Y positions of the wind turbines in the plant
        """

        # Initialize layout variables
        layout_x = []
        layout_y = []

        # Find the closest square root
        close_square, dist = self.getClosestPerfectSquare(n_wt)
        side_length = int(np.sqrt(close_square))

        # Build a square grid
        for i in range(side_length):
            for k in range(side_length):
                layout_x.append(i*grid_spc*D)
                layout_y.append(k*grid_spc*D)

        # Check dist and determine what to do
        if dist == 0:
            # do nothing
            pass
        elif dist > 0:
            # square>n_wt : remove locations
            del(layout_x[close_square-dist:close_square])
            del(layout_y[close_square-dist:close_square])
        else:
            # square < n_w_t : add a partial row
            for i in range(abs(dist)):
                layout_x.append(np.sqrt(close_square)*grid_spc*D)
                layout_y.append(i*grid_spc*D)

        return layout_x, layout_y

    def space_constraint(self):
        dist = [np.min([np.np.sqrt((self.x[i] - self.x[j])**2 + \
                (self.y[i] - self.y[j])**2) \
                for j in range(self.nturbs) if i != j]) \
                for i in range(self.nturbs)]

        return dist

    def distance_from_boundaries(self):  
        x = self.x
        y = self.y

        dist_out = []

        for k in range(self.nturbs):
            dist = []
            in_poly = self.point_inside_polygon(self.x[k],
                                                 self.y[k],
                                                 self.boundaries)

            for i in range(len(self.boundaries)):
                self.boundaries = np.array(self.boundaries)
                p1 = self.boundaries[i]
                if i == len(self.boundaries) - 1:
                    p2 = self.boundaries[0]
                else:
                    p2 = self.boundaries[i + 1]

                px = p2[0] - p1[0]
                py = p2[1] - p1[1] 
                norm = px*px + py*py

                u = ((self.x[k] - self.boundaries[i][0])*px + \
                     (self.y[k] - self.boundaries[i][1])*py)/float(norm)

                if u <= 0:
                    xx = p1[0]
                    yy = p1[1]
                elif u >=1:
                    xx = p2[0]
                    yy = p2[1]
                else:
                    xx = p1[0] + u*px
                    yy = p1[1] + u*py

                dx = self.x[k] - xx
                dy = self.y[k] - yy
                dist.append(np.np.sqrt(dx*dx + dy*dy))

            dist = np.array(dist)
            if in_poly:
                dist_out.append(np.min(dist))
            else:
                dist_out.append(-np.min(dist))

        dist_out = np.array(dist_out)

        return dist_out

    def point_inside_polygon(self, x, y, poly):
        n = len(poly)
        inside =False

        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y)*(p2x - p1x)/(p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def plot_layout_opt_results(self, sol):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        print()
        print('theta opt: ', sol.getDVs()['theta'])
        # locsx = sol.getDVs()['x']
        # locsy = sol.getDVs()['y']

        plt.figure(figsize=(9,6))
        fontsize= 16
        plt.plot(self.x0, self.y0, 'ob')
        plt.plot(self.x, self.y, 'or')
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel('x (m)', fontsize=fontsize)
        plt.ylabel('y (m)', fontsize=fontsize)
        plt.axis('equal')
        plt.grid()
        plt.tick_params(which='both', labelsize=fontsize)
        plt.legend(['Old locations', 'New locations'], loc='lower center', \
            bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fontsize)

        # verts = self.boundaries
        # for i in range(len(verts)):
        #     if i == len(verts)-1:
        #         plt.plot([verts[i][0], verts[0][0]], \
        #                  [verts[i][1], verts[0][1]], 'b')        
        #     else:
        #         plt.plot([verts[i][0], verts[i+1][0]], \
        #                  [verts[i][1], verts[i+1][1]], 'b')

        plt.show()

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def nturbs(self):
        """
        This property returns the number of turbines in the FLORIS 
        object.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = len(self.fi.floris.farm.turbine_map.turbines)
        return self._nturbs

    @property
    def rotor_diameter(self):
        return self.fi.floris.farm.turbine_map.turbines[0].rotor_diameter