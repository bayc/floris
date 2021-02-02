# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from .optimization import Optimization


class PowerDensityOptimization2D(Optimization):
    """
    PowerDensityOptimization1D is a subclass of the
    :py:class:`~.tools.optimization.scipy.optimization.Optimization` class
    that performs layout optimization in 1 dimension. TODO: What is this single
    dimension?
    """

    def __init__(
        self,
        fi,
        wd,
        ws,
        freq,
        AEP_initial,
        nturbs_x,
        nturbs_y,
        spacing_x_init,
        spacing_y_init,
        x0=None,
        bnds=None,
        min_dist=None,
        opt_method="SLSQP",
        opt_options=None,
    ):
        """
        Instantiate PowerDensityOptimization1D object with a FlorisInterface
        object and assigns parameter values.

        Args:
            fi (:py:class:`floris.tools.floris_interface.FlorisInterface`):
                Interface used to interact with the Floris object.
            wd (np.array): An array of wind directions (deg).
            ws (np.array): An array of wind speeds (m/s).
            freq (np.array): An array of the frequencies of occurance
                correponding to each pair of wind direction and wind speed
                values.
            AEP_initial (float): The initial Annual Energy
                Production used for normalization in the optimization (Wh)
                (TODO: Is Watt-hours the correct unit?).
            x0 (iterable, optional): The initial turbine locations,
                ordered by x-coordinate and then y-coordiante
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]) (m). If none are
                provided, x0 initializes to the current turbine locations.
                Defaults to None.
            bnds (iterable, optional): Bounds for the optimization
                variables (pairs of min/max values for each variable (m)). If
                none are specified, they are set to some example values (TODO:
                what is the significance of these example values?). Defaults to
                None.
            min_dist (float, optional): The minimum distance to be
                maintained between turbines during the optimization (m). If not
                specified, initializes to 2 rotor diameters. Defaults to None.
            opt_method (str, optional): The optimization method used by
                scipy.optimize.minize. Defaults to 'SLSQP'.
            opt_options (dict, optional): Optimization options used by
                scipy.optimize.minize. If none are specified, they are set to
                {'maxiter': 100, 'disp': True, 'iprint': 2, 'ftol': 1e-9}.
                Defaults to None.
        """
        super().__init__(fi)
        self.max_x = self.fi.layout_x[-1]
        self.epsilon = np.finfo(float).eps
        self.counter = 0

        if opt_options is None:
            self.opt_options = {"maxiter": 100, "disp": True, "iprint": 2, "ftol": 1e-9}

        self.reinitialize_opt(
            wd=wd,
            ws=ws,
            freq=freq,
            AEP_initial=AEP_initial,
            x0=x0,
            bnds=bnds,
            min_dist=min_dist,
            opt_method=opt_method,
            opt_options=opt_options,
            nturbs_x=nturbs_x,
            nturbs_y=nturbs_y,
            spacing_x_init=spacing_x_init,
            spacing_y_init=spacing_y_init,
        )

    # def _powDens_opt(self, optVars):
    #     locs_x = optVars[0 : self.nturbs]
    #     locs_y = optVars[self.nturbs : 2 * self.nturbs]
    #     locs_x_unnorm = [
    #         self._unnorm(valx, self.bndx_min, self.bndx_max) for valx in locs_x
    #     ]
    #     locs_y_unnorm = [
    #         self._unnorm(valy, self.bndy_min, self.bndy_max) for valy in locs_y
    #     ]
    #     turb_controls = [
    #         optVars[2 * self.nturbs + i * self.nturbs : 3 * self.nturbs + i * self.nturbs]
    #         for i in range(len(self.wd))
    #     ]
    #     turb_controls_unnorm = [
    #         self._unnorm(yaw, self.yaw_min, self.yaw_max) for yaw in turb_controls
    #     ]

    #     self._change_coordinates(locs_x_unnorm, locs_y_unnorm)

    #     layout_dist = self._avg_dist(locs)
    #     AEP_sum = self._AEP_single_wd(self.wd[0], self.ws[0], turb_controls_unnorm[0])

    #     return layout_dist / self.layout_dist_initial

    def _gen_locs_from_optVars(self, optVars):
        # spacing_x_norm = optVars[0]
        # spacing_y_norm = optVars[1]

        # spacing_x = self._unnorm(spacing_x_norm, self.min_dist, self.spacing_x_init)
        # spacing_y = self._unnorm(spacing_y_norm, self.min_dist, self.spacing_y_init)

        spacing_x_norm = optVars[0]
        # spacing_y_norm = optVars[1]

        spacing_x = self._unnorm(spacing_x_norm, self.min_dist, self.spacing_x_init)
        spacing_y = self._unnorm(spacing_x_norm, self.min_dist, self.spacing_y_init)

        layout_x = [
            i * spacing_x for i in range(self.nturbs_x) for j in range(self.nturbs_y)
        ]
        layout_y = [
            j * spacing_y for i in range(self.nturbs_x) for j in range(self.nturbs_y)
        ]

        return layout_x, layout_y

    def _powDens_opt(self, optVars):
        # locsx = optVars[0 : self.nturbs]
        # locsy = optVars[self.nturbs : 2 * self.nturbs]

        # locsx_unnorm = [
        #     self._unnorm(valx, self.bndx_min, self.bndx_max) for valx in locsx
        # ]
        # locsy_unnorm = [
        #     self._unnorm(valy, self.bndy_min, self.bndy_max) for valy in locsy
        # ]

        locs_x_unnorm, locs_y_unnorm = self._gen_locs_from_optVars(optVars)

        self._change_coordinates(locs_x_unnorm, locs_y_unnorm)
        opt_area = self.find_layout_area(locs_x_unnorm + locs_y_unnorm)

        # AEP_sum = 0.0

        # for i in range(len(self.wd)):
        #     for j, turbine in enumerate(self.fi.floris.farm.turbine_map.turbines):
        #         turbine.yaw_angle = turb_controls_unnorm[i][j]

        # AEP_sum = AEP_sum + self._AEP_single_wd(
        #     self.wd[i], self.ws[i], self.freq[i]
        # )

        # print('AEP ratio: ', AEP_sum/self.AEP_initial)

        # return -1 * AEP_sum / self.AEP_initial * self.initial_area / opt_area
        # return -1 * self.initial_area / opt_area
        return opt_area / self.initial_area

    # def _avg_dist(self, locs):
    #     dist = []
    #     for i in range(len(locs) - 1):
    #         dist.append(locs[i + 1] - locs[i])

    #     return np.mean(dist)

    def find_layout_area(self, locs):
        """
        This method returns the area occupied by the wind farm.

        Args:
            locs (iterable): A list of the turbine coordinates, organized as
                [x1, x2, ..., xn, y1, y2, ..., yn] (m).

        Returns:
            float: The area occupied by the wind farm (m^2).
        """
        locsx = locs[0 : self.nturbs]
        locsy = locs[self.nturbs :]

        points = zip(locsx, locsy)
        points = np.array(list(points))

        hull = self.convex_hull(points)

        area = self.polygon_area(
            np.array([val[0] for val in hull]), np.array([val[1] for val in hull])
        )

        return area

    def convex_hull(self, points):
        """
        Finds the vertices that describe the convex hull shape given the input
        coordinates.

        Args:
            points (iterable((float, float))): Coordinates of interest.

        Returns:
            list: Vertices describing convex hull shape.
        """
        # find two hull points, U, V, and split to left and right search
        u = min(points, key=lambda p: p[0])
        v = max(points, key=lambda p: p[0])
        left, right = self.split(u, v, points), self.split(v, u, points)

        # find convex hull on each side
        return [v] + self.extend(u, v, left) + [u] + self.extend(v, u, right) + [v]

    def polygon_area(self, x, y):
        """
        Calculates the area of a polygon defined by its (x, y) vertices.

        Args:
            x (iterable(float)): X-coordinates of polygon vertices.
            y (iterable(float)): Y-coordinates of polygon vertices.

        Returns:
            float: Area of polygon.
        """
        # coordinate shift
        x_ = x - x.mean()
        y_ = y - y.mean()

        correction = x_[-1] * y_[0] - y_[-1] * x_[0]
        main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
        return 0.5 * np.abs(main_area + correction)

    def split(self, u, v, points):
        # TODO: Provide description of this method.
        # return points on left side of UV
        return [p for p in points if np.cross(p - u, v - u) < 0]

    def extend(self, u, v, points):
        # TODO: Provide description of this method.
        if not points:
            return []

        # find furthest point W, and split search to WV, UW
        w = min(points, key=lambda p: np.cross(p - u, v - u))
        p1, p2 = self.split(w, v, points), self.split(u, w, points)
        return self.extend(w, v, p1) + [w] + self.extend(u, w, p2)

    def _change_coordinates(self, locs_x, locs_y):
        # Parse the layout coordinates
        layout_array = [locs_x, locs_y]

        # Update the turbine map in floris
        self.fi.reinitialize_flow_field(layout_array=layout_array)

    def _set_opt_bounds(self):
        # self.bnds = [(0.0, 1.0) for _ in range(2*self.nturbs)]
        # self.bnds = [
        #     (0.0, 0.0),
        #     (0.083333, 0.25),
        #     (0.166667, 0.5),
        #     (0.25, 0.75),
        #     (0.33333, 1.0),
        #     (0.0, 1.0),
        #     (0.0, 1.0),
        #     (0.0, 1.0),
        #     (0.0, 1.0),
        #     (0.0, 1.0),
        # ]

        self.bnds = []
        # inc = 1/(self.nturbs - 1)
        # min_inc = self.min_dist/self.max_x

        # for i in range(self.nturbs):
        #     x1 = i*min_inc
        #     x2 = i*inc
        #     self.bnds.append((x1, x2))

        # for i in range(self.nturbs):
        #     self.bnds.append((0.0, 1.0))

        for i in range(len(self.x0)):
            self.bnds.append((0.0, 1.0))

    def _AEP_single_wd(self, wd, ws, freq, yaw):
        self.fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
        self.fi.calculate_wake(yaw_angles=yaw)

        turb_powers = [turbine.power for turbine in self.fi.floris.farm.turbines]
        return np.sum(turb_powers) * freq * 8760

    def _AEP_constraint(self, optVars):
        # locs_x = optVars[0 : self.nturbs]
        # locs_y = optVars[self.nturbs : 2 * self.nturbs]
        # locs_x_unnorm = [
        #     self._unnorm(valx, self.bndx_min, self.bndx_max) for valx in locs_x
        # ]
        # locs_y_unnorm = [
        #     self._unnorm(valy, self.bndy_min, self.bndy_max) for valy in locs_y
        # ]
        locs_x_unnorm, locs_y_unnorm = self._gen_locs_from_optVars(optVars)

        turb_controls = [
            optVars[1 + i * self.nturbs : 1 + i * self.nturbs + self.nturbs]
            for i in range(len(self.wd))
        ]
        turb_controls_unnorm = [
            self._unnorm(yaw, self.yaw_min, self.yaw_max) for yaw in turb_controls
        ]
        self._change_coordinates(locs_x_unnorm, locs_y_unnorm)

        aep_opt = self.fi.get_farm_AEP(
            self.wd, self.ws, self.freq, yaw=turb_controls_unnorm
        )

        # aep_opt = 0
        # for i in range(len(self.wd)):
        #     aep_opt = aep_opt + self._AEP_single_wd(self.wd[i], self.ws[i], self.freq[i], turb_controls_unnorm[i])

        # return (
        #     self._AEP_single_wd(self.wd[0], self.ws[0], turb_controls_unnorm[0]) / self.AEP_initial - 1
        # ) * 1000000.0

        return -1 * np.abs(aep_opt / self.AEP_initial - 1)

    def _space_constraint(self, x_in, min_dist):
        # x = np.nan_to_num(x_in[0 : self.nturbs])
        # y = np.nan_to_num(x_in[self.nturbs : 2 * self.nturbs])

        x_unnorm, y_unnorm = self._gen_locs_from_optVars(x_in)
        x = self._norm(np.array(x_unnorm), self.min_dist, self.spacing_x_init)
        y = self._norm(np.array(y_unnorm), self.min_dist, self.spacing_y_init)

        dist = [
            np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            for i in range(self.nturbs)
            for j in range(self.nturbs)
            if i != j
        ]

        # print('spacing constraint: ', np.min(dist))
        return np.min(dist)

    def _generate_constraints(self):
        tmp1 = {
            "type": "ineq",
            "fun": lambda x, *args: self._space_constraint(x, self.min_dist),
            "args": (self.min_dist,),
        }
        tmp2 = {"type": "ineq", "fun": lambda x, *args: self._AEP_constraint(x)}
        con_strs = []
        for i in range(self.nturbs - 1):
            con_str = (
                "x["
                + str(i + 1)
                + "] - x["
                + str(i)
                + "] - "
                + str(self._norm(self.min_dist, self.bndx_min, self.bndx_max))
            )
            con_strs.append(con_str)

        cons = [tmp1, tmp2]
        for i in range(self.nturbs - 1):
            eval_str = (
                'cons.append({"type": "ineq", "fun": lambda x, *args: '
                + con_strs[i]
                + "})"
            )
            exec(eval_str)
            # eval('cons.append(con_num_str)')

        # print(cons)
        # lkj

        # self.cons = [tmp1, tmp2]
        self.cons = cons

    def _optimize(self):
        self.residual_plant = minimize(
            self._powDens_opt,
            self.x0,
            method=self.opt_method,
            bounds=self.bnds,
            constraints=self.cons,
            options=self.opt_options,
        )

        opt_results = self.residual_plant.x

        return opt_results

    def optimize(self):
        """
        This method finds the optimized layout of wind turbines for power
        production given the provided frequencies of occurance of wind
        conditions (wind speed, direction).

        Returns:
            opt_locs (iterable): A list of the optimized x, y locations of each
            turbine (m).
        """
        print("=====================================================")
        print("Optimizing turbine layout...")
        print("Number of parameters to optimize = ", len(self.x0))
        print("=====================================================")

        opt_vars_norm = self._optimize()

        print("Optimization complete!")

        # opt_locs_x = [
        #     self._unnorm(valx, self.bndx_min, self.bndx_max)
        #     for valx in opt_vars_norm[0 : self.nturbs]
        # ]

        # opt_locs_y = [
        #     self._unnorm(valy, self.bndy_min, self.bndy_max)
        #     for valy in opt_vars_norm[self.nturbs : 2 * self.nturbs]
        # ]
        print("optvarsnorm: ", opt_vars_norm)
        opt_locs_x, opt_locs_y = self._gen_locs_from_optVars(opt_vars_norm)

        opt_yaw = [
            self._unnorm(yaw, self.yaw_min, self.yaw_max) for yaw in opt_vars_norm[1:]
        ]
        opt_yaw = [
            opt_yaw[i * self.nturbs : (i + 1) * self.nturbs]
            for i in range(len(self.wd))
        ]

        self.opt_area = self.find_layout_area(opt_locs_x + opt_locs_y)

        return [opt_locs_x, opt_locs_y, opt_yaw]

    def reinitialize_opt(
        self,
        wd=None,
        ws=None,
        freq=None,
        AEP_initial=None,
        x0=None,
        bnds=None,
        min_dist=None,
        yaw_lims=None,
        opt_method=None,
        opt_options=None,
        nturbs_x=None,
        nturbs_y=None,
        spacing_x_init=None,
        spacing_y_init=None,
    ):
        """
        This method reinitializes any optimization parameters that are
        specified. Otherwise, the current parameter values are kept.

        Args:
            wd (np.array): An array of wind directions (deg). Defaults to None.
            ws (np.array): An array of wind speeds (m/s). Defaults to None.
            freq (np.array): An array of the frequencies of occurance
                correponding to each pair of wind direction and wind speed
                values. Defaults to None.
            AEP_initial (float): The initial Annual Energy
                Production used for normalization in the optimization (Wh)
                (TODO: Is Watt-hours the correct unit?). Defaults to None.
            x0 (iterable, optional): The initial turbine locations,
                ordered by x-coordinate and then y-coordiante
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]) (m). If none are
                provided, x0 initializes to the current turbine locations.
                Defaults to None.
            bnds (iterable, optional): Bounds for the optimization
                variables (pairs of min/max values for each variable (m)). If
                none are specified, they are set to some example values (TODO:
                what is the significance of these example values?). Defaults to
                None.
            min_dist (float, optional): The minimum distance to be
                maintained between turbines during the optimization (m). If not
                specified, initializes to 2 rotor diameters. Defaults to None.
            opt_method (str, optional): The optimization method used by
                scipy.optimize.minize. Defaults to None.
            opt_options (dict, optional): Optimization options used by
                scipy.optimize.minize. Defaults to None.
        """
        # if boundaries is not None:
        #     self.boundaries = boundaries
        #     self.bndx_min = np.min([val[0] for val in boundaries])
        #     self.bndy_min = np.min([val[1] for val in boundaries])
        #     self.boundaries_norm = [[self._norm(val[0], self.bndx_min, \
        #                           self.bndx_max)] for val in self.boundaries]
        # self.bndx_min = np.min(
        #     [coord.x1 for coord in self.fi.floris.farm.turbine_map.coords]
        # )
        # self.bndx_max = np.max(
        #     [coord.x1 for coord in self.fi.floris.farm.turbine_map.coords]
        # )
        # self.bndy_min = np.min(
        #     [coord.x2 for coord in self.fi.floris.farm.turbine_map.coords]
        # )
        # self.bndy_max = np.max(
        #     [coord.x2 for coord in self.fi.floris.farm.turbine_map.coords]
        # )
        if min_dist is not None:
            self.min_dist = min_dist
        else:
            self.min_dist = 2 * self.fi.floris.farm.turbines[0].rotor_diameter
        self.bndx_min = self.min_dist
        self.bndx_max = spacing_x_init
        self.bndy_min = self.min_dist
        self.bndy_max = spacing_y_init
        self.spacing_x_init = spacing_x_init
        self.spacing_y_init = spacing_y_init
        self.nturbs_x = nturbs_x
        self.nturbs_y = nturbs_y
        if yaw_lims is not None:
            self.yaw_min = yaw_lims[0]
            self.yaw_max = yaw_lims[1]
        else:
            self.yaw_min = 0.0
            self.yaw_max = 25.0
        if wd is not None:
            self.wd = wd
        if ws is not None:
            self.ws = ws
        if freq is not None:
            self.freq = freq
        if AEP_initial is not None:
            print("AEP initial 2: ", AEP_initial)
            self.AEP_initial = AEP_initial
        # else:
        # self.AEP_initial = self.fi.get_farm_AEP(self.wd, self.ws, self.freq)
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [
                self._norm(coord.x1, self.bndx_min, self.bndx_max)
                for coord in self.fi.floris.farm.turbine_map.coords
            ] + [0.0] * self.nturbs * len(self.wd)
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds()
        if opt_method is not None:
            self.opt_method = opt_method
        if opt_options is not None:
            self.opt_options = opt_options

        self._generate_constraints()
        # self.layout_dist_initial = np.max(self.x0[0:self.nturbs]) \
        #    - np.min(self.x0[0:self.nturbs])
        # self.layout_dist_initial = self._avg_dist(self.x0[0 : self.nturbs])
        # print('initial dist: ', self.layout_dist_initial)

        self.layout_x_orig = [
            coord.x1 for coord in self.fi.floris.farm.turbine_map.coords
        ]
        self.layout_y_orig = [
            coord.x2 for coord in self.fi.floris.farm.turbine_map.coords
        ]

        self.initial_area = self.find_layout_area(
            self.layout_x_orig + self.layout_y_orig
        )

    def plot_layout_opt_results(self):
        """
        This method plots the original and new locations of the turbines in a
        wind farm after layout optimization.
        """
        # locsx_old = [
        #     self._unnorm(valx, self.bndx_min, self.bndx_max)
        #     for valx in self.x0[0 : self.nturbs]
        # ]
        # locsy_old = [
        #     self._unnorm(valy, self.bndy_min, self.bndy_max)
        #     for valy in self.x0[self.nturbs : 2 * self.nturbs]
        # ]
        # locsx = [
        #     self._unnorm(valx, self.bndx_min, self.bndx_max)
        #     for valx in self.residual_plant.x[0 : self.nturbs]
        # ]
        # locsy = [
        #     self._unnorm(valy, self.bndy_min, self.bndy_max)
        #     for valy in self.residual_plant.x[self.nturbs : 2 * self.nturbs]
        # ]
        locsx_old, locsy_old = self._gen_locs_from_optVars([1.0, 1.0])
        locsx, locsy = self._gen_locs_from_optVars(self.residual_plant.x)

        plt.figure(figsize=(4, 3))
        fontsize = 10
        plt.plot(locsx_old, locsy_old, "ob")
        plt.plot(locsx, locsy, "or")
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel("x (m)", fontsize=fontsize)
        plt.ylabel("y (m)", fontsize=fontsize)
        plt.axis("equal")
        plt.grid()
        plt.tick_params(which="both", labelsize=fontsize)
        plt.legend(
            ["Old locations", "New locations"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=fontsize,
        )
