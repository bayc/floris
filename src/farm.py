# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from typing import Dict, List, Union

import attr
import numpy as np
import xarray as xr

from .turbine import Turbine
from .utilities import Vec3, FromDictMixin, iter_validator, attrs_array_converter


class FarmController:
    def __init__(self, n_wind_directions: int, n_wind_speeds: int, n_turbines: int) -> None:
        # TODO: This should hold the yaw settings for each turbine for each wind speed and wind direction

        # Initialize the yaw settings to an empty array
        self.yaw_angles = np.zeros((n_wind_speeds, n_turbines))

    def set_yaw_angles(self, yaw_angles: np.ndarray) -> None:
        """
        Set the yaw angles for each wind turbine at each atmospheric
        condition.

        Args:
            yaw_angles (np.ndarray): Array of dimensions (n wind directions,
            n wind speeds, n turbines)
        """
        # if yaw_angles.ndim != 3:
        #     raise ValueError("yaw_angles must be set for each turbine at each atmospheric condition.")
        self.yaw_angles[:] = yaw_angles

def create_turbines(mapping: Dict[str, dict]) -> Dict[str, Turbine]:
    return {t_id: Turbine.from_dict(config) for t_id, config in mapping.items()}


def generate_turbine_tuple(turbine: Turbine) -> tuple:
    exclusions = ("power_thrust_table", "model_string")
    return attr.astuple(
        turbine, filter=lambda attribute, value: attribute.name not in exclusions
    )


def generate_turbine_attribute_order(turbine: Turbine) -> List[str]:
    exclusions = ("power_thrust_table", "model_string")
    mapping = attr.asdict(
        turbine, filter=lambda attribute, value: attribute.name not in exclusions
    )
    return list(mapping.keys())


@attr.s(auto_attribs=True)
class Farm(FromDictMixin):
    """Farm is where wind power plants should be instantiated from a YAML configuration
    file. The Farm will create a heterogenous set of turbines that compose a windfarm,
    validate the inputs, and then create a vectorized representation of the the turbine
    data.

    Farm is the container class of the FLORIS package. It brings
    together all of the component objects after input (i.e., Turbine,
    Wake, FlowField) and packages everything into the appropriate data
    type. Farm should also be used as an entry point to probe objects
    for generating output.

    Args:
        turbine_id (List[str]): The turbine identifiers to map each turbine to one of
            the turbine classifications in `turbine_map`.
        turbine_map (Dict[str, Union[dict, Turbine]]): The dictionary mapping of unique
            turbines at the wind power plant. Takes either a pre-generated `Turbine`
            object, or a dictionary that will be used to generate the `Turbine` object.
        layout_x (Union[List[float], np.ndarray]): The x-coordinates for the turbines at
            the wind power plant.
        layout_y (Union[List[float], np.ndarray]): The y-coordinates for the turbines at
            the wind power plant.
        wtg (List[str]): The WTG ID values for each turbine. This field acts as metadata
            only.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    turbine_id: List[str] = attr.ib(validator=iter_validator(list, str))
    turbine_map: Dict[str, Union[dict, Turbine]] = attr.ib(converter=create_turbines)
    layout_x: Union[List[float], np.ndarray] = attr.ib(converter=attrs_array_converter)
    layout_y: Union[List[float], np.ndarray] = attr.ib(converter=attrs_array_converter)
    wtg_id: List[str] = attr.ib(
        factory=list,
        on_setattr=attr.setters.validate,
        validator=iter_validator(list, str),
    )

    coordinates: List[Vec3] = attr.ib(init=False)

    rotor_diameter: np.ndarray = attr.ib(init=False)
    hub_height: np.ndarray = attr.ib(init=False)
    pP: np.ndarray = attr.ib(init=False)
    pT: np.ndarray = attr.ib(init=False)
    generator_efficiency: np.ndarray = attr.ib(init=False)
    # power_thrust_table: np.ndarray = attr.ib(init=False)  # NOTE: Is this only necessary for the creation of the interpolations?
    fCp_interp: np.ndarray = attr.ib(init=False)
    fCt_interp: np.ndarray = attr.ib(init=False)
    power_interp: np.ndarray = attr.ib(init=False)
    rotor_radius: np.ndarray = attr.ib(init=False)
    rotor_area: np.ndarray = attr.ib(init=False)
    array_data: xr.DataArray = attr.ib(init=False)

    # Pre multi-turbine
    # i  j  k  l  m
    # wd ws x  y  z

    # With multiple turbines per floris ez (aka Chris)
    # i  j  k    l  m  n
    # wd ws t_ix x  y  z

    def __attrs_post_init__(self) -> None:
        self.coordinates = [
            Vec3([x, y, self.turbine_map[t_id].hub_height])
            for x, y, t_id in zip(self.layout_x, self.layout_y, self.turbine_id)
        ]

        self.generate_farm_points()

        # TODO: Enable the farm controller
        # # Turbine control settings indexed by the turbine ID
        self.farm_controller = FarmController(1, 1)  # TODO placeholder
        self.farm_controller.set_yaw_angles(np.zeros((len(self.layout_x))))

    @layout_x.validator
    def check_x_len(self, instance: str, value: Union[List[float], np.ndarray]) -> None:
        if len(value) < len(self.turbine_id):
            raise ValueError("Not enough `layout_x` values to match the `turbine_id`s")
        if len(value) > len(self.turbine_id):
            raise ValueError("Too many `layout_x` values to match the `turbine_id`s")

    @layout_y.validator
    def check_y_len(self, instance: str, value: Union[List[float], np.ndarray]) -> None:
        if len(value) < len(self.turbine_id):
            raise ValueError("Not enough `layout_y` values to match the `turbine_id`s")
        if len(value) > len(self.turbine_id):
            raise ValueError("Too many `layout_y` values to match the `turbine_id`s")

    @wtg_id.validator
    def check_wtg_id(self, instance: str, value: Union[list, List[str]]) -> None:
        if len(value) == 0:
            self.wtg_id = [
                f"t{str(i).zfill(4)}" for i in 1 + np.arange(len(self.turbine_id))
            ]
        elif len(value) < len(self.turbine_id):
            raise ValueError("There are too few `wtg_id` values")
        elif len(value) > len(self.turbine_id):
            raise ValueError("There are too many `wtg_id` values")

    def generate_farm_points(self) -> None:
        # Create an array of turbine values and the column ordering
        arbitrary_turbine = self.turbine_map[self.turbine_id[0]]
        column_order = generate_turbine_attribute_order(arbitrary_turbine)
        turbine_array = np.array(
            [generate_turbine_tuple(self.turbine_map[t_id]) for t_id in self.turbine_id]
        )

        # TODO: how to handle multiple data types xarray
        column_ix = {col: i for i, col in enumerate(column_order)}
        self.rotor_diameter = turbine_array[:, column_ix["rotor_diameter"]].astype(float)
        self.rotor_radius = turbine_array[:, column_ix["rotor_radius"]].astype(float)
        self.rotor_area = turbine_array[:, column_ix["rotor_area"]].astype(float)
        self.hub_height = turbine_array[:, column_ix["hub_height"]].astype(float)
        self.pP = turbine_array[:, column_ix["pP"]].astype(float)
        self.pT = turbine_array[:, column_ix["pT"]].astype(float)
        self.generator_efficiency = turbine_array[:, column_ix["generator_efficiency"]].astype(float)
        self.fCt_interp = turbine_array[:, column_ix["fCt_interp"]]
        # TODO: should we have both fCp_interp and power_interp
        self.fCp_interp = turbine_array[:, column_ix["fCp_interp"]]
        self.power_interp = turbine_array[:, column_ix["power_interp"]]

        self.data_array = xr.DataArray(
            turbine_array,
            coords=dict(wtg_id=self.wtg_id, turbine_attributes=column_order),
            attrs=dict(layout_x=self.layout_x, layout_y=self.layout_y),
        )

    def customize_turbine(self) -> None:
        # TODO: a method to update a turbine property? Is this needed?
        # DO THE WORK
        # self.generate_farm_points()
        pass

    def sort_turbines(self, by: str) -> np.ndarray:
        """Sorts the turbines by the given dimension.

        Args:
            by (str): The dimension to sort by; should be one of x or y.

        Returns:
            np.ndarray: The index order for retrieving data from `data_array` or any
                other farm object.
        """

        if by == "x":
            return np.argsort(self.layout_x)
        elif by == "y":
            return np.argsort(self.layout_y)
        else:
            raise ValueError("`by` must be set to one of 'x' or 'y'!")
