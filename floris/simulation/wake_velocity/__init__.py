# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from floris.simulation.wake_velocity.none import NoneVelocityDeficit
from floris.simulation.wake_velocity.cumulative_gauss_curl import CumulativeGaussCurlVelocityDeficit
from floris.simulation.wake_velocity.gauss import GaussVelocityDeficit
from floris.simulation.wake_velocity.jensen import JensenVelocityDeficit
from floris.simulation.wake_velocity.turbopark import TurbOParkVelocityDeficit
from floris.simulation.wake_velocity.curl import CurledWakeVelocityDeficit
