# import numpy as np
import autograd.numpy as np  # thinly-wrapped numpy
# from autograd import grad    # the only autograd function you may ever need
from jax import grad

# import matplotlib.pyplot as plt
import floris.tools as wfct

def grad_func(yaw_angles, fi):
    fi.calculate_wake(yaw_angles=yaw_angles)

    # return fi.get_farm_power()
    turb_powers = [
        turbine.power for turbine in fi.floris.farm.turbines
    ]
    print('here!')
    [print('turb powers: ', val._value) for val in turb_powers]
    return np.sum(turb_powers)

fi = wfct.floris_interface.FlorisInterface("example_input.json")
# D = fi.floris.farm.flow_field.turbine_map.turbines[0].rotor_diameter
# layout_x = [0, 0]
# layout_y = [0, 1000]
# fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

yaw_angles = [0.5, 0.5, 0.5, 0.5]

# power = fi.get_farm_power_for_yaw_angle(yaw_angles=yaw_angles)

# yaw_grad = grad(fi.get_farm_power_for_yaw_angle)

yaw_grad = grad(grad_func)

print('--------------------------')
print('starting grad calculation')
print('--------------------------')
print(yaw_grad(yaw_angles, fi))

