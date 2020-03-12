# import numpy as np
# import autograd.numpy as np  # thinly-wrapped numpy
# from autograd import grad    # the only autograd function you may ever need
import jax.numpy as np
from jax import grad

# import matplotlib.pyplot as plt
import floris.tools as wfct

def grad_func_yaw(yaw_angles, fi):
    fi.calculate_wake(yaw_angles=yaw_angles)

    # return fi.get_farm_power()
    turb_powers = [
        turbine.power for turbine in fi.floris.farm.turbines
    ]
    print('here!', np.sum(turb_powers))
    return np.sum(turb_powers)

def grad_func_layout(layout, fi):
    nturbs = len(fi.layout_x)

    layout_x = layout[0:nturbs]
    layout_y = layout[nturbs:]
    # print('x: ', layout_x)
    # print('y: ', layout_y)
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
    # print('here')
    fi.calculate_wake()

    # return fi.get_farm_power()
    turb_powers = [
        turbine.power for turbine in fi.floris.farm.turbines
    ]
    print('here!', np.sum(turb_powers))
    return np.sum(turb_powers)

fi = wfct.floris_interface.FlorisInterface("example_input.json")

D = fi.floris.farm.flow_field.turbine_map.turbines[0].rotor_diameter
nturbs_x = 5
nturbs_y = 5
layout_x = [i*7*D for i in range(nturbs_x)]*nturbs_y
layout_y = [i*7*D for i in range(nturbs_y)]
layout_y = [[val]*nturbs_x for val in layout_y]
layout_y = [val for vals in layout_y for val in vals]
print('x: ', layout_x)
print('y: ', layout_y)
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# yaw_angles = [0.5, 0.5, 0.5, 0.5]
yaw_angles = [0.5]*nturbs_x*nturbs_y
layout = [5., 5., 5., 5., 5., 5., 5., 5.]

# power = fi.get_farm_power_for_yaw_angle(yaw_angles=yaw_angles)

# yaw_grad = grad(fi.get_farm_power_for_yaw_angle)

yaw_grad = grad(grad_func_yaw)
# layout_grad = grad(grad_func_layout)

print('--------------------------')
print('starting grad calculation')
print('--------------------------')
print(yaw_grad(yaw_angles, fi))
# print(layout_grad(layout, fi))

