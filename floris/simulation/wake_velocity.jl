
struct Vec3
    x1::Float64
    x2::Float64
    x3::Float64
end

model_grid_resolution = Vec3(250.0, 100.0, 75.0)
model_string = "jensen"

function func(x_locations, y_locations, z_locations, turbine, turbine_coord, deflection_field, flow_field)
    we = 0.05

    # define the boundary of the wake model ... y = mx + b
    m = we
    x = x_locations .- turbine_coord.x1
    b = turbine.rotor_radius

    boundary_line = m .* x .+ b

    y_upper = boundary_line .+ turbine_coord.x2 + deflection_field
    y_lower = -1 * boundary_line .+ turbine_coord.x2 + deflection_field

    z_upper = boundary_line .+ turbine.hub_height
    z_lower = -1 * boundary_line .+ turbine.hub_height

    # calculate the wake velocity
    c = (turbine.rotor_diameter ./ (2 * we .* (x_locations .- turbine_coord.x1) .+ turbine.rotor_diameter)).^2

    # filter points upstream and beyond the upper and lower bounds of the wake
    # c[x_locations .- turbine_coord.x1 .< 0] = 0
    # c[y_locations .> y_upper] = 0
    # c[y_locations .< y_lower] = 0
    # c[z_locations .> z_upper] = 0
    # c[z_locations .< z_lower] = 0
    
    return 2 * turbine.aI * c .* flow_field.u_initial, zeros(size(flow_field.u_initial)), zeros(size(flow_field.u_initial))
end
