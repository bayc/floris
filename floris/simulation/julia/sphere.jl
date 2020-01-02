
mutable struct Sphere <: AbstractShape
    r::Float64
    name::String
end

function volume(shape::Sphere)
    return 4 / 3 * pi * shape.r^3
end

function edit_shape(shape::Sphere; r=nothing, name=nothing)
    if r !== nothing
        shape.r = r
    end
    if name !== nothing
        shape.name = name
    end
end
