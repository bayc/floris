
mutable struct Cube <: AbstractShape    
    x::Float64
    y::Float64
    z::Float64
    name::String
end

function volume(shape::Cube)
    return shape.x * shape.y * shape.z
end

function edit_shape(shape::Cube; x=nothing, y=nothing, z=nothing, name=nothing)
    if x !== nothing
        shape.x = x
    end
    if y !== nothing
        shape.y = y
    end
    if z !== nothing
        shape.z = z
    end
    if name !== nothing
        shape.name = name
    end
end

function change_name(shape::Cube, name_object)
    println("change_name: ", name_object.new_name)
end
