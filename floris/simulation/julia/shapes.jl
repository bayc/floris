abstract type AbstractShape end

mutable struct Cube <: AbstractShape    
    x::Float64
    y::Float64
    z::Float64
    name::String
end

mutable struct Sphere <: AbstractShape
    r::Float64
    name::String
end

mutable struct Pyramid <: AbstractShape
    l::Float64
    w::Float64
    h::Float64
    name::String
end

function volume(shape::Cube)
    return shape.x * shape.y * shape.z
end

function volume(shape::Sphere)
    return 4 / 3 * pi * shape.r^3
end

function volume(shape::Pyramid)
    return shape.l * shape.w * shape.h / 3
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
    return shape
end

function edit_shape(shape::Sphere; r=nothing, name=nothing)
    if r !== nothing
        shape.r = r
    end
    if name !== nothing
        shape.name = name
    end
    return shape
end

function edit_shape(shape::Pyramid; l=nothing, w=nothing, h=nothing, name=nothing)
    if l !== nothing
        shape.l = l
    end
    if w !== nothing
        shape.w = w
    end
    if h !== nothing
        shape.h = h
    end
    if name !== nothing
        shape.name = name
    end
    return shape
end