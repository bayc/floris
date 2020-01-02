
mutable struct Pyramid <: AbstractShape
    l::Float64
    w::Float64
    h::Float64
    name::String
end

function volume(shape::Pyramid)
    return shape.l * shape.w * shape.h / 3
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
end
