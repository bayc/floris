abstract type AbstractShape end

struct Cube <: AbstractShape    
    x
    y
    z
    name
end

struct Sphere <: AbstractShape
    r
    name
end

struct Pyramid <: AbstractShape
    l
    w
    h
    name
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
