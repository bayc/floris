# from julia.api import LibJulia

# api = LibJulia.load()
# api.sysimage = "sys.so"
# api.init_julia()

import julia

class NameObject():
    def __init__(self, new_name):
        self.new_name = new_name

# Load the `shapes` module
shapes = julia.Julia(compiled_modules=False)
shapes.include("shapes.jl")

# Instantiate a Cube and do some stuff with it
cube = shapes.Cube(
    2.0,
    12.0,
    1.0,
    "cube"
)
print(cube.x)
print(shapes.volume(cube))
shapes.edit_shape(cube, x=1.)
print(cube.x)
print(shapes.volume(cube))
shapes.change_name(cube, NameObject("cyoob"))

# Instantiate a Sphere and do some stuff with it
sphere = shapes.Sphere(
    2,
    "sphere"
)
print(sphere.r)
print(shapes.volume(sphere))
