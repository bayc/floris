from julia.api import LibJulia

api = LibJulia.load()
api.sysimage = "sys.so"
api.init_julia()

# from julia import Main
import julia

j = julia.Julia()

j.include("shapes.jl")

x = 2.
y = 12.
z = 1.
name = 'cube1'

cube = j.Cube(y=y, x=x, z=z, name=name)

print(cube.x)