# %%
include("MPC.jl");

# %%
N = 50; δt = 0.005;

Q  = Matrix(Diagonal([1.,1,1,1,1,1])) * 1.
R  = Matrix(Diagonal([1.,1,1])) * .1
Qf = Matrix(Diagonal([1.,1,1,1,1,1])) * 10.

n = size(Q)[1]; m = size(R)[1];
Np = (N-1)*(n+m)
Nd = (N-1)*n

ctrl = OSQPController(Q, R, Qf, δt, N, Np, Nd);

# %%
include("MPC.jl");

mₜ = 4.709;             # Mass of satellite Megagrams
mₛ = 5.972e21;          # Mass of earth Megagrams
G = 8.6498928e-19;      # Gravitational constant Megameters^3/Hour^2
earthRadius = 6.37814;  # Radius of earth Megameters
orbitRadius = earthRadius + 1
μ = sqrt(G * mₛ / ((orbitRadius)^3))

num_steps = 3500
x_init = [-.5, 0, 0, 0.1, 0., 0]
x_hist, u_hist = simulate(ctrl, x_init; num_steps=num_steps, verbose=false);

# %%
using Plots
plot([x_hist[i][1] for i in 1:length(x_hist)])
plot!([x_hist[i][2] for i in 1:length(x_hist)])

# %% Build CSV for Kirtan
using DataFrames
using CSV

tmp = DataFrame(hcat(x_hist...)', ["x_tc", "y_tc", "z_tc", "vx_tc", "vy_tc", "vz_tc"]);
CSV.write("x_hist.csv", tmp)

# %%
plot([x_hist[i][1] for i in 1:length(x_hist)],
     [x_hist[i][2] for i in 1:length(x_hist)])

# %%
plot([u_hist[i][1] for i in 1:length(u_hist)])
plot!([u_hist[i][2] for i in 1:length(u_hist)])
# plot!([u_hist[i][3] for i in 1:length(u_hist)])

# %%
using Plots
using Images, FileIO
img = load("graphics/planet-earth.png")

scale = .95

p_st = [[orbitRadius * cos(μ * (2π) * δt * (i-1)),
         orbitRadius * sin(μ * (2π) * δt * (i-1))] for i in 1:num_steps]

anim = @animate for i ∈ 1:num_steps
    scatter([p_st[i][1]], [p_st[i][2]], xlims=(-1.2*orbitRadius,1.2*orbitRadius),
            ylims=(-1.2*orbitRadius,1.2*orbitRadius), aspect_ratio=:equal)
    scatter!([p_st[i][1] + x_hist[i][1]], [p_st[i][2] + x_hist[i][2]])
    plot!([-earthRadius*scale, earthRadius*scale],
          [-earthRadius*scale, earthRadius*scale],
          img[end:-1:1, :], yflip = false)
end every 1;

# %%
fps = 30
gif(anim, "graphics/anim_fps$fps.gif", fps=fps)
