# %%
include("MPC.jl");

# %%
N = 50; δt = 0.01; #N = 100

Q  = Matrix(Diagonal([1.,1,1,1,1,1])) * 10.
R  = Matrix(Diagonal([1.,1,1])) * .1
Qf = Matrix(Diagonal([1.,1,1,1,1,1])) * 10.

n = size(Q)[1]; m = size(R)[1];
Np = (N-1)*(n+m)
Nd = (N-1)*n

ctrl = OSQPController(Q, R, Qf, δt, N, Np, Nd);

# %%
include("MPC.jl");

mₜ = 4.709;             # Mass of satellite Megagrams
mₛ = 5.97e21;           # Mass of earth Megagrams
G = 8.65e-19;           # Gravitational constant Megameters^3/Hour^2 Megagrams
earthRadius = 6.371;    # Radius of earth Megameters
orbitRadius = earthRadius + 1.
μ = sqrt(G * mₛ / ((orbitRadius)^3))    # radians / hour

num_steps = 1000
x_init = [-.5, 0, 0, 0.1, 0., 0]
x_hist, u_hist = simulate(ctrl, x_init; num_steps=num_steps, verbose=false);

# %%
using Plots
plot([x_hist[i][1] for i in 1:length(x_hist)])
plot!([x_hist[i][2] for i in 1:length(x_hist)])

# %%    Build CSV for Kirtan
using DataFrames
using CSV

tmp = DataFrame(hcat(x_hist...)', ["x_tc", "y_tc", "z_tc", "vx_tc", "vy_tc", "vz_tc"]);
CSV.write("data/x_hist_position.csv", tmp);

# %%    Build Rotation matricies
using Rotations

p_st = [[orbitRadius * cos(μ * δt * (i-1)),
         orbitRadius * sin(μ * δt * (i-1))] for i in 1:num_steps]
p_tc = Matrix(CSV.read("data/x_hist_position.csv", DataFrame))[:, 1:3];

tmp = Matrix(CSV.read("data/x_hist_quaternion.csv", DataFrame));
R_sc = [RotMatrix(UnitQuaternion(tmp[i,1:4]))[1:2,1:2] for i in 1:size(tmp)[1]];
R_st = [RotMatrix(UnitQuaternion(tmp[i,8:11]))[1:2,1:2] for i in 1:size(tmp)[1]];

# %%
using Plots
using Images, FileIO
img = load("graphics/planet-earth.png")

scale = .95
tCoor = [1  0; 0  1]';
cCoor = [1  0; 0  1]';

anim = @animate for i ∈ 1:num_steps
    scatter([p_st[i][1]], [p_st[i][2]],
            xlims=(-1.3*orbitRadius,1.3*orbitRadius),
            ylims=(-1.3*orbitRadius,1.3*orbitRadius),
            aspect_ratio=:equal,
            label="Target",
            size=(750,750))
    scatter!([p_st[i][1] + x_hist[i][1]], [p_st[i][2] + x_hist[i][2]],
             label="Chaser")

    quiver!([p_st[i][1]], [p_st[i][2]],
            quiver=([(R_st[i] * tCoor)[1,1]],
                    [(R_st[i] * tCoor)[2,1]]),
            aspect_ratio=:equal,
            color=1)
    quiver!([p_st[i][1]], [p_st[i][2]],
            quiver=([(R_st[i] * tCoor)[1,2]],
                    [(R_st[i] * tCoor)[2,2]]),
            color=2)

    quiver!([p_st[i][1] + x_hist[i][1]], [p_st[i][2] + x_hist[i][2]],
            quiver=([(R_sc[i] * cCoor)[1,1]],
                    [(R_sc[i] * cCoor)[2,1]]),
            color=1)
    quiver!([p_st[i][1] + x_hist[i][1]], [p_st[i][2] + x_hist[i][2]],
            quiver=([(R_sc[i] * cCoor)[1,2]],
                    [(R_sc[i] * cCoor)[2,2]]),
            color=2)

    plot!([-earthRadius*scale, earthRadius*scale],
          [-earthRadius*scale, earthRadius*scale],
          img[end:-1:1, :], yflip = false)
end every 1;

# %%
fps = 20
gif(anim, "graphics/anim_fps$fps.mp4", fps=fps)
