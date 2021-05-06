# %%
include("MPC.jl");

# %%
N = 50; δt = 0.01; #N = 100

Q  = Matrix(Diagonal([20.,20,20,20, 1,1,1, 1,1,1,1, 1,1,1])) * 5.
R  = Matrix(Diagonal([1.,1,1])) * 1.
Qf = Matrix(Diagonal([20.,20,20,20, 1.1,1.1,1.1, 1,1,1,1, 1,1,1])) * 5.

n = size(Q)[1]; m = size(R)[1];
Np = (N-1)*(n-2+m)
Nd = (N-1)*(n-2)

num_steps = 1000

ctrl = OSQPController(Q, R, Qf, δt, N, Np, Nd);

# %%
include("MPC.jl");
mₜ = 4.709;             # Mass of satellite Megagrams
mₛ = 5.97e21;           # Mass of earth Megagrams
G = 8.65e-19;           # Gravitational constant Megameters^3/Hour^2 Megagrams
earthRadius = 6.371;    # Radius of earth Megameters
orbitRadius = earthRadius + 1.0
μ = sqrt(G * mₛ / ((orbitRadius)^3))    # radians / hour

x_init = [0.877583, 0.0, 0.0, -0.479426, 0, 0, 1,
          1., 0, 0, 0, 0, 0, μ]
x_hist, u_hist, cost_hist = simulate(ctrl, x_init; num_steps=num_steps, verbose=false);

# %%
using DataFrames
using CSV

tmp = DataFrame(hcat(x_hist...)', ["q_cw", "q_cx", "q_cy", "q_cz", "w_cx", "w_cy", "w_cz",
                                   "q_tw", "q_tx", "q_ty", "q_tz", "w_tx", "w_ty", "w_tz"]);
CSV.write("x_hist.csv", tmp)

# %%
using Plots
plot([1:length(cost_hist);], cost_hist, title="Orientation + Angular Velocity Cost",
     xlabel="Simulation Step", legend=false, size=(1000,666), yaxis=:log)
savefig("graphics/rotation_cost.png")

# %%
plot([x_hist[i][7] for i in 1:length(x_hist)])
plot!([x_hist[i][14] for i in 1:length(x_hist)])

# %%
plot([x_hist[i][8] for i in 1:length(x_hist)])
plot!([x_hist[i][11] for i in 1:length(x_hist)])
plot!([x_hist[i][1] for i in 1:length(x_hist)])
plot!([x_hist[i][4] for i in 1:length(x_hist)])

# %%
x_hist[end][1:4]

# %%
x_hist[end][8:11]

# %%
