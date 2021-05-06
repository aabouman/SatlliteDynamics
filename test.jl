include("MPC.jl");

# %%
N = 50; δt = 0.01; #N = 100

Q  = Matrix(Diagonal([1,1,1,1,1,1,1,0,0,0,0,0,0])) * 10.
R  = Matrix(Diagonal([1.,1,1])) * 5.
Qf = Matrix(Diagonal([2.,2.,2.,2.,1.1, 1.1, 1.1, 0,0,0,0,0,0])) * 10.

n = size(Q)[1]; m = size(R)[1];

ctrl = OSQPController(Q, R, Qf, δt, N, (N-1)*(n-1));


# %%
include("MPC.jl");
mₜ = 4.709;             # Mass of satellite Megagrams
mₛ = 5.97e21;           # Mass of earth Megagrams
G = 8.65e-19;           # Gravitational constant Megameters^3/Hour^2 Megagrams
earthRadius = 6.371;    # Radius of earth Megameters
orbitRadius = earthRadius + 1.0
μ = sqrt(G * mₛ / ((orbitRadius)^3))    # radians / hour

x_init = [1., 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, μ]
x_hist, u_hist, cost_hist = simulate(ctrl, x_init; num_steps=3000, verbose=false);

# %%
using DataFrames
using CSV

tmp = DataFrame(hcat(x_hist...)', ["q_w", "q_x", "q_y", "q_z", "w_cx", "w_cy", "w_cz",
                                   "e_x", "e_y", "e_z", "w_tx", "w_ty", "w_tz"]);
CSV.write("x_hist.csv", tmp)

# %%
using Plots
plot(cost_hist)
# %%
cost_hist[end]
# %%
x_hist[end][7]
# %%
plot([x_hist[i][7] for i in 1:length(x_hist)])

# %%
plot([x_hist[i][1] for i in 1:length(x_hist)])
plot!([x_hist[i][2] for i in 1:length(x_hist)])
plot!([x_hist[i][3] for i in 1:length(x_hist)])
plot!([x_hist[i][4] for i in 1:length(x_hist)])

# %%
RotXYZ(UnitQuaternion(x_hist[end][1:4])).theta3 .% (2pi)

# %%
RotXYZ(x_hist[end][8:10]...).theta3 .% (2pi)

# %%
using Rotations

# %%
dist(q1, q2) = min(1 + q1'*q2, 1 - q1'*q2)
eul_err = [dist(x_hist[i][1:4], params(UnitQuaternion(RotXYZ(x_hist[i][8:10]...))))
           for i = 1:length(x_hist)];
# eul_err = hcat(eul_err...)
# plot(eul_err)

# %%
plot(eul_err)

# %%
hcat(eul_err...)
