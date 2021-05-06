include("MPC.jl");

# %%
N = 50; δt = 0.01; #N = 100

Q  = Matrix(Diagonal([1.,1.,1.,1.,1,1,1,0,0,0,0,0,0])) * 10.
R  = Matrix(Diagonal([1.,1,1])) * 1.
Qf = Matrix(Diagonal([5.,5.,5.,5.,1,1,1,0,0,0,0,0,0])) * 10.

n = size(Q)[1]; m = size(R)[1];

ctrl = OSQPController(Q, R, Qf, δt, N, (N-1)*(n-1));

# %%
include("MPC.jl");

x_init = [1., 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 2*π/2]
x_hist, u_hist, cost_hist = simulate(ctrl, x_init; num_steps=5000, verbose=false);

# %%
plot(cost_hist)

# %%
plot([x_hist[i][7] for i in 1:length(x_hist)])

# %%
using Plots
plot([x_hist[i][1] for i in 1:length(x_hist)])
plot!([x_hist[i][2] for i in 1:length(x_hist)])
plot!([x_hist[i][3] for i in 1:length(x_hist)])
plot!([x_hist[i][4] for i in 1:length(x_hist)])

# %%
RotXYZ(UnitQuaternion(x_hist[end][1:4]))

# %%
(RotXYZ(x_hist[end][8:10]...))
(0.0, 0.0, 157.08).%(2*pi)
