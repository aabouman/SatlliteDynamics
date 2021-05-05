include("MPC.jl");

# %%
n = 18; m = 6;
N = 100; δt = 0.001;

Q  = Matrix(Diagonal([1.,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0])) * 1.
R  = Matrix(Diagonal([1.,1,1,1,1,1])) * .1
Qf = Matrix(Diagonal([1.,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0])) * 10.

ctrl = OSQPController(Q, R, Qf, δt, N, (N-1)*(n));

# %%
include("MPC.jl");

x_init = [-.1, 0, 0, 1., 0, 0, 0, 10., 0., 0, 0, 0, 0,
          0, 0, 0, 0, 0, 2*π/2]
x_hist, u_hist = simulate(ctrl, x_init; num_steps=100, verbose=true)
# %%
μ = sqrt(G * mₛ / (earthRadius^3)) 
(μ^2)*3
# %%
using Plots
plot([x_hist[i][1] for i in 1:length(x_hist)])
# %%


# %%
plot([x_hist[i][2] for i in 1:length(x_hist)])

# %%
plot([x_hist[i][1] for i in 1:length(x_hist)],
     [x_hist[i][2] for i in 1:length(x_hist)])

# %%
plot([x_hist[i][8] for i in 1:length(x_hist)])


# %%
plot([x_hist[i][9] for i in 1:length(x_hist)])

# %%
plot([u_hist[i][1] for i in 1:length(u_hist)])
plot!([u_hist[i][2] for i in 1:length(u_hist)])
plot!([u_hist[i][3] for i in 1:length(u_hist)])

# %%
plot([u_hist[i][4] for i in 1:length(u_hist)])
plot!([u_hist[i][5] for i in 1:length(u_hist)])
plot!([u_hist[i][6] for i in 1:length(u_hist)])
