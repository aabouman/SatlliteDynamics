using Plots
plotly()

include("MPC.jl");

# %%
n = 12; m = 6;
N = 10000; δt = 0.001;

x_init1 = [6.371, 0, 0, 1., 0, 0, 0, 0, 15.44, 0, 0, 0, 3,
           1, 0, 0, 1., 0, 0, 0, 0, .1, 0, 0, 0, 0]
roll1 = rollout(x_init1, [zeros(6) for _ in 1:N], δt);

x1s = [roll1[i][1] for i in 1:length(roll1)]
y1s = [roll1[i][2] for i in 1:length(roll1)]
x2s = [roll1[i][14] for i in 1:length(roll1)]
y2s = [roll1[i][15] for i in 1:length(roll1)]

x_init2 = [7.371, 0, 0, 1., 0, 0, 0, 0, 15.54, 0, 0, 0, 3,
           0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0]
roll2 = rollout(x_init2, [zeros(6) for _ in 1:N], δt);

x3s = [roll2[i][1] for i in 1:length(roll1)];
y3s = [roll2[i][2] for i in 1:length(roll1)];

# %%
nrg = systemEnergy.(roll1);

# %%
plot(nrg)

# %%
plot(x1s)

# %%
plot(x1s, y1s)
# plot!(x1s+x2s, y1s+y2s)
#plot!(x3s, y3s)

# %%
using LinearAlgebra: Diagonal

n = 13; m = 6;
N = 50; δt = 0.001;
num_steps = 1000
Np = (N-1)*(n-1+m)
Nd = (N-1)*(n-1)

Q = Matrix(Diagonal([1.,1,1, 2,2,2,2, 1,1,1, 1,1,1])) * 1.
R = Matrix(Diagonal([1.,1,1,1,1,1])) * .1
Qf = Matrix(Diagonal([1.,1,1, 2,2,2,2, 1,1,1, 1,1,1])) * 200.

ctrl = OSQPController(Q, R, Qf, δt, N, Np, Nd);

# %%
using LinearAlgebra: det
include("MPC.jl");

x_init1 = [6.371, 0, 0, 1., 0, 0, 0, 0, 15.44, 0, 0, 0, 0,
           1,     0, 0, 1., 0, 0, 0, 0,   .1,  0, 0, 0, 0]

x_hist, u_hist, cost_hist = simulate(ctrl, x_init1; num_steps=num_steps, verbose=false);

# %%
using Plots
plot([1:length(cost_hist);], cost_hist,
     title="Orientation + Angular Velocity Cost",
     xlabel="Simulation Step",
     legend=false,
     size=(700, 500),
     # yaxis=:log
     )
# savefig("graphics/rotation_cost.png")

# %%
xTs = [x_hist[i][1] for i in 1:length(x_hist)];
yTs = [x_hist[i][2] for i in 1:length(x_hist)];

xCs = xTs + [x_hist[i][14] for i in 1:length(x_hist)];
yCs = yTs + [x_hist[i][15] for i in 1:length(x_hist)];

plot(1:length(x_hist), xTs, yTs, size=(700, 700), label="Target", legend=:bottomleft)
plot!(1:length(x_hist), xCs, yCs, label="Chaser")

# %%
u_hist

# %%
δq = [0.1, 0.1, 0.1]
