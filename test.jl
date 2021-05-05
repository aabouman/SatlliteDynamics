include("MPC.jl");

# %%
N = 100; δt = 0.001;

Q  = Matrix(Diagonal([0.,0,0,0,1,1,1,0,0,0,0,0,0])) * 10.
R  = Matrix(Diagonal([1.,1,1])) * 1
Qf = Matrix(Diagonal([0.,0,0,0,1,1,1,0,0,0,0,0,0])) * 10.

n = size(Q)[1]; m = size(R)[1];

ctrl = OSQPController(Q, R, Qf, δt, N, (N-1)*(n-1));

# %%
include("MPC.jl");

x_init = [1., 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 2*π/2]
x_hist, u_hist = simulate(ctrl, x_init; num_steps=1000, verbose=false);

# %%
x_hist[end][1:4]

# %%
x_hist[end][8:10]

# %%
UnitQuaternion(RotXYZ(x_hist[end][8:10]...))

# %%
include("MPC.jl");
u_init = rand(3)
x_init = rand(13)
x_init[1:4] .= normalize(x_init[1:4])
x_init[8:9] .= 0; x_init[11:12] .= 0;
x_init

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
