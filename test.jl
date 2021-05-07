using Plots
include("dynamics.jl")
include("MPC.jl");

# %%
n = 12; m = 6;
N = 10000; δt = 0.001;

xₛₜ_init = [6.371, 0, 0, 1., 0, 0, 0, 0, 15.44, 0, 0, 0, 0,
           1, 0, 0, 1., 0, 0, 0, 0, .1, 0, 0, 0, 0]
roll = rollout(xₛₜ_init, [zeros(6) for _ in 1:N], δt);

x1s = [roll[i][1] for i in 1:length(roll)]
y1s = [roll[i][2] for i in 1:length(roll)]
x2s = [roll[i][14] for i in 1:length(roll)]
y2s = [roll[i][15] for i in 1:length(roll)]

xₛₜ_init2 = [7.371, 0, 0, 1., 0, 0, 0, 0, 15.54, 0, 0, 0, 0,
            0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0]
roll2 = rollout(xₛₜ_init2, [zeros(6) for _ in 1:N], δt);

x3s = [roll2[i][1] for i in 1:length(roll)];
y3s = [roll2[i][2] for i in 1:length(roll)];

# %%
plot(x1s,y1s)
plot!(x1s+x2s, y1s+y2s)
plot!(x3s, y3s)

# %%
jacobian(xₛₜ_init2, rand(6))[1]

# %%
n = 12; m = 6;
N = 1000; δt = 0.001;

Q = Matrix(Diagonal([1.,1,1,0,0,0,0,1,1,1,0,0,0])) * 10.
R = Matrix(Diagonal([1.,1,1,1,1,1])) * 1.
Qf = Matrix(Diagonal([1.,1,1,0,0,0,0,1,1,1,0,0,0])) * 100.

ctrl = OSQPController(Q, R, Qf, δt, N, (N-1)*(n));

# %%
# xₛc_init = [earthRadius+2, 0, 0, 1., 0, 0, 0, 0, 28.4, 0, 0, 0, 0]
# xₛₜ_init = [0, earthRadius+1, 0, 1., 0, 0, 0, 28.4, 0, 0, 0, 0, 0]
xₛc_init = [earthRadius+2, 0, 0, 1., 0, 0, 0, 28.4, 0, 0, 0, 0, 0]
xₛₜ_init = [earthRadius+2, 0, 0, 1., 0, 0, 0, 28.4, 0, 0, 0, 0, 0]

simulate(ctrl, xₛc_init, xₛₜ_init; num_steps=10000);

# %%
include("dynamics.jl")
include("MPC.jl");

x1 = rand(13)
x1[4:7] = normalize(x1[4:7])
x2 = rand(13)
x2[4:7] = normalize(x2[4:7])

dx = state_error(x1, x2)

# %%
state_error_inv(x2, state_error(x1, x2)) ≈ x1
