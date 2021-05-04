include("dynamics.jl")
include("MPC.jl");

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
