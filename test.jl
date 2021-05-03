include("dynamics.jl")
include("MPC.jl")

# %%
n = 13; m = 6;
N = 100

Q = Matrix(Diagonal([1.,1,1,0,0,0,0,1,1,1,0,0,0])) * 10
R = zeros(m,m)
Qf = Matrix(Diagonal([1.,1,1,0,0,0,0,1,1,1,0,0,0])) * 100

ctrl = OSQPController(Q, R, Qf, 0.001, N, (N-1)*(n+1));

# %%
xₛc_init = [earthRadius+2, 0, 0, 1., 0, 0, 0, 0, 28.4, 0, 0, 0, 0]
xₛₜ_init = [0, earthRadius+1, 0, 1., 0, 0, 0, 28.4, 0, 0, 0, 0, 0]

simulate(ctrl, xₛc_init, xₛₜ_init; num_steps=1000, δt=0.001);
