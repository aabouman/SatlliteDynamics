# %%
import Pkg
Pkg.add("Plots")
# %%
include("MPC.jl");

# %%
n = 18; m = 6;
N = 100; δt = 1/3600 #0.001;

Q  = Matrix(Diagonal([1.,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0])) * 1.
R  = Matrix(Diagonal([1.,1,1,1,1,1])) * .1
Qf = Matrix(Diagonal([1.,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0])) * 10.

ctrl = OSQPController(Q, R, Qf, δt, N, (N-1)*(n));

# %%
include("MPC.jl");

x_init = [-.1, 0, 0, 1., 0, 0, 0, 10., 0., 0, 0, 0, 0,
          0, 0, 0, 0, 0, 2*π/2]
x_hist, u_hist, x_ref_hist = simulate(ctrl, x_init; num_steps=3600, verbose=true)


# %%

μ = sqrt(G * mₛ / (earthRadius^3))
(μ^2)*3
# %%
using Plots
plot([x_hist[i][1] for i in 1:length(x_hist)])
# %%
# plot([x_ref_hist[i][1] for i in 1:length(x_ref_hist)])

# %%
plot!([x_hist[i][2] for i in 1:length(x_hist)])

# %%
plot([x_hist[i][1] for i in 1:length(x_hist)],
     [x_hist[i][2] for i in 1:length(x_hist)])

# %%
plot!([x_hist[i][9] for i in 1:length(x_hist)])


# %%
plot!([x_hist[i][9] for i in 1:length(x_hist)])

# %%
plot([u_hist[i][1] for i in 1:length(u_hist)])
plot!([u_hist[i][2] for i in 1:length(u_hist)])
# plot!([u_hist[i][3] for i in 1:length(u_hist)])

# %%
plot([u_hist[i][4] for i in 1:length(u_hist)])
plot!([u_hist[i][5] for i in 1:length(u_hist)])
plot!([u_hist[i][6] for i in 1:length(u_hist)])


# %% checking rotation is indentity
RotMatrix(UnitQuaternion([1.0, 0., 0., 0.]))
# %%
x_hist[end]

# %%
temp = [zeros(6) for i in 1:10000]
rollout(xₛₜ_init, temp, δt)[end]

# %%
x_hist[end] - rollout(xₛₜ_init, temp, δt)[end]

# %%
[x_hist[] for i in 1:length(x_hist)]

# %%
q = normalize(rand(4))

# %%
attitude_jacobian(q)

# %%
Rotations.lmult(UnitQuaternion(q)) * hmat()

# %%
Rotations.∇differential(UnitQuaternion(q))

# %%
ForwardDiff.jacobian(x->(1/sqrt(1+norm(x)^2) * vcat([1], x)), [1,2,3])

# %%
ForwardDiff.jacobian(x->(x[2:4] ./ x[1]), q)

# %%
Rotations.∇differential(UnitQuaternion(q))

# %%
ForwardDiff.jacobian(q->params(RodriguesParam(UnitQuaternion(q))), q)

# %%
ForwardDiff.jacobian(q->params(MRP(UnitQuaternion(q))), q)

# %%
Rotations.jacobian(MRP, UnitQuaternion(q))
