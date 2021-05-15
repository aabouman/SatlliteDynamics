using Plots
include("MPC.jl");

# %%
n = 12; m = 6;
N = 2000; δt = 0.01;

x_init1 = [6.371, 0, 0, 1., 0, 0, 0, 0, 15.44, 0, 0, 0, μ,
           1, 0, 0, 1., 0, 0, 0, 0, .1, 0, 0, 0, 0]
roll1 = rollout(x_init1, [zeros(6) for _ in 1:N], δt);

x1s = [roll1[i][1] for i in 1:length(roll1)]
y1s = [roll1[i][2] for i in 1:length(roll1)]
q1s = [roll1[i][4] for i in 1:length(roll1)]
q2s = [roll1[i][5] for i in 1:length(roll1)]
q3s = [roll1[i][6] for i in 1:length(roll1)]
q4s = [roll1[i][7] for i in 1:length(roll1)]

x2s = [roll1[i][14] for i in 1:length(roll1)]
y2s = [roll1[i][15] for i in 1:length(roll1)]

x_init2 = [7.371, 0, 0, 1., 0, 0, 0, 0, 15.54, 0, 0, 0, 0,
           0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0]
roll2 = rollout(x_init2, [zeros(6) for _ in 1:N], δt);

x3s = [roll2[i][1] for i in 1:length(roll1)];
y3s = [roll2[i][2] for i in 1:length(roll1)];

# %% plotting positions
plot(x1s,y1s)
plot!(x1s+x2s, y1s+y2s)
plot!(x3s, y3s)
plot(x2s, y2s)

# plotting Quaternions
plot(q1s)
plot!(q2s)
plot!(q3s)
plot!(q4s)

# %%
using LinearAlgebra: Diagonal

n = 13; m = 6;
N = 50; δt = 0.01;
num_steps = 450
Np = (N-1)*(n-1+m)
Nd = (N-1)*(n-1)

# Q = Matrix(Diagonal([1.,1,1, 2,2,2,2, 1,1,1, 1,1,1,
#                      1.,1,1, 2,2,2,2, 1,1,1, 1,1,1])) * 1.
# R = Matrix(Diagonal([1.,1,1,1,1,1])) * .1
# Qf = Matrix(Diagonal([1.,1,1, 2,2,2,2, 1,1,1, 1,1,1,
#                       1.,1,1, 2,2,2,2, 1,1,1, 1,1,1])) * 10.

Q = Matrix(Diagonal([1.,1,1, .2,.2,.2,.2, 1,1,1, 1,1,1])) * 1.
R = Matrix(Diagonal([1.,1,1,1,1,1])) * .01
Qf = Matrix(Diagonal([1.,1,1, .2,.2,.2,.2, 1,1,1, 1,1,1])) * 10.


ctrl = OSQPController(Q, R, Qf, δt, N, Np, Nd);


# %%
using LinearAlgebra: det
mₜ = 4.709;             # Mass of satellite Megagrams
mₛ = 5.97e21;           # Mass of earth Megagrams
G = 8.65e-19;           # Gravitational constant Megameters^3/Hour^2 Megagrams
earthRadius = 6.371;    # Radius of earth Megameters
orbitRadius = earthRadius + 1.0
μ = sqrt(G * mₛ / ((orbitRadius)^3))    # radians / hour

include("MPC.jl");

x_init1 = [6.371, 0, 0, 1., 0, 0, 0, 0, 15.44, 0, 0, 0, μ,
           1,     0, 0, 1., 0, 0, 0, 0,   .1,  0, 0, 0, 0]

x_hist, u_hist, cost_hist = simulate(ctrl, x_init1; num_steps=num_steps, verbose=false)

# %%
using Plots
plot([1:length(cost_hist);], cost_hist, title="Cost",
     xlabel="Simulation Step", legend=false, size=(500, 400)) #yaxis=:log
# savefig("graphics/rotation_cost.png")

# %% Positon error plots
plot([x_hist[i][14] for i = 1:length(x_hist)], title="relative position of chaser wrt target",
     xlabel="Simulation Step", legend=true, size=(500, 400) , label = "chaser-x")
plot!([x_hist[i][15] for i = 1:length(x_hist)],
      xlabel="Simulation Step", legend=true, size=(500, 400) , label = "chaser-y")
plot!([x_hist[i][16] for i = 1:length(x_hist)],
   xlabel="Simulation Step", legend=true, size=(500, 400) , label = "chaser-z")

# %% Velocity error Plots
plot([x_hist[i][21] for i = 1:length(x_hist)], title="relative velocity of chaser wrt target",
     xlabel="Simulation Step", legend=true, size=(500, 400) , label = "chaser-x")
plot!([x_hist[i][22] for i = 1:length(x_hist)],
      xlabel="Simulation Step", legend=true, size=(500, 400) , label = "chaser-y")
plot!([x_hist[i][23] for i = 1:length(x_hist)],
   xlabel="Simulation Step", legend=true, size=(500, 400) , label = "chaser-z")


# %%
plot([x_hist[i][17] for i in 1:length(x_hist)] , title= "MPC: Orientation", label="chaser_q1" ,line = (2, :dash, "blue"), legendfontsize=10)
plot!([x_hist[i][18] for i in 1:length(x_hist)] , line = (2, :dash, "red") , label="chaser_q2" , legend=:bottomleft)
plot!([x_hist[i][19] for i in 1:length(x_hist)] , line = (2, :dash, "green") , label="chaser_q3")
plot!([x_hist[i][20] for i in 1:length(x_hist)] , line = (2, :dash, "black") , label="chaser_q4")
plot!([x_hist[i][4] for i in 1:length(x_hist)] , line = (2, "blue") , label="target_q1")
plot!([x_hist[i][5] for i in 1:length(x_hist)], line = (2, "red") , label="target_q2")
plot!([x_hist[i][6] for i in 1:length(x_hist)] , line = (2, "green") , label="target_q3")
plot!([x_hist[i][7] for i in 1:length(x_hist)] , line = (2, "black") , label="target_q4" , xlabel="timestep" , ylabel="Quaternions")
# savefig("graphics/MPC_orientation.png")
# %% Angular velocity plot
plot([x_hist[i][13] for i = 1:length(x_hist)], title="relative velocity of chaser wrt target",
     xlabel="Simulation Step", size=(500, 400) , label = "target-ωz")
plot!([x_hist[i][26] for i = 1:length(x_hist)],
      xlabel="Simulation Step", size=(500, 400) , label = "chaser-ωz" )

# %%
u_hist
# %%
ctrl.Xref[end]
# %%
[discreteJacobian(x_init1, rand(6), δt)[1]

# %%
q1 = rand(UnitQuaternion)

p = MRP(q1) #use this as phi.x , phi.y, phi.z


q_tmp = UnitQuaternion(p)

FD_jac = ForwardDiff.jacobian(x -> (q = UnitQuaternion(MRP(x[1],x[2],x[3]));
                                        SVector(q.w, q.x, q.y, q.z)),
                                        SVector(p.x, p.y, p.z))
FD_jac_new = normalize(FD_jac)


R_jac = Rotations.jacobian(UnitQuaternion, p)

p_ = RotXYZ(q1)
FD_jac = ForwardDiff.jacobian(x -> (q = UnitQuaternion(RotXYZ(x[1],x[2],x[3]));
                                        SVector(q.w, q.x, q.y, q.z)),
                                        SVector(p_))

_p = rotation_axis(q1)
FD_jac = ForwardDiff.jacobian(x -> (UnitQuaternion(rotation_axis(x)); SVector(q.w, q.x, q.y, q.z)),
                                        SVector(_p))

# %% checking for CayleyMap

q = rand(UnitQuaternion)
q_param = params(q)
ϕ = q_param[2:4]/q_param[1]

f(ϕ) = [1; ϕ]/(1 + norm(ϕ)^2)^(0.5)

fd_jac_2 = ForwardDiff.jacobian(x->f(x), ϕ)

fd_jac_2_norm = normalize(fd_jac_2)
# %% compare with G(q)
jac_tmp = lmult(q)*hmat() #∇differential(UnitQuaternion(q)) ##same!

# %%
u = zeros(6)
x = rand(26)
x[17:20] .= params(q1)
δt = 0.01
A_tmp = discreteJacobian(x, u, δt)[1]

# %% qaut jac is 17:20, 17:20
q_jac = A_tmp[17:20, 17:20]

# %%
q_jac*lmult(q1)*hmat()
