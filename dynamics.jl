# %%
using LinearAlgebra: normalize, norm, ×
using Rotations: lmult, vmat, hmat, RotMatrix, UnitQuaternion
using ForwardDiff
using StaticArrays

Jₜ = [1  0  0; 0  1  0; 0  0  1];
mₜ = 419.709;
mₛ = 5.972e21;
G = 8.6498928e-19;

function dynamics(x::Vector, u::Vector)
    xStatic = SVector{length(x)}(x)
    uStatic = SVector{length(u)}(u)
    dynamics(xStatic, uStatic)
end

function dynamics(x::SVector{13}, u::SVector{6})
    pₛₜ = @SVector [x[1], x[2], x[3]]
    qₛₜ = normalize(@SVector [x[4], x[5], x[6], x[7]])
    vₛₜ = @SVector [x[8], x[9], x[10]]
    ωₛₜ = @SVector [x[11], x[12], x[13]]

    𝑓ₜ = @SVector [u[1], u[2], u[3]]
    𝜏ₜ = @SVector [u[4], u[5], u[6]]

    ṗₛₜ = vₛₜ
    Rₛₜ = RotMatrix(UnitQuaternion(qₛₜ...))
    v̇ₛₜ = - inv(Rₛₜ) * (G * mₛ / norm(pₛₜ)^3) * pₛₜ + 𝑓ₜ / mₜ  - ωₛₜ × vₛₜ

    ω̇ₛₜ = Jₜ \ (𝜏ₜ - ωₛₜ × (Jₜ * ωₛₜ))
    q̇ₛₜ = 0.5 * lmult(qₛₜ) * hmat() * ωₛₜ
    return [ṗₛₜ; q̇ₛₜ; v̇ₛₜ; ω̇ₛₜ]
end

function jacobian(x::Vector, u::Vector)
    A = ForwardDiff.jacobian(x_temp->dynamics(x_temp, u), x)
    B = ForwardDiff.jacobian(u_temp->dynamics(x, u_temp), u)

    return (A, B)
end

function rk4(x, u, h)

    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5*h*k1,u)
    k3 = dynamics(x + 0.5*h*k2,u)
    k4 = dynamics(x + h*k3,u)
    xnext = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return xnext
end

function simulate()
    return
end;

# %%
δt = 0.001; iters = 10000
x = zeros(13); u = zeros(6) #rand(6);
x[1:3] = [6.3710, 0, 0]
x[4:7] = [1., 0, 0, 0]
x[8:10] = [0, 28.4, 0]

x_hist = zeros(iters, length(x))

for i in 1:iters
    x_hist[i,:] .= x

    ẋ = dynamics(x, u)
    x += ẋ * δt

    pₛₜ, qₛₜ, vₛₜ, ωₛₜ = x[1:3], x[4:7], x[8:10], x[11:13]
    qₛₜ = normalize(qₛₜ);
end

pₛₜ_hist = x_hist[:,1:3];

# %%
using Plots

plot(pₛₜ_hist[:,1], pₛₜ_hist[:,2])
