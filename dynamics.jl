# %%
using LinearAlgebra: normalize, norm, ×
using Rotations: lmult, hmat, RotMatrix, UnitQuaternion, RotationError, add_error, rotation_error, params
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
    v̇ₛₜ = 𝑓ₜ / mₜ  - ωₛₜ × vₛₜ

    ω̇ₛₜ = Jₜ \ (𝜏ₜ - ωₛₜ × (Jₜ * ωₛₜ))
    q̇ₛₜ = 0.5 * lmult(qₛₜ) * hmat() * ωₛₜ
    return [ṗₛₜ; q̇ₛₜ; v̇ₛₜ; ω̇ₛₜ]
end

function jacobian(x::Vector, u::Vector)
    A = ForwardDiff.jacobian(x_temp->dynamics(x_temp, u), x)
    B = ForwardDiff.jacobian(u_temp->dynamics(x, u_temp), u)

    return (A, B)
end

function discreteDynamics(x::Vector, u::Vector, δt::Real)
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * δt * k1, u)
    k3 = dynamics(x + 0.5 * δt * k2, u)
    k4 = dynamics(x + δt * k3, u)
    xnext = x + (δt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return xnext
end

function rollout(x0::Vector, Utraj::Vector, δt::Real)
    N = length(Utraj)
    Xtraj = [zeros(length(x0)) for _ in 1:N+1]
    Xtraj[1] = x0

    for i in 1:N
        Xtraj[i+1] = discreteDynamics(Xtraj[i], Utraj[i], δt)
    end

    return Xtraj
end

function state_error(x, xref)
    ip, iq, iv, iw = 1:3, 4:7, 8:10, 11:13
    q = x[iq]; qref = xref[iq]
    qe = Vector(rotation_error(UnitQuaternion(q), UnitQuaternion(qref),
                               CayleyMap()))
    dx = [x[ip] - xref[ip]; qe; x[iv] - xref[iv]; x[iw] - xref[iw]]
    return dx
end

function state_error_inv(xref, dx)
    ip, iq, iv, iw = 1:3, 4:7, 8:10, 11:13

    p_new = xref[ip] + dx[1:3]
    q_new = add_error(UnitQuaternion(xref[iq]),
                      RotationError(SVector{3}(dx[4:6]), CayleyMap()))
    q_new = params(q_new)
    v_new = xref[iv] + dx[7:9]
    w_new = xref[iw] + dx[10:12]

    return vcat(p_new, q_new, v_new, w_new)
end

function attitude_jacobian(q)
    q̂ = [ 0    -q[4]  q[3];
          q[4]  0    -q[2];
         -q[3]  q[2]  0]
    return SMatrix{4,3}(vcat(-q[2:end]', q[1] * I(3) + q̂))
end


function state_error_jacobian(x)
    ip, iq, iv, iw = 1:3, 4:7, 8:10, 11:13
    q = x[iq]
    M = blockdiag(sparse(I(3)),
                  sparse(attitude_jacobian(q)),
                  sparse(I(6)))

    return Matrix(M)
end
