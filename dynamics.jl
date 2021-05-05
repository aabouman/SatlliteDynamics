# %%
using LinearAlgebra: normalize, norm, ×, I
using Rotations: lmult, hmat, RotMatrix, UnitQuaternion, RotationError
using Rotations: CayleyMap, add_error, rotation_error, params, RotXYZ
using ForwardDiff
using StaticArrays

J_c = I(3)
mₜ = 419.709;
mₛ = 5.972e21;
G = 8.6498928e-19;
earthRadius = 6.37814;  # Megameters
μ = sqrt(G * mₛ / (earthRadius^3))


function dynamics(x::Vector, u::Vector)::SVector{19}
    xStatic = SVector{length(x)}(x)
    uStatic = SVector{length(u)}(u)
    return dynamics(xStatic, uStatic)
end


function dynamics(x::SVector{19}, u::SVector{6})::SVector{19}
    p_tc = SVector{3}(x[1:3])
    q_sc = normalize(SVector{4}(x[4:7]))
    v_tc = SVector{3}(x[8:10])
    ω_sc = SVector{3}(x[11:13])

    q_st = SVector{3}(x[14:16])  # Use Euler angles (X, Y, Z) for TRN orientation param
    @assert q_st[1] ≈ 0 && q_st[2] ≈ 0  # Only rotates about Z
    ω_st = SVector{3}(x[17:19])
    @assert q_st[1] ≈ 0 && q_st[2] ≈ 0  # Only rotates about Z

    𝑓_c = SVector{3}(u[1:3])
    𝜏_c = SVector{3}(u[4:6])

    R_tc = RotMatrix(RotXYZ(q_st...))' * RotMatrix(UnitQuaternion(q_sc))

    # Chaser wrt Target
    ṗ_tc = v_tc
    v̇_tc = ([3*(μ^2)*p_tc[1] + 2*μ*v_tc[2]; -2*μ*v_tc[1]; -(μ^2)*p_tc[3]] +
            R_tc * 𝑓_c)
    # Chaser wrt Inertial
    ω̇_sc = J_c \ (𝜏_c - ω_sc × (J_c * ω_sc))
    q̇_sc = 0.5 * lmult(q_sc) * hmat() * ω_sc
    # Target wrt Inertial
    ω̇_st = SVector{3}(zeros(3))    # Constant velocity
    q̇_st = ω_st

    return [ṗ_tc; q̇_sc; v̇_tc; ω̇_sc; q̇_st; ω̇_st]
end

function jacobian(x::Vector, u::Vector)
    A = ForwardDiff.jacobian(x_temp->dynamics(x_temp, u), x)
    B = ForwardDiff.jacobian(u_temp->dynamics(x, u_temp), u)
    return (A, B)
end

function discreteDynamics(x::Vector, u::Vector, δt::Real)::SVector{19}
    xnew = SVector{length(x)}(x)
    unew = SVector{length(u)}(u)

    k1 = dynamics(xnew, unew)
    k2 = dynamics(xnew + 0.5 * δt * k1, unew)
    k3 = dynamics(xnew + 0.5 * δt * k2, unew)
    k4 = dynamics(xnew + δt * k3, unew)
    xnext = xnew + (δt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return xnext
end

function rollout(x0::Vector, Utraj::Vector, δt::Real)
    N = length(Utraj)+1
    Xtraj = [zeros(length(x0)) for _ in 1:N]
    Xtraj[1] = x0

    for i in 1:N-1
        Xtraj[i+1] = discreteDynamics(Xtraj[i], Utraj[i], δt)
    end

    return Xtraj
end

function state_error(x, xref)
    ip, iq, iv, iw, iq2, iw2 = 1:3, 4:7, 8:10, 11:13, 14:16, 17:19
    q = x[iq]; qref = xref[iq]
    qe = Vector(rotation_error(UnitQuaternion(q), UnitQuaternion(qref),
                               CayleyMap()))
    dx = [x[ip]-xref[ip]; qe; x[iv]-xref[iv]; x[iw]-xref[iw];
          x[iq2]-xref[iq2]; x[iw2]-xref[iw2]]
    return dx
end

function state_error_inv(xref, dx)
    ip, iq, iv, iw, iq2, iw2 = 1:3, 4:7, 8:10, 11:13, 14:16, 17:19

    p_new = xref[ip] + dx[1:3]      # Position of chaser wrt target
    q_new = add_error(UnitQuaternion(xref[iq]),
                      RotationError(SVector{3}(dx[4:6]), CayleyMap()))
    q_new = params(q_new)           # orientation of chaser wrt inertial
    v_new = xref[iv] + dx[7:9]      # Velocity of chaser wrt target
    w_new = xref[iw] + dx[10:12]    # Angular velocity of chaser wrt inertial
    q2_new = xref[iq2] + dx[13:15]  # Euler angles of target wrt inertial
    w2_new = xref[iw2] + dx[16:18]  # Angular velocity of target wrt inertial

    return vcat(p_new, q_new, v_new, w_new, q2_new, w2_new)
end

function attitude_jacobian(q)
    q̂ = [ 0    -q[4]  q[3];
          q[4]  0    -q[2];
         -q[3]  q[2]  0]
    return SMatrix{4,3}(vcat(-q[2:end]', q[1] * I(3) + q̂))
end


function state_error_jacobian(x)
    ip, iq, iv, iw, iq2, iw2 = 1:3, 4:7, 8:10, 11:13, 14:16, 17:19
    q = x[iq]
    M = blockdiag(sparse(I(3)),
                  sparse(attitude_jacobian(q)),
                  sparse(I(12)))
    return Matrix(M)
end
