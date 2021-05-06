# %%
using LinearAlgebra: normalize, norm, ×, I
using Rotations: RotMatrix, UnitQuaternion, RotXYZ, RotationError, params, lmult, hmat
using Rotations: CayleyMap, add_error, rotation_error,  kinematics, ∇differential
using ForwardDiff
using StaticArrays

J_c = I(3)
num_states = 13
num_inputs = 3

function dynamics(x::Vector, u::Vector)::SVector{num_states}
    xStatic = SVector{length(x)}(x)
    uStatic = SVector{length(u)}(u)
    return dynamics(xStatic, uStatic)
end


function dynamics(x::SVector{num_states}, u::SVector{num_inputs})::SVector{num_states}
    q_sc = normalize(SVector{4}(x[1:4]))
    ω_sc = SVector{3}(x[5:7])

    q_st = SVector{3}(x[8:10])  # Use Euler angles (X, Y, Z) for TRN orientation param
    @assert q_st[1] ≈ 0 && q_st[2] ≈ 0  # Only rotates about Z
    ω_st = SVector{3}(x[11:13])
    @assert ω_st[1] ≈ 0 && ω_st[2] ≈ 0  # Only rotates about Z

    𝜏_c = SVector{3}(u[1:3])

    R_tc = RotMatrix(RotXYZ(q_st...))' * RotMatrix(UnitQuaternion(q_sc))

    # Chaser wrt Inertial
    ω̇_sc = J_c \ (𝜏_c - ω_sc × (J_c * ω_sc))
    q̇_sc = 1/2 * lmult(q_sc) * hmat() * ω_sc
    # q̇_sc = kinematics(UnitQuaternion(q_sc), ω_sc)

    # Target wrt Inertial
    ω̇_st = SVector{3}(zeros(3))    # Constant velocity
    q̇_st = ω_st

    return [q̇_sc; ω̇_sc; q̇_st; ω̇_st]
end


function jacobian(x::Vector, u::Vector)
    A = ForwardDiff.jacobian(x_temp->dynamics(x_temp, u), x)
    B = ForwardDiff.jacobian(u_temp->dynamics(x, u_temp), u)
    return (A, B)
end


function discreteDynamics(x::Vector, u::Vector, δt::Real)::SVector{num_states}
    xnew = SVector{num_states}(x)
    unew = SVector{num_inputs}(u)

    k1 = dynamics(xnew, unew)
    k2 = dynamics(xnew + 0.5 * δt * k1, unew)
    k3 = dynamics(xnew + 0.5 * δt * k2, unew)
    k4 = dynamics(xnew + δt * k3, unew)
    xnext = xnew + (δt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return xnext
end


function discreteJacobian(x::Vector, u::Vector, δt::Real)
    A = ForwardDiff.jacobian(x_temp->discreteDynamics(x_temp, u, δt), x)
    B = ForwardDiff.jacobian(u_temp->discreteDynamics(x, u_temp, δt), u)
    return (A, B)
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
    iq, iw, iq2, iw2 = 1:4, 5:7, 8:10, 11:13
    q = x[iq]; qref = xref[iq]
    qe = Vector(rotation_error(UnitQuaternion(q), UnitQuaternion(qref),
                               CayleyMap()))
    dx = [qe; x[iw]-xref[iw]; x[iq2]-xref[iq2]; x[iw2]-xref[iw2]]
    return dx
end


function state_error_inv(xref, dx)
    iq, iw, iq2, iw2 = 1:4, 5:7, 8:10, 11:13

    q_new = params(add_error(UnitQuaternion(xref[iq]),
                             RotationError(SVector{3}(dx[1:3]), CayleyMap()))) # orientation of chaser wrt inertial
    w_new = xref[iw] + dx[4:6]    # Angular velocity of chaser wrt inertial
    q2_new = xref[iq2] + dx[7:9]  # Euler angles of target wrt inertial
    w2_new = xref[iw2] + dx[10:12]  # Angular velocity of target wrt inertial

    return vcat(q_new, w_new, q2_new, w2_new)
end


function state_error_jacobian(x)
    iq, iw, iq2, iw2 = 1:4, 5:7, 8:10, 11:13
    q = x[iq]
    M = blockdiag(sparse(∇differential(UnitQuaternion(q))),
                  sparse(I(9)))
    return Matrix(M)
end


function slerp(qa::UnitQuaternion, qb::UnitQuaternion, N::Int64)
    function slerpHelper(qa::UnitQuaternion{T}, qb::UnitQuaternion{T}, t::T) where {T}
        coshalftheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;

        if coshalftheta < 0
            qm = -qb
            coshalftheta = -coshalftheta
        else
            qm = qb
        end
        abs(coshalftheta) >= 1.0 && return params(qa)

        halftheta    = acos(coshalftheta)
        sinhalftheta = sqrt(one(T) - coshalftheta * coshalftheta)

        if abs(sinhalftheta) < 0.001
            return params(UnitQuaternion(T(0.5) * (qa.w + qb.w),
                          T(0.5) * (qa.x + qb.x),
                          T(0.5) * (qa.y + qb.y),
                          T(0.5) * (qa.z + qb.z)))
        end

        ratio_a = sin((one(T) - t) * halftheta) / sinhalftheta
        ratio_b = sin(t * halftheta) / sinhalftheta

        temp = params(UnitQuaternion(qa.w * ratio_a + qm.w * ratio_b,
                                     qa.x * ratio_a + qm.x * ratio_b,
                                     qa.y * ratio_a + qm.y * ratio_b,
                                     qa.z * ratio_a + qm.z * ratio_b))
        return temp
    end

    ts = range(0., 1., length=N)
    return [slerpHelper(qa, qb, t) for t in ts]
end


function stateInterpolate_CW(x_init::Vector, N::Int64, δt::Real)
    # initial
    q1, w1 = x_init[1:4], x_init[5:7]
    # final
    q2, w2 = x_init[8:10], x_init[11:13]
    q2_final = q2 + w2 * δt * (N-1)
    quat_final = UnitQuaternion(RotXYZ(q2_final...))

    # quaternion
    q2s = range(q2, q2_final, length=N)
    w2s = fill(w2, N)

    qs = [params(UnitQuaternion(RotXYZ(q2...))) for q2 in q2s]
    ws = fill(w2, N)

    return [[qs[i]; ws[i]; q2s[i]; w2s[i]] for i in 1:N]
end
