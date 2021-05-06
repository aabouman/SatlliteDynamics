# %%
using LinearAlgebra: normalize, norm, ×, I
using Rotations: RotMatrix, UnitQuaternion, RotXYZ, RotationError, params
using Rotations: CayleyMap, add_error, rotation_error, kinematics, ∇differential
using ForwardDiff
using StaticArrays

J_c = I(3)
num_states = 14
num_inputs = 3

function dynamics(x::Vector, u::Vector)::SVector{num_states}
    xStatic = SVector{length(x)}(x)
    uStatic = SVector{length(u)}(u)
    return dynamics(xStatic, uStatic)
end


function dynamics(x::SVector{num_states}, u::SVector{num_inputs})::SVector{num_states}
    q_sc = normalize(SVector{4}(x[1:4]))
    ω_sc = SVector{3}(x[5:7])

    q_st = normalize(SVector{4}(x[8:11]))
    ω_st = SVector{3}(x[12:14])
    @assert ω_st[1] ≈ 0 && ω_st[2] ≈ 0  # Only rotates about Z

    𝜏_c = SVector{3}(u[1:3])

    R_tc = RotMatrix(UnitQuaternion(q_st))' * RotMatrix(UnitQuaternion(q_sc))

    # Chaser wrt Inertial
    ω̇_sc = J_c \ (𝜏_c - ω_sc × (J_c * ω_sc))
    q̇_sc = kinematics(UnitQuaternion(q_sc), ω_sc)

    # Target wrt Inertial
    ω̇_st = SVector{3}(zeros(3))    # Constant velocity
    q̇_st = kinematics(UnitQuaternion(q_st), ω_st)

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
    iq, iw, iq2, iw2 = 1:4, 5:7, 8:11, 12:14
    q = x[iq]; qref = xref[iq]
    q2 = x[iq2]; q2ref = xref[iq2]
    qe = Vector(rotation_error(UnitQuaternion(q), UnitQuaternion(qref),
                               CayleyMap()))
    q2e = Vector(rotation_error(UnitQuaternion(q2), UnitQuaternion(q2ref),
                                CayleyMap()))
    dx = [qe; x[iw]-xref[iw]; q2e; x[iw2]-xref[iw2]]
    return dx
end


function state_error_inv(xref, dx)
    iq, iw, iq2, iw2 = 1:4, 5:7, 8:11, 12:14

    dq = dx[1:3]; qref = xref[iq]
    dq2 = dx[7:9]; q2ref = xref[iq2]

    # orientation of chaser wrt inertial
    q_new = params(add_error(UnitQuaternion(qref),
                             RotationError(SVector{3}(dq), CayleyMap())))

    # orientation of chaser wrt inertial
    q2_new = params(add_error(UnitQuaternion(q2ref),
                              RotationError(SVector{3}(dq2), CayleyMap())))

    w_new = xref[iw] + dx[4:6]    # Angular velocity of chaser wrt inertial
    w2_new = xref[iw2] + dx[10:12]  # Angular velocity of target wrt inertial

    return vcat(q_new, w_new, q2_new, w2_new)
end


function state_error_jacobian(x)
    iq, iw, iq2, iw2 = 1:4, 5:7, 8:11, 12:14
    q = x[iq]
    M = blockdiag(sparse(∇differential(UnitQuaternion(q))),
                  sparse(I(3)),
                  sparse(∇differential(UnitQuaternion(q))),
                  sparse(I(3)))
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
    q2, w2 = x_init[8:11], x_init[12:14]

    Us = [zeros(num_inputs) for _ in 1:N]
    roll = rollout(x_init, Us, δt)
    q2s = [roll[i][8:11] for i in 1:N]
    w2s = [roll[i][12:14] for i in 1:N]

    qs = deepcopy(q2s)
    ws = deepcopy(w2s)

    return [[qs[i]; ws[i]; q2s[i]; w2s[i]] for i in 1:N]
end
