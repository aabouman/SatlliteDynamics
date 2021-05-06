# %%
using LinearAlgebra: normalize, norm, ×, I
using Rotations: lmult, hmat, RotMatrix, UnitQuaternion, RotationError
using Rotations: CayleyMap, add_error, rotation_error, params, RotXYZ, ∇differential
using ForwardDiff
using StaticArrays

num_states = 6
num_inputs = 3


function dynamics(x::Vector, u::Vector)::SVector{num_states}
    xStatic = SVector{length(x)}(x)
    uStatic = SVector{length(u)}(u)
    return dynamics(xStatic, uStatic)
end


function dynamics(x::SVector{num_states}, u::SVector{num_inputs})::SVector{num_states}
    p_tc = SVector{3}(x[1:3])
    v_tc = SVector{3}(x[4:6])

    𝑓_c = SVector{3}(u[1:3])

    # Chaser wrt Target
    ṗ_tc = v_tc
    v̇_tc = ([3*(μ^2)*p_tc[1] + 2*μ*v_tc[2]; -2*μ*v_tc[1]; -(μ^2)*p_tc[3]] +
            𝑓_c) # R_tc * 𝑓_c

    return [ṗ_tc; v̇_tc]
end


function jacobian(x::Vector, u::Vector)
    A = ForwardDiff.jacobian(x_temp->discreteDynamics(x_temp, u, δt), x)
    B = ForwardDiff.jacobian(u_temp->discreteDynamics(x, u_temp, δt), u)
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
    ip, iv = 1:3, 4:6
    dx = [x[ip]-xref[ip]; x[iv]-xref[iv]]
    return dx
end


function state_error_inv(xref, dx)
    ip, iv = 1:3, 4:6

    p_new = xref[ip] + dx[1:3]      # Position of chaser wrt target
    v_new = xref[iv] + dx[4:6]      # Velocity of chaser wrt target

    return vcat(p_new, v_new)
end


function state_error_jacobian(x)
    ip, iv = 1:3, 4:6

    M = blockdiag(sparse(I(6)))
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
    p1, v1 = x_init[1:3], x_init[4:6]

    ps = range(p1, zeros(3), length=N)
    vs = range(v1, zeros(3), length=N)

    return [[ps[i]; vs[i]] for i in 1:N]
end
