using LinearAlgebra: normalize, norm, ×, I, inv
using Rotations: RotMatrix, UnitQuaternion, RotXYZ, RotationError, params, hmat, rmult, lmult
using Rotations: CayleyMap, add_error, rotation_error, kinematics, ∇differential, skew
using ForwardDiff
using StaticArrays

Jₜ = [1  0  0;
     0  1  0;
     0  0  1];
J꜀ = [1  0  0;
      0  1  0;
      0  0  1];
mₜ = 419.709;
m꜀ = 419.709;
mₛ = 5.972e21;
G = 8.6498928e-19;

num_states = 12
num_inputs = 3


function dynamics(x::Vector, u::Vector)
    xStatic = SVector{length(x)}(x)
    uStatic = SVector{length(u)}(u)
    dynamics(xStatic, uStatic)
end

function dynamics(x::SVector{num_states}, u::SVector{num_inputs})
    # Extract target state
    p_st_s = @SVector [x[1], x[2], x[3]]
    # q_st_s = normalize(@SVector [x[4], x[5], x[6], x[7]])
    v_st_s = @SVector [x[4], x[5], x[6]]
    # ω_t_t = @SVector [x[11], x[12], x[13]]
    # Extract chaser state
    p_sc_s = @SVector [x[7], x[8], x[9]]
    # q_sc_s = normalize(@SVector [x[17], x[18], x[19], x[20]])
    v_sc_s = @SVector [x[10], x[11], x[12]]
    # ω_c_c = @SVector [x[24], x[25], x[26]]
    # Extract input
    𝑓_c = @SVector [u[1], u[2], u[3]]
    # Building helpful rot matricies
    # Rst = RotMatrix(UnitQuaternion(q_st_s))
    # Rsc = RotMatrix(UnitQuaternion(q_sc_s))

# =========================================================================== #
#                           Target Dyanmics
# =========================================================================== #
    # Target Translational Dynamics written in spatial frame
    p_st_s_dot = v_st_s
    v_st_s_dot = -(G * mₛ)/norm(p_st_s)^3 * p_st_s
    # Target Rotational Dynamics written in target frame
    # q_st_s_dot = kinematics(UnitQuaternion(q_st_s), ω_t_t)  # Quaternion kinematics
    # ω_t_t_dot = Jₜ \ (-ω_t_t × (Jₜ * ω_t_t))            # Body velocity dynamics
# =========================================================================== #
#                           Chaser Dyanmics
# =========================================================================== #
    # Target Translational Dynamics written in spatial frame
    p_sc_s_dot = v_sc_s
    v_sc_s_dot = -(G * mₛ)/norm(p_sc_s)^3 * p_sc_s  +  𝑓_c/m꜀
    # Target Rotational Dynamics written in target frame
    # q_sc_s_dot = kinematics(UnitQuaternion(q_sc_s), ω_c_c)  # Quaternion kinematics
    # ω_c_c_dot = Jₜ \ (𝜏_c - ω_c_c × (Jₜ * ω_c_c))            # Body velocity dynamics

    return [p_st_s_dot; v_st_s_dot;
            p_sc_s_dot; v_sc_s_dot]
end


function jacobian(x::Vector, u::Vector)
    A = ForwardDiff.jacobian(x_temp->dynamics(x_temp, u), x)
    B = ForwardDiff.jacobian(u_temp->dynamics(x, u_temp), u)

    return (A, B)
end


function discreteDynamics(x::SVector{num_states}, u::SVector{num_inputs}, δt::Real)
    k1 = dynamics(x, u)
    k2 = dynamics(x .+ 0.5 * δt * k1, u)
    k3 = dynamics(x .+ 0.5 * δt * k2, u)
    k4 = dynamics(x .+ δt * k3, u)
    xnext = x + (δt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return xnext
end

function discreteJacobian(x::Vector, u::Vector, δt::Real)
    x = SVector{length(x)}(x)
    u = SVector{length(u)}(u)

    A = ForwardDiff.jacobian(x_temp->discreteDynamics(x_temp, u, δt), x)
    B = ForwardDiff.jacobian(u_temp->discreteDynamics(x, u_temp, δt), u)
    return (A, B)
end

function systemEnergy(x::Vector)
    pₛₜˢ = x[1:3]
    # qₛₜ = x[4:7]
    vₛₜˢ = x[4:6]
    # ωₛₜᵗ = x[11:13]

    Rₛₜ = RotMatrix(UnitQuaternion(qₛₜ))
    ωₛₜˢ = Rₛₜ * ωₛₜᵗ

    NRG_target = (mₜ/2 * vₛₜˢ' * vₛₜˢ) + (1/2 * ωₛₜˢ' * Jₜ * ωₛₜˢ) - (G*mₛ*mₜ / norm(pₛₜˢ))

    return NRG_target
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


function state_error(x::Vector, xref::Vector)
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26

    q1 = x[iq1]; q1ref = xref[iq1]
    q1e = Vector(rotation_error(UnitQuaternion(q1),
                                UnitQuaternion(q1ref),
                                CayleyMap()))
    q2 = x[iq2]; q2ref = xref[iq2]
    q2e = Vector(rotation_error(UnitQuaternion(q2),
                                UnitQuaternion(q2ref),
                                CayleyMap()))

    dx = [x[ip1] - xref[ip1]; q1e; x[iv1] - xref[iv1]; x[iw1] - xref[iw1];
          x[ip2] - xref[ip2]; q2e; x[iv2] - xref[iv2]; x[iw2] - xref[iw2]]
    return dx
end


function state_error_half(x::Vector, xref::Vector)
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    #ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26

    q1 = x[iq1]; q1ref = xref[iq1]
    q1e = Vector(rotation_error(UnitQuaternion(q1),
                                UnitQuaternion(q1ref),
                                CayleyMap()))

    dx = [x[ip1] - xref[ip1]; q1e; x[iv1] - xref[iv1]; x[iw1] - xref[iw1]]
    return dx
end


function state_error_inv(xref::Vector, dx::Vector)
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26

    idp1, idq1, idv1, idw1 =  1:3,   4:6,   7:9,  10:12
    idp2, idq2, idv2, idw2 = 13:15, 16:18, 19:21, 22:24

    dq1 = dx[idq1]; q1ref = xref[iq1]
    dq2 = dx[idq2]; q2ref = xref[iq2]

    q1 = params(add_error(UnitQuaternion(q1ref),
                          RotationError(SVector{3}(dq1), CayleyMap())))
    q2 = params(add_error(UnitQuaternion(q2ref),
                          RotationError(SVector{3}(dq2), CayleyMap())))

    return [xref[ip1] + dx[idp1]; q1; xref[iv1] + dx[idv1]; xref[iw1] + dx[idw1];
            xref[ip2] + dx[idp2]; q2; xref[iv2] + dx[idv2]; xref[iw2] + dx[idw2]]
end


function state_error_jacobian(x::Vector)
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26

    q1 = x[iq1]; q2 = x[iq2]
    M = blockdiag(sparse(I(3)),
                  sparse(∇differential(UnitQuaternion(q1))),
                  sparse(I(9)),
                  sparse(∇differential(UnitQuaternion(q2))),
                  sparse(I(6)))
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
