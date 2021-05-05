# %%
using LinearAlgebra: normalize, norm, ×
using Rotations: lmult, hmat, RotMatrix, UnitQuaternion, RotationError, add_error, rotation_error, params
using ForwardDiff
using StaticArrays

mₜ = 419.709;
mₛ = 5.972e21;
G = 8.6498928e-19;
earthRadius = 6.37814;  # Megameters
n = sqrt(G * mₛ / (earthRadius^3))
Ω = [0; 0; 2*pi/1.5] #change 24 to solar time or whatever its called

# struct TargetSatellite
#     mₜ::Real
#     Jₜ::Matrix
#     n::Real
#     initialState::Vector
#     Ω::Vector
#     Xtraj::Vector
#     function TargetSatellite(; mₜ=419.709, Jₜ=[1  0  0; 0  1  0; 0  0  1],
#                              n=sqrt(G * mₛ / (earthRadius^3)),
#                              initialState=[0,0,0, 1,0,0,0, 0,0,0, 0,0,2*π*n],
#                              Ω=initialState[11:13])
#         Xtraj = []
#         return new(mₜ, Jₜ, n, initialState, Ω, Xtraj)
#     end
# end

# function target_quaternion(sat::TargetSatellite, simStep::Int64,
#                            mpcStep::Int64, δt::Real)::Vector
#     temp = RotXYZ(UnitQuaternion(sat[4:7]))
#     @assert temp.theta1 ≈ 0 && temp.theta2 ≈ 0
#
#     newZang = temp.theta3 + Ω[3] * δt * (simStep + mpcStep - 1)
#
#     quat = params(UnitQuaternion(RotXYZ(x=0, y=0, z=newZang)))
#
#     return quat
# end

function dynamics_CW(x::Vector, u::Vector)
    p_tc = x[1:3]
    q_sc = normalize(x[4:7])
    v_tc = x[8:10]
    ω_sc = x[11:13]

    q_st = x[14:16]  # Use Euler angles (X, Y, Z) for TRN orientation param
    @assert q_st[1] ≈ 0 && q_st[2] ≈ 0  # Only rotates about Z

    𝑓_c = [u[1], u[2], u[3]]
    𝜏_c = [u[4], u[5], u[6]]

    R_tc = RotMatrix(RotXYZ(q_st...))' * RotMatrix(UnitQuaternion(q_sc))

    ṗ_tc = v_tc
    v̇_tc = ([3*(n^2)*p_tc[1] + 2*n*v_tc[2]; -2*n*v_tc[1]; -(n^2)*p_tc[3]] +
            Rtc * 𝑓_c)

    ω̇_sc = J_c \ (𝜏_c - ω_sc × (J_c * ω_sc))
    q̇_sc = 0.5 * lmult(q_sc) * hmat() * ω_sc

    q̇_st = zeros(3)
    q̇_st[3] = Ω[3]

    return [ṗ_tc; q̇_sc; v̇_tc; ω̇_sc; q̇_st]
end

function jacobian(x::Vector, u::Vector, sat::TargetSatellite,
                  simStep::Int64, mpcStep::Int64, δt::Real)
    A = ForwardDiff.jacobian(x_temp->dynamics(x_temp, u, sat, simStep, mpcStep, δt), x)
    B = ForwardDiff.jacobian(u_temp->dynamics(x, u_temp, sat, simStep, mpcStep, δt), u)
    return (A, B)
end

function discreteDynamics(x::Vector, u::Vector, sat::TargetSatellite,
                          simStep::Int64, mpcStep::Int64, δt::Real)
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
