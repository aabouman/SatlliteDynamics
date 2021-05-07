using LinearAlgebra: normalize, norm, ×, I
using Rotations: RotMatrix, UnitQuaternion, RotXYZ, RotationError, params, hmat, rmult, lmult
using Rotations: CayleyMap, add_error, rotation_error, kinematics, ∇differential
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

num_states = 14
num_inputs = 3

function dynamics(x::Vector, u::Vector)
    xStatic = SVector{length(x)}(x)
    uStatic = SVector{length(u)}(u)
    dynamics(xStatic, uStatic)
end

function dynamics(x::SVector{26}, u::SVector{6})
    pₛₜˢ = @SVector [x[1], x[2], x[3]]
    qₛₜˢ = normalize(@SVector [x[4], x[5], x[6], x[7]])
    vₛₜᵗ = @SVector [x[8], x[9], x[10]]
    ωₛₜᵗ = @SVector [x[11], x[12], x[13]]

    pₜ꜀ᵗ = @SVector [x[14], x[15], x[16]]
    qₛ꜀ˢ = normalize(@SVector [x[17], x[18], x[19], x[20]])
    vₜ꜀ᵗ = @SVector [x[21], x[22], x[23]]
    ωₛ꜀ᶜ = @SVector [x[24], x[25], x[26]]

    𝑓꜀ = @SVector [u[1], u[2], u[3]]
    𝜏꜀ = @SVector [u[4], u[5], u[6]]

    # Building helpful rot matricies
    Rₛₜ = RotMatrix(UnitQuaternion(qₛₜˢ))
    Rₛ꜀ = RotMatrix(UnitQuaternion(qₛ꜀ˢ))
    Rₜₛ = (Rₛₜ)'
    R꜀ₛ = (Rₛ꜀)'
    Rₜ꜀ = Rₜₛ * Rₛ꜀
    R꜀ₜ = R꜀ₛ * Rₛₜ

# =========================================================================== #
#                           Target Dyanmics
# =========================================================================== #
    # Target Rotational Dynamics written in target frame
    ω̇ₛₜᵗ = Jₜ \ (-ωₛₜᵗ × (Jₜ * ωₛₜᵗ))            # Body velocity dynamics
    q̇ₛₜˢ = kinematics(UnitQuaternion(qₛₜˢ), ωₛₜᵗ)  # Quaternion kinematics
    # Target Translational Dynamics written in spatial frame
    ṗₛₜˢ = Rₛₜ * vₛₜᵗ
    v̇ₛₜᵗ = Rₜₛ * (-(G * mₛ)/norm(pₛₜˢ)^3 * pₛₜˢ) - ωₛₜᵗ × vₛₜᵗ

# =========================================================================== #
#                           Chaser Dyanmics
# =========================================================================== #
    # Chaser rotational dynamics written in chaser frame
    ω̇ₛ꜀ᶜ = J꜀ \ (𝜏꜀ - ωₛ꜀ᶜ × (J꜀ * ωₛ꜀ᶜ))            # Body velocity dynamics
    # Chaser rotational kinematics written in spatial frame
    q̇ₛ꜀ˢ = kinematics(UnitQuaternion(qₛ꜀ˢ), ωₛ꜀ᶜ)  # Quaternion kinematics

    # Useful definitions
    pₛₜᵗ = Rₜₛ * pₛₜˢ
    pₛ꜀ᵗ = pₛₜᵗ + pₜ꜀ᵗ
    vₛ꜀ᵗ = vₛₜᵗ + vₜ꜀ᵗ
    ṗₛₜᵗ = vₛₜᵗ
    ṗₛ꜀ᵗ = (-ωₛₜᵗ × pₛ꜀ᵗ) + vₛ꜀ᵗ
    # Chaser translational kinematics written in target frame
    ṗₜ꜀ᵗ = ṗₛ꜀ᵗ - ṗₛₜᵗ

    # Useful definitions
    pₛ꜀ˢ = Rₛₜ * pₛ꜀ᵗ
    vₛ꜀ˢ = Rₛₜ * vₛ꜀ᵗ
    vₛ꜀ᶜ = R꜀ₜ * vₛ꜀ᵗ
    v̇ₛ꜀ᶜ = R꜀ₛ * (-G * mₛ * pₛ꜀ˢ / norm(pₛ꜀ˢ)^3) + 𝑓꜀ / m꜀ - ωₛ꜀ᶜ × vₛ꜀ᶜ
    Ṙₛₜ = hmat()' * (lmult(SVector{4}(q̇ₛₜˢ)) * rmult(SVector{4}(qₛₜˢ))' +
                    lmult(SVector{4}(qₛₜˢ)) * rmult(SVector{4}(q̇ₛₜˢ))') * hmat()
    Ṙₛ꜀ = hmat()' * (lmult(SVector{4}(q̇ₛ꜀ˢ)) * rmult(SVector{4}(qₛ꜀ˢ))' +
                     lmult(SVector{4}(qₛ꜀ˢ)) * rmult(SVector{4}(q̇ₛ꜀ˢ))') * hmat()
    Ṙₜ꜀ = ((Ṙₛₜ)' * Rₛ꜀ + (Rₛₜ)' * Ṙₛ꜀)
    v̇ₛ꜀ᵗ = Ṙₜ꜀ * vₛ꜀ᶜ + Rₜ꜀ * v̇ₛ꜀ᶜ
    # Chaser translational dynamics written in target frame
    v̇ₜ꜀ᵗ = v̇ₛ꜀ᵗ - v̇ₛₜᵗ

    return [ṗₛₜˢ; q̇ₛₜˢ; v̇ₛₜᵗ; ω̇ₛₜᵗ;
            ṗₜ꜀ᵗ; q̇ₛ꜀ˢ; v̇ₜ꜀ᵗ; ω̇ₛ꜀ᶜ]
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
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26

    q1 = x[iq1]; q1ref = xref[iq1]
    q1e = Vector(rotation_error(UnitQuaternion(q1), UnitQuaternion(q1ref),
                                CayleyMap()))

    q2 = x[iq2]; q2ref = xref[iq2]
    q2e = Vector(rotation_error(UnitQuaternion(q2), UnitQuaternion(q2ref),
                                CayleyMap()))


    dx = [x[ip1] - xref[ip1]; q1e; x[iv1] - xref[iv1]; x[iw1] - xref[iw1];
          x[ip2] - xref[ip2]; q2e; x[iv2] - xref[iv2]; x[iw2] - xref[iw2]]
    return dx
end

function state_error_inv(xref, dx)
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26

    idp1, idq1, idv1, idw1 =  1:3,   4:6,   7:9,  10:12
    idp2, idq2, idv2, idw2 = 13:15, 16:18, 19:21, 22:24

    dq1 = dx[4:6]; qref = xref[iq]
    dq2 = dx[4:6]; q2ref = xref[iq2]

    q1 = params(add_error(UnitQuaternion(xref[iq1]),
                          RotationError(SVector{3}(dx[idq1]), CayleyMap())))
    q2 = params(add_error(UnitQuaternion(xref[iq1]),
                          RotationError(SVector{3}(dx[idq1]), CayleyMap())))

    return [xref[ip1] + dx[idp1]; q1; xref[iv1] + dx[idv1]; xref[iw1] + dx[idw1];
            xref[ip2] + dx[idp2]; q2; xref[iv2] + dx[idv2]; xref[iw2] + dx[idw2]]
end

function state_error_jacobian(x)
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26

    q1 = x[iq1]; q2 = x[iq2]
    M = blockdiag(sparse(∇differential(UnitQuaternion(q1))),
                  sparse(I(3)),
                  sparse(∇differential(UnitQuaternion(q2))),
                  sparse(I(3)))
    return Matrix(M)
end
