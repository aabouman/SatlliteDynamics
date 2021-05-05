# %%
using LinearAlgebra: normalize, norm, ×, I
using Rotations: lmult, hmat, RotMatrix, UnitQuaternion, RotationError, add_error, rotation_error, params, RotXYZ
using ForwardDiff
using StaticArrays

J_c = I(3)
mₜ = 419.709;
mₛ = 5.972e21;
G = 8.6498928e-19;
earthRadius = 6.37814;  # Megameters
n = sqrt(G * mₛ / (earthRadius^3))
Ω = [0; 0; 2*pi/1.5] #change 24 to solar time or whatever its called

function dynamics(x::Vector, u::Vector)
    xStatic = SVector{length(x)}(x)
    uStatic = SVector{length(u)}(u)
    dynamics(xStatic, uStatic)
end

function dynamics(x::SVector{16}, u::SVector{6})
    p_tc = @SVector x[1:3]
    q_sc = normalize(@SVector x[4:7])
    v_tc = @SVector x[8:10]
    ω_sc = @SVector x[11:13]

    q_st = @SVector x[14:16]  # Use Euler angles (X, Y, Z) for TRN orientation param
    # @assert q_st[1] ≈ 0 && q_st[2] ≈ 0  # Only rotates about Z

    𝑓_c = @SVector [u[1], u[2], u[3]]
    𝜏_c = @SVector [u[4], u[5], u[6]]

    R_tc = RotMatrix(RotXYZ(q_st...))' * RotMatrix(UnitQuaternion(q_sc))

    ṗ_tc = v_tc
    v̇_tc = ([3*(n^2)*p_tc[1] + 2*n*v_tc[2]; -2*n*v_tc[1]; -(n^2)*p_tc[3]] +
            R_tc * 𝑓_c)

    ω̇_sc = J_c \ (𝜏_c - ω_sc × (J_c * ω_sc))
    q̇_sc = 0.5 * lmult(q_sc) * hmat() * ω_sc

    q̇_st = @SVector zeros(3)
    q̇_st[3] = Ω[3]

    return [ṗ_tc; q̇_sc; v̇_tc; ω̇_sc; q̇_st]
end

function jacobian(x::Vector, u::Vector)
    x_new = vcat(x, zeros(3))

    A = ForwardDiff.jacobian(x_temp->dynamics(x_temp, u), x_new)[1:end-3, 1:end-3]
    B = ForwardDiff.jacobian(u_temp->dynamics(x_new, u_temp), u)[1:end-3, 1:end]
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
    N = length(Utraj)+1
    Xtraj = [zeros(length(x0)) for _ in 1:N]
    Xtraj[1] = x0

    for i in 1:N-1
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
