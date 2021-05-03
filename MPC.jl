using OSQP
using SparseArrays
using LinearAlgebra
using Rotations

earthRadius = 6.37814  # Megameters

"""
    MPCController

An MPC controller that uses a solver of type `S` to solve a QP at every iteration.

It will track the reference trajectory specified by `Xref`, `Uref` and `times`
with an MPC horizon of `Nmpc`. It will track the terminal reference state if
the horizon extends beyond the reference horizon.
"""
struct MPCController{S}
    P::SparseMatrixCSC{Float64,Int} # Cost quadratic matrix
    q::Vector{Float64}              # Cost linear matrix

    C::SparseMatrixCSC{Float64,Int} # Constraint matrix
    lb::Vector{Float64}             # Lower bound on constraints
    ub::Vector{Float64}             # Upper bound on constraints

    Nmpc::Int                       # MPC horizon
    solver::S                       # Solver

    Xref::Vector{Vector{Float64}}   # X reference trajectory
    Uref::Vector{Vector{Float64}}   # U reference trajectory

    Q::Matrix{Float64}              # Cost on states
    R::Matrix{Float64}              # Cost on inputs
    Qf::Matrix{Float64}             # Cost on final state

    δt::Real                        # Time step duration
end

"""
    OSQPController(n,m,N,Nref,Nd)

Generate an `MPCController` that uses OSQP to solve the QP.
Initializes the controller with matrices consistent with `n` states,
`m` controls, and an MPC horizon of `N`, and `Nref` constraints.

Use `Nref` to initialize a reference trajectory whose length may differ from the
horizon length.
`Nd` is the number of dual variables.
"""
function OSQPController(Q::Matrix, R::Matrix, Qf::Matrix, δt::Real, N::Integer, Nd::Integer)
    n = size(Q)[1]

    m = size(R)[2]
    Np = (N-1)*(n-1+m)   # number of primals

    P = spzeros(Np, Np)
    q = zeros(Np)

    C = spzeros(Nd, Np)
    lb = zeros(Nd)
    ub = zeros(Nd)

    Xref = [zeros(n) for k = 1:N]
    Uref = [zeros(m) for k = 1:N-1]

    solver = OSQP.Model()

    MPCController{OSQP.Model}(P, q, C, lb, ub, N, solver, Xref, Uref, Q, R, Qf, δt)
end


"""
    buildQP!(ctrl, A, B, Q, R, Qf; kwargs...)

Build the QP matrices `P` and `A` for the MPC problem. Note that these matrices
should be constant between MPC iterations.

Any keyword arguments will be passed to `initialize_solver!`.
"""
function buildQP!(ctrl::MPCController{OSQP.Model}, X, U)
    #Build QP matrices for OSQP
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1])
    m = length(ctrl.Uref[1])

    # Initialize the included solver
    OSQP.setup!(ctrl.solver, P=ctrl.P, q=ctrl.q, A=ctrl.C, l=ctrl.lb, u=ctrl.ub,
                polish=1)

    update_QP!(ctrl::MPCController{OSQP.Model}, X, U)

    return nothing
end


"""
    update_QP!(ctrl::MPCController, x, time)

Update the vectors in the QP problem for the current state `x` and time `time`.
This should update `ctrl.q`, `ctrl.lb`, and `ctrl.ub`.
"""
function update_QP!(ctrl::MPCController{OSQP.Model}, X, U)
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1]) - 1  #remember n = 12 not 13
    m = length(ctrl.Uref[1])

    iq = 4:7
    Iq = Diagonal(SA[0,0,0,1,1,1, 0,0,0, 0,0,0])

    println("X[1]= " , X[1])
    println("Xref[1] " , ctrl.Xref[1])
    #remember q has u subsumed => q[k] is 19x1 and not 13x1 {u,x}
    q = [[-ctrl.R * (U[i] - ctrl.Uref[i]); -ctrl.Q * (X[i+1] - ctrl.Xref[i+1])] for i in 1:N-1]
    q[end][m+1:end] .= -ctrl.Qf * (X[end] - ctrl.Xref[end]) #overwriting the last value

    Qtilde = [state_error_jacobian(X[i+1])' * ctrl.Q * state_error_jacobian(X[i+1]) - Iq * (q[i][m .+ iq]' * X[i+1][iq])
              for i in 1:(N-1)]
    Qtilde[end] = state_error_jacobian(X[end])' * ctrl.Qf * state_error_jacobian(X[end]) - Iq * (q[end][m .+ iq]' * X[end][iq])

    qtilde = [blockdiag(sparse(I(m)), sparse(state_error_jacobian(X[i+1])')) * q[i] for i in 1:N-1]

    # Building the Cost QP
    ctrl.P .= blockdiag([blockdiag(sparse(ctrl.R), sparse(Qtilde[i])) for i=1:N-1]...)

    ctrl.q .= vcat(qtilde...)

    # Computing the Dynamics constraints
    A = [state_error_jacobian(X[i+1])' *
         jacobian(ctrl.Xref[i], ctrl.Uref[i])[1] *
         state_error_jacobian(X[i]) for i in 1:N-1]
    B = [(state_error_jacobian(X[i+1])' *
          jacobian(ctrl.Xref[i], ctrl.Uref[i])[2]) for i in 1:N-1]

    dynConstMat = blockdiag([sparse([B[i]  -I(n)]) for i in 1:(N-1)]...)
    dynConstMat += blockdiag(spzeros(n, m),
                             [sparse([A[i]  zeros(n, m)]) for i in 2:(N-2)]...,
                             sparse([A[end]  zeros(n, m+n)]))
    earthRadiusConstMat =
        blockdiag([sparse([zeros(m)'  normalize(xref[1:3])'  zeros(n-3)';]) for xref in ctrl.Xref[2:end]]...)

    # Concatenate the dynamics constraints and the earth radius constraint
    ctrl.C .= vcat(dynConstMat, earthRadiusConstMat)

    # Compute the equality constraints
    dynConstlb = vcat(-A[1] * state_error(X[1], ctrl.Xref[1]), zeros((N-2)*n))
    dynConstub = vcat(-A[1] * state_error(X[1], ctrl.Xref[1]), zeros((N-2)*n))

    earthRadiusConstlb = [-norm(xref[1:3]) + earthRadius for xref in ctrl.Xref[2:end]]
    earthRadiusConstub = [Inf for xref in ctrl.Xref[2:end]]

    # Concatenate the dynamics constraints and earth radius constraint bounds
    ctrl.lb .= vcat(dynConstlb, earthRadiusConstlb)
    ctrl.ub .= vcat(dynConstub, earthRadiusConstub)

    OSQP.update!(ctrl.solver, P=ctrl.P, q=ctrl.q, A=ctrl.C, l=ctrl.lb, u=ctrl.ub)
    return nothing
end


"""
    find the reference trajectory using
    initial chaser position and
    final target position
"""
function stateInterpolate(x_init, x_final, N)
    # initial
    p1, q1, v1, w1 = x_init[1:3], x_init[4:7], x_init[8:10], x_init[11:13]
    # final
    p2, q2, v2, w2 = x_final[1:3], x_final[4:7], x_final[8:10], x_final[11:13]
    # quaternion
    ps = range(p1, p2, length=N)
    qs = slerp(UnitQuaternion(q1), UnitQuaternion(q2), N)
    vs = range(v1, v2, length=N)
    ws = range(w1, w2, length=N)

    # Concatenate
    xref = vcat(hcat(ps...), hcat(qs...), hcat(vs...), hcat(ws...))

    return [xref[:,i] for i in 1:N]
end

function updateRef!(ctrl::MPCController{OSQP.Model}, Xₖ, Uₖ, Xₜₖ, Uₜₖ)
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1])
    m = length(ctrl.Uref[1])

    ctrl.Xref .= stateInterpolate(Xₖ[1], Xₜₖ[end], N)
    ctrl.Uref .= Uₖ

    return nothing
end

function solve_QP!(ctrl::MPCController{OSQP.Model}, Xₜₖ, Uₜₖ)
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1]) - 1 #remember n = 12 not 13 as dealing with errors
    m = length(ctrl.Uref[1])

    results = OSQP.solve!(ctrl.solver)

    Uₖ₊₁ = ctrl.Uref + kron(I(N-1), blockdiag(I(m), spzeros(n,n))) * results.x
    xdelta = [state_error_inv(results.x[(n+m)*(i-1) .+ m+1:m+n]) for i=1:N-1]
    Xₖ₊₁ = ctrl.Xref[2:end] + xdelta
    Xₖ₊₁ = vcat(Xₖ₊₁, [discreteDynamics(Xₖ₊₁[end], Uₖ₊₁[end], ctrl.δt)])

    Xₜₖ₊₁ = rollout(Xₜₖ[2], Uₜₖ, ctrl.δt)
    Uₜₖ₊₁ = Uₜₖ

    return Xₖ₊₁, Uₖ₊₁, Xₜₖ₊₁, Uₜₖ₊₁
end

"""
controller function is called as controller(x), where x is a state vector
(length 13)
"""
function simulate(ctrl::MPCController{OSQP.Model}, xₛc_init::Vector, xₛₜ_init::Vector;
                  num_steps=1000, δt=0.001)
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1])
    m = length(ctrl.Uref[1])

    num_steps ≥ N || error("Number of steps being simulated must be ≥ the controller time horizon")

    Uₖ = [zeros(m) for _ in 1:N-1]
    Uₜₖ = [zeros(m) for _ in 1:N-1]
    Xₜₖ = rollout(xₛₜ_init, Uₜₖ, ctrl.δt)
    Xₖ = stateInterpolate(xₛc_init, Xₜₖ[end], N)

    #buld initial reference trajectory
    updateRef!(ctrl, Xₖ, Uₖ, Xₜₖ, Uₜₖ)

    buildQP!(ctrl, Xₖ, Uₖ)

    x_hist = [zeros(n) for _ in 1:num_steps+1]
    u_hist = [zeros(n) for _ in 1:num_steps]

    x_hist[1] = xₛc_init

    for i in 1:num_steps
        println("step = " , i)
        println("X[1] in loop " , X[1])
        println("Xref[1] in loop " , ctrl.Xtrl[1])
        updateRef!(ctrl, Xₖ, Uₖ, Xₜₖ, Uₜₖ)
        println("Xref after update " , ctrl.Xref[1])
        update_QP!(ctrl, Xₖ, Uₖ)
        Xₖ, Uₖ, Xₜₖ, Uₜₖ = solve_QP!(ctrl, Xₜₖ, Uₜₖ)

        x_hist[i+1] = Xₖ[1]
        u_hist[i] = Uₖ[1]

        println("############################")
    end

    return x_hist, u_hist
end;


function slerp(qa::UnitQuaternion, qb::UnitQuaternion, N::Int64)

    function slerpHelper(qa::UnitQuaternion{T}, qb::UnitQuaternion{T}, t::T) where {T}
        # Borrowed from Quaternions.jl
        # http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/
        coshalftheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;

        if coshalftheta < 0
            qm = -qb
            coshalftheta = -coshalftheta
        else
            qm = qb
        end
        abs(coshalftheta) >= 1.0 && return Rotations.params(qa)

        halftheta    = acos(coshalftheta)
        sinhalftheta = sqrt(one(T) - coshalftheta * coshalftheta)

        if abs(sinhalftheta) < 0.001
            return Rotations.params(UnitQuaternion(T(0.5) * (qa.w + qb.w),
                                                   T(0.5) * (qa.x + qb.x),
                                                   T(0.5) * (qa.y + qb.y),
                                                   T(0.5) * (qa.z + qb.z)))
        end

        ratio_a = sin((one(T) - t) * halftheta) / sinhalftheta
        ratio_b = sin(t * halftheta) / sinhalftheta

        temp = Rotations.params(UnitQuaternion(qa.w * ratio_a + qm.w * ratio_b,
                                               qa.x * ratio_a + qm.x * ratio_b,
                                               qa.y * ratio_a + qm.y * ratio_b,
                                               qa.z * ratio_a + qm.z * ratio_b)
                                )
        return temp
    end

    ts = range(0., 1., length=N)
    return [slerpHelper(qa, qb, t) for t in ts]
end
