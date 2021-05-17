using OSQP
using SparseArrays
using Rotations

include("dynamics.jl")

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
function OSQPController(Q::Matrix, R::Matrix, Qf::Matrix, δt::Real, N::Integer,
                        Np::Integer, Nd::Integer)
    n = size(Q)[1]
    m = size(R)[2]

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


function cost(ctrl::MPCController{OSQP.Model}, x_next)
    p1 = x_next[1:3]
    q1 = x_next[4:7]
    v1 = x_next[8:10]
    w1 = x_next[11:13]
    q2 = x_next[17:20]
    w2 = x_next[24:26]

    J1 = p1' * ctrl.Q[1:3, 1:3] * p1
    J2 = ctrl.Q[4,4] .* min(1 + q1' * q2, 1 - q1' * q2)
    J3 = v1' * ctrl.Q[8:10, 8:10] * v1
    J4 = (w1 - w2)' * ctrl.Q[11:13, 11:13] * (w1 - w2)

    return J1 + J2 + J3 + J4
end


"""
    buildQP!(ctrl, A, B, Q, R, Qf; kwargs...)

Build the QP matrices `P` and `A` for the MPC problem. Note that these matrices
should be constant between MPC iterations.

Any keyword arguments will be passed to `initialize_solver!`.
"""
function buildQP!(ctrl::MPCController{OSQP.Model}, X, U)
    #Build QP matrices for OSQP
    iq = BitArray([0,0,0, 1,1,1,1, 0,0,0, 0,0,0])
    Iq = Diagonal(SA[0,0,0, 1,1,1, 0,0,0, 0,0,0])
    i1, i2 = 1:13, 14:26
    id1, id2 = 1:12, 13:24
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26
    i𝑓, i𝜏 = 1:3, 4:6

    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1][i2]) - 1
    m = length(ctrl.Uref[1])

    # Construct specific translation/rotational
    X_tmp1 = [[zeros(3); X[i][iq2]; zeros(3); X[i][iw2]] - ctrl.Xref[i][i2]
              for i = 1:N]
    U_tmp1 = [[zeros(3); U[i][i𝜏]] - ctrl.Uref[i]
             for i = 1:N-1]

    q = [[ctrl.R * U_tmp1[i]; ctrl.Q * X_tmp1[i+1]]
         for i in 1:N-1]
    q[end][m+1:end] .= ctrl.Qf * X_tmp1[end]  # Overwriting the last value
    qtilde = [blockdiag(sparse(I(m)),
                        sparse(state_error_jacobian(X[i])[i2, id2]')) * q[i]
              for i in 1:N-1]

    # println("Size of q: ", size(ctrl.q))
    # println("Size of qtilde: ", size(vcat(qtilde...)))

    # Building the Cost linear term
    ctrl.q .= vcat(qtilde...)

    Qtilde = [state_error_jacobian(X[i])[i2, id2]' * ctrl.Q * state_error_jacobian(X[i])[i2, id2] -
              sign(X[i][i2][iq]' * ctrl.Xref[i][i2][iq]) * Iq * (X[i][i2][iq]' * ctrl.Xref[i][i2][iq])
              for i in 2:(N)]
    Qtilde[end] = (state_error_jacobian(X[end])[i2, id2]' * ctrl.Qf * state_error_jacobian(X[end])[i2, id2] -
                   sign(X[end][i2][iq]' * ctrl.Xref[end][i2][iq]) * Iq * (X[end][i2][iq]' * ctrl.Xref[end][i2][iq]))
    # Building the Cost QP
    ctrl.P .= blockdiag([blockdiag(sparse(ctrl.R), sparse(Qtilde[i])) for i=1:N-1]...)

    # Computing the Dynamics constraints

    # TAKEN WRT X_TMP
    X_tmp2 = [[ctrl.Xref[i][i1];
               zeros(3); ctrl.Xref[i][iq2]; zeros(3); ctrl.Xref[i][iw2]]
              for i = 1:N]
    U_tmp2 = [[zeros(3); ctrl.Uref[i][i𝜏]] for i = 1:N-1]

    A = [state_error_jacobian(X[i+1])[i2, id2]' *
         discreteJacobian(X_tmp2[i], U_tmp2[i], ctrl.δt)[1][i2, i2] *
         state_error_jacobian(X[i])[i2, id2] for i in 1:N-1]
    B = [state_error_jacobian(X[i+1])[i2, id2]' *
         discreteJacobian(X_tmp2[i], U_tmp2[i], ctrl.δt)[2][i2,:] for i in 1:N-1]

    dynConstMat = blockdiag([sparse([B[i]  -I(n)]) for i in 1:(N-1)]...)
    dynConstMat += blockdiag(spzeros(n, m),
                             [sparse([A[i]  zeros(n, m)]) for i in 2:(N-2)]...,
                             sparse([A[end]  zeros(n, m+n)]))

    # Concatenate the dynamics constraints and the earth radius constraint
    ctrl.C .= vcat(dynConstMat)

    # Construct specific translation/rotational
    X_tmp3 = [[X[i][i1]; zeros(3); ctrl.Xref[i][iq2]; zeros(3); ctrl.Xref[i][iw2]]
              for i = 1:N]

    # Compute the equality constraints
    dynConstlb = vcat(-A[1] * state_error(X[1], X_tmp3[1])[id2], zeros((N-2)*n))
    dynConstub = vcat(-A[1] * state_error(X[1], X_tmp3[1])[id2], zeros((N-2)*n))

    # Concatenate the dynamics constraints and earth radius constraint bounds
    ctrl.lb .= vcat(dynConstlb)
    ctrl.ub .= vcat(dynConstub)

    # Initialize the included solver
    OSQP.setup!(ctrl.solver, P=ctrl.P, q=ctrl.q, A=ctrl.C, l=ctrl.lb, u=ctrl.ub,
             polish=1, verbose=0)
    return nothing
end


function stateInterpolate_CW(x_init::Vector, N::Int64, δt::Real)
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26

    roll = rollout(x_init, [zeros(num_inputs) for _ in 1:N], δt)

    # Build the rollout of target quaternion and angular velocity
    p1s = [roll[i][ip1] for i in 1:N]
    q1s = [roll[i][iq1] for i in 1:N]
    v1s = [roll[i][iv1] for i in 1:N]
    w1s = [roll[i][iw1] for i in 1:N]

    # Build the reference trajectory for the chaser's orientaiton and angular
    # velocity. This is just the trajectory of the target
    p2s = range(x_init[ip2], zeros(3), length=N)
    # p2s = [zeros(3) for i in 1:N]
    q2s = deepcopy(q1s)
    v2s = range(x_init[iv2], zeros(3), length=N)
    # v2s = [zeros(3) for i in 1:N]
    w2s = deepcopy(w1s)

    return [[p1s[i]; q1s[i]; v1s[i]; w1s[i];
             p2s[i]; q2s[i]; v2s[i]; w2s[i]] for i in 1:N]
end


function updateRef!(ctrl::MPCController{OSQP.Model}, x_init)
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1]) - 2
    m = length(ctrl.Uref[1])

    ctrl.Xref .= stateInterpolate_CW(x_init, N, ctrl.δt)
    ctrl.Uref .= [zeros(m) for _ in N-1] # Uₖ

    return nothing
end


function solve_QP!(ctrl::MPCController{OSQP.Model}, x_start::Vector)
    i1, i2 = 1:13, 14:26
    id1, id2 = 1:12, 13:24
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26
    i𝑓, i𝜏 = 1:3, 4:6

    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1][i2]) - 1
    m = length(ctrl.Uref[1])

    results = OSQP.solve!(ctrl.solver)

    ΔU = [results.x[(n+m)*(i-1) .+ (1:m)] for i=1:N-1]
    Uₖ₊₁ = ctrl.Uref + ΔU

    u_curr = Uₖ₊₁[1]  # Recover u₁

    x_next = Vector(discreteDynamics(x_start, u_curr, ctrl.δt))

    X = rollout(x_start, Uₖ₊₁, ctrl.δt)

    return x_next, u_curr, X, Uₖ₊₁
end


"""
controller function is called as controller(x), where x is a state vector
(length 13)
"""
function simulate(ctrl::MPCController{OSQP.Model}, x_init::Vector;
                  num_steps=1000, verbose=false)
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1])
    m = length(ctrl.Uref[1])

    num_steps ≥ N || error("Number of steps being simulated must be ≥ the controller time horizon")

    #initialize trajectory
    x_next = x_init
    u_curr = zeros(m)

    X = [x_init for _ in 1:N]
    U = [zeros(m) for _ in 1:N-1]

    x_hist = []; u_hist = []; cost_hist = []

    x_hist = vcat(x_hist, [x_next])

    for i in 1:num_steps
        !verbose && print("step = $i\r")

        updateRef!(ctrl, x_next)

        !verbose || println("step = " , i)
        !verbose || println("\tx_curr = " , x_next)
        !verbose || println("\tXref[2] = " , ctrl.Xref[2])

        #need to build QP each time as P updating
        buildQP!(ctrl, X, U)

        x_next, u_curr, X, U = solve_QP!(ctrl, x_next)
        x_hist = vcat(x_hist, [x_next])
        u_hist = vcat(u_hist, [u_curr])

        !verbose || println("\tu_curr = " , u_curr)
        !verbose || println("COST = " , cost(ctrl, x_next))
        !verbose || println("############################")

        cost_hist = vcat(cost_hist, cost(ctrl, x_next))
    end

    return x_hist, u_hist, cost_hist
end;
