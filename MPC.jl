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
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1]) - 1
    m = length(ctrl.Uref[1])

    iq = BitArray([0,0,0, 1,1,1,1, 0,0,0, 0,0,0])
    Iq = Diagonal(SA[0,0,0, 1,1,1, 0,0,0, 0,0,0])
    ip1, iq1, iv1, iw1 =  1:3,   4:7,   8:10, 11:13
    ip2, iq2, iv2, iw2 = 14:16, 17:20, 21:23, 24:26
    i𝑓, i𝜏 = 1:3, 4:6

    i11, i12 = 1:13, 1:12
    i21, i22 = 14:26, 13:24

    # Construct specific translation/rotational

    q = [[ctrl.R * (U[i] - ctrl.Uref[i]); ctrl.Q * (X[i+1][i21] - ctrl.Xref[i+1])]
         for i in 1:N-1]
    q[end][m+1:end] .= ctrl.Qf * (X[end][i21] - ctrl.Xref[end])  # Overwriting the last value
    qtilde = [blockdiag(sparse(I(m)), sparse(state_error_jacobian(X[i])[i21,i22]')) * q[i]
              for i in 1:N-1]
    # Building the Cost linear term
    ctrl.q .= vcat(qtilde...)

    Qtilde = [state_error_jacobian(X[i+1])[i21,i22]' * ctrl.Q * state_error_jacobian(X[i+1])[i21,i22] -
              sign(X[i+1][iq2]' * ctrl.Xref[i+1][iq1]) * Iq * (X[i+1][iq2]' * ctrl.Xref[i+1][iq1])
              for i in 1:(N-1)]
    Qtilde[end] = (state_error_jacobian(X[end])[i21,i22]' * ctrl.Qf * state_error_jacobian(X[end])[i21,i22] -
                   sign(X[end][iq2]' * ctrl.Xref[end][iq1]) * Iq * (X[end][iq2]' * ctrl.Xref[end][iq1]))
    # Building the Cost QP
    ctrl.P .= blockdiag([blockdiag(sparse(ctrl.R), sparse(Qtilde[i])) for i=1:N-1]...)

    # Computing the Dynamics constraints

    # TAKEN WRT X_TMP
    X_tmp2 = [[X[i][i11];ctrl.Xref[i]] for i = 1:N]

    A = [state_error_jacobian(X[i+1])[i21,i22]' *
         discreteJacobian(X_tmp2[i], ctrl.Uref[i], ctrl.δt)[1][i21,i21] *
         state_error_jacobian(X[i])[i21,i22] for i in 1:N-1]
    B = [state_error_jacobian(X[i+1])[i21,i22]' *
         discreteJacobian(X_tmp2[i], ctrl.Uref[i], ctrl.δt)[2][i21,:] for i in 1:N-1]

    dynConstMat = blockdiag([sparse([B[i]  -I(n)]) for i in 1:(N-1)]...)
    dynConstMat += blockdiag(spzeros(n, m),
                             [sparse([A[i]  zeros(n, m)]) for i in 2:(N-2)]...,
                             sparse([A[end]  zeros(n, m+n)]))

    # Concatenate the dynamics constraints and the earth radius constraint
    ctrl.C .= vcat(dynConstMat)

    # println("The determinate of the KKT condition matrix: ",
    #         det([ctrl.P  ctrl.C';
    #              ctrl.C  zeros(size(ctrl.C)[1], size(ctrl.C)[1])])
    #          )

    # Compute the equality constraints |
    """
        the Position and velocity lb and ub become zero when we take in spatial frame
    """
    dynConstlb = vcat(-A[1] * state_error_half(X[1][i21], ctrl.Xref[1]), zeros((N-2)*n))
    dynConstub = vcat(-A[1] * state_error_half(X[1][i21], ctrl.Xref[1]), zeros((N-2)*n))

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

    # Position/velocity of chaser WRT target
    p2, v2 = x_init[ip2], x_init[iv2]

    roll = rollout(x_init, [zeros(num_inputs) for _ in 1:N], δt)

    # Build the rollout of target quaternion and angular velocity
    p1s = [roll[i][ip1] for i in 1:N]
    q1s = [roll[i][iq1] for i in 1:N]
    v1s = [roll[i][iv1] for i in 1:N]
    w1s = [roll[i][iw1] for i in 1:N]

    # Build the reference trajectory for the chaser's orientaiton and angular
    # velocity. This is just the trajectory of the target
    # p2s = range(p2, p1s[end], length=N)
    p2s = deepcopy(p1s)

    q2s = deepcopy(q1s)

    # v2s = range(v2, v1s[end], length=N)
    v2s = deepcopy(v1s)

    w2s = deepcopy(w1s)

    return [[p2s[i]; q2s[i]; v2s[i]; w2s[i]] for i in 1:N]
end


function updateRef!(ctrl::MPCController{OSQP.Model}, x_init)
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1]) - 1
    m = length(ctrl.Uref[1])

    ctrl.Xref .= stateInterpolate_CW(x_init, N, ctrl.δt)
    ctrl.Uref .= [zeros(m) for _ in N-1] # Uₖ

    return nothing
end


function solve_QP!(ctrl::MPCController{OSQP.Model}, x_start::Vector)
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1]) - 1 #remember n = 12 not 13 as dealing with errors
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
        !verbose || println("\tXref[1] = " , ctrl.Xref[1])
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
