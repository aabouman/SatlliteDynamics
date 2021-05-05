using OSQP
using SparseArrays
using LinearAlgebra
using Rotations: UnitQuaternion, params, RotXYZ

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

function cost(ctrl::MPCController{OSQP.Model}, x_next)

    #converting euler of desired to UnitQuaternion
    q_ref =  UnitQuaternion(RotXYZ(x_next[8:11]...))
    J1 = ctrl.Q[1,1].*min(1 + x_next[1:4]'*q_ref, 1 - x_next[1:4]'*q_ref)
    J2 = (x_next[5:7] - x_next[11:13])'*ctrl.Q[5:7]*(x_next[5:7] - x_next[11:13])
    println("cost = " , J1 + J2)
    return J1 + J2
end

"""
    buildQP!(ctrl, A, B, Q, R, Qf; kwargs...)

Build the QP matrices `P` and `A` for the MPC problem. Note that these matrices
should be constant between MPC iterations.

Any keyword arguments will be passed to `initialize_solver!`.
"""
function buildQP!(ctrl::MPCController{OSQP.Model}, X)
    #Build QP matrices for OSQP
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1]) - 1  #remember n = 18 not 19
    m = length(ctrl.Uref[1])

    iq = 1:4
    Iq = Diagonal(SA[1,1,1, 0,0,0, 0,0,0, 0,0,0])

    q = [[-ctrl.R * ctrl.Uref[i]; -ctrl.Q * ctrl.Xref[i+1]]
         for i in 1:N-1]
    q[end][m+1:end] .= -ctrl.Qf * ctrl.Xref[end] # Overwriting the last value
    qtilde = [blockdiag(sparse(I(m)), sparse(state_error_jacobian(ctrl.Xref[i+1])')) * q[i]
              for i in 1:N-1]

    # Building the Cost linear term
    ctrl.q .= vcat(qtilde...)

    Qtilde = [state_error_jacobian(ctrl.Xref[i+1])' * ctrl.Q * state_error_jacobian(ctrl.Xref[i+1]) -
              Iq * (X[i][iq]' * ctrl.Xref[i+1][iq]) #q[i][m .+ (iq)]
              for i in 1:(N-1)]
    Qtilde[end] = state_error_jacobian(ctrl.Xref[end])' * ctrl.Qf * state_error_jacobian(ctrl.Xref[end]) -
                  Iq * (X[end][iq]' * ctrl.Xref[end][iq])
    # Building the Cost QP
    ctrl.P .= blockdiag([blockdiag(sparse(ctrl.R), sparse(Qtilde[i])) for i=1:N-1]...)

    # Computing the Dynamics constraints
    A = [state_error_jacobian(ctrl.Xref[i+1])' *
         discreteJacobian(ctrl.Xref[i], ctrl.Uref[i], ctrl.δt)[1] *
         state_error_jacobian(ctrl.Xref[i]) for i in 1:N-1]
    B = [state_error_jacobian(ctrl.Xref[i+1])' *
         discreteJacobian(ctrl.Xref[i], ctrl.Uref[i], ctrl.δt)[2] for i in 1:N-1]

    dynConstMat = blockdiag([sparse([B[i]  -I(n)]) for i in 1:(N-1)]...)
    dynConstMat += blockdiag(spzeros(n, m),
                             [sparse([A[i]  zeros(n, m)]) for i in 2:(N-2)]...,
                             sparse([A[end]  zeros(n, m+n)]))

    # Concatenate the dynamics constraints and the earth radius constraint
    ctrl.C .= vcat(dynConstMat)

    # Compute the equality constraints
    dynConstlb = vcat(zeros((N-1)*n)) #-A[1] * state_error(X[1], ctrl.Xref[1]),
    dynConstub = vcat(zeros((N-1)*n)) #-A[1] * state_error(X[1], ctrl.Xref[1]),

    # Concatenate the dynamics constraints and earth radius constraint bounds
    ctrl.lb .= vcat(dynConstlb)
    ctrl.ub .= vcat(dynConstub)

    # Initialize the included solver
    OSQP.setup!(ctrl.solver, P=ctrl.P, q=ctrl.q, A=ctrl.C, l=ctrl.lb, u=ctrl.ub,
                polish=1, verbose=0)
    return nothing
end

function updateRef!(ctrl::MPCController{OSQP.Model}, x_init)
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1]) - 1
    m = length(ctrl.Uref[1])

    ctrl.Xref .= stateInterpolate_CW(x_init, N, ctrl.δt)
    ctrl.Uref .= [zeros(m) for _ in N-1] # Uₖ

    return nothing
end

function solve_QP!(ctrl::MPCController{OSQP.Model}, x_start::Vector)#Xₖ::Vector)
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1]) - 1 #remember n = 12 not 13 as dealing with errors
    m = length(ctrl.Uref[1])

    results = OSQP.solve!(ctrl.solver)

    ΔU = [results.x[(n+m)*(i-1) .+ (1:m)] for i=1:N-1]
    Uₖ₊₁ = ctrl.Uref + ΔU

    # Xₖ₊₁ = [state_error_inv(ctrl.Xref[i+1], results.x[(n+m)*(i-1) .+ (m+1:m+n)])
    #         for i=1:N-1]

    u_curr = Uₖ₊₁[1]  # Recover u₁

    x_next = Vector(discreteDynamics(x_start, u_curr, ctrl.δt))

    X = rollout(x_start, Uₖ₊₁, ctrl.δt)

    return x_next, u_curr, X
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

    x_hist = []; u_hist = []; cost_hist = []

    x_hist = vcat(x_hist, [x_next])

    for i in 1:num_steps
        !verbose && print("step = $i\r")

        updateRef!(ctrl, x_next)

        !verbose || println("step = " , i)
        !verbose || println("\tx_curr = " , x_next)
        !verbose || println("\tXref[2] = " , ctrl.Xref[2])

        #need to build QP each time as P updating
        buildQP!(ctrl, X)

        x_next, u_curr, X = solve_QP!(ctrl, x_next)
        x_hist = vcat(x_hist, [x_next])
        u_hist = vcat(u_hist, [u_curr])

        !verbose || println("\tu_curr = " , u_curr)
        !verbose || println("COST = " , cost(ctrl, x_next))
        !verbose || println("############################")

        cost_hist = vcat(cost_hist, cost(ctrl, x_next))
        # all(x_next[1:3].+1 .≈ 1.) && println("\nDone!") && break
    end

    return x_hist, u_hist
end;
