using OSQP
using SparseArrays

"""
    MPCController

An MPC controller that uses a solver of type `S` to solve a QP at every iteration.

It will track the reference trajectory specified by `Xref`, `Uref` and `times`
with an MPC horizon of `Nmpc`. It will track the terminal reference state if
the horizon extends beyond the reference horizon.
"""
struct MPCController{S}
    P::SparseMatrixCSC{Float64,Int}
    q::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int}

    lb::Vector{Float64}
    ub::Vector{Float64}

    Nmpc::Int
    solver::S

    Xref::Vector{Vector{Float64}}
    Uref::Vector{Vector{Float64}}

    times::Vector{Float64}
end

"""
    OSQPController(n,m,N,Nref,Nd)

Generate an `MPCController` that uses OSQP to solve the QP.
Initializes the controller with matrices consistent with `n` states,
`m` controls, and an MPC horizon of `N`, and `Nref` constraints.

Use `Nref` to initialize a reference trajectory whose length may differ from the
horizon length.
"""
function OSQPController(n::Integer, m::Integer, N::Integer, Nref::Integer=N, Nd::Integer=(N-1)*n)
    Np = (N-1)*(n+m)   # number of primals
    P = spzeros(Np, Np)
    q = zeros(Np)
    A = spzeros(Nd, Np)
    lb = zeros(Nd)
    ub = zeros(Nd)
    Xref = [zeros(n) for k = 1:Nref]
    Uref = [zeros(m) for k = 1:Nref]
    tref = zeros(Nref)
    solver = OSQP.Model()
    MPCController{OSQP.Model}(P, q, A, lb, ub, N, solver, Xref, Uref, tref)
end

function onehot(length::Integer, one_spot::Integer)
    @assert(1 <= one_spot <= length)
    ret = zeros(length)
    ret[one_spot] = 1
    return ret
end

"""
    buildQP!(ctrl, A, B, Q, R, Qf; kwargs...)

Build the QP matrices `P` and `A` for the MPC problem. Note that these matrices
should be constant between MPC iterations.

Any keyword arguments will be passed to `initialize_solver!`.
"""
function buildQP!(ctrl::MPCController{OSQP.Model}, A, B, Q, R, Qf; kwargs...)
    #Build QP matrices for OSQP
    N = length(ctrl.Xref)
    n = length(ctrl.Xref[1])
    m = length(ctrl.Uref[1])

    # Building the Cost QP
    ctrl.P .= blockdiag(sparse(R),
                        kron(I(ctrl.Nmpc-2), blockdiag(sparse(Q), sparse(R))),
                        sparse(Qf));

    # Computing the Dynamics constraints
    As = [jacobian(ctrl.Xref[i], ctrl.Uref[i])[1] for i in 1:N]
    Bs = [jacobian(ctrl.Xref[i], ctrl.Uref[i])[2] for i in 1:N-1]

    dynConstMat = blockdiag([sparse([Bs[i]  -I(n)]) for i in 1:(N-1)]...)
    dynConstMat += blockdiag(spzeros(n, m),
                             [sparse([As[i]  zeros(n, m)]) for i in 2:(N-2)]...,
                             sparse([As[end]  zeros(n, m+n)]))

    ctrl.A .= dynConstMat

    # Initialize the included solver
    #    If you want to use your QP solver, you should write your own
    #    method for this function
    initialize_solver!(ctrl; kwargs...)
    return nothing
end

get_k(ctrl::MPCController, t) = searchsortedlast(ctrl.times, t)

"""
    initialize_solver!(ctrl::MPCController; kwargs...)

Initialize the internal solver once the QP matrices are initialized in the
controller.
"""
function initialize_solver!(ctrl::MPCController{OSQP.Model}; tol=1e-6, verbose=false)
    OSQP.setup!(ctrl.solver, P=ctrl.P, q=ctrl.q, A=ctrl.A, l=ctrl.lb, u=ctrl.ub,
                verbose=verbose, eps_rel=tol, eps_abs=tol, polish=1)
end

"""
    update_QP!(ctrl::MPCController, x, time)

Update the vectors in the QP problem for the current state `x` and time `time`.
This should update `ctrl.q`, `ctrl.lb`, and `ctrl.ub`.
"""
function update_QP!(ctrl::MPCController{OSQP.Model}, x, time)

    q = vcat([[zeros(m); Qilc*(X[i]-data.Xref[i])] for i in 2:(Nh-1)]...,
                 [zeros(m); Qf*(X[Nh]-data.Xref[Nh])])

    for k = 1:(ctrl.Nmpc-1)
        ind = get_k(ctrl, time + ctrl.times[k+1])
        ctrl.q[(k-1)*(n+m)+1:k*(n+m)] .= vcat(-R*(Uref[ind]-ueq), -Q*(Xref[ind]-xeq))
    end
    ind = get_k(ctrl, time + ctrl.times[ctrl.Nmpc])
    ctrl.q[end-n-m+1:end] .= vcat(-R*(Uref[ind]-ueq), -Qf*(Xref[ind]-xeq))

    # Compute the equality constraints
    eqConlb = vcat(-A * (x - xeq), zeros((ctrl.Nmpc-1) * size(A)[1] - size(x)[1]))
    eqConub = vcat(-A * (x - xeq), zeros((ctrl.Nmpc-1) * size(A)[1] - size(x)[1]))

    # Compute the inequality constraints
    p_zinConlb = repeat([0], ctrl.Nmpc-1)
    p_zinConub = repeat([1e8], ctrl.Nmpc-1)
    θinConlb = repeat([-5*π/180], ctrl.Nmpc-1)
    θinConub = repeat([5*π/180], ctrl.Nmpc-1)
    ϕinConlb = repeat([-10*π/180], ctrl.Nmpc-1)
    ϕinConub = repeat([10*π/180], ctrl.Nmpc-1)
    TinConlb = repeat([model.m*model.g*0.75 - xeq[7]], ctrl.Nmpc-1)
    TinConub = repeat([model.m*model.g*1.50 - xeq[7]], ctrl.Nmpc-1)
    inConlb = vcat(p_zinConlb, θinConlb, ϕinConlb, TinConlb)
    inConub = vcat(p_zinConub, θinConub, ϕinConub, TinConub)

    if size(ctrl.lb)[1] > size(eqConlb)[1]
        ctrl.lb .= vcat(eqConlb, inConlb)
        ctrl.ub .= vcat(eqConub, inConub)
    else
        ctrl.lb .= eqConlb
        ctrl.ub .= eqConub
    end

    return nothing
end

"""
    get_control(ctrl::MPCController, x, t)

Get the control from the MPC solver by solving the QP.
If you want to use your own QP solver, you'll need to change this
method.
"""
function get_control(ctrl::MPCController{OSQP.Model}, x, time)
    # Update the QP
    update_QP!(ctrl, x, time)
    OSQP.update!(ctrl.solver, q=ctrl.q, l=ctrl.lb, u=ctrl.ub)

    # Solve QP
    results = OSQP.solve!(ctrl.solver)
    Δu = results.x[1:2]

    k = get_k(ctrl, time)
    return ctrl.Uref[k] + Δu
end
