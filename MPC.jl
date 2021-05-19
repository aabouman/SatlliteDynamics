# %% codecell
import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); Pkg.instantiate()
using LinearAlgebra
using Ipopt

using ForwardDiff
using RobotDynamics

using MathOptInterface
const MOI = MathOptInterface

using SparseArrays

include("quadratic_cost.jl")
include("dynamics.jl")
include("utils.jl")

# %%
"""
    HybridNLP{n,m,L,Q}

Represents a (N)on(L)inear (P)rogram of a trajectory optimization problem,
with a dynamics model of type `L`, a quadratic cost function, horizon `T`,
and initial and final state `x0`, `xf`.

The kth state and control can be extracted from the concatenated state vector `Z` using
`Z[nlp.xinds[k]]`, and `Z[nlp.uinds[k]]`.

# Constructor
    HybridNLP(model, obj, tf, N, M, x0, xf, [integration])

# Basic Methods
    Base.size(nlp)    # returns (n,m,T)
    num_ineq(nlp)     # number of inequality constraints
    num_eq(nlp)       # number of equality constraints
    num_primals(nlp)  # number of primal variables
    num_duals(nlp)    # total number of dual variables
    packZ(nlp, X, U)  # Stacks state `X` and controls `U` into one vector `Z`

# Evaluating the NLP
The NLP supports the following API for evaluating various pieces of the NLP:

    eval_f(nlp, Z)         # evaluate the objective
    grad_f!(nlp, grad, Z)  # gradient of the objective
    eval_c!(nlp, c, Z)     # evaluate the constraints
    jac_c!(nlp, c, Z)      # constraint Jacobian
"""
struct HybridNLP{n, m, Q} <: MOI.AbstractNLPEvaluator
    obj::Vector{QuadraticCost{n,m,Float64}}  # objective function
    N::Int                                   # number of knot points
    δt::Float64
    x0::MVector{n,Float64}                   # initial condition
    Xref::Vector{SVector{n,Float64}}
    Uref::Vector{SVector{m,Float64}}
    xinds::Vector{SVector{n,Int}}            # Z[xinds[k]] gives states for time step k
    uinds::Vector{SVector{m,Int}}            # Z[uinds[k]] gives controls for time step k
    cinds::UnitRange{Int}                    # indices for each of the constraints
    lb::Vector{Float64}                      # lower bounds on the constraints
    ub::Vector{Float64}                      # upper bounds on the constraints
    zL::Vector{Float64}                      # lower bounds on the primal variables
    zU::Vector{Float64}                      # upper bounds on the primal variables
    rows::Vector{Int}                        # rows for Jacobian sparsity
    cols::Vector{Int}                        # columns for Jacobian sparsity
    function HybridNLP(obj::Vector{<:QuadraticCost{n,m}},
                    N::Integer, dt::Float64, x0::AbstractVector, integration::Type{<:QuadratureRule}=RK4
        ) where {n,m}

        Xref = [@SVector zeros(n) for k = 1:N]
        Uref = [@SVector zeros(m) for k = 1:N-1]

        # Create indices
        uinds = [SVector{m}((k-1)*(n+m) .+ (1:m)) for k = 1:N-1]
        xinds = [SVector{n}((k-1)*(n+m) .+ (m+1:n+m)) for k = 1:N-1]

        # TODO: specify the constraint indices
        c_dyn_inds = 1 : (N-1)*n

        # TODO: specify the bounds on the constraints
        m_nlp = (N-1)*(n+m)
        lb = fill(0.,m_nlp)
        ub = fill(0.,m_nlp)

        #lower bounds
        lb[c_dyn_inds] .= 0.

        #upper bounds
        ub[c_dyn_inds] .= 0.

        # Other initialization
        cinds = c_dyn_inds
        n_nlp = (N-1)*(m+n)
        zL = fill(-Inf, n_nlp)
        zU = fill(+Inf, n_nlp)
        rows = Int[]
        cols = Int[]

        δt = dt

        new{n, m, integration}(
            obj, N, δt, x0, Xref, Uref, xinds, uinds, cinds, lb, ub, zL, zU, rows, cols
        )
    end
end
Base.size(nlp::HybridNLP{n,m}) where {n,m} = (n,m,nlp.N)
num_primals(nlp::HybridNLP{n,m}) where {n,m} =  (m+n)*(nlp.N-1)
num_duals(nlp::HybridNLP) = nlp.cinds[end][end]

include("moi.jl")

"""
    packZ(nlp, X, U)

Take a vector state vectors `X` and controls `U` and stack them into a single vector Z.
"""
function packZ(nlp, X, U)
    Z = zeros(num_primals(nlp))
    for k = 1:nlp.N-1
        Z[nlp.xinds[k]] = X[k]
        Z[nlp.uinds[k]] = U[k]
    end
    # Z[nlp.xinds[end]] = X[end]
    return Z
end

"""
    unpackZ(nlp, Z)

Take a vector of all the states and controls and return a vector of state vectors `X` and
controls `U`.
"""
function unpackZ(nlp, Z)
    X = [Z[xi] for xi in nlp.xinds]
    U = [Z[ui] for ui in nlp.uinds]

    return X, U
end


# includes the interface to Ipopt

"""
    eval_f(nlp, Z)

Evaluate the objective, returning a scalar.
"""
function eval_f(nlp::HybridNLP, Z)
    J = 0.0
    xi,ui = nlp.xinds, nlp.uinds
    for k = 1:nlp.N-1
        x,u = Z[xi[k]], Z[ui[k]]
        J += stagecost(nlp.obj[k], x, u)
    end
    J += termcost(nlp.obj[end], Z[xi[end]])
    return J
end

"""
    grad_f!(nlp, grad, Z)

Evaluate the gradient of the objective at `Z`, storing the result in `grad`.
"""
function grad_f!(nlp::HybridNLP{n,m}, grad, Z) where {n,m}
    xi,ui = nlp.xinds, nlp.uinds
    obj = nlp.obj
    for k = 1:nlp.N-1
        x, u = Z[xi[k]], Z[ui[k]]
        grad[xi[k]] = obj[k].Q*x + obj[k].q
        grad[ui[k]] = obj[k].R*u + obj[k].r
    end
    grad[xi[end]] = obj[end].Q*Z[xi[end]] + obj[end].q
    return nothing
end

"""
    dynamics_constraint!(nlp, c, Z)

Calculate the dynamics constraints for the hybrid dynamics.
"""
function dynamics_constraint!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
    xi, ui = nlp.xinds, nlp.uinds

    N = nlp.N                      # number of time steps

    X, U = unpackZ(nlp, Z)

    A = [discreteJacobian(nlp.Xref[i], nlp.Uref[i], nlp.δt)[1] for i in 1:N-1]

    dynConstlb = vcat(-A[1] * (X[1] - nlp.Xref[1]), zeros((N-2)*n))
    dynConstub = vcat(-A[1] * (X[1] - nlp.Xref[1]), zeros((N-2)*n))

    c =  vcat(dynConstlb)

    return c  # for easy Jacobian checking
end

"""
    eval_c!(nlp, c, Z)

Evaluate all the constraints
"""
function eval_c!(nlp::HybridNLP, c, Z)
    xi = nlp.xinds
    dynamics_constraint!(nlp, c, Z)
end

"""
    dynamics_jacobian!(nlp, jac, Z)

Calculate the Jacobian of the dynamics constraints, storing the result in the matrix `jac`.
"""
function dynamics_jacobian!(nlp::HybridNLP{n,m}, jac, Z, Xref, Uref) where {n,m}

    xi, ui = nlp.xinds, nlp.uinds

    N = nlp.N

    X, U = unpackZ(nlp, Z)

    A = [discreteJacobian(nlp.Xref[i], Uref[i], nlp.δt)[1] for i in 1:N-1]
    B = [discreteJacobian(nlp.Xref[i], Uref[i], nlp.δt)[2] for i in 1:N-1]

    dynConstMat = blockdiag([sparse([B[i]  -I(n)]) for i in 1:(N-1)]...)
    dynConstMat += blockdiag(spzeros(n, m),
                             [sparse([A[i]  zeros(n, m)]) for i in 2:(N-2)]...,
                             sparse([A[end]  zeros(n, m+n)]))

    # Concatenate the dynamics constraints and the earth radius constraint
    jac .= vcat(dynConstMat)

    return nothing
end

"""
    jac_c!(nlp, jac, Z)

Evaluate the constraint Jacobians.
"""
function jac_c!(nlp::HybridNLP{n,m}, jac, Z) where {n,m}
    xi,ui = nlp.xinds, nlp.uinds

    dynamics_jacobian!(nlp, jac, Z, nlp.Xref, nlp.Uref)

    return nothing
end

"""
    reference_trajectory
"""
function stateInterpolate_CW(x_init, N::Int64, δt::Real)
    x_init = Vector(x_init)

    ip1, iv1 =  1:3,   4:6
    ip2, iv2 =  7:9,   10:12

    # Position/velocity of chaser WRT target
    p2, v2 = x_init[ip2], x_init[iv2]

    roll = rollout(x_init, [zeros(num_inputs) for _ in 1:N], δt)

    # Build the rollout of target quaternion and angular velocity
    p1s = [roll[i][ip1] for i in 1:N]
    v1s = [roll[i][iv1] for i in 1:N]

    # Build the reference trajectory for the chaser's orientaiton and angular
    # velocity. This is just the trajectory of the target
    p2s = deepcopy(p1s)
    v2s = deepcopy(v1s)

    return [[p1s[i]; v1s[i]; p2s[i]; v2s[i]] for i in 1:N]
end

function updateRef(Xref, Uref, x_init,dt)
    N = length(Xref)
    n = length(Xref[1]) - 1
    m = length(Uref[1])

    Xref = stateInterpolate_CW(x_init, N, dt)
    Uref = [zeros(m) for _ in 1:(N-1)] # Uₖ

    return Xref, Uref
end

"""
    update objective after each nlp step
    use error_State_jacobian here
"""
function update_obj(Xref, Uref, Q, R, Qf)
    obj = map(1:N-1) do k
        LQRCost(Q, R, Xref[k], Uref[k])
    end
    push!(obj, LQRCost(Qf, R, Xref[end], Uref[end]))
    return obj
end

function simulate(nlp_steps::Int, x_init::Vector, u_curr::Vector, Q::Matrix,
                  R::Matrix, Qf::Matrix; dt=0.01, num_steps=1000, verbose=false)
    N = nlp_steps
    n = length(x_init)
    m = length(u_curr)

    #initialize trajectory
    x_next = x_init
    u_curr = zeros(m)

    x_hist = []; u_hist = []; cost_hist = []
    x_hist = vcat(x_hist, [x_next])

    Xref = [zeros(n) for _ in 1:N]
    Uref = [zeros(m) for _ in 1:(N-1)]

    # Reference Trajectory
    Xref, Uref =  updateRef(Xref, Uref, x_init, dt) #reference_trajectory(model, times)

    Xguess = Xref #[x + 0.1*randn(length(x)) for x in Xref]
    Uguess = Uref #[u + 0.1*randn(length(u)) for u in Uref]

    # Objective
    obj = update_obj(Xref, Uref, Q, R, Qf)

    num_steps ≥ N || error("Number of steps being simulated must be ≥ the controller time horizon")

    for i in 1:num_steps
        !verbose && print("step = $i\r")

        Xref, Uref = updateRef(Xref, Uref, x_next, dt)
        obj = update_obj(Xref, Uref, Q, R, Qf)
        !verbose || println("step = " , i)
        !verbose || println("\tx_curr = " , x_next)
        !verbose || println("\tXref[1] = " , Xref[1])
        !verbose || println("\tXref[2] = " , Xref[2])

        #need to build nlp each time as grad and hesian changing
        # how to make nlp aware of the quaternion: need error state
        # use only for translation
        nlp = HybridNLP(obj, N, dt, x_next);

        nlp.Xref .= Xref
        nlp.Uref .= Uref

        Z0 = packZ(nlp, Xref, Uref);

        Z_sol, solver = solve(Z0, nlp, c_tol=1e-6, tol=1e-6)
        Xsol, Usol = unpackZ(nlp, Z_sol)

        x_next = Xsol[1]
        u_curr = Usol[1]
        # x_next, u_curr, X, U = solve_QP!(ctrl, x_next)
        x_hist = vcat(x_hist, [x_next])
        u_hist = vcat(u_hist, [u_curr])

        !verbose || println("\tu_curr = " , u_curr)
        !verbose || println("COST = " , stagecost(obj[1], x_next, u_curr))
        !verbose || println("############################")

        cost_hist = vcat(cost_hist, stagecost(obj[1], x_next, u_curr))
    end

    return x_hist, u_hist, cost_hist
end;
