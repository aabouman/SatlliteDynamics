# %% codecell
import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); Pkg.instantiate()
# Pkg.add("WebIO")
using LinearAlgebra
using ForwardDiff
using RobotZoo
using RobotDynamics
using Ipopt
using MathOptInterface
using TrajOptPlots
const MOI = MathOptInterface
using Random
using Test
include("quadratic_cost.jl")
include("walker.jl")
include("utils.jl")
# %%
# import Pkg; Pkg.add("Ipopt")
# %% codecell
# Pkg.add("WebIO")
# %% markdown
# # Q2: Hybrid Trajectory Optimization  (40 pts)
# In this problem you'll use a direct method to optimize a walking trajectory for a simple biped model, using the hybrid dynamics formulation. You'll pre-specify a gait sequence and solve the problem using Ipopt, a high-quality open-source interior point nonlinear programming solver (actually developed here at CMU!).
#
# ## The Dynamics
# Our system is modeled as three point masses: one for the body and one for each foot. The state is defined as the x and y positions and velocities of these masses, for a total of 6 degrees of freedom and 12 states. The legs are connected to the body with prismatic joints. The system has three control inputs: a force along each leg, and the torque between the legs.
# Reference the code block below for a quick overview of the API we've implemented for you. You're encouraged to look at the code in [src/walker.jl](https://github.com/Optimal-Control-16-745/hw4_solutions/blob/master/src/walker.jlhttps://github.com/Optimal-Control-16-745/hw4_solutions/blob/master/src/walker.jl).
# %% codecell
model = SimpleWalker()
x,u = rand(model)  # generate some random states and controls
dt = 0.1

# evaluate the discrete dynamics using RK4
stance1_dynamics_rk4(model, x, u, dt)
stance2_dynamics_rk4(model, x, u, dt)

# jump maps
jump1_map(x)
jump2_map(x)

# evaluate the discrete dynamics Jacobians
stance1_jacobian(model, x, u, dt)
stance2_jacobian(model, x, u, dt)

# jump map Jacobian
jump1_jacobian()
jump2_jacobian();

# visualizer
x = zeros(12)
x[2] = 1
vis = Visualizer()
set_mesh!(vis, model)
visualize!(vis, model, SVector{12}(x))
render(vis)
# %% markdown
# ## The Problem Formulation
# The trajectory optimization problem we're solving has the following form:
#
# $$
# \begin{aligned}
# &\text{minimize} && \frac{1}{2} (x_N - \bar{x}_N)^T Q_N (x_N - \bar{x}) +
# \frac{1}{2}\sum_{k=1}^{N-1} (x_k - \bar{x}_k)^T Q (x_k - \bar{x}_k) + (u_k - \bar{u}_k)^T R_k (u_k - \bar{u}_k) \\
# &\text{subject to} && x_1 = x_\text{init} \\
#                   &&& x_N = x_\text{goal} \\
#                   &&& f_1(x_k,u_k) = x_{k+1}, && \forall k \in \mathcal{M}_1 \setminus \mathcal{J}_1 \\
#                   &&& f_2(x_k,u_k) = x_{k+1}, && \forall k \in \mathcal{M}_2 \setminus \mathcal{J}_2 \\
#                   &&& g_2(f_1(x_k,u_k)) = x_{k+1}, && \forall k \in \mathcal{J}_1 \\
#                   &&& g_1(f_2(x_k,u_k)) = x_{k+1}, && \forall k \in \mathcal{J}_2 \\
#                   &&& y^{(1)}_k = 0, && \forall k \in \mathcal{M}_1 \\
#                   &&& y^{(2)}_k = 0, && \forall k \in \mathcal{M}_2 \\
#                   &&& 0.5 < ||r^{(b)}_k - r^{(i)}_k|| < 1.5, && \forall k \in \mathcal{M}_i, \; i \in \{1,2\} \\
# \end{aligned}
# $$
# where $\bar{x}$ and $\bar{u}$ are reference states and controls. The first 2 constraints are the initial and terminal constraints, and the last constraint is a bound on the length of the prismatic leg joints.
#
# The other constraints encode the hybrid dynamics. For this problem we have 2 different "modes," each corresponding to when one foot is on the ground (we don't consider the cases when both or neither feet are on the ground). Every knot point is assigned either to $\mathcal{M}_1$ or $\mathcal{M}_2$, but not both. To simplify our problem and obtain a nice, even walking gait, we'll assign $M$ adjacent time steps to the one mode, and then alternate. For a trajectory of 45 time steps, we'll have:
# $$
# \mathcal{M}_1 = \{1\text{:}5,11\text{:}15,21\text{:}25,31\text{:}35,41\text{:}45\} \\
# \mathcal{M}_2 = \{6\text{:}10,16\text{:}20,26\text{:}30,36\text{:}40\}
# $$
#
# The jump map sets $\mathcal{J}_1$ and $\mathcal{J}_2$ are the indices where the mode of the next time step is different than the current, i.e. $\mathcal{J}_i \equiv \{k+1 \notin \mathcal{M}_i \; | \; k \in \mathcal{M}_i\}$.
#
# Lastly, constraints 7 and 8 require that the height of the foot is zero for the corresponding mode.
# %% markdown
# ## Part (a): Setting up the NLP (3 pts)
# As a first step, we'll set up the variables we'll need to evaluate our constraints (we've already implemented the cost functions for you). Your constraints should be ordered as follows:
#
# $$ \begin{bmatrix}
# c_\text{init} \\
# c_\text{goal} \\
# c_\text{dynamics} \\
# c_\text{stance} \\
# c_\text{length} \\
# \end{bmatrix}$$
# which are of length $n$, $n$, $Nn + (N-1)m$, $N$, and $2N$, respectively. The dynamics, stance, and length constraints should be ordered by time step.
# %% codecell
# TASK: Complete the constructor for the HybridNLP type (3 pts)

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
struct HybridNLP{n,m,L,Q} <: MOI.AbstractNLPEvaluator
    model::L                                 # dynamics model
    obj::Vector{QuadraticCost{n,m,Float64}}  # objective function
    N::Int                                   # number of knot points
    M::Int                                   # number of steps in each mode
    Nmodes::Int                              # number of modes
    tf::Float64                              # total time (sec)
    x0::MVector{n,Float64}                   # initial condition
    xf::MVector{n,Float64}                   # final condition
    times::Vector{Float64}                   # vector of times
    modes::Vector{Int}                       # mode ID
    xinds::Vector{SVector{n,Int}}            # Z[xinds[k]] gives states for time step k
    uinds::Vector{SVector{m,Int}}            # Z[uinds[k]] gives controls for time step k
    cinds::Vector{UnitRange{Int}}            # indices for each of the constraints
    lb::Vector{Float64}                      # lower bounds on the constraints
    ub::Vector{Float64}                      # upper bounds on the constraints
    zL::Vector{Float64}                      # lower bounds on the primal variables
    zU::Vector{Float64}                      # upper bounds on the primal variables
    rows::Vector{Int}                        # rows for Jacobian sparsity
    cols::Vector{Int}                        # columns for Jacobian sparsity
    function HybridNLP(model, obj::Vector{<:QuadraticCost{n,m}},
            tf::Real, N::Integer, M::Integer, x0::AbstractVector, xf::AbstractVector, integration::Type{<:QuadratureRule}=RK4
        ) where {n,m}
        # Create indices
        xinds = [SVector{n}((k-1)*(n+m) .+ (1:n)) for k = 1:N]
        uinds = [SVector{m}((k-1)*(n+m) .+ (n+1:n+m)) for k = 1:N-1]
        times = collect(range(0, tf, length=N))

        # Specify the mode sequence
        modes = map(1:N) do k
            isodd((k-1) ÷ M + 1) ? 1 : 2
        end
        Nmodes = Int(ceil(N/M))

        # TODO: specify the constraint indices
        c_init_inds = 1:n
        c_term_inds = n+1:2*n
        c_dyn_inds = 2*n+1 : 2*n + (N-1)*n
        c_stance_inds = 2*n + (N-1)*n+1 : 2*n + (N-1)*n + N
        c_length_inds = 2*n + (N-1)*n + N + 1 : 2*n + (N-1)*n + N + 2*N

        # TODO: specify the bounds on the constraints
        m_nlp = 2*n + (N-1)*n + N + 2*N
        lb = fill(0.,m_nlp)
        ub = fill(0.,m_nlp)

        #lower bounds
        lb[c_init_inds] .= 0.
        lb[c_term_inds] .= 0.
        lb[c_dyn_inds] .= 0.
        lb[c_stance_inds] .= 0.
        lb[c_length_inds] .= 0.5

        #upper bounds
        ub[c_init_inds] .= 0.
        ub[c_term_inds] .= 0.
        ub[c_dyn_inds] .= 0.
        ub[c_stance_inds] .= 0.
        ub[c_length_inds] .= 1.5

        # Other initialization
        cinds = [c_init_inds, c_term_inds, c_dyn_inds, c_stance_inds, c_length_inds]
        n_nlp = n*N + (N-1)*m
        zL = fill(-Inf, n_nlp)
        zU = fill(+Inf, n_nlp)
        rows = Int[]
        cols = Int[]

        new{n,m,typeof(model), integration}(
            model, obj,
            N, M, Nmodes, tf, x0, xf, times, modes,
            xinds, uinds, cinds, lb, ub, zL, zU, rows, cols
        )
    end
end
Base.size(nlp::HybridNLP{n,m}) where {n,m} = (n,m,nlp.N)
num_primals(nlp::HybridNLP{n,m}) where {n,m} = n*nlp.N + m*(nlp.N-1)
num_duals(nlp::HybridNLP) = nlp.cinds[end][end]

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
    Z[nlp.xinds[end]] = X[end]
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

function TrajOptPlots.visualize!(vis, nlp::HybridNLP, Z)
    TrajOptPlots.visualize!(vis, nlp.model, nlp.tf, unpackZ(nlp, Z)[1])
end

# includes the interface to Ipopt
include("moi.jl")
# %% markdown
# ## Costs (provided)
# %% codecell
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
        x,u = Z[xi[k]], Z[ui[k]]
        grad[xi[k]] = obj[k].Q*x + obj[k].q
        grad[ui[k]] = obj[k].R*u + obj[k].r
    end
    grad[xi[end]] = obj[end].Q*Z[xi[end]] + obj[end].q
    return nothing
end
# %% markdown
# ## Part (b): Reference Trajectory (3 pts)
# A good reference trajectory is often critical for trajectory optimization. Design a reference trajectory that just translates the walker from the start to the finish (remember to make the velocities consistent with your state trajectory). The height of the body should be 1m off the ground, and the feet should have a height of zero. The robot should start at a x location of -1.5m and end at 1.5m.
# %% codecell
"""
    reference_trajectory(model, times)

Return a reference trajectory that translates the walker from an x position of `xinit` to `xterm`,
with a nominal body height of `height` meters.
"""
function reference_trajectory(model::SimpleWalker, times;
        xinit = -1.5,
        xterm = +1.5,
        height = 1.0,
    )
    # Some useful variables
    n,m = size(model)
    tf = times[end]
    N = length(times)
    Δx = xterm - xinit
    mb,g = model.mb, model.g

    # TODO: Design the reference trajectory
    xref = zeros(n,N)
    uref = zeros(m,N)

    dt = times[2] - times[1]

    #body
    xref[1,1:end] = range(xinit, xterm, length=N)
    xref[2,1:end] .= height
    #leg 1
#     xref[3,1:5] = range(xinit,xinit + Δx/5, length=5)
#     xref[3,6:10] .= xref[3,5]
#     xref[3,11:15] = range(xref[3,10],xref[3,10]+Δx/5, length=5)
#     xref[3,16:20] .= xref[3,15]
#     xref[3,21:25] = range(xref[3,15],xref[3,15]+Δx/5, length=5)
#     xref[3,26:30] .= xref[3,25]
#     xref[3,31:35] = range(xref[3,25],xref[3,25]+Δx/5, length=5)
#     xref[3,36:40] .= xref[3,35]
#     xref[3,41:45] = range(xref[3,35],xref[3,35]+Δx/5, length=5)
    xref[3,1:end] = range(xinit, xterm, length=N)
    xref[4,1:end] .= 0.

    #leg 2
#     xref[5,1:5] .= xinit
#     xref[5,6:10] = range(xinit,xinit+Δx/4, length=5)
#     xref[5,11:15] .= xref[5,10]
#     xref[5,16:20] .= range(xref[5,10],xref[5,10]+Δx/4, length=5)
#     xref[5,21:25] .= xref[5,20]
#     xref[5,26:30] .= range(xref[5,20],xref[5,20]+Δx/4, length=5)
#     xref[5,31:35] .= xref[5,30]
#     xref[5,36:40] .= range(xref[5,30],xref[5,30]+Δx/4, length=5)
#     xref[5,41:45] .= xref[5,40]
    xref[5,1:end] = range(xinit, xterm, length=N)
    xref[6,1:end] .= 0.

    #velcoity
    #xref[7,1:end] = range(xref[1,2]-xref[1,1], xref[1,N]-xref[1,N-1],length=N)
    xref[7,1:end-1] = range((xref[1,2]-xref[1,1])/dt, (xref[1,N-1]-xref[1,N-2])/dt,length=N-1)
    xref[7,end] = 0.
    xref[8,1:end] .= 0.

    for i=1:N-1
        xref[9,i] = (xref[3,i+1] - xref[3,i])/dt
        xref[11,i] = (xref[5,i+1] - xref[5,i])/dt
    end

#     xref[7:end,1:end] .= xref[7:end,1:end]/dt

    uref[1,1:end] .= 0.5*mb*g
    uref[2,1:end] .= 0.5*mb*g

    # Convert to a trajectory
    Xref = [SVector{n}(x) for x in eachcol(xref)]
    Uref = [SVector{m}(u) for u in eachcol(uref)]
    return Xref, Uref
end
# %% codecell
# using Plots
# %% codecell
# plot(Xref,inds=9:10)
# %% markdown
# ### Problem Definition
# %% codecell
# Dynamics model
model = SimpleWalker()

# Discretization
tf = 4.4
dt = 0.1
N = Int(ceil(tf/dt)) + 1
M = 5
times = range(0,tf, length=N);

# Reference Trajectory
Xref,Uref = reference_trajectory(model, times)

# Objective
Random.seed!(1)
Q = Diagonal([1; 10; fill(1.0, 4); 1; 10; fill(1.0, 4)]);
R = Diagonal(fill(1e-3,3))
Qf = Q;
obj = map(1:N-1) do k
    LQRCost(Q,R,Xref[k],Uref[k])
end
push!(obj, LQRCost(Qf, R*0, Xref[N], Uref[1]))

# Define the NLP
nlp = HybridNLP(model, obj, tf, N, M, Xref[1], Xref[end]);
# %% codecell
using Statistics
@testset "Part b" begin
    Xref, Uref = reference_trajectory(model, times)
    Xdiff = diff(Xref)
    xdiff = mean(Xdiff)
    @test xdiff[2] == 0
    @test xdiff[4] == 0
    @test xdiff[[1,3,5]] ≈ fill(3/(N-1), 3) atol=1e-2
    udiff = mean(diff(Uref))
    @test udiff ≈ zeros(3) atol=1e-4
    @test Uref[1][1] ≈ model.mb*model.g*0.5 atol=1e-3
    @test Uref[1][2] ≈ model.mb*model.g*0.5 atol=1e-3
    @test Uref[1][3] ≈ 0
end;
# %% markdown
# ## Part (c): Constraints (17 pts)
# As you can probably guess looking at the problem definition above, the tricky part of the optimization problem is all in the constraints. Implement the methods below to specify the constraints for our hybrid trajectory optimization problem.
# %% codecell
# TASK: Implement the following methods
#       1. dynamics_constraint! (6 pts)
#       2. stance_constraint!   (4 pts)
#       3. length_constraint!   (4 pts)
#       4. eval_c!              (3 pts)
"""
    dynamics_constraint!(nlp, c, Z)

Calculate the dynamics constraints for the hybrid dynamics.
"""
function dynamics_constraint!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
    xi,ui = nlp.xinds, nlp.uinds
    model = nlp.model
    N = nlp.N                      # number of time steps
    M = nlp.M                      # time steps per mode
    Nmodes = nlp.Nmodes            # number of mode sequences (N ÷ M)

    # Grab a view of the indices for the dynamics constraints
    d = reshape(view(c, nlp.cinds[3]),n,N-1)

    dt = nlp.times[2] - nlp.times[1]
    # TODO: calculate the hybrid dynamics constraints
    #  TIP: remember to include the jump map when the mode changes!

    for kk = 1:4

        ks = 10*(kk-1)

        for k=ks+1:ks+4
            d[(k-1)*n + 1: k*n] = stance1_dynamics_rk4(model, Z[xi[k]], Z[ui[k]], dt) - Z[xi[k+1]]
        end

        k = ks+5
        d[(k-1)*n + 1: k*n] = jump2_map(stance1_dynamics_rk4(model, Z[xi[k]], Z[ui[k]], dt)) - Z[xi[k+1]]

        for k=ks+6:ks+9
            d[(k-1)*n + 1: k*n] = stance2_dynamics_rk4(model, Z[xi[k]], Z[ui[k]], dt) - Z[xi[k+1]]
        end

        k = ks+10
        d[(k-1)*n + 1: k*n] = jump1_map(stance2_dynamics_rk4(model, Z[xi[k]], Z[ui[k]], dt)) - Z[xi[k+1]]

    end

    for k=41:44
        d[(k-1)*n + 1: k*n] = stance1_dynamics_rk4(model, Z[xi[k]], Z[ui[k]], dt) - Z[xi[k+1]]
    end


    return d  # for easy Jacobian checking
end

"""
    stance_constraint!(nlp, c, Z)

Calculate the stance constraint for each time step, i.e. that the height of
appropriate leg must be zero.
"""
function stance_constraint!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
    # Create a view of the portion for the stance constraints
    d = view(c, nlp.cinds[4])

    # Some useful variables
    xi,ui = nlp.xinds, nlp.uinds
    N = nlp.N                      # number of time steps
    M = nlp.M                      # time steps per mode
    Nmodes = nlp.Nmodes            # number of mode sequences (N ÷ M)

    # TODO: Calculate the stance constraints
    switch = 1
    for k=1:N

        if switch == 1
            d[k] = Z[xi[k]][4]
        else
            d[k] = Z[xi[k]][6]

        end

        if k%M == 0

            switch = switch*(-1)
        end

    end

    return d  # for easy Jacobian checking
end

"""
    length_constraint!(nlp, c, Z)

Calculate the length constraints, i.e. that the length of each leg must
be between `nlp.model.ℓ_min` and `nlp.model.ℓ_max`.
"""
function length_constraint!(nlp::HybridNLP{n,m}, c, Z) where {n,m}
    # Create a view for the portion for the length constraints
    d = view(c, nlp.cinds[5])

    # Some useful variables
    xi,ui = nlp.xinds, nlp.uinds
    N = nlp.N                      # number of time steps
    M = nlp.M                      # time steps per mode
    Nmodes = nlp.Nmodes            # number of mode sequences (N ÷ M)

    # TODO: Calculate the length constraints

    for k = 1:N

        d[2*k - 1] = norm(Z[xi[k]][1:2] - Z[xi[k]][3:4])

        d[2*k] = norm(Z[xi[k]][1:2] - Z[xi[k]][5:6])

    end


    return d   # for easy Jacobian checking
end

"""
    eval_c!(nlp, c, Z)

Evaluate all the constraints
"""
function eval_c!(nlp::HybridNLP, c, Z)
    xi = nlp.xinds
    c[nlp.cinds[1]] .= Z[xi[1]] - nlp.x0
    c[nlp.cinds[2]] .= Z[xi[end]] - nlp.xf
    dynamics_constraint!(nlp, c, Z)
    stance_constraint!(nlp, c, Z)
    length_constraint!(nlp, c, Z)
end
# %% codecell
@testset "Part c: Constraints" begin
    Z = rand(num_primals(nlp))
    X,U = unpackZ(nlp, Z)
    c = zeros(num_duals(nlp))
    n,m,N = size(nlp)

    @testset "Dynamics constraints" begin
        d = dynamics_constraint!(nlp, c, Z)
        @test length(d) == n*(N-1)

        @test d[1:n] ≈ stance1_dynamics_rk4(model, X[1], U[1], dt) - X[2]
        @test d[n*(M-1) .+ (1:n)] ≈ jump2_map(stance1_dynamics_rk4(model, X[M], U[M], dt)) - X[M+1]
        @test d[n*M .+ (1:n)] ≈ stance2_dynamics_rk4(model, X[M+1], U[M+1], dt) - X[M+2]
    end

    @testset "Stance constraints" begin
        d = stance_constraint!(nlp, c, Z)
        @test length(d) == N
        @test d[1:M] ≈ [x[4] for x in X[1:M]]
        @test d[M .+ (1:M)] ≈ [x[6] for x in X[M .+ (1:M)]]
    end

    @testset "Length constraints" begin
        d = length_constraint!(nlp, c, Z)
        @test length(d) == 2N
        @test d[1] ≈ norm(X[1][1:2] - X[1][3:4])
        @test d[2] ≈ norm(X[1][1:2] - X[1][5:6])
        @test d[3] ≈ norm(X[2][1:2] - X[2][3:4])
        @test d[4] ≈ norm(X[2][1:2] - X[2][5:6])
    end
end;
# %% markdown
# ## Part (d): Constraint Jacobians (17 pts)
# As you've probably guessed, we'll also need the Jacobians of our constraints. While we can use methods like automatic differentiation or finite differencing, for trajectory optimization problems it's often very easy to write down the Jacobians by hand, and get large speedups as a result.
#
# **NOTE**: Since the way Ipopt deals with sparsity is a little complicated, to keep things simple we'll just treat the Jacobian as dense, so we won't see a huge speedup here. But it's still really good practice!
#
# **EXTRA CREDIT (10pts)**: Leverage the sparsity in the constraint Jacobian. You'll need to do a little digging into the interface expected by Julia's optimization solver wrapper, MathOptInterface. To get started, refer to [this section](https://jump.dev/MathOptInterface.jl/dev/reference/nonlinear/#Functions) in the documention. Basically, you'll deal directly with the vector of non-zero entries, rather than a sparse matrix, and specify the row and column of these entries a-priori. Feel free to use the `row` and `col` fields in `HybridNLP` provided for this purpose.
# %% codecell
# TASK: Implement the following methods
#       1. dynamics_jacobian! (9 pts)
#       2. jac_c!             (8 pts)

"""
    dynamics_jacobian!(nlp, jac, Z)

Calculate the Jacobian of the dynamics constraints, storing the result in the matrix `jac`.
"""
function dynamics_jacobian!(nlp::HybridNLP{n,m}, jac, Z) where {n,m}
    # Create a view of the portion of the Jacobian for the dynamics constraints
    D = view(jac, nlp.cinds[3], :)

    # Some useful variables
    xi,ui = nlp.xinds, nlp.uinds
    model = nlp.model
    N = nlp.N                      # number of time steps
    M = nlp.M                      # time steps per mode
    Nmodes = nlp.Nmodes            # number of mode sequences (N ÷ M)

    #println("before")
    #println(norm(jac))

    dt = nlp.times[2] - nlp.times[1]
    # TODO: Calculate the dynamics Jacobians
    for kk = 1:4
        #println("kk " , kk)
        ks = 10*(kk-1)    #0, 10, 20, 30

        for k=ks+1:ks+4
            #println("here1")
            D[(k-1)*n+1:k*n,(k-1)*(n+m)+1:k*(n+m)] = stance1_jacobian(model, Z[xi[k]], Z[ui[k]], dt)
            D[(k-1)*n+1:k*n,k*(n+m)+1:k*(n+m)+n] = -I(n)
        end
        k = ks+5

        D[(k-1)*n+1:k*n,(k-1)*(n+m)+1:k*(n+m)]=jump2_jacobian()*stance1_jacobian(model, Z[xi[k]], Z[ui[k]], dt)
        D[(k-1)*n+1:k*n,k*(n+m)+1:k*(n+m)+n]= -I(n)

        for k=ks+6:ks+9
            #println("here2")
            inds = (k-1)*(n+m)
            D[(k-1)*n+1:k*n,(k-1)*(n+m)+1:k*(n+m)] = stance2_jacobian(model, Z[xi[k]], Z[ui[k]], dt)
            D[(k-1)*n+1:k*n,k*(n+m)+1:k*(n+m)+n] = -I(n)
        end
        k = ks+10
        inds = (k-1)*(n+m)
        D[(k-1)*n+1:k*n,(k-1)*(n+m)+1:k*(n+m)]=jump1_jacobian()*stance2_jacobian(model, Z[xi[k]], Z[ui[k]], dt)
        D[(k-1)*n+1:k*n,k*(n+m)+1:k*(n+m)+n]= -I(n)

    end

    for k=41:44
        D[(k-1)*n+1:k*n,(k-1)*(n+m)+1:k*(n+m)] = stance1_jacobian(model, Z[xi[k]], Z[ui[k]], dt)
        D[(k-1)*n+1:k*n,k*(n+m)+1:k*(n+m)+n] = -I(n)
    end
    #println("after")
    #println(norm(jac))

    return nothing
end

"""
    jac_c!(nlp, jac, Z)

Evaluate the constraint Jacobians.
"""
function jac_c!(nlp::HybridNLP{n,m}, jac, Z) where {n,m}
    xi,ui = nlp.xinds, nlp.uinds

    # Create views for each portion of the Jacobian
    jac_init = view(jac, nlp.cinds[1], xi[1])
    jac_term = view(jac, nlp.cinds[2], xi[end])
    jac_dynamics = view(jac, nlp.cinds[3], :)
    jac_stance = view(jac, nlp.cinds[4], :)
    jac_length = view(jac, nlp.cinds[5], :)

    # TODO: Calculate all the Jacobians
    #  TIP: You should call dynamics_jacobian!, and you probably won't need jac_dynamics
    #  TIP: You can write extra functions for the other constraints, or just do them here (they're pretty easy)
    #  TIP: Consider starting with ForwardDiff and then implement analytically (you won't get full points if you don't
    #       implement the Jacobians analytically)

    #println("size ini " , size(jac_init) , " " , size(jac_term))
    jac_init[:,1:n] = I(n)
    jac_term[:,end-n+1:end] = I(n)
    dynamics_jacobian!(nlp,jac,Z)
    #stance jacob
    switch = 1

    for k=1:N

        if switch == 1

            jac_stance[k,(k-1)*(n+m)+4] = 1.0
        else
            jac_stance[k,(k-1)*(n+m)+6] = 1.0

        end

        if k%5 == 0

            switch = switch*(-1)
        end

    end
    #println("here6")
    #length jacob
    for k = 1:N
        jac_length[2*k-1,(k-1)*(n+m)+1] = (Z[xi[k][1]] - Z[xi[k]][3])/norm(Z[xi[k]][1:2] - Z[xi[k]][3:4])
        jac_length[2*k-1,(k-1)*(n+m)+2] = (Z[xi[k][2]] - Z[xi[k]][4])/norm(Z[xi[k]][1:2] - Z[xi[k]][3:4])

        jac_length[2*k-1,(k-1)*(n+m)+3] = -(Z[xi[k][1]] - Z[xi[k]][3])/norm(Z[xi[k]][1:2] - Z[xi[k]][3:4])
        jac_length[2*k-1,(k-1)*(n+m)+4] = -(Z[xi[k][2]] - Z[xi[k]][4])/norm(Z[xi[k]][1:2] - Z[xi[k]][3:4])

        jac_length[2*k,(k-1)*(n+m)+1] = (Z[xi[k][1]] - Z[xi[k]][5])/norm(Z[xi[k]][1:2] - Z[xi[k]][5:6])
        jac_length[2*k,(k-1)*(n+m)+2] = (Z[xi[k][2]] - Z[xi[k]][6])/norm(Z[xi[k]][1:2] - Z[xi[k]][5:6])

        jac_length[2*k,(k-1)*(n+m)+5] = -(Z[xi[k][1]] - Z[xi[k]][5])/norm(Z[xi[k]][1:2] - Z[xi[k]][5:6])
        jac_length[2*k,(k-1)*(n+m)+6] = -(Z[xi[k][2]] - Z[xi[k]][6])/norm(Z[xi[k]][1:2] - Z[xi[k]][5:6])

    end

    #println("yo")

    return nothing
end
# %% codecell
@testset "Part (d): Constraint Jacobians" begin
    Z = randn(num_primals(nlp))
    n,m,N = size(nlp)
    jac = zeros(num_duals(nlp), num_primals(nlp))

    @testset "Dynamics Jacobian" begin
        jac_dyn = ForwardDiff.jacobian(x->dynamics_constraint!(nlp, zeros(eltype(x), num_duals(nlp)), x), Z)
        dynamics_jacobian!(nlp, jac, Z)
        @test jac[nlp.cinds[3], :] ≈ jac_dyn
    end

    @testset "Initial and Final Constraint" begin
        jac_c!(nlp, jac, Z)
        @test jac[1:n,1:n] ≈ I(n)
        @test jac[n+1:2n,end-n+1:end] ≈ I(n)
    end

    @testset "Stance Constraint" begin
        jac_stance = ForwardDiff.jacobian(x->stance_constraint!(nlp, zeros(eltype(x), num_duals(nlp)), x), Z)
        @test jac[nlp.cinds[4], :] ≈ jac_stance
    end

    @testset "Length Constraint" begin
        jac_length = ForwardDiff.jacobian(x->length_constraint!(nlp, zeros(eltype(x), num_duals(nlp)), x), Z)
        @test jac[nlp.cinds[5], :] ≈ jac_length
    end
end;
# %% markdown
# ## Part (e): Solve (0 pts)
# We now have all the pieces! Now let's set up the problem and check out the result.
#
# ### Problem Definition
# %% codecell
# Initial guess
Random.seed!(1)
Xguess = [x + 0.1*randn(length(x)) for x in Xref]
Uguess = [u + 0.1*randn(length(u)) for u in Uref]
Z0 = packZ(nlp, Xguess, Uguess);
# %% markdown
# ### Solve
# **NOTE**: If the solve fails (especially if you get an error about the restoration phase failing), try running it a couple more times. Sometimes Ipopt is a little finicky.
#
# **TIP**: Try solving with coarser tolerances at first (e.g. `c_tol = 1e-4, tol=1e-2`) while you dial it in so it doesn't take as long.
#
# **TIP**: With tolerances of `1e-6`, it takes about 90 iterations and converges to a cost of about 248.
# %% codecell
Z_sol, solver = solve(Z0, nlp, c_tol=1e-6, tol=1e-6) #1e-6 , 1e-6
# %% codecell
@testset "Part (e): Solve" begin
    Xsol,Usol = unpackZ(nlp,Z_sol)
    @test norm(Xsol[1] - nlp.x0) < 1e-6
    @test norm(Xsol[end] - nlp.xf) < 1e-6
    @test norm([x[4] for x in Xsol[nlp.modes .== 1]], Inf) < 1e-6
    @test norm([x[6] for x in Xsol[nlp.modes .== 2]], Inf) < 1e-6

    @test eval_f(nlp, Z_sol) < 250

    @test all(x->0.5 < x < 1.5, [norm(x[1:2] - x[3:4]) for x in Xsol[nlp.modes .== 1]])
    @test all(x->0.5 < x < 1.5, [norm(x[1:2] - x[5:6]) for x in Xsol[nlp.modes .== 2]])
end;
# %% markdown
# ## Visualizer
# %% codecell
vis = Visualizer()
set_mesh!(vis, model)
render(vis)
# %% codecell
visualize!(vis, nlp, Z_sol)
