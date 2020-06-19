module Pumas

using FiniteDiff, Reexport, StatsBase,
      StaticArrays, Distributed, LabelledArrays,
      TreeViews, CSV, ForwardDiff, DiffResults, Optim, PDMats,
      Missings, RecipesBase, RecursiveArrayTools, Quadrature,
      Statistics, DiffEqSensitivity, QuantileRegressions,
      DiffEqJump
using LinearAlgebra
using Base.Threads # for bootstrap
using AdvancedHMC: DiagEuclideanMetric, Hamiltonian, NUTS, Leapfrog, find_good_stepsize, StanHMCAdaptor, MassMatrixAdaptor, StepSizeAdaptor, MultinomialTS, GeneralisedNoUTurn
using StatsFuns: logistic

using DataStructures: OrderedDict, OrderedSet
using MacroTools
using ModelingToolkit

import Base: keys

import MCMCChains: Chains

import DiffResults: DiffResult

import DataInterpolations, ExponentialUtilities

@reexport using OrdinaryDiffEq, Unitful
@reexport using Distributions, DataFrames

const Numeric = Union{AbstractVector{<:Number}, Number}

function opt_minimizer end
include("nca/NCA.jl")
include("ivivc/IVIVC.jl")

include("data_parsing/io.jl")

include("dsl/model_macro.jl")

include("models/params.jl")
include("models/simulated_observations.jl")
include("models/model_api.jl")
include("models/model_utils.jl")

include("estimation/transforms.jl")
include("estimation/likelihoods.jl")
include("estimation/inference.jl")
include("estimation/bayes.jl")
include("estimation/diagnostics.jl")
include("estimation/gsa.jl")
include("estimation/vpc.jl")
include("estimation/show.jl")

include("analytical_solutions/standard_models.jl")
include("analytical_solutions/analytical_problem.jl")
include("analytical_solutions/analytical_solution_type.jl")

include("simulate_methods/utils.jl")
include("simulate_methods/diffeqs.jl")
include("simulate_methods/analytical.jl")

include("uq/expectation.jl")

include("plotting/plotting.jl")

@reexport using .NCA

example_data(filename) = joinpath(joinpath(@__DIR__, ".."),"examples/"*filename*".csv")

export Subject, Population, DosageRegimen, TimeToEvent
export PumasModel, init_param, init_randeffs, sample_randeffs
export simobs, pre
export tad, eventnum
export conditional_nll
export wresiduals, empirical_bayes
export ηshrinkage, ϵshrinkage
export read_pumas, example_data
export @model, @nca
export infer, inspect, vpc

export expectation, KoopmanExpectation, MonteCarloExpectation

# From LinearAlgebra
export diagm, Diagonal, I

# From Statistics
export mean, std, var

# From StatsBase
export aic, bic, coef, coeftable, deviance, fit, informationmatrix, predict, stderror, vcov, bootstrap

# From StatsFuns
export logistic

# From DiffEqSensitivity
export gsa

# Print summary license intormation after new installations and upgrades. This is implemented
# by printing the info from the toplevel such that it's triggered only during precompilation.
# For ordinary users, procompilation will mostly happen after updates and new installation but
# will also happen sometimes when the user switches environment.
function print_license()
    printstyled("Important Note:", bold=true)
    print("""
 Pumas.jl is a proprietary package. It is free to use for non-commercial
academic teaching and research purposes. For commercial users, license fees apply. Please
refer to End User License Agreement (https://juliacomputing.com/eula) for details. Please
contact sales@juliacomputing.com for purchase.
""")
end
print_license()

end # module
