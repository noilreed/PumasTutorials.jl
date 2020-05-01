module Pumas

using FiniteDiff, Reexport, StatsBase,
      StaticArrays, Distributed, LabelledArrays, GLM,
      TreeViews, CSV, ForwardDiff, DiffResults, Optim, PDMats,
      Missings, RecipesBase, RecursiveArrayTools, Quadrature,
      Statistics, DiffEqSensitivity
using LinearAlgebra
using Base.Threads # for bootstrap
using AdvancedHMC: DiagEuclideanMetric, Hamiltonian, NUTS, Leapfrog, find_good_stepsize, StanHMCAdaptor, MassMatrixAdaptor, StepSizeAdaptor, MultinomialTS, GeneralisedNoUTurn
using StatsFuns: logistic

using DataStructures: OrderedDict, OrderedSet
using MacroTools
using ModelingToolkit

import MCMCChains: Chains

import DiffResults: DiffResult

import DataInterpolations

@reexport using OrdinaryDiffEq, Unitful
@reexport using Distributions, DataFrames

const Numeric = Union{AbstractVector{<:Number}, Number}

function opt_minimizer end
include("nca/NCA.jl")
include("ivivc/IVIVC.jl")

include("data_parsing/data_types.jl")
include("data_parsing/data_read.jl")

include("dsl/model_macro.jl")

include("models/params.jl")
include("models/simulated_observations.jl")
include("models/model_api.jl")
include("models/model_utils.jl")

include("estimation/transforms.jl")
include("estimation/likelihoods.jl")
include("estimation/bootstrap.jl")
include("estimation/bayes.jl")
include("estimation/diagnostics.jl")
include("estimation/gsa.jl")
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

export Subject, Population, DosageRegimen
export PumasModel, init_param, init_randeffs, sample_randeffs
export simobs, pre
export tad, eventnum
export conditional_nll
export wresiduals, empirical_bayes
export ηshrinkage, ϵshrinkage
export read_pumas, example_data
export @model, @nca
export infer, inspect

export expectation, KoopmanExpectation, MonteCarloExpectation

# From LinearAlgebra
export diagm, Diagonal, I

# From Statistics
export mean, std, var

# From StatsBase
export aic, bic, coef, coeftable, deviance, fit, informationmatrix, predict, stderror, vcov, bootstrap

# From StatsFuns
export logistic

# From SensitivityDiffEq
export gsa

end # module
