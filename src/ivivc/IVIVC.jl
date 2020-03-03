module IVIVC

using Reexport
using RecipesBase
using OrdinaryDiffEq
using ForwardDiff
using CSV, DataFrames
using Parameters

@reexport using DataInterpolations, Optim, ..NCA

abstract type Ivivc end

Base.size(A::Ivivc) = size(A.subjects)
Base.length(A::Ivivc) = length(A.subjects)
Base.getindex(A::Ivivc, i) = A.subjects[i]
Base.setindex!(A::Ivivc, x, i) = A.subjects[i] = x

include("type.jl")
include("data_parsing.jl")
include("models.jl")
include("deconvo_methods.jl")
include("stats.jl")
include("plot_rec.jl")
include("utils.jl")
include("model_validation.jl")
include("main.jl")

export InVitroForm, InVitroData
export InVivoForm, InVivoData
export read_vitro, read_vivo, read_uir
export emax, emax_ng, weibull, double_weibull, makoid
export estimate_fdiss, get_avail_models, estimate_fabs
export estimate_uir, get_avail_vivo_models, to_csv
export calc_input_rate, wagner_nelson, IVIVCModel, predict_vivo, percentage_prediction_error
export loglikelihood, nullloglikelihood, dof, nobs, deviance, mss,
       rss, aic, aicc, bic, r2
end # module
