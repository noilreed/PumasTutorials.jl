using Test, SafeTestsets
using Pumas, StatsBase

if group == "All" || group == "NLME_Basic"
  @time @safetestset "Maximum-likelihood interface" begin
    @time @safetestset "Types (constructors, api, etc...)" begin
      include("types.jl")
    end
    @time @safetestset "Naive estimation" begin
      include("single_subject.jl")
    end
    @time @safetestset "Show methods" begin
      include("show.jl")
    end
    @time @safetestset "Soft failing infer" begin
      include("infer_error.jl")
    end
    @time @safetestset "VPC" begin
      include("vpc.jl")
    end
  end
end

if group == "All" || group == "NLME_ML1"
  @time @safetestset "Maximum-likelihood models 1" begin
    @time @safetestset "Theophylline" begin
      include("theophylline.jl")
    end
    @time @safetestset "Analytical solutions" begin
      include("analytical_solutions.jl")
    end
  end
end

if group == "All" || group == "NLME_ML2"
  @time @safetestset "Maximum-likelihood models 2" begin
    @time @safetestset "BOV" begin
      include("bov.jl")
    end
    @time @safetestset "Poisson" begin
      include("poisson_model.jl")
    end
    @time @safetestset "Negative binomial" begin
      include("negativebinomial.jl")
    end
    @time @safetestset "Ordinal" begin
      include("ordinal.jl")
    end
    @time @safetestset "Bolus" begin
      include("bolus.jl")
    end
    @time @safetestset "Missing observations" begin
      include("missings.jl")
    end
    @time @safetestset "Global Sensitivity Analysis" begin
      include("gsa.jl")
    end
  end
end

if group == "All" || group == "NLME_ML3"
  @time @safetestset "Maximum-likelihood models 3" begin
    @time @safetestset "Time-varying covariates" begin
      include("tvcov.jl")
    end
    @time @safetestset "Multiple dependent variables" begin
      include("mdv.jl")
    end
  end
end

if group == "All" || group == "NLME_ML4"
  @time @safetestset "Maximum-likelihood models 4" begin
    @time @safetestset "Simple Model" begin
      include("simple_model.jl")
    end
    @time @safetestset "Simple Model (logistic regression)" begin
      include("logistic.jl")
    end
    @time @safetestset "Simple Model with T-distributed error model" begin
      include("simple_model_tdist.jl")
    end
    @time @safetestset "Simple Model with Gamma-distributed error model" begin
      include("simple_model_gamma.jl")
    end
    @time @safetestset "Simple Model with Exponential-distributed error model" begin
      include("simple_model_exponential.jl")
    end
    @time @safetestset "Simple Model disagnostics" begin
      include("simple_model_diagnostics.jl")
    end
    @time @safetestset "Theophylline NLME.jl" begin
      include("theop_nlme.jl")
    end
    @time @safetestset "Wang" begin
      include("wang.jl")
    end
    @time @safetestset "Dose control parameters" begin
      include("dosecontrolparameters.jl")
    end
  end
end

if group == "ALL" || group == "NLME_ML5"
  @time @safetestset "Maximum-likelihood models 5" begin
    @time @safetestset "Information matrix" begin
      include("information.jl")
    end
    @time @safetestset "Medium size ODE system (HCV model)" begin
      include("hcv.jl")
    end
    @time @safetestset "Time-to-event" begin
      include("timetoevent.jl")
    end
  end
end

if group == "All" || group == "NLME_Bayes"
  @time @safetestset "Bayesian models" begin
    include("bayes.jl")
  end
end

@testset "Check that all NLME test files are run" begin
  dirfiles     = Set(filter(t -> t ∉ ("runtests.jl", "testmodels.jl"), readdir(@__DIR__())))
  includefiles = Set(map(t -> first(match(r"include\(\"(.*\.jl)", t).captures),
    filter(t -> occursin(r"include\(", t) && occursin(r"\.jl", t),
      readlines(joinpath(@__DIR__(), "runtests.jl")))))
  @test dirfiles == includefiles
end

