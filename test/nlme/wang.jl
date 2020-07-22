using Test
using Pumas, CSV

@testset "Models from the Wang paper" begin

  df   = DataFrame(CSV.File(example_data("wang")), copycols=true)
  # Add initial events following Wang's model
  df[!,:amt] .= missing
  df[!,:cmt] .= missing
  allowmissing!(df)
  df_doses = DataFrame(id=unique(df.id), time=0.0, dv=missing, amt=10.0, cmt=1)
  df = vcat(df, df_doses)
  sort!(df, [:id, :time])

  data = read_pumas(df)
  n = sum(subject -> length(subject.observations.dv), data)

  @testset "Additive error model" begin

    wang_additive = @model begin
      @param begin
        θ ∈ RealDomain(init=0.5)
        Ω ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
        σ ∈ RealDomain(init=sqrt(0.1))
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        CL = θ * exp(η[1])
        Vc = 1.0
      end

      @vars begin
        conc = Central / Vc
      end

      @dynamics Central1

      @derived begin
        dv ~ @. Normal(conc, σ)
      end
    end

    param = init_param(wang_additive)

    @test -2*loglikelihood(wang_additive, data, param, Pumas.FO()) - n*log(2π)  ≈  0.026 atol = 1e-3
    @test -2*loglikelihood(wang_additive, data, param, Pumas.FOCE()) - n*log(2π) ≈ -2.059 atol = 1e-3
  end

  @testset "Proportional error model" begin

    wang_proportional = @model begin
      @param begin
        θ ∈ RealDomain(init=0.5)
        Ω ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
        σ ∈ RealDomain(lower=0.001, upper=10.0, init=sqrt(0.1))
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        CL = θ * exp(η[1])
        Vc = 1.0
      end

      @vars begin
        conc = Central / Vc
      end

      @dynamics Central1

      @derived begin
        dv ~ @. Normal(conc, conc*σ)
      end
    end

    param = init_param(wang_proportional)

    @test -2*loglikelihood(wang_proportional, data, param, Pumas.FO()) - n*log(2π)    ≈ 39.213 atol = 1e-3
    @test_throws ArgumentError loglikelihood(wang_proportional, data, param, Pumas.FOCE())
    @test -2*loglikelihood(wang_proportional, data, param, Pumas.FOCEI()) - n*log(2π) ≈ 39.458 atol = 1e-3
  end

  @testset "Exponential error model" begin
    wang_exponential = @model begin
      @param begin
        θ ∈ RealDomain(init=0.5)
        Ω ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
        σ ∈ RealDomain(lower=0.001, upper=10.0, init=sqrt(0.1))
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        CL = θ[1] * exp(η[1])
        Vc = 1.0
      end

      @vars begin
        conc = Central / Vc
      end

      @dynamics Central1

      @derived begin
        dv ~ @. LogNormal(log(conc), σ)
      end
    end

    # NONMEM computes the exponential error model differently. Instead of deriving the
    # FO/FOCE(I) approximations from the Laplace approximation, NONMEM's version is based
    # on linearization. The two approaches are only equivalent when the model is Gaussian.
    # Hence we test by log transforming the model

    # First we load a new verison of data and log transform dv
    _df = DataFrame(CSV.File(example_data("wang")))
    _df[!,:dv] = log.(_df[!,:dv])
    _df[!,:amt] .= missing
    _df[!,:cmt] .= missing
    data_log = read_pumas(_df)

    # Add initial events following Wang's model
    for i in eachindex(data_log)
      push!(data_log[i].events, Pumas.Event(10.0, 0.0, 1, 1))
    end

    # Define a model similar to wang_exponential but using Normal instead of LogNormal since
    # dv has been log transformed
    wang_exponential_log = @model begin
      @param begin
        θ ∈ RealDomain(init=0.5)
        Ω ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
        σ ∈ RealDomain(lower=0.01, upper=1.0, init=0.1)
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        CL = θ[1] * exp(η[1])
        Vc = 1.0
      end

      @vars begin
        conc = Central / Vc
      end

      @dynamics Central1

      @derived begin
        dv ~ @. Normal(log(conc), σ)
      end
    end

    # Now use density transformations to test that the two formulations are equivalent
    param = init_param(wang_exponential)

    # Compute the correction term which is sum(log(dg⁻¹(y)/dy)) where g=exp in our case so g⁻¹=log
    correction_term = sum(sum(log.(d.observations.dv)) for d in data)
    @test loglikelihood(wang_exponential, data, param, Pumas.FO())    ≈ loglikelihood(wang_exponential_log, data_log, param, Pumas.FO())    - correction_term
    @test loglikelihood(wang_exponential, data, param, Pumas.FOCE())  ≈ loglikelihood(wang_exponential_log, data_log, param, Pumas.FOCE())  - correction_term
    @test loglikelihood(wang_exponential, data, param, Pumas.FOCEI()) ≈ loglikelihood(wang_exponential_log, data_log, param, Pumas.FOCEI()) - correction_term
  end
end
