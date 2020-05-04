using Pumas, Test, CSV

@testset "One compartment intravenous bolus study" begin
  #No. of subjects= 100, Dose = 100 or 250mg,DV=Plasma concentration, ug/ml
  #Time = hrs, CL = L/hr, V=L

  pkdata = CSV.read(example_data("event_data/CS1_IV1EST_PAR"))
  pkdata[!,:dv] .= pkdata[!, :CONC]
  pkdata[!,:CMT] .= 1
  data = read_pumas(pkdata,cvs = [:AGE, :WT, :SCR, :CLCR], dvs = [:dv],
                        id=:ID, time=:TIME, amt=:AMT, evid=:MDV, cmt=:CMT)


  @testset "proportional error model" begin
    mdl_proportional = Dict()
    mdl_proportional["analytical"] = @model begin
      @param begin
        θ ∈ VectorDomain(2, lower=[0.0,0.0], upper=[20.0,20.0])
        Ω ∈ PDiagDomain(2)
        σ_prop ∈ RealDomain(lower=0.0001)
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        CL = θ[1] * exp(η[1])
        Vc = θ[2] * exp(η[2])
      end

      @vars begin
        conc = Central / Vc
      end

      @dynamics Central1

      @derived begin
        dv ~ @. Normal(conc, abs(conc)*σ_prop)
      end
    end

    mdl_proportional["solver"] = @model begin
      @param begin
        θ ∈ VectorDomain(2, lower=[0.00,0.00], upper=[20.0,20.0])
        Ω ∈ PDiagDomain(2)
        σ_prop ∈ RealDomain(lower=0.0001)
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        CL = θ[1] * exp(η[1])
        Vc = θ[2] * exp(η[2])
      end

      @vars begin
        conc = Central / Vc
      end

      @dynamics begin
         Central' =  - (CL/Vc)*Central
      end

      @derived begin
        dv ~ @. Normal(conc, abs(conc)*σ_prop)
      end
    end

    param_proportional = (
      θ = [0.6, 10.0],
      Ω = Diagonal([0.04, 0.01]),
      σ_prop = sqrt(0.1)
    )

    @testset "FO estimation of $dyntype model" for dyntype in ("analytical", "solver")
      result = fit(mdl_proportional[dyntype], data, param_proportional, Pumas.FO())
      param = coef(result)

      @test param.θ      ≈ [3.5592e-01, 8.5888e+00] rtol=1e-3
      @test param.Ω.diag ≈ [3.0186e-01, 4.2789e-01] rtol=1e-3
      @test param.σ_prop ≈ sqrt(9.9585e-02)         rtol=1e-3
    end

    @testset "FOCE estimation of $dyntype model" for dyntype in ("analytical", "solver")
      @test_throws ArgumentError deviance(mdl_proportional[dyntype], data, param_proportional, Pumas.FOCE())
    end

    @testset "FOCEI estimation of $dyntype model" for dyntype in ("analytical", "solver")
      result = fit(mdl_proportional[dyntype], data, param_proportional, Pumas.FOCEI())
      param = coef(result)

      @test param.θ      ≈ [3.91e-01, 7.56e+00] rtol=1e-3
      @test param.Ω.diag ≈ [1.60e-01, 1.68e-01] rtol=3e-3
      @test param.σ_prop ≈ sqrt(1.01e-1)        rtol=1e-3
    end

    @testset "LaplaceI estimation of $dyntype model" for dyntype in ("analytical", "solver")
      result = fit(mdl_proportional[dyntype], data, param_proportional, Pumas.LaplaceI())
      param = coef(result)

      @test param.θ      ≈ [3.7400e-01, 7.5009e+00] rtol=1e-3
      @test param.Ω.diag ≈ [1.5123e-01, 1.6798e-01] rtol=3e-3
      @test param.σ_prop ≈ sqrt(1.0025e-01)         rtol=1e-3
    end
  end

  @testset "proportional+additive error model" begin
    mdl_proportional_additive = Dict()
    mdl_proportional_additive["analytical"] = @model begin
      @param begin
       θ ∈ VectorDomain(2, lower=[0.0,0.0], upper=[20.0,20.0])
       Ω ∈ PDiagDomain(2)
       σ_add ∈ RealDomain(lower=0.0001)
       σ_prop ∈ RealDomain(lower=0.0001)
      end

      @random begin
       η ~ MvNormal(Ω)
      end

      @pre begin
       CL = θ[1] * exp(η[1])
       Vc = θ[2] * exp(η[2])
      end

      @vars begin
       conc = Central / Vc
      end

      @dynamics Central1

      @derived begin
       dv ~ @. Normal(conc, sqrt((conc*σ_prop)^2 + σ_add^2))
      end
    end

    mdl_proportional_additive["solver"] = @model begin
      @param begin
       θ ∈ VectorDomain(2, lower=[0.0,0.0], upper=[20.0,20.0])
       Ω ∈ PDiagDomain(2)
       σ_add ∈ RealDomain(lower=0.0001)
       σ_prop ∈ RealDomain(lower=0.0001)
      end

      @random begin
       η ~ MvNormal(Ω)
      end

      @pre begin
       CL = θ[1] * exp(η[1])
       Vc = θ[2] * exp(η[2])
      end

      @vars begin
       conc = Central / Vc
      end

      @dynamics begin
         Central' =  - (CL/Vc)*Central
      end

      @derived begin
       dv ~ @. Normal(conc, sqrt((conc*σ_prop)^2 + σ_add^2))
      end
    end

    param_proportional_additive = (
      θ = [0.6, 10.0],
      Ω = Diagonal([0.04, 0.01]),
      σ_add = sqrt(2.0),
      σ_prop = sqrt(0.1)
    )

    @testset "FO estimation of $dyntype model" for dyntype in ("analytical", "solver")
      result = fit(mdl_proportional_additive[dyntype], data, param_proportional_additive, Pumas.FO())
      param = coef(result)

      @test param.θ      ≈ [4.2224e-01, 6.5691e+00] rtol=1e-3
      @test param.Ω.diag ≈ [2.1552e-01, 2.2674e-01] rtol=5e-3
      @test param.σ_add  ≈ sqrt(7.9840e+00)         rtol=1e-3
      @test param.σ_prop ≈ sqrt(2.0831e-02)         rtol=1e-3
    end

    @testset "FOCE estimation of $dyntype model" for dyntype in ("analytical", "solver")
      @test_throws ArgumentError deviance(mdl_proportional_additive[dyntype], data, param_proportional_additive, Pumas.FOCE())
    end

    @testset "FOCEI estimation of $dyntype model" for dyntype in ("analytical", "solver")
      result = fit(mdl_proportional_additive[dyntype], data, param_proportional_additive, Pumas.FOCEI())
      param = coef(result)

      @test param.θ      ≈ [4.17e-01, 7.16e+00] rtol=1e-3
      @test param.Ω.diag ≈ [1.71e-01, 1.98e-01] rtol=5e-3
      @test param.σ_add  ≈ sqrt(8.57e+00)       rtol=1e-3
      @test param.σ_prop ≈ sqrt(1.47e-02)       rtol=3e-3
    end

    @testset "LaplaceI estimation of $dyntype model" for dyntype in ("analytical", "solver")
      result = fit(mdl_proportional_additive[dyntype], data, param_proportional_additive, Pumas.LaplaceI())
      param = coef(result)

      @test param.θ      ≈ [4.1156e-01, 7.1442e+00] rtol=1e-3
      @test param.Ω.diag ≈ [1.7226e-01, 1.9856e-01] rtol=5e-3
      @test param.σ_add  ≈ sqrt(8.5531e+00)         rtol=1e-3
      @test param.σ_prop ≈ sqrt(1.4630e-02)         rtol=3e-3
    end
  end
end
