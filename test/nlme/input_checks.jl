using Pumas, CSV, Test, Random
@testset "read_pumas vs DataFrame" begin
  df = DataFrame(CSV.File(example_data("event_data/CS1_IV1EST_PAR")))
  model = @model begin end
  param = NamedTuple()
  @test_throws ArgumentError("The second argument to fit was a DataFrame instead of a Population. Please use read_pumas to construct a Population from a DataFrame.") fit(model, df, param, Pumas.FOCEI())  
  @test_throws ArgumentError("The second argument to simobs was a DataFrame instead of a Population. Please use read_pumas to construct a Population from a DataFrame.") simobs(model, df, param)  
  @test_throws ArgumentError("The second argument to solve was a DataFrame instead of a Population. Please use read_pumas to construct a Population from a DataFrame.") solve(model, df, param)  
end

@testset "Dosage compartments" begin
    mdl_diffeq = @model begin
      @param begin
        θCL ∈ RealDomain(lower = 0.0, upper=1.0)
        θV  ∈ RealDomain(lower = 0.0)
        ω   ∈ RealDomain(lower = 0.0, upper=1.0)
      end

      @random begin
        η ~ Normal(0.0, ω)
      end

      @pre begin
        Ka = 1
        CL = θCL
        Vc  = θV
      end

      @vars begin
        μ := Central / Vc
        p := logistic(μ/10 + η)
      end

      @dynamics begin
          Depot' = -Ka*Depot
          Central' = Ka*Depot-CL/Vc*Central
      end

      @derived begin
          y ~ @. Bernoulli(p)
      end
    end
    mdl_diffeq_toofew = @model begin
      @param begin
        θCL ∈ RealDomain(lower = 0.0, upper=1.0)
        θV  ∈ RealDomain(lower = 0.0)
        ω   ∈ RealDomain(lower = 0.0, upper=1.0)
      end

      @random begin
        η ~ Normal(0.0, ω)
      end

      @pre begin
        CL = θCL
        Vc  = θV
      end

      @vars begin
        μ := Central / Vc
        p := logistic(μ/10 + η)
      end

      @dynamics begin
          Central' = -CL/Vc*Central
      end

      @derived begin
          y ~ @. Bernoulli(p)
      end
    end

    dr1 = DosageRegimen(100, time=0.0, cmt=1)
    dr2 = DosageRegimen(100, time=0.0, cmt=3)
    dr3 = DosageRegimen(100, time=0.0, cmt=:Central)
    dr4 = DosageRegimen(100, time=0.0, cmt=:Depot)
    dr5 = DosageRegimen(100, time=0.0, cmt=2)

    _t = obstimes = range(0.5, stop=24, step=0.5)

    par_init = (
      θCL = 0.3,
      θV  = 1.1,
      ω   = 0.1,
      )

    n = 5
   	# Check that a few different dosage regimens either errors or doesn't error
    pop_skeleton = [Subject(id=i, events=dr1, time=_t) for i in 1:n]
    pop_skeleton = [Subject(id=i, events=dr2, time=_t) for i in 1:n]
    @test_throws ArgumentError Pumas._check_dose_compartments(mdl_diffeq_toofew, pop_skeleton[1], par_init)
    pop_skeleton = [Subject(id=i, events=dr3, time=_t) for i in 1:n]
    Pumas._check_dose_compartments(mdl_diffeq_toofew, pop_skeleton[1], par_init)
    pop_skeleton = [Subject(id=i, events=dr4, time=_t) for i in 1:n]
    @test_throws ArgumentError Pumas._check_dose_compartments(mdl_diffeq_toofew, pop_skeleton[1], par_init)
    pop_skeleton = [Subject(id=i, events=DosageRegimen(dr1, dr4), time=_t) for i in 1:n]
    @test_throws ArgumentError Pumas._check_dose_compartments(mdl_diffeq_toofew, pop_skeleton[1], par_init)
    pop_skeleton = [Subject(id=i, events=DosageRegimen(dr1, dr5), time=_t) for i in 1:n]
    @test_throws ArgumentError Pumas._check_dose_compartments(mdl_diffeq_toofew, pop_skeleton[1], par_init)
    pop_skeleton = [Subject(id=i, events=DosageRegimen(dr4, dr1), time=_t) for i in 1:n]
    @test_throws ArgumentError Pumas._check_dose_compartments(mdl_diffeq_toofew, pop_skeleton[1], par_init)
    pop_skeleton = [Subject(id=i, events=DosageRegimen(dr5, dr1), time=_t) for i in 1:n]
    @test_throws ArgumentError Pumas._check_dose_compartments(mdl_diffeq_toofew, pop_skeleton[1], par_init)


  # Test that fit, simobs, solve errors for main functions when called on populations
  Random.seed!(123)
  # First for integer dosage regimen compartment
  pop_skeleton = [Subject(id=i, events=dr4, time=_t) for i in 1:n]
  pop_sim = simobs(mdl_diffeq, pop_skeleton, par_init, ensemblealg=EnsembleSerial())

  pop_est = Subject.(pop_sim)

  @test_throws ArgumentError fit(mdl_diffeq_toofew, pop_est, par_init, Pumas.FOCE())
  @test_throws ArgumentError simobs(mdl_diffeq_toofew, pop_est, par_init)
  @test_throws ArgumentError solve(mdl_diffeq_toofew, pop_est, par_init)

  # Second for symbol dosage regimen compartment
  pop_skeleton = [Subject(id=i, events=dr5, time=_t) for i in 1:n]
  pop_sim = simobs(mdl_diffeq, pop_skeleton, par_init, ensemblealg=EnsembleSerial())

  pop_est = Subject.(pop_sim)

  @test_throws ArgumentError fit(mdl_diffeq_toofew, pop_est, par_init, Pumas.FOCE())
  @test_throws ArgumentError simobs(mdl_diffeq_toofew, pop_est, par_init)
  @test_throws ArgumentError solve(mdl_diffeq_toofew, pop_est, par_init)
end
