using Pumas, Test, CSV, Random

@testset "nmtran" begin
  data = read_pumas(example_data("event_data/data1"))
  @test_nowarn show(data)

  @test getproperty.(data[1].events, :time) == 0:12:36

  @testset "Subject comparison" begin
    datacopy = deepcopy(data)
    @test data == datacopy
    @test hash(data) == hash(datacopy)
  end

  for ev in data[1].events
    @test ev.amt == data[1].events[1].amt
    @test ev.evid == data[1].events[1].evid
    @test ev.cmt == data[1].events[1].cmt
    @test ev.rate == data[1].events[1].rate
    @test ev.ss == data[1].events[1].ss
    @test ev.ii == data[1].events[1].ii
    @test ev.rate_dir == data[1].events[1].rate_dir
    @test ev.cmt == 1
  end

  @testset "Dosage Regimen" begin
    gen_subject = Subject(events = DosageRegimen(100, ii = 12, addl = 3))
    @test data[1].id == gen_subject.id
    @test data[1].covariates == gen_subject.covariates
    @test data[1].events == gen_subject.events
    e1 = DosageRegimen(100, ii = 24, addl = 6)
    e2 = DosageRegimen(50,  ii = 12, addl = 13)
    e3 = DosageRegimen(200, ii = 24, addl = 2)
    @test isa(DosageRegimen(e1, e2, e3), DosageRegimen)
  end
end

@testset "Time Variant Covariates" begin
  df = DataFrame(CSV.File(example_data("time_varying_covariates")))
  df[!,:amt] .= missing
  data = read_pumas(df, covariates = [:weight, :dih])
  @test data[1].covariates(0).dih == 2
end

@testset "Chronological Observations" begin
  data = DataFrame(id = 1, time = [0.0, 1, 2], dv = rand(3), evid = 0, amt = missing)
  @test isa(read_pumas(data), Population)
  append!(data, DataFrame(id = 1, time = 0.5, dv = rand(), evid = 0, amt=missing))
  @test_throws Pumas.PumasDataError("Time is not monotonically increasing between reset dose events (evid=3 or evid=4)") read_pumas(data)
  @test_throws Pumas.PumasDataError("Time is not monotonically increasing within a manually constructed subject") Subject(observations=DataFrame(x=[2:3;], time=1:-1:0))
end

@testset "event_data" begin
  data = DataFrame(id = 1, time = [0, 1, 2, 2], amt = zeros(4), dv = missing, evid = 1)
  @test_throws Pumas.PumasDataError("Some dose-related data items must be non-zero when evid = 1") read_pumas(data)
  @test isa(read_pumas(data, event_data = false), Population)
  @test isa(DosageRegimen(100, rate = -2, cmt = 2, ii = 24, addl = 3), DosageRegimen)
  @test isa(DosageRegimen(0, time = 50, evid = 3, cmt = 2), DosageRegimen)
  @test_throws ArgumentError("amt must be 0 for evid = 3") DosageRegimen(1, time = 50, evid = 3, cmt = 2)
  choose_covariates() = (isPM = rand(["yes", "no"]), Wt = rand(55:80))
  generate_population(events, nsubs = 24) =
    Population([ Subject(id = i, events = events, covariates = choose_covariates()) for i ∈ 1:nsubs ])
  firstdose = DosageRegimen(100, ii = 12, addl = 3, rate = 50)
  seconddose = DosageRegimen(0, time = 50, evid = 3, cmt = 2)
  thirddose = DosageRegimen(120, time = 54, ii = 16, addl = 2)
  ev = reduce(DosageRegimen, [firstdose, seconddose, thirddose])
  @test isa(generate_population(ev), Population)
end

@testset "Population Constructors" begin
  e1 = DosageRegimen(100, ii = 24, addl = 6)
  e2 = DosageRegimen(50,  ii = 12, addl = 13)
  e3 = DosageRegimen(200, ii = 24, addl = 2)
  s1 = map(i -> Subject(id = i, events = e1), 1:5)
  s2 = map(i -> Subject(id = i, events = e2), 6:8)
  s3 = map(i -> Subject(id = i, events = e3), 9:10)
  pop1 = Population(s1)
  pop2 = Population(s2)
  pop3 = Population(s3)
  @test Population(s1, s2, s3) == Population(pop1, pop2, pop3)
end

@testset "Missing Observables" begin
  df = DataFrame(
    id   = 1,
    time = [0, 1, 2, 4],
    evid = [1, 1, 0, 0],
    dv1  = [missing, missing, 1, missing],
    dv2  = [missing, missing, missing, 2],
    x    = [missing, missing, 0.5, 0.75],
    amt  = [0.25, 0.25, 0, 0],
    cmt  = 1)
  data = read_pumas(df,
    covariates = [:x],
    observations = [:dv1, :dv2])
  @test isa(data, Population)
  isa(data[1].observations[1], NamedTuple{(:dv1,:dv2),NTuple{2, Union{Missing,Float64}}})

  df = DataFrame(id=[1], time=0.0, dv=[0.0], evid=0, cv1=1, cv2=missing, amt=missing)
  @test_throws Pumas.PumasDataError("covariate cv2 for subject with ID 1 had no non-missing values") read_pumas(df, covariates=[:cv2])
  @test_throws Pumas.PumasDataError("covariate cv2 for subject with ID 1 had no non-missing values")  read_pumas(df, covariates=[:cv1, :cv2])
end

@testset "DosageRegimen (w/ or w/o offset)" begin
  e1 = DosageRegimen(100, ii = 24, addl = 6)
  e2 = DosageRegimen(50, ii = 12, addl = 13)
  events = DosageRegimen(e1, e2)
  @test events.data[!,:time] == [0, 0]
  events = DosageRegimen(e1, e2, offset = 10)
  @test events.data[!,:time] == [0, 178]
end

@testset "DataFrames Constructors" begin
  e1 = DosageRegimen(100, ii = 24, addl = 6)
  e2 = DosageRegimen(50, ii = 12, addl = 13)
  e3 = DosageRegimen(200, ii = 24, addl = 2)
  events = reduce(DosageRegimen, [e1, e2, e3])
  data = DataFrame(events, true)
  @test size(data, 1) == 24
end

@testset "MDV" begin
  data = DataFrame(id = 1, time = 0.0, amt = missing, dv = 0, evid = 0, mdv = 1)
  output = read_pumas(data)
  @test ismissing(output[1].observations.dv[1])
end

@testset "amt = rate * duration" begin
  e1 = DosageRegimen(100, rate = 25)
  e2 = DosageRegimen(100, duration = 4)
  @test e1.data == e2.data
  @test_throws AssertionError DosageRegimen(100, duration = 4, rate = 20)
end

@testset "Test show methods for DosageRegimen" begin
  dr = DosageRegimen(2, ii = 24, addl=2)

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), dr) == """
DosageRegimen
│ Row │ time    │ cmt   │ amt     │ evid │ ii      │ addl  │ rate    │ duration │ ss   │
│     │ Float64 │ Int64 │ Float64 │ Int8 │ Float64 │ Int64 │ Float64 │ Float64  │ Int8 │
├─────┼─────────┼───────┼─────────┼──────┼─────────┼───────┼─────────┼──────────┼──────┤
│ 1   │ 0.0     │ 1     │ 2.0     │ 1    │ 24.0    │ 2     │ 0.0     │ 0.0      │ 0    │"""

  @test sprint((io, t) -> show(io, MIME"text/html"(), t), dr) == """
<table class="data-frame"><thead><tr><th></th><th>time</th><th>cmt</th><th>amt</th><th>evid</th><th>ii</th><th>addl</th><th>rate</th><th>duration</th><th>ss</th></tr><tr><th></th><th>Float64</th><th>Int64</th><th>Float64</th><th>Int8</th><th>Float64</th><th>Int64</th><th>Float64</th><th>Float64</th><th>Int8</th></tr></thead><tbody><p>1 rows × 9 columns</p><tr><th>1</th><td>0.0</td><td>1</td><td>2.0</td><td>1</td><td>24.0</td><td>2</td><td>0.0</td><td>0.0</td><td>0</td></tr></tbody></table>"""
end

@testset "Dataset format checks" begin
  # no error
  df1 = DataFrame(id = [1,1,1,1,1,2,2,2,2,2],
                time = [0,1,2,3,4,0,1,2,3,4],
                amt=[10,0,0,0,0,10,0,0,0,0],
                cmt=[1,2,2,2,2,1,2,2,2,2],
                evid=[1,0,0,0,0,1,0,0,0,0],
                dv=[missing,8,6,4,2,missing,8,6,4,2],
                age=[45,45,45,45,45,50,50,50,50,50],
                sex = ["M","M","M","M","M","F","F","F","F","F"],
                crcl =[90,85,75,72,70,110,110,110,110,110])
  @test_nowarn read_pumas(df1, covariates=[:age,:sex,:crcl])

  # dv observation is present at the time of dose
  df2 = DataFrame(id = [1,1,1,1,1,2,2,2,2,2],
                time = [0,1,2,3,4,0,1,2,3,4],
                amt=[10,0,0,0,0,10,0,0,0,0],
                cmt=[1,2,2,2,2,1,2,2,2,2],
                evid=[1,0,0,0,0,1,0,0,0,0],
                dv=[10,8,6,4,2,10,8,6,4,2],
                age=[45,45,45,45,45,50,50,50,50,50],
                sex = ["M","M","M","M","M","F","F","F","F","F"],
                crcl =[90,85,75,72,70,110,110,110,110,110])
  @test_throws Pumas.PumasDataError read_pumas(df2, covariates=[:age,:sex,:crcl])

  # We expect the dv column to be of numeric type.
  df4 = DataFrame(id = [1,1,1,1,1,2,2,2,2,2],
                time = [0,1,2,3,4,0,1,2,3,4],
                amt=[10,missing,missing,missing,missing,10,missing,missing,missing,missing],
                cmt=[1,2,2,2,2,1,2,2,2,2],
                evid=[1,0,0,0,0,1,0,0,0,0],
                dv=[missing,8,6,4,"<LOQ",missing,8,6,4,2],
                age=[45,45,45,45,45,50,50,50,50,50],
                sex = ["M","M","M","M","M","F","F","F","F","F"],
                crcl =[90,85,75,72,70,110,110,110,110,110])
  @test_throws Pumas.PumasDataError read_pumas(df4, amt=:amt, covariates=[:age,:sex,:crcl])

  # evid is missing with event_data = true (default)
  df5 = DataFrame(id = [1,1,1,1,1,2,2,2,2,2],
                time = [0,1,2,3,4,0,1,2,3,4],
                amt=[10,missing,missing,missing,missing,10,missing,missing,missing,missing],
                cmt=[1,2,2,2,2,1,2,2,2,2],
                #evid=[1,0,0,0,0,1,0,0,0,0],
                dv=[missing,8,6,4,2,missing,8,6,4,2],
                age=[45,45,45,45,45,50,50,50,50,50],
                sex = ["M","M","M","M","M","F","F","F","F","F"],
                crcl =[90,85,75,72,70,110,110,110,110,110])
  # @test_throws Pumas.PumasDataError read_pumas(df5, amt=:amt, covariates=[:age,:sex,:crcl])

  # cmt column should be positive
  df6 = DataFrame(id = [1,1,1,1,1,2,2,2,2,2],
                time = [0,1,2,3,4,0,1,2,3,4],
                amt=[10,missing,missing,missing,missing,10,missing,missing,missing,missing],
                cmt=[-1,2,2,2,2,-1,2,2,2,2],
                evid=[1,0,0,0,0,1,0,0,0,0],
                dv=[missing,8,6,4,2,missing,8,6,4,2],
                age=[45,45,45,45,45,50,50,50,50,50],
                sex = ["M","M","M","M","M","F","F","F","F","F"],
                crcl =[90,85,75,72,70,110,110,110,110,110])
  @test_throws Pumas.PumasDataError read_pumas(df6, amt=:amt, covariates=[:age,:sex,:crcl])

  #  We expect the amt column to be of numeric type.
  df7 = DataFrame(id = [1,1,1,1,1,2,2,2,2,2],
                time = [0,1,2,3,4,0,1,2,3,4],
                amt=["10","","","","",10,"","","",""],
                cmt=["Depot","Central","Central","Central","Central","Depot","Central","Central","Central","Central"],
                evid=[1,0,0,0,0,1,0,0,0,0],
                dv=[missing,8,6,4,2,missing,8,6,4,2],
                age=[45,45,45,45,45,50,50,50,50,50],
                sex = ["M","M","M","M","M","F","F","F","F","F"],
                crcl =[90,85,75,72,70,110,110,110,110,110])
  @test_throws Pumas.PumasDataError read_pumas(df7, amt=:amt, covariates=[:age,:sex,:crcl])

  # dataset has column addl but not ii
  df8 = DataFrame(id = [1,1,1,1,1,2,2,2,2,2],
                time = [0,1,2,3,4,0,1,2,3,4],
                 amt=[10,0,0,0,0,10,0,0,0,0],
                addl = [5,0,0,0,0,5,0,0,0,0],
                cmt=["Depot","Central","Central","Central","Central","Depot","Central","Central","Central","Central"],
                evid=[1,0,0,0,0,1,0,0,0,0],
                dv=[missing,8,6,4,2,missing,8,6,4,2],
                age=[45,45,45,45,45,50,50,50,50,50],
                sex = ["M","M","M","M","M","F","F","F","F","F"],
                crcl =[90,78,75,72,70,110,110,110,110,110])
  @test_throws Pumas.PumasDataError read_pumas(df8, amt=:amt, covariates=[:age,:sex,:crcl])

  # ii must be positive for addl > 0
  df9 = DataFrame(id = [1,1,1,1,1,2,2,2,2,2],
                time = [0,1,2,3,4,0,1,2,3,4],
                 amt=[10,0,0,0,0,10,0,0,0,0],
                addl = [5,0,0,0,0,5,0,0,0,0],
                ii = [0,0,0,0,0,0,0,0,0,0],
                cmt=["Depot","Central","Central","Central","Central","Depot","Central","Central","Central","Central"],
                evid=[1,0,0,0,0,1,0,0,0,0],
                dv=[missing,8,6,4,2,missing,8,6,4,2],
                age=[45,45,45,45,45,50,50,50,50,50],
                sex = ["M","M","M","M","M","F","F","F","F","F"],
                crcl =[90,78,75,72,70,110,110,110,110,110])
  @test_throws Pumas.PumasDataError read_pumas(df9, amt=:amt, covariates=[:age,:sex,:crcl])

  # addl must be positive for ii > 0
  df10 = DataFrame(id = [1,1,1,1,1,2,2,2,2,2],
                time = [0,1,2,3,4,0,1,2,3,4],
                amt=[10,0,0,0,0,10,0,0,0,0],
                addl = [0,0,0,0,0,0,0,0,0,0],
                ii = [12,0,0,0,0,12,0,0,0,0],
                cmt=["Depot","Central","Central","Central","Central","Depot","Central","Central","Central","Central"],
                evid=[1,0,0,0,0,1,0,0,0,0],
                dv=[missing,8,6,4,2,missing,8,6,4,2],
                age=[45,45,45,45,45,50,50,50,50,50],
                sex = ["M","M","M","M","M","F","F","F","F","F"],
                crcl =[90,78,75,72,70,110,110,110,110,110])
  @test_throws Pumas.PumasDataError read_pumas(df10, amt=:amt, covariates=[:age,:sex,:crcl])

  # evid must be nonzero when amt > 0 or addl and ii are positive
  df12 = DataFrame(id = [1,1,1,1,1,2,2,2,2,2],
                time = [0,1,2,3,4,0,1,2,3,4],
                 amt=[10,0,0,0,0,10,0,0,0,0],
                addl = [5,0,0,0,0,5,0,0,0,0],
                ii = [12,0,0,0,0,12,0,0,0,0],
                cmt=["Depot","Central","Central","Central","Central","Depot","Central","Central","Central","Central"],
                evid=[0,0,0,0,0,0,0,0,0,0],
                dv=[missing,8,6,4,2,missing,8,6,4,2],
                age=[45,45,45,45,45,50,50,50,50,50],
                sex = ["M","M","M","M","M","F","F","F","F","F"],
                crcl =[90,78,75,72,70,110,110,110,110,110])
  @test_throws Pumas.PumasDataError read_pumas(df12, amt=:amt, covariates=[:age,:sex,:crcl])
end

@testset "Covartime inclusion in time vector" begin
  Random.seed!(8765492)
  covar3() = (WT = fill(rand(13.1:18.3), 2),)
  covar4() = (WT = rand(13.1:18.3),)
  e3 = DosageRegimen(320,
                     time = 0,
                     cmt = 1,
                     addl = 3,
                     ii = 24)

  pop1 = map(i -> Subject(id=i, events=e3, covariates=covar4()), 2:5)
  pop1df = DataFrame(pop1)
  @test pop1df.WT == reduce(vcat, map(i->fill(pop1[i].covariates(0.0).WT, 4), 1:length(pop1)))
  pop2 = map(i -> Subject(id=i, events=e3, covariates=covar3(), covariates_time = (WT = range(0, 8; length=2))),2:5)
  pop2df = DataFrame(pop2)
  @test pop2df.WT == reduce(vcat, map(i->fill(pop2[i].covariates(0.0).WT, 5), 1:length(pop2)))
end

@testset "test that cmt is carried down" begin
  ev = DosageRegimen([10,10], cmt=[1,2], time=[0,12])
  pop =  Population(map(i->Subject(id=i, events=ev),1:3))
  popdf = DataFrame(pop)
  @test all(popdf.cmt .== [1,2,1,2,1,2])

  theopp = DataFrame(read_pumas(example_data("event_data/THEOPP"), covariates = [:SEX,:WT]))
  @test all(theopp.cmt .== Ref(1))
end
@testset "integer obstimes" begin
  # from https://discourse.pumas.ai/t/help-with-fitting/50/54
  # the test is simply that dff runs in the end of the testset 
  pk_13 = @model begin

    @param begin
      tvcl ∈ RealDomain(lower=0)
      tvvc ∈ RealDomain(lower=0)
      tvcld ∈ RealDomain(lower=0)
      tvvt ∈ RealDomain(lower=0)
      Ω ∈ PDiagDomain(4)
      σ_prop ∈ RealDomain(lower=0)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Cl = tvcl*exp(η[1])
      Vc = tvvc*exp(η[2])
      Cld = tvcld*exp(η[3])
      Vt = tvvt*exp(η[4])
    end

    @dynamics begin
      Central' = -(Cl/Vc)*Central -(Cld/Vc)*Central +(Cld/Vt)*Peripheral
      Peripheral' = (Cld/Vc)*Central -(Cld/Vt)*Peripheral
    end

    @derived begin
      cp = @. Central/Vc
      dv ~ @. Normal(cp, sqrt(cp^2*σ_prop))
    end
  end

  par = (tvcl = 0.344708, # L/min/kg
          tvvc = 2.8946, # L/kg
          tvcld = 0.178392, # L/min/kg
          tvvt = 2.18368, # L/kg
          Ω = Diagonal([0.0001, 0.0001, 0.0001, 0.0001]),
          σ_prop = 0.01)

  ev1 = DosageRegimen(400, time=0, cmt=1)
  ev2 = DosageRegimen(800, time=0, cmt=1, rate=30.769)
  com = DosageRegimen(ev1,ev2)
  sub = Subject(id=1, events=com)
  # The issue was that these obstimes are integers and that caused problems in the DataFrame constructor
  sim = simobs(pk_13, sub, par, obstimes=[2, 5, 10, 15, 20, 25, 30, 33, 35, 37, 40, 45, 50, 60, 70, 90, 110, 120, 150])
  # Test that dff is constructed by converting obstimes to floats internally
  ddf = DataFrame(sim)
  @test eltype(ddf.time) == Float64
end