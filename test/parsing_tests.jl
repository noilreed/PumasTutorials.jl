using PuMaS, Test, CSV

@testset "nmtran" begin
  data = process_nmtran(example_nmtran_data("event_data/data1"))
  @test_nowarn show(data)

  @test getproperty.(data[1].events, :time) == 0:12:36

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
    gen_subject = Subject(evs = DosageRegimen(100, ii = 12, addl = 3))
    @test data[1].id == gen_subject.id
    @test data[1].covariates == gen_subject.covariates
    @test data[1].events == gen_subject.events
  end
end
@testset "Time Variant Covariates" begin
  data = process_nmtran(example_nmtran_data("time_variant_cvs"), [:weight, :dih])
  @test data[1].covariates.weight |> (x -> isa(x, Vector{Int}) && length(x) == 9)
  @test data[1].covariates.dih == 2
end
@testset "Chronological Observations" begin
  data = DataFrame(time = [0, 1, 2, 2], dv = rand(4), evid = 0)
  @test isa(process_nmtran(data), Population)
  append!(data, DataFrame(time = 1, dv = rand(), evid = 0))
  @test_throws AssertionError process_nmtran(data)
  @test_throws AssertionError Subject(obs = [ PuMaS.Observation(t, (x = x,)) for (t, x)
                                              ∈ zip(1:-1:0, 2:3) ])
end
@testset "event_data" begin
  data = DataFrame(time = [0, 1, 2, 2], amt = zeros(4), dv = rand(4), evid = 1)
  @test isa(process_nmtran(data), Population)
  @test_throws AssertionError process_nmtran(data, event_data = true)
end
@testset "Population Constructors" begin
  e1 = DosageRegimen(100, ii = 24, addl = 6)
  e2 = DosageRegimen(50,  ii = 12, addl = 13)
  e3 = DosageRegimen(200, ii = 24, addl = 2)
  s1 = map(i -> Subject(id = i, evs = e1), 1:5)
  s2 = map(i -> Subject(id = i, evs = e2), 6:8)
  s3 = map(i -> Subject(id = i, evs = e3), 9:10)
  pop1 = Population(s1)
  pop2 = Population(s2)
  pop3 = Population(s3)
  @test Population(s1, s2, s3) == Population(pop1, pop2, pop3)
end
