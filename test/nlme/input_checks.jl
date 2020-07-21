using Pumas, CSV, Test
@testset "read_pumas vs DataFrame" begin
  df = DataFrame(CSV.File(example_data("event_data/CS1_IV1EST_PAR")))
  model = @model begin end
  param = NamedTuple()
  @test_throws ArgumentError("The second argument to fit was a DataFrame instead of a Population. Please use read_pumas to construct a Population from a DataFrame.") fit(model, df, param, Pumas.FOCEI())  
  @test_throws ArgumentError("The second argument to simobs was a DataFrame instead of a Population. Please use read_pumas to construct a Population from a DataFrame.") simobs(model, df, param)  
  @test_throws ArgumentError("The second argument to solve was a DataFrame instead of a Population. Please use read_pumas to construct a Population from a DataFrame.") solve(model, df, param)  
end
