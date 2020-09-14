using Pumas, SafeTestsets

@time begin
    @time @safetestset "Pumas Integration Tests" begin include("pumas_tests.jl") end
end
