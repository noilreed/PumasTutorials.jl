using Pumas, Pumas.IVIVC, Test

# read vitro data
vitro_data = @test_nowarn read_vitro(Pumas.example_data("ivivc_test_data/vitro_data"))
vitro_subs = vitro_data.subjects

@test typeof(vitro_subs) <: AbstractVector
@test eltype(vitro_subs) <: AbstractDict{String, Pumas.IVIVC.InVitroForm}

@test length(vitro_data) == length(vitro_subs)
@test size(vitro_data) == size(vitro_subs)

k = keys(vitro_data[1])
@test "slow" in k
@test "medium" in k
@test "fast" in k

@test vitro_data[1]["slow"].time[1:4] == [0.5, 1.0, 1.5, 2.0]
@test vitro_data[1]["medium"].conc[1:4] == [0.186, 0.299, 0.384, 0.464]
@test vitro_data[1]["fast"].form == "fast"
@test vitro_data[1]["fast"].id == 1
@test vitro_data[1]["slow"].id == 1
@test vitro_data[1]["medium"].id == 1

# read vivo data
vivo_data = @test_nowarn read_vivo(Pumas.example_data("ivivc_test_data/vivo_data"))
vivo_subs = vivo_data.subjects

@test typeof(vivo_subs) <: AbstractVector
@test eltype(vivo_subs) <: AbstractDict{String, Pumas.IVIVC.InVivoForm}

@test length(vivo_data) == length(vivo_subs)
@test size(vivo_data) == size(vivo_subs)

k = keys(vivo_data[1])
@test "slow" in k
@test "medium" in k
@test "fast" in k

@test vivo_data[1]["slow"].time[1:4] == [0.0, 0.5, 1.0, 1.5]
@test vivo_data[1]["medium"].conc[1:4] == [0.0, 5.182, 26.3, 45.17]
@test vivo_data[1]["fast"].form == "fast"
@test vivo_data[1]["fast"].id == 1
@test vivo_data[1]["slow"].id == 1
@test vivo_data[1]["medium"].id == 1

# Exception tests
using CSV
using Pumas.IVIVC: ___read_vitro, ___read_uir, ___read_vivo

file = Pumas.example_data("ivivc_test_data/vitro_data")
df = CSV.read(file)
rename!(df, :conc => :concc)
data = @test_throws ArgumentError ___read_vitro(df)

file = Pumas.example_data("ivivc_test_data/vivo_data")
df = CSV.read(file)
rename!(df, :id => :idd)
data = @test_throws ArgumentError ___read_vivo(df)

file = Pumas.example_data("ivivc_test_data/uir_data")
df = CSV.read(file)
rename!(df, :dose => :does)
data = @test_throws ArgumentError ___read_uir(df)
