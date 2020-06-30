using Pumas.NCA, Test, CSV
using Pumas
using Random

file = Pumas.example_data("nca_test_data/patient_data_test_sk")
df = DataFrame(CSV.File(file))
timeu, concu, amtu = u"hr", u"mg/L", u"mg"
df[!,:route] .= "inf"
data = @test_nowarn read_nca(df, id=:ID, time=:Time, conc=:Prefilter_Conc, verbose=false, route=:route, amt=:Amount, duration=:Infusion_Time, timeu=timeu, concu=concu, amtu=amtu)
@test data[1].dose === NCADose(0.0timeu, 750amtu, 0.25timeu, NCA.IVInfusion)
@test NCA.mrt(data; auctype=:last)[!, :mrt] == NCA.aumclast(data)[!, :aumclast]./NCA.auclast(data)[!, :auclast] .- 0.25timeu/2
@test NCA.mrt(data; auctype=:inf)[!, :mrt] == NCA.aumc(data)[!, :aumc]./NCA.auc(data)[!, :auc] .- 0.25timeu/2
@test !ismissing(NCA.vss(data)[!, 1])

# 1029
allowmissing!(df)
df.Prefilter_Conc[1]=missing
data = read_nca(df, id=:ID, time=:Time, conc=:Prefilter_Conc, verbose=true, route=:route, amt=:Amount, duration=:Infusion_Time, timeu=timeu, concu=concu, amtu=amtu)
@test_nowarn NCA.aumclast(data)

inf_data = DataFrame(CSV.File(IOBuffer("""
id,time,conc,amt,duration,rate,route
1,0,0,2551,0.5,5101,inf
1,0.25,85.13,,,,
1,0.416666667,109.31,,,,
1,0.583333333,46.49,,,,
1,0.666666667,24.66,,,,
1,0.75,15.74,,,,
1,1,9.86,,,,
1,1.5,2.94,,,,
1,2,2.4,,,,
1,2.5,1.04,,,,
1,4.5,0.21,,,,
1,6.5,0.08,,,,
1,8.5,0.1,,,,
1,24.5,,,,,
""")))
@test inf_data.route[1] === Inf
dfnca = @test_nowarn read_nca(inf_data, id=:id, time=:time, conc=:conc)
@test dfnca[1].dose.formulation === NCA.IVInfusion
