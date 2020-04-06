using Pumas.NCA, Test, CSV
using Pumas

multiple_doses_file = Pumas.example_data("nca_test_data/dapa_IV_ORAL")
mdata = copy(CSV.read(multiple_doses_file))
msol = CSV.read(Pumas.example_data("nca_test_data/dapa_IV_ORAL_sol"))

timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg"
mdata[!,:route] .= map(f -> f=="ORAL" ? "ev" : "iv", mdata.FORMULATION)
mdata[!,:ii] .= 24
didxs = findall(x->!ismissing(x) && !iszero(x), mdata.AMT)
mdata[rand(didxs), :AMT] = 10

mncapop = @test_nowarn read_nca(mdata, id=:ID, time=:TIME, conc=:COBS, amt=:AMT, route=:route, occasion=:OCC,
                                     timeu=timeu, concu=concu, amtu=amtu)
@test ustrip.(NCA.doseamt(mncapop)[!, end]) == mdata[didxs, :AMT]

@test_throws ArgumentError NCA.interpextrapconc(mncapop[1], 22timeu, method=:linear)

# test caching
@test_nowarn NCA.auc(mncapop)
@test NCA.auc(mncapop[1]; method=:linear) != NCA.auc(mncapop[1]; method=:linlog)
@test NCA.auc(mncapop[1]; method=:linear) != NCA.auc(mncapop[1]; method=:linuplogdown)

lambdazdf = @test_nowarn NCA.lambdaz(mncapop)
@test size(lambdazdf, 2) == 3
@test lambdazdf[!,:lambdaz] isa Vector
@test lambdazdf[!,:occasion] == repeat(collect(1:4), 24)
@test lambdazdf[!,:id] == repeat(collect(1:24), inner=4)
@test_nowarn NCA.lambdazr2(mncapop)
@test_nowarn NCA.lambdazadjr2(mncapop)
@test_nowarn NCA.lambdazintercept(mncapop)
@test_nowarn NCA.lambdaztimefirst(mncapop)
@test_nowarn NCA.lambdaznpoints(mncapop)

@test_nowarn NCA.bioav(mncapop, ithdose=1)
@test all(vcat(NCA.tlag(mncapop)[!,:tlag]...) .=== float.(map(x -> ismissing(x) ? x : x*timeu, msol[!, :Tlag])))
@test_nowarn NCA.mrt(mncapop; auctype=:inf)
@test_nowarn NCA.mrt(mncapop; auctype=:last)
@test_nowarn NCA.cl(mncapop, ithdose=1)
@test NCA.tmin(mncapop[1])[1] == 20timeu
@test NCA.cmin(mncapop[1])[1] == mncapop[1].conc[1][end-1]
@test NCA.cminss(mncapop[1])[1] == NCA.cmin(mncapop[1])[1]*NCA.accumulationindex(mncapop[1])[1]
@test NCA.cmaxss(mncapop[1])[1] == NCA.cmax(mncapop[1])[1]*NCA.accumulationindex(mncapop[1])[1]
@test NCA.fluctuation(mncapop[1])[1] == (100 .*(NCA.cmaxss(mncapop[1]) .- NCA.cminss(mncapop[1]))./NCA.cavgss(mncapop[1]))[1]
@test NCA.fluctuation(mncapop[1], usetau=true) == (100 .*(NCA.cmaxss(mncapop[1]) .- NCA.ctau(mncapop[1]))./NCA.cavgss(mncapop[1]))
@test NCA.accumulationindex(mncapop[1]) == 1 ./(1 .-exp.(-NCA.lambdaz(mncapop[1]).*NCA.tau(mncapop[1])))
@test NCA.swing(mncapop[1])[1] == ((NCA.cmaxss(mncapop[1]) .- NCA.cminss(mncapop[1]))./NCA.cminss(mncapop[1]))[1]
@test NCA.c0(mncapop[1])[1] == mncapop[1].conc[1][1]
@test NCA.auc_back_extrap_percent(mncapop[1])[1] == 0
@test all(ismissing, NCA.auc_back_extrap_percent(mncapop[1])[2:end])

@test_nowarn ncareport1 = NCAReport(mncapop[1])

df = NCAReport(mncapop)
NCA.cleancache!(mncapop)
@test all(i->all(x->x == 0, mncapop[i].points), eachindex(mncapop))
@test all(i->all(x->x == -oneunit(mncapop[i].auc_last[1]), mncapop[i].auc_last), eachindex(mncapop))
@test all(i->all(x->x == -oneunit(mncapop[i].aumc_last[1]), mncapop[i].aumc_last), eachindex(mncapop))
@test count(!ismissing, df.cl_obs) == 24
@test count(!ismissing, df.cl_f_obs) == 72
@test count(!ismissing, df.cl_pred) == 24
@test count(!ismissing, df.cl_f_pred) == 72
df = NCAReport(mncapop, sigdigits=2)
@test df.lambda_z[2] == round(ustrip(df.lambda_z[2]), sigdigits=2)*oneunit(df.lambda_z[2])

# check retcode
mdata2 = mdata[1:1522, :]
mncapop2 = @test_nowarn read_nca(mdata2, id=:ID, time=:TIME, conc=:COBS, amt=:AMT, route=:route, occasion=:OCC,
                                     timeu=timeu, concu=concu, amtu=amtu)
@test NCA.lambdaz(mncapop2, verbose=false)[end, end] === missing
@test NCA.retcode(mncapop2)[end, end] === :NotEnoughDataAfterCmax
@test NCA.lambdaz(mncapop2, verbose=false)[end, end] === missing
@test NCA.retcode(mncapop2)[end, end] === :NotEnoughDataAfterCmax

data1 = CSV.read(IOBuffer("""
  id,time,tad,conc,amt,occasion,formulation
  1,0.0,0,0.755,0.705,1,oral
  1,1.0,1,0.55,0,1,oral
  1,2.0,2,0.65,0,1,oral
  1,12.0,0,0.473,0.29,2,oral
  1,13.0,1,0.235,0,2,oral
  1,14.0,2,0.109,0,2,oral
  2,0.0,0,0.341,0.932,1,oral
  2,1.0,1,0.557,0,1,oral
  2,2.0,2,0.159,0,1,oral
  2,3.0,0,0.307,0,2,oral
  2,3.5,0,0.226,0,3,oral
  3,0.0,0,0.763,0.321,1,oral
  3,1.0,1,0.96,0,1,oral
  3,2.0,2,0.772,0,1,oral
  3,12.0,0,0.941,0.656,2,oral
  3,13.0,1,0.204,0,2,oral
  3,14.0,2,0.0302,0,2,oral
  """))
data2 = CSV.read(IOBuffer("""
  id,time,tad,conc,amt,occasion,formulation
  1,0.0,0,0.755,0.705,1,oral
  1,1.0,1,0.55,0,1,oral
  1,2.0,2,0.65,0,1,oral
  1,12.0,0,0.473,0.29,2,oral
  1,13.0,1,0.235,0,2,oral
  1,14.0,2,0.109,0,2,oral
  2,0.0,0,0.341,0.932,1,oral
  2,1.0,1,0.557,0,1,oral
  2,2.0,2,0.159,0,1,oral
  3,0.0,0,0.763,0.321,1,oral
  3,1.0,1,0.96,0,1,oral
  3,2.0,2,0.772,0,1,oral
  3,12.0,0,0.941,0.656,2,oral
  3,13.0,1,0.204,0,2,oral
  3,14.0,2,0.0302,0,2,oral
  """))
for df in (data1, data2)
  df[!,:route] .= "ev"
  @test_throws AssertionError read_nca(df, timeu=timeu, concu=concu, amtu=amtu);
end

df = DataFrame()
df[!,:time] = collect(0:9)
df[!,:conc] .= [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]
df[!,:amt] .=  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
df[!,:route] .= "ev"
df[!,:ss] .=   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
df[!,:ii] .= df.ss .* 24
df[!,:occasion] .= [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
df[!,:id] .= 1
subj = @test_nowarn read_nca(df, llq=0concu, timeu=timeu, concu=concu, amtu=amtu)[1]
@test !subj.dose[1].ss
@test subj.dose[2].ss

valid_dose = NCADose(1, 0.1, 0, NCA.IVBolus)
subj = NCASubject(1:2, 1:2, dose=[NCADose(0, 0.1, 0, NCA.IVBolus), NCADose(10, 0.1, 0, NCA.IVBolus), valid_dose])
subj.dose[1] === valid_dose
