using Pumas.NCA, Test, Pumas

@test_nowarn NCA.checkconctime([1,2,3,4], 1:4)
@test_nowarn NCA.checkconctime([1,2,missing,4], 1:4)
@test_throws ArgumentError NCA.checkconctime([1,2,missing,4], 1:5)

@test_throws ArgumentError NCA.checkconctime(Int[])
@test_throws ArgumentError NCA.checkconctime([1,2], Int[])
@test_throws ArgumentError NCA.checkconctime(Set([1,2]))
@test_throws ArgumentError NCA.checkconctime([1,2], Set([1,2]))
@test_throws ArgumentError NCA.checkconctime([missing, true])
@test_logs (:warn, "All concentration data is missing") NCA.checkconctime([missing, missing])
@test_logs (:warn, "Negative concentrations found at index 1") NCA.checkconctime([missing, -1])

@test_throws ArgumentError NCA.checkconctime([1,2], [missing, 2])
@test_throws ArgumentError NCA.checkconctime([1,2], Set([1, 2]))
@test_throws ArgumentError NCA.checkconctime([1,2], [2, 1])
@test_throws ArgumentError NCA.checkconctime([1,2], [1, 1])

conc, t = NCA.cleanmissingconc([1,2,missing,4], 1:4)
@test conc == [1,2,4]
@test eltype(conc) === Int
@test t == [1,2,4]

conc, t = NCA.cleanmissingconc([1,2,missing,4], 1:4, missingconc=100)
@test conc == [1,2,100,4]
@test eltype(conc) === Int
@test t === 1:4

conc, t = NCA.cleanblq(zeros(6), 1:6, concblq=:drop)
@test isempty(conc)
@test isempty(t)
conc, t = NCA.cleanblq(zeros(6), 1:6, concblq=:keep)
@test conc == zeros(6)
@test t == 1:6
conc, t = NCA.cleanblq(ones(6), 1:6, concblq=:drop)
@test conc == ones(6)
@test t == 1:6
conc, t = NCA.cleanblq([0,1,1,3,0], 1:5, concblq=Dict(:first=>:drop, :middle=>:keep, :last=>:drop))
@test conc == [1,1,3]
@test t == 2:4
df = DataFrame(id=1, conc=[0,1,1,3,0], time=1:5)
subj = read_nca(df, time=:time, conc=:conc, verbose=false, concblq=:drop)[1]
@test subj.conc == [1,1,3]
subj = read_nca(df, time=:time, conc=:conc, verbose=false, concblq=Dict(:first=>:drop, :middle=>:keep, :last=>:keep))[1]
@test subj.conc == [1,1,3,0]

io = IOBuffer()
show(io, NCASubject([1,2,3.]*u"mg/L", (1:3)*u"hr"))

@test NCA.choosescheme(0, 1, 0, 1, 1, 10, :linuplogdown) === NCA.Linear

conc = [0.0,  0.010797988784995802,  0.0311661014235923,  0.0,  0.08643594149959559,  0.018489935844277383,  0.11633550035087399,  0.04838749939561179,  0.028858964356788423,  0.03488486420528403,  0.018256694779941317,  0.0]
t = [0.0,   0.5,   1.0,   2.0,   3.0,   4.0,   8.0,  12.0,  16.0,  24.0,  36.0,  48.0]
@test NCA.auclast(NCASubject(conc, t, concblq=:keep), method=:linuplogdown) ≈ 1.409803969

conc = [0.0
 0.01912264430801591
 0.027436690181501182
 0.04886246640378432
 0.025108477059121975
 0.08984585803215825
 0.06592118556711972
 0.0
 0.04706891166332337
 0.03787903464152118
 0.016417346582615896
 0.013251953705527533
]
@test NCA.thalf(NCASubject(conc, t, concblq=:keep)) ≈ 16.23778094

@test_throws ArgumentError NCA.lambdaz(Float64[], Float64[])
