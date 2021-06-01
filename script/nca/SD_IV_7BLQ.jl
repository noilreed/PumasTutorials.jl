
using Pumas, PumasTutorials, CSV


data7BLQ = CSV.read("./tutorials/nca/data/single_dose_IVbolus_7BLQ.csv",missingstring="NA")
data7BLQ

data18BLQ = CSV.read("./tutorials/nca/data/single_dose_IVbolus_18BLQ.csv",missingstring="NA")
data18BLQ


timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg"


pop7 = read_nca(data7BLQ, id=:ID, time=:time, conc=:DV, amt=:DOSE, ii=24timeu,
    route=:Formulation,timeu=timeu, concu=concu, amtu=amtu)
pop18 = read_nca(data18BLQ, id=:ID, time=:time, conc=:DV, amt=:DOSE, ii=24timeu,
    route=:Formulation, timeu=timeu, concu=concu, amtu=amtu)


pop7 = read_nca(data7BLQ, id=:ID, time=:time, conc=:DV, amt=:DOSE, ii=24timeu,
    route=:Formulation,timeu=timeu, concu=concu, amtu=amtu,llq=0.4concu)
pop18 = read_nca(data18BLQ, id=:ID, time=:time, conc=:DV, amt=:DOSE, ii=24timeu,
    route=:Formulation,timeu=timeu, concu=concu, amtu=amtu,llq=0.4concu)


NCA.auc(pop7,auctype=:last,method=:linear)
NCA.auc(pop18,auctype=:last,method=:linear)


NCA.auc(pop7,auctype=:inf,method=:linuplogdown)


NCA.auc_extrap_percent(pop7)


NCA.auc(pop7[1], interval=(0,12).*timeu)


pop7[1]


NCA.auc(pop7[1], interval=[(0,12).*timeu,(0,6).*timeu])


NCA.lambdaz(pop7)


NCA.lambdaz(pop7[1], threshold=3)


NCA.lambdaz(pop7[1], idxs=[18,19,20])


NCA.lambdaz(pop7[1], slopetimes=[18.5,19,19.5].*timeu)


NCA.interpextrapconc(pop7, 22timeu, method=:linear)


cmax = NCA.cmax(pop7[1])


normalizedose(cmax,pop7[1])


AUClast = NCA.auc(pop7[1],auctype=:last)
normalizedose(AUClast,pop7[1])


NCA.lambdazr2(pop7)
NCA.lambdazadjr2(pop7)
NCA.lambdazintercept(pop7)
NCA.lambdaztimefirst(pop7)
NCA.lambdaznpoints(pop7)

NCA.tmax(pop7)
NCA.cmin(pop7)
NCA.tmin(pop7)

NCA.tlast(pop7)
NCA.clast(pop7)

NCA.aumc(pop7)
NCA.aumclast(pop7)

NCA.thalf(pop7)

NCA.cl(pop7)

NCA.vss(pop7)
NCA.vz(pop7)

NCA.accumulationindex(pop7)


pop7[24]

NCA.cl(pop7[24])


report = NCAReport(pop7)
report = NCA.to_dataframe(report)


report = NCAReport(pop7,pred=true)
report = NCA.to_dataframe(report)


CSV.write("./tutorials/nca/report_SD_IVbolus_7BLQ.csv", report)

