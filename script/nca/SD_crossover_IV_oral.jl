
using Pumas, PumasTutorials, CSV, Plots


data = PumasTutorials.tutorial_data("data/nca","SD_crossover_IV_oral")
data = CSV.read(data,missingstring="NA")
first(data,10)


timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg"


pop = read_nca(data, id=:ID, time=:time, conc=:DV, amt=:DOSE, ii=24timeu,
    route=:Formulation, occasion=:OCC,timeu=timeu, concu=concu, amtu=amtu)


pop = read_nca(data, id=:ID, time=:time, conc=:DV, amt=:DOSE, ii=24timeu,
    route=:Formulation, occasion=:OCC,timeu=timeu, concu=concu, amtu=amtu, lloq=0.4concu)


NCA.auc(pop,auctype=:last,method=:linear)


NCA.auc(pop,auctype=:inf,method=:linuplogdown)


NCA.auc(pop[1], interval=(0,12).*timeu)


NCA.auc(pop[1], interval=[(0,12).*timeu,(0,6).*timeu])


NCA.auc_extrap_percent(pop)


NCA.lambdaz(pop)


NCA.lambdaz(pop, threshold=3)


NCA.lambdaz(pop, idxs=[18,19,20])


NCA.lambdaz(pop, slopetimes=[18.5,19,19.5].*timeu)


cmax = NCA.cmax(pop[1])

pop[1]


normalizedose(cmax,pop[1])


AUClast = NCA.auc(pop[1],auctype=:last)
normalizedose(AUClast,pop[1])


NCA.lambdazr2(pop)
NCA.lambdazadjr2(pop)
NCA.lambdazintercept(pop)
NCA.lambdaztimefirst(pop)
NCA.lambdaznpoints(pop)

NCA.tmax(pop)
NCA.cmin(pop)
NCA.tmin(pop)

NCA.tlast(pop)
NCA.clast(pop)

NCA.aumc(pop)
NCA.aumclast(pop)

NCA.thalf(pop)

NCA.cl(pop,ithdose=1)

NCA.vss(pop)
NCA.vz(pop)

NCA.accumulationindex(pop)


pop[2]
NCA.cl(pop[2])


report = NCAReport(pop,ithdose=1)
report = NCA.to_dataframe(report)


report = NCAReport(pop,ithdose=1,pred=true)
report = NCA.to_dataframe(report)


CSV.write("./tutorials/nca/report_SD_crossover_IV_oral.csv", report)


using PuMaSTutorials
PuMaSTutorials.tutorial_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

