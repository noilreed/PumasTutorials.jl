
using Pumas, PumasTutorials, CSV, Plots


data = PumasTutorials.tutorial_data("data/nca","SD_IV_infusion")
data = CSV.read(data,missingstring="NA")
first(data,10)


timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg"


pop = read_nca(data, id=:ID, time=:time, conc=:DV, amt=:DOSE, ii=24timeu,
    route=:Formulation, duration=:Infusion_Time,timeu=timeu,
    concu=concu, amtu=amtu,lloq=0.4concu)


plot(pop)


NCA.auc(pop,auctype=:last,method=:linear)


NCA.auc(pop,auctype=:inf,method=:linuplogdown)


NCA.auc(pop, interval=(0,12).*timeu)


NCA.auc(pop, interval=[(0,12).*timeu,(0,6).*timeu])


NCA.lambdaz(pop)


NCA.lambdaz(pop, threshold=3)
NCA.lambdaz(pop, idxs=[18,19,20])
NCA.lambdaz(pop, slopetimes=[18.5,19,19.5].*timeu)


cmax = NCA.cmax(pop[1])


 NCA.normalizedose(cmax,pop[1])


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

NCA.cl(pop)

NCA.vss(pop)
NCA.vz(pop)


report = NCAReport(pop)
report = NCA.to_dataframe(report)


report = NCAReport(pop,pred=true)
report = NCA.to_dataframe(report)


CSV.write("./tutorials/nca/report_SD_IV_infusion.csv", report)


using PumasTutorials
PumasTutorials.tutorial_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

