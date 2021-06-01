
using Pumas, PumasTutorials, CSV


data = PumasTutorials.tutorial_data("data/nca","multiple_dose_IVbolus_7BLQ_test")
data = CSV.read(data,missingstring="NA")
first(data,10)


timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg"


pop = read_nca(data, id=:ID, time=:time, conc=:DV, amt=:DOSE, ii=24timeu,
    route=:Formulation, occasion=:OCC,timeu=timeu, concu=concu, amtu=amtu,llq=0.4concu)


NCA.auc(pop,auctype=:last,method=:linear)


report = NCAReport(pop)
report = NCA.to_dataframe(report)


names(report)


CSV.write("./tutorials/nca/report_MD_IVbolus_7BLQ.csv", report)

