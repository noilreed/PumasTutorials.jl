
using Pumas, PumasTutorials, CSV


data = PumasTutorials.tutorial_data("data/nca","SD_oral_2analytes")
data = CSV.read(data,missingstring="NA")
first(data,10)


timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg"


pop = read_nca(data, id=:ID, time=:time, conc=:DV, amt=:DOSE, ii=24timeu, group=:Analyte,
    route=:Formulation, timeu=timeu, concu=concu, amtu=amtu,lloq=0.4concu)


NCA.auc(pop,auctype=:last,method=:linear)


report = NCAReport(pop)
report = NCA.to_dataframe(report)


CSV.write("./tutorials/nca/report_SD_oral_2analytes.csv", report)


using PumasTutorials
PumasTutorials.tutorial_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

