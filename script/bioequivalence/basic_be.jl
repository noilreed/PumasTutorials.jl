
ENV["COLUMNS"] = 130;
ENV["LINES"] = 30;


using Pkg
Pkg.activate(joinpath(@__DIR__, "tutorials", "bioequivalence"))
using Bioequivalence


SLF2014 = Bioequivalence.testdata("SLF2014_8")
show(SLF2014)


id = 1
period = 1
sequence = "TR"
endpoint = 168.407


Crossover = pumas_be(SLF2014)
show(Crossover)


using DataFrames
rename!(SLF2014, "id" => "subj")
try
    pumas_be(SLF2014)
catch err
    err
end


Crossover = pumas_be(SLF2014, id = :subj)
show(Crossover)


show(Crossover.design)


show(Crossover.data)


show(Crossover.data_stats.total)


show(Crossover.data_stats.used_for_analysis)


show(Crossover.data_stats.sequence)


show(Crossover.data_stats.period)


show(Crossover.data_stats.formulation)


using StatsBase
r2(Crossover.model)


deviance(Crossover.model)


show(Crossover.model_stats.Wald)


show(Crossover.model_stats.lsmeans)


data = Bioequivalence.testdata("PJ2017_4_3")
show(data)


auc = pumas_be(data)
cmax = pumas_be(data, endpoint = :Cmax)
output = Dict(endpoint => pumas_be(data, endpoint = endpoint) for endpoint in setdiff(propertynames(data), (:id, :sequence, :period)))


show(output[:AUC])


show(output[:Cmax])

