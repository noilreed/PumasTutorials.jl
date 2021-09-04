
using PumasTutorials, NCA, NCAUtilities, CSV, DataFrames, StatsPlots, Statistics, Unitful


data = DataFrame(CSV.File(joinpath(dirname(pathof(PumasTutorials)), "..", "tutorials", "nca", "NCA_tutorial.csv")))
first(data, 6)


single_dose_data  = filter(x -> x.TIME <= 24 , data)


timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg";


pop        = read_nca(  single_dose_data,
                        id    = :ID,
                        time  = :TIME,
                        observations  = :CONC,
                        amt   = :AMT,
                        route = :ROUTE,
                        llq   = 0.001concu,
                        timeu = timeu,
                        concu = concu,
                        amtu  = amtu
                      )


NCAUtilities.observations_vs_time(pop[1])


report0     = run_nca(pop, sigdigits=3)


vz        = NCA.vz(pop, sigdigits=3)  # Volume of Distribution/F, in this case since the drug is given orally
cl        = NCA.cl(pop, sigdigits=3)  # Clearance/F, in this case since the drug is given orally
lambdaz   = NCA.lambdaz(pop, threshold=3, sigdigits=3)  # Terminal Elimination Rate Constant, threshold=3 specifies the max no. of time point used for calculation
lambdaz_1 = NCA.lambdaz(pop, slopetimes=[8,12,16].*timeu, sigdigits=3) # slopetimes in this case specifies the exact time point you want for the calculation
thalf     = NCA.thalf(pop[4], sigdigits=3) # Half-life calculation for 4th individual
cmax_d    = NCA.cmax(pop, normalize=true, sigdigits=3) # Dose Normalized Cmax
mrt       = NCA.mrt(pop, sigdigits=3) # Mean residence time
aumc      = NCA.aumc(pop, method=:linlog, sigdigits=3) # AUMC calculation, using :linlog method
rename!(lambdaz_1, Dict(:lambdaz => "lambdaz_specf")) #since we have two lambdaz calculation rename one column to merge in a dataframe
df_1      = innerjoin(vz,cl,lambdaz, lambdaz_1,cmax_d,mrt,aumc, on=[:id], makeunique=true)


auc0_12   = NCA.auc(pop, interval=(0,12).*timeu, method=:linuplogdown, sigdigits=3) #various other methods are :linear, :linlog
auc12_24  = NCA.auc(pop, interval=(12,24).*timeu, method=:linuplogdown, sigdigits=3)
final     = innerjoin(report0.reportdf, auc0_12, auc12_24, on = [:id], makeunique=true)


nca_output_single = select(final, :doseamt, :kel, :half_life, :cmax, :tmax, :auclast, :aucinf_obs, :vz_f_obs, :cl_f_obs, :aumcinf_obs, :auc0_12, :auc12_24)


Stat_report = describe(nca_output_single)


multiple_dose_data           = filter(x -> x.TIME > 24 , data)


pop1       = read_nca( multiple_dose_data,
                       id    = :ID,
                       time  = :TIME,
                       observations  = :CONC,
                       amt   = :AMT,
                       route = :ROUTE,
                       ii    = :II,        # please specify II for Multiple-dose NCA
                       llq   = 0.001concu,
                       timeu = timeu,
                       concu = concu,
                       amtu  = amtu)
report1    = run_nca(pop1, sigdigits=3)


Stat_report1        = summarize(report1.reportdf, parameters = [:doseamt, :kel, :half_life, :tmax, :cmax, :auclast, :auc_tau_obs, :vz_f_obs, :cl_f_obs, :aumcinf_obs, :tau, :cavgss])


CSV.write("NCA_single_dose.csv", final)      # Single-NCA final report
CSV.write("NCA_multiple_dose.csv", report1.reportdf)  # Multiple-NCA final report

