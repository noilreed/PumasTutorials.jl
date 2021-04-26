
using Dates


using Pumas, GlobalSensitivity, CairoMakie, PumasPlots


peg_inf_model = @model begin

  @param begin
    logKa   ∈  RealDomain()
    logKe   ∈  RealDomain()
    logVd   ∈  RealDomain()
    logn    ∈  RealDomain()
    logd    ∈  RealDomain()
    logc    ∈  RealDomain()
    logEC50 ∈  RealDomain()
    ω²Ka   ∈ RealDomain(lower=0.0)
    ω²Ke   ∈ RealDomain(lower=0.0)
    ω²Vd   ∈ RealDomain(lower=0.0)
    ω²n    ∈ RealDomain(lower=0.0)
    ω²δ    ∈ RealDomain(lower=0.0)
    ω²c    ∈ RealDomain(lower=0.0)
    ω²EC50 ∈ RealDomain(lower=0.0)
    σ²PK ∈ RealDomain(lower=0.0)
    σ²PD ∈ RealDomain(lower=0.0)
  end

  @random begin
    ηKa   ~ Normal(0.0, sqrt(ω²Ka))
    ηKe   ~ Normal(0.0, sqrt(ω²Ke))
    ηVd   ~ Normal(0.0, sqrt(ω²Vd))
    ηn    ~ Normal(0.0, sqrt(ω²n))
    ηδ    ~ Normal(0.0, sqrt(ω²δ))
    ηc    ~ Normal(0.0, sqrt(ω²c))
    ηEC50 ~ Normal(0.0, sqrt(ω²EC50))
  end

  @pre begin
    # constants
    p = 100.0
    d = 0.001
    e = 1e-7
    s = 20000.0
    log_Ka   = logKa   + ηKa
    log_Ke   = logKe   + ηKe
    log_Vd   = logVd   + ηVd
    log_n    = logn    + ηn
    log_d    = logd    + ηδ
    log_c    = logc    + ηc
    log_EC50 = logEC50 + ηEC50
  end

  @init begin
    T = exp(log_c + log_d)/(p*e)
    I = (s*e*p - d*exp(log_c + log_d))/(p*exp(log_d)*e)
    W = (s*e*p - d*exp(log_c + log_d))/(exp(log_c + log_d)*e)
  end

  @dynamics begin
    X' = -exp(log_Ka)*X
    A' = exp(log_Ka)*X - exp(log_Ke)*A
    T' = s - T*(e*W + d)
    I' = e*W*T - exp(log_d)*I
    W' = p/((A/exp(log_Vd)/exp(log_EC50) + 1e-100)^exp(log_n) + 1)*I - exp(log_c)*W
  end

  @derived begin
    conc   = @. A/exp(log_Vd)
    log10W = @. log10(max.(W, 1e-5))
    yPK ~ @. Normal(A/exp(log_Vd), sqrt(σ²PK))
    yPD ~ @. Normal(log10W, sqrt(σ²PD))
    nca1 := @nca conc
    cmax_pk = NCA.cmax(nca1)
    nca2 := @nca log10W
    cmax_pd = NCA.cmax(nca2)
  end

end


peg_inf_dr = DosageRegimen(180.0)
t = collect(0.0:1.0:28.0)
_pop = map(i -> Subject(id=i, observations=(yPK=[], yPD=[]), events=peg_inf_dr, time=t), 1:3)


param_PKPD = (
  logKa   = log(0.80),
  logKe   = log(0.15),
  logVd   = log(100.0),
  logn    = log(2.0),
  logd    = log(0.20),
  logc    = log(7.0),
  logEC50 = log(0.12),
  # random effects variance parameters, set to 0 to neutralize effect of random effects
  ω²Ka   = 0.0,
  ω²Ke   = 0.0,
  ω²Vd   = 0.0,
  ω²n    = 0.0,
  ω²δ    = 0.0,
  ω²c    = 0.0,
  ω²EC50 = 0.0,
  # variance parameter in proportional error model
  σ²PK = 0.04,
  σ²PD = 0.04)


simdata = simobs(peg_inf_model, _pop, param_PKPD, ensemblealg=EnsembleThreads())
sim_plot(peg_inf_model ,simdata, observations=[:yPD,:yPK], all = true)


p_range_low =  (logKa   = log(0.80)/2,
                logKe   = log(0.15)/2,
                logVd   = log(100.0)/2,
                logn    = log(2.0)/2,
                logd    = log(0.20)/2,
                logc    = log(7.0)/2,
                logEC50 = log(0.12)/2, )

p_range_high = (logKa   = log(0.80)*2,
                logKe   = log(0.15)*2,
                logVd   = log(100.0)*2,
                logn    = log(2.0)*2,
                logd    = log(0.20)*2,
                logc    = log(7.0)*2,
                logEC50 = log(0.12)*2, )


morris_ = Pumas.gsa(
  peg_inf_model,
  _pop,param_PKPD,
  GlobalSensitivity.Morris(),
  [:cmax_pk,:cmax_pd],
  p_range_low,
  p_range_high,
  ensemblealg=EnsembleThreads())


keys_ = [keys(p_range_low)...]
cmax_pk_meansstar = [morris_.means_star[1,:][key] for key in keys_]
cmax_pd_meansstar = [morris_.means_star[2,:][key] for key in keys_]

fig = Figure(resolution = (1200, 400))
plot_pk = barplot(fig[1,1], string.(keys_), cmax_pk_meansstar, axis = (xlabel ="Parameters", title="Cmax PK Morris means"))
plot_pd = barplot(fig[1,2], string.(keys_), cmax_pd_meansstar, axis = (xlabel ="Parameters", title="Cmax PD Morris means"))
display(fig)


sobol_ = Pumas.gsa(
  peg_inf_model,
  _pop,
  param_PKPD,
  GlobalSensitivity.Sobol(),
  [:cmax_pk,:cmax_pd],
  p_range_low,
  p_range_high,
  N=4000,
  ensemblealg=EnsembleThreads())


cmax_pk_s1 = [sobol_.first_order[1,:][key] for key in keys_]
cmax_pd_s1 = [sobol_.first_order[2,:][key] for key in keys_]
cmax_s1 = hcat(cmax_pk_s1, cmax_pd_s1)

fig = Figure(resolution = (1200, 400))
plot_s1 = heatmap(fig[1,1], 1:2, 1:7, cmax_s1, axis = (xticks = (1:2, ["Cmax_PK","Cmax_PD"]), yticks = (1:7, string.(keys_)), title="First Order Indices"), colormap = "darkrainbow")

cmax_pk_st = [sobol_.total_order[1,:][key] for key in keys_]
cmax_pd_st = [sobol_.total_order[2,:][key] for key in keys_]
cmax_st = hcat(cmax_pk_st, cmax_pd_st)

plot_st = heatmap(fig[1,2], 1:2, 1:7, cmax_st,  axis = (xticks = (1:2, ["Cmax_PK","Cmax_PD"]), yticks = (1:7, string.(keys_)), title="First Order Indices"), colormap = "darkrainbow")
display(fig)

