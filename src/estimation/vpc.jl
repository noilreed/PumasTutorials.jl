struct PopVPC
    data_quantiles::DataFrame
    data::Array{Population}
    stratify_by::Union{Nothing,Array{Symbol}}
    dv::Symbol
end

struct VPC
    simulated_quantiles::DataFrame
    popvpc::PopVPC
    level::Float64
end

abstract type VPCType end

struct DiscreteVPC{Boolean} <: VPCType
    idvdiscrete::Boolean
    numbins::Int
end
DiscreteVPC() = DiscreteVPC(true,0)


struct ContinuousVPC <: VPCType end

function _npqreg(yname::Symbol, xname::Symbol, df::AbstractDataFrame, τ::Real, method; xrange=nothing, bandwidth=nothing)
    ft = QuantileRegressions.npqreg(df[!,yname], df[!,xname], τ, method; xrange=xrange, h=bandwidth)
    return DataFrame(time=ft[1], quantile=ft[2], τ=τ)
end

function discrete_count(count_vals, df::AbstractDataFrame)
    proportions = vcat([[count(i -> val == i, df.dv)]./size(df,1) for val in count_vals], [[df.idv[1]]])
    colnames = vcat([Symbol("dv_$val") for val in count_vals], [:idv])
    DataFrame(proportions, colnames)
end

function discretize_covariates(population, cvname::Symbol, numstrat)
    sub_covariates = vcat([getproperty(subject.covariates(0.0), cvname) for subject in population]...)
    unqcovariates = unique(sub_covariates)
    if length(unqcovariates) > numstrat
        boundvals = [quantile(unqcovariates,i/numstrat) for i in 1:numstrat]
        f_ = function(el)
            for i in 1:numstrat
                if el <= boundvals[i]
                    return boundvals[i]
                end
            end
        end
        sub_covariates = map(f_, sub_covariates)
    end

    covariates = vcat([[sub_covariates[j] for i in 1:(subject.covariates isa ConstantInterpolationStructArray ? length(subject.covariates.u) : length(subject.time))] for (j,subject) in enumerate(population)]...)

    return covariates
end

function _vpc(
    population::Population,
    vpctype::DiscreteVPC;
    dv::Symbol,
    idv::Symbol,
    stratify_by,
    count_vals,
    numstrats=stratify_by === nothing ? nothing : [4 for i in 1:length(stratify_by)]
    )

    # Convert observations to DataFrame
    df = DataFrame(population, include_events=false)

    # Use hardcoded names
    df.idv = df[!,idv]
    df.dv  = df[!,dv]

    if isa(stratify_by, Array{Symbol})
        for (numstrat,cvname) in zip(numstrats,stratify_by)
            df[!,cvname] .= discretize_covariates(population, cvname, numstrat)
        end
    end

    # filter out missing obs
    df = filter(i -> !ismissing(i.dv), df)

    if vpctype.idvdiscrete
        data_quantiles = combine(t -> discrete_count(count_vals, t), groupby(df, stratify_by === nothing ? [:idv] : [stratify_by..., :idv]))
    else
        binned_idv = cut(df.idv, vpctype.numbins)
        idv = Int.([binned_idv[i].level for i in 1:length(binned_idv)])
        df.idv = idv
        data_quantiles = combine(t -> discrete_count(count_vals, t), groupby(df, stratify_by === nothing ? [:idv] : [stratify_by..., :idv]))
    end
    return data_quantiles, [read_pumas(DataFrame(df), id = :id, observations = [:dv], event_data=false) for df in groupby(df, stratify_by === nothing ? [] : stratify_by)]
end

function _vpc(
    population::Population,
    qreg_method,
    vpctype::ContinuousVPC;
    dv::Symbol,
    idv::Symbol,
    stratify_by,
    quantiles::NTuple{3,Float64}=(0.1, 0.5, 0.9),
    bandwidth=2,
    numstrats=stratify_by === nothing ? nothing : [4 for i in 1:length(stratify_by)],
    )

    # Convert observations to DataFrame
    df = stratify_by === nothing ? DataFrame(population, include_covariates=false, include_events=false) : DataFrame(population, include_events=false)

    # Use hardcoded names
    df.idv = df[!,idv]
    df.dv  = df[!,dv]

    if isa(stratify_by, Array{Symbol})
        for (numstrat,cvname) in zip(numstrats,stratify_by)
            df[!,cvname] .= discretize_covariates(population, cvname, numstrat)
        end
    end

    # filter out missing obs
    df = filter(i -> !ismissing(i.dv), df)

    data_quantiles = mapreduce(vcat, quantiles) do τ
        return combine(t -> _npqreg(:dv, :idv, t, τ, qreg_method; xrange=sort(unique(t.idv)), bandwidth=bandwidth), groupby(df, stratify_by === nothing ? [] : stratify_by))
    end

    return data_quantiles, [read_pumas(DataFrame(df), id = :id, observations = [:dv], event_data=false) for df in groupby(df, stratify_by === nothing ? [] : stratify_by)]
end

function quantile_discrete(df::AbstractDataFrame, quantiles, names_)
    quantiles_disc_sim = [[quantile(df[!,name], quantiles)...] for name in names_]
    return hcat(DataFrame(idv = [df.idv[1] for i in 1:3], τ = [quantiles...]), DataFrame(quantiles_disc_sim, names_))
end

function _vpc(
    m::PumasModel,
    population::Population,
    param::NamedTuple,
    reps::Integer,
    qreg_method,
    vpctype::VPCType;
    dv::Symbol,
    idv::Symbol,
    stratify_by,
    quantiles::NTuple{3,Float64}=(0.1, 0.5, 0.9),
    level::Real=0.95,
    ensemblealg=EnsembleSerial(),
    bandwidth=2,
    numstrats=stratify_by === nothing ? nothing : [4 for i in 1:length(stratify_by)],
    sim_idvs = nothing,
    count_vals = unique(skipmissing(DataFrame(population)[!,dv]))
    )

    prediction_probabilities = ((1 - level)/2, 0.5, (1 + level)/2)
    sim_quantiles_v = Array{Any}(undef, reps)
    if sim_idvs === nothing
        mintime,maxtime = extrema(vcat([sub.time for sub in population]...))
        num_timepoints = maximum([length(sub.time) for sub in population]) #needs some heuristic guided selection of the times
        sim_idvs = range(mintime, stop = maxtime, length = num_timepoints)
    end

    for i in 1:reps
        # Simulate a new population
        if vpctype isa ContinuousVPC
            sim_pop = Subject.(simobs(m, population, param, ensemblealg=ensemblealg, obstimes = sim_idvs))
            sim_quantiles_v[i], = _vpc(sim_pop, qreg_method, vpctype; quantiles=quantiles, dv=dv, bandwidth=bandwidth, stratify_by=stratify_by, numstrats = numstrats, idv = idv)
        else
            sim_pop = Subject.(simobs(m, population, param, ensemblealg=ensemblealg))
            sim_quantiles_v[i], = _vpc(sim_pop, vpctype; dv=dv, count_vals = count_vals, stratify_by = stratify_by, numstrats = numstrats, idv = idv)
        end
    end

    sim_quantiles_df = reduce(vcat, sim_quantiles_v)
    if vpctype isa ContinuousVPC
        sim_quantile_interval =  combine(t -> [quantile(t.quantile, prediction_probabilities)...]', groupby(sim_quantiles_df, stratify_by === nothing ? [:τ, :time] : [:τ, :time, stratify_by...]))
        rename!(sim_quantile_interval, stratify_by === nothing ? [:τ, :time, :lower, :middle, :upper] : [:τ, :time, stratify_by..., :lower, :middle, :upper])
    else
        names_ = filter(s -> occursin("dv_",s), names(sim_quantiles_df))
        sim_quantile_interval = combine(t -> quantile_discrete(t, prediction_probabilities, names_), groupby(sim_quantiles_df, stratify_by === nothing ? [:idv] : [:idv, stratify_by...]))
    end
    return sim_quantile_interval
end

function vpc(
    m::PumasModel,
    population::Population,
    param::NamedTuple,
    reps::Integer=499,
    qreg_method=IP(),
    vpctype::VPCType = ContinuousVPC();
    dv::Symbol = keys(population[1].observations)[1],
    stratify_by = nothing,
    quantiles::NTuple{3,Float64}=(0.1, 0.5, 0.9),
    level::Real=0.95,
    ensemblealg=EnsembleSerial(),
    bandwidth=2,
    numstrats= stratify_by === nothing ? nothing : [4 for i in 1:length(stratify_by)],
    idv = :time,
    count_vals = unique(skipmissing(DataFrame(population)[!,dv])),
    sim_idvs = nothing
    )
    if vpctype isa ContinuousVPC
        _vpc_data, pop_stratified = _vpc(population, qreg_method, vpctype; dv = dv, stratify_by = stratify_by, quantiles = quantiles, bandwidth = bandwidth, numstrats = numstrats, idv = idv)
        _vpc_simulated = _vpc(m, population, param, reps, qreg_method, vpctype; dv = dv, stratify_by = stratify_by, quantiles = quantiles, level = level, ensemblealg = ensemblealg, bandwidth = bandwidth, numstrats = numstrats, idv = idv, sim_idvs = sim_idvs)
    else
        _vpc_data, pop_stratified = _vpc(population, vpctype; dv = dv, stratify_by = stratify_by, numstrats = numstrats, idv = idv, count_vals = count_vals)
        _vpc_simulated = _vpc(m, population, param, reps, qreg_method, vpctype; dv = dv, stratify_by = stratify_by, level = level, ensemblealg = ensemblealg, numstrats = numstrats, idv = idv, count_vals = count_vals)
    end
    return VPC(_vpc_simulated, PopVPC(_vpc_data,pop_stratified, stratify_by, dv), level)
end

function vpc(
    population::Population,
    qreg_method=IP(),
    vpctype::VPCType= ContinuousVPC();
    dv::Symbol = keys(population[1].observations)[1],
    stratify_by = nothing,
    idv = :time,
    count_vals = unique(skipmissing(DataFrame(population)[!,dv])),
    kwargs...)
    if vpctype isa ContinuousVPC
        _vpc_data, pop_stratified = _vpc(population, qreg_method, vpctype; stratify_by = stratify_by, dv = dv, idv = idv, kwargs...)
    else
        _vpc_data, pop_stratified = _vpc(population, vpctype; stratify_by = stratify_by, dv = dv, idv = idv, count_vals = count_vals, kwargs...)
    end
    return PopVPC(_vpc_data, pop_stratified, stratify_by, dv)
end

"""
 vpc(fpm::FittedPumasModel,
        reps::Integer = 499,
        qreg_method = IP(),
        vpctype::VPCType = ContinuousVPC();
        dv::Symbol = keys(fpm.data[1].observations)[1],
        stratify_by = nothing,
        quantiles::NTuple{3,Float64}=(0.1, 0.5, 0.9),
        level::Real=0.95,
        ensemblealg=EnsembleSerial(),
        bandwidth=2,
        numstrats=stratify_by === nothing ? nothing : [4 for i in 1:length(stratify_by)])

 Computes the quantiles for VPC for a `FittedPumasModel` with simulated prediction intervals around the empirical quantiles based on `reps` simulated populations.

 The following keyword arguments are supported:
  - `quantiles::NTuple{3,Float64}`: A three-tuple of the quantiles for which the quantiles will be computed. The default is `(0.1, 0.5, 0.9)` which computes the
                                    10th, 50th and 90th percentile.
  - `level::Real`: Probability level to use for the simulated prediction intervals. The default is `0.95`.
  - `dv::Symbol`: The name of the dependent variable to use for the VPCs. The default is the first dependent variable in the dataset.
  - `stratify_by`: The covariates to be used for stratification. Takes an array of the `Symbol`s of the stratification covariates.
  - `ensemblealg`: This is passed to the `simobs` call while the `reps` simulations. For more description check the docs for `simobs`.
  - `bandwidth`: The kernel bandwidth in the quantile regression. If you are seeing `NaN`s or an error, increasing the bandwidth should help in most cases.
                With higher values of the `bandwidth` you will get more smoothened plots of the quantiles so it's a good idea to check with your data the right `bandwidth`.
  - `numstrats`: The number of strata to divide into based on the unique values of the covariate, takes an array with the number of strata for the corresponding covariate
                passed in `stratify_by`. It takes a default of `4` for each of the covariates.

  While plotting the obtained `VPC` object with `plot` the following keyword arguments allow the option to include or exclude various components with `true` or `false` respectively:

  - `observations`: Scatter plot of the true observations.
  - `simquantile_medians`: The median quantile regression of each quantile from the simulations.
  - `observed_quantiles`: The quantile regressions for the true observations.
  - `ci_bands`: Shaded region between the upper and lower confidence levels of each quantile from the simulations.

  `observations` and `simquantile_medians` are set to `false` by default.

  For most users the method used in quantile regression is not going to be of concern, but if you see large run times switching `qreg_method` to `IP(true)` should help in improving the
  performance with a tradeoff in the accuracy of the fitting.
"""
vpc(fpm::FittedPumasModel,
    reps::Integer=499,
    qreg_method=IP(),
    vpctype::VPCType= ContinuousVPC();
    dv::Symbol = keys(fpm.data[1].observations)[1],
    stratify_by = nothing,
    quantiles::NTuple{3,Float64}=(0.1, 0.5, 0.9),
    level::Real=0.95,
    ensemblealg=EnsembleSerial(),
    bandwidth=2,
    numstrats= stratify_by === nothing ? nothing : [4 for i in 1:length(stratify_by)],
    idv = :time,
    count_vals = unique(skipmissing(DataFrame(fpm.data)[!,dv])),
    sim_idvs = nothing
    ) = vpc(fpm.model, fpm.data, coef(fpm), reps, qreg_method, vpctype;
            dv = dv, stratify_by = stratify_by, quantiles = quantiles,
            level = level, ensemblealg = ensemblealg, bandwidth = bandwidth,
            numstrats = numstrats, idv = idv, count_vals = count_vals, sim_idvs = sim_idvs)


@recipe function f(vpc::PopVPC;observations=true, observed_quantiles = true)
    scatterlabel = ["Observed data", "Observation quantiles"]
    if observations == true
        if vpc.stratify_by === nothing
            for (i,sub) in enumerate(vpc.data[1])
                @series begin
                    seriestype --> :scatter
                    markercolor --> :blue
                    markeralpha --> 0.2
                    label --> (i == 1 ? scatterlabel[1] : "")
                    legend --> :outertop
                    title --> ""
                    sub
                end
            end
        else
            for (pltno,data_strat) in enumerate(vpc.data)
                for (i,sub) in enumerate(data_strat)
                    @series begin
                        subplot --> pltno
                        seriestype --> :scatter
                        markercolor --> :blue
                        markeralpha --> 0.2
                        label --> (i == 1 && pltno == 1 ? scatterlabel[1] : "")
                        legend --> :outertop
                        title --> ""
                        sub
                    end
                end
            end
        end
    end
    if observed_quantiles == true
        if vpc.stratify_by === nothing
            data_quantiles = groupby(vpc.data_quantiles, :τ)
            for i in 1:3
                @series begin
                    linewidth --> 2
                    label --> (i == 1 ? scatterlabel[2] : "")
                    seriescolor --> :red
                    legend --> :outertop
                    data_quantiles[i][!,:time], data_quantiles[i][!,:quantile]
                end
            end
        else
            data_quantiles = groupby(vpc.data_quantiles, vpc.stratify_by)
            colinds = data_quantiles.cols
            colnames = names(data_quantiles)[colinds]
            layout --> good_layout(length(data_quantiles))
            for (pltno,data_quantile) in enumerate(data_quantiles)
                df_data_quantile = groupby(DataFrame(data_quantile),:τ)
                for i in 1:3
                    @series begin
                        subplot --> pltno
                        title --> "Stratified on " * string(["$(colnames[j]): $(data_quantile[1,colinds[j]]) " for j in 1:length(colinds)]...)
                        linewidth --> 2
                        label --> ((i == 1 && pltno == 1) ? scatterlabel[2] : "")
                        seriescolor --> :red
                        legend --> :outertop
                        df_data_quantile[i][!,:time], df_data_quantile[i][!,:quantile]
                    end
                end
            end
        end
    end
end

@recipe function f(vpc::VPC; observations = false, simquantile_medians = false, observed_quantiles = true, ci_bands = true)

    scatterlabel = ["Observed data", "Observation quantiles", "Simulated quantiles", "Simulated $(vpc.level*100)% CI"]
    if vpc.popvpc.stratify_by === nothing
        sim_quantiles = groupby(vpc.simulated_quantiles, :τ)
        for i in 1:3
            @series begin
                if !simquantile_medians
                    linealpha --> 0
                else
                    label --> (i == 1 ? scatterlabel[3] : "")
                end

                if ci_bands == true
                    ribbon --> [sim_quantiles[i][!,:middle] .- sim_quantiles[i][!,:lower],sim_quantiles[i][!,:upper] .- sim_quantiles[i][!,:middle]]
                    fillalpha --> 0.2
                    label --> (i == 1 ? scatterlabel[4] : "")
                end

                seriescolor --> :black
                legend --> :outertop
                sim_quantiles[i][!,:time],sim_quantiles[i][!,:middle]
            end
        end
    else
        sim_quantiles = groupby(vpc.simulated_quantiles, vpc.popvpc.stratify_by)
        colinds = sim_quantiles.cols
        colnames = names(vpc.simulated_quantiles)[colinds]
        layout --> good_layout(length(sim_quantiles))
        for (pltno,sim_quantile) in enumerate(sim_quantiles)
            df_sim_quantile = groupby(sim_quantile, :τ)
            for i in 1:3
                @series begin
                    subplot --> pltno

                    if !simquantile_medians
                        linealpha --> 0
                    else
                        label --> ((i == 1 && pltno == 1) ? scatterlabel[3] : "")
                    end

                    if ci_bands == true
                        ribbon --> [df_sim_quantile[i][!,:middle] .- df_sim_quantile[i][!,:lower],df_sim_quantile[i][!,:upper] .- df_sim_quantile[i][!,:middle]]
                        fillalpha --> 0.2
                        label --> ((i == 1 && pltno == 1) ? scatterlabel[4] : "")
                    end

                    title --> "Stratified on " * string(["$(colnames[j]): $(sim_quantile[1,colinds[j]]) " for j in 1:length(colinds)]...)
                    seriescolor --> :black
                    legend --> :outertop
                    df_sim_quantile[i][!,:time],df_sim_quantile[i][!,:middle]
                end
            end
        end
    end
    observed_quantiles --> observed_quantiles
    observations --> observations
    vpc.popvpc
end
