struct PopVPC
    data_quantiles::DataFrame
    data::Population
    stratify_by::Union{Nothing,Array{Symbol}}
    dv::Symbol
end

struct VPC
    simulated_quantiles::DataFrame
    popvpc::PopVPC
end

function _npqreg(yname::Symbol, xname::Symbol, df::AbstractDataFrame, τ::Real, method; xrange=nothing, bandwidth=nothing)
    ft = QuantileRegressions.npqreg(df[!,yname], df[!,xname], τ, method; xrange=xrange, h=bandwidth)
    return DataFrame(time=ft[1], quantile=ft[2], τ=τ)
end

function discretize_cvs(population, cvname::Symbol, numstrat)
    cvs = vcat([[subject.covariates[cvname] for j in 1:length(subject.time)] for subject in population]...)
    unqcvs = unique(cvs)
    if length(unqcvs) > numstrat
        boundvals = [quantile(unqcvs,i/numstrat) for i in 1:numstrat]
        f_ = function(el)
            for i in 1:numstrat
                if el <= boundvals[i]
                    return boundvals[i]
                end
            end
        end
        cvs = map(f_, cvs)
    end
    return cvs
end

function _vpc(
    population::Population, 
    qreg_method=IP();
    dv::Symbol,
    stratify_by,
    quantiles::NTuple{3,Float64}=(0.1, 0.5, 0.9),
    bandwidth=2,
    numstrats=stratify_by === nothing ? nothing : [4 for i in 1:length(stratify_by)])

    # Convert observations to DataFrame
    df = stratify_by === nothing ? DataFrame(population, include_events=false, include_covariates = false) : DataFrame(population, include_events=false)

    # Use hardcoded names
    df.idv = df.time
    df.dv  = df[!,dv]
    if isa(stratify_by, Array{Symbol})
        for (numstrat,cvname) in zip(numstrats,stratify_by)
            df[!,cvname] .= discretize_cvs(population, cvname, numstrat)
        end
    end

    # filter out missing obs
    df = filter(i -> !ismissing(i.dv), df)

    _xrange = sort(unique(df.idv))

    data_quantiles = mapreduce(vcat, quantiles) do τ
        return combine(t -> _npqreg(:dv, :idv, t, τ, qreg_method; xrange=_xrange, bandwidth=bandwidth), groupby(df, stratify_by === nothing ? [] : stratify_by))
    end

    return data_quantiles
end


function _vpc(
    m::PumasModel,
    population::Population,
    param::NamedTuple,
    reps::Integer=499, 
    qreg_method=IP();
    dv::Symbol,
    stratify_by,
    quantiles::NTuple{3,Float64}=(0.1, 0.5, 0.9),
    level::Real=0.95,
    ensemblealg=EnsembleSerial(),
    bandwidth=2,
    numstrats=stratify_by === nothing ? nothing : [4 for i in 1:length(stratify_by)])

    prediction_probabilities = ((1 - level)/2, 0.5, (1 + level)/2)
    sim_quantiles_v = Array{Any}(undef, reps)

    for i in 1:reps
        # Simulate a new population
        sim_pop = Subject.(simobs(m, population, param, ensemblealg=ensemblealg))
        sim_quantiles_v[i] = _vpc(sim_pop, qreg_method; quantiles=quantiles, dv=dv, bandwidth=bandwidth, stratify_by=stratify_by, numstrats = numstrats)
    end

    sim_quantiles_df = reduce(vcat, sim_quantiles_v)

    sim_quantile_interval =  combine(t -> [quantile(t.quantile, prediction_probabilities)...]', groupby(sim_quantiles_df, stratify_by === nothing ? [:τ, :time] : [:τ, :time, stratify_by...]))
    rename!(sim_quantile_interval, stratify_by === nothing ? [:τ, :time, :lower, :middle, :upper] : [:τ, :time, stratify_by..., :lower, :middle, :upper])

    return sim_quantile_interval
end

function vpc(
    m::PumasModel,
    population::Population,
    param::NamedTuple,
    reps::Integer=499, 
    qreg_method=IP();
    dv::Symbol = keys(population[1].observations)[1], 
    stratify_by = nothing,
    quantiles::NTuple{3,Float64}=(0.1, 0.5, 0.9),
    level::Real=0.95,
    ensemblealg=EnsembleSerial(),
    bandwidth=2,
    numstrats= stratify_by === nothing ? nothing : [4 for i in 1:length(stratify_by)])
    _vpc_data = _vpc(population, qreg_method; dv = dv, stratify_by = stratify_by, quantiles = quantiles, bandwidth = bandwidth, numstrats = numstrats)
    _vpc_simulated = _vpc(m, population, param, reps, qreg_method;dv = dv, stratify_by = stratify_by, quantiles = quantiles, level = level, ensemblealg = ensemblealg, bandwidth = bandwidth, numstrats = numstrats)
    return VPC(_vpc_simulated, PopVPC(_vpc_data, population, stratify_by, dv))
end

function vpc(population::Population,
    reps::Integer=499, qreg_method=IP();
    dv::Symbol = keys(population[1].observations)[1],
    stratify_by = nothing,
    kwargs...)
    _vpc_data = _vpc(population, qreg_method;stratify_by = stratify_by, dv = dv, kwargs...)
    return PopVPC(_vpc_data, population, stratify_by, dv)
end

"""
 vpc(fpm::FittedPumasModel, reps::Integer=499, qreg_method=IP();
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

  While plotting the obtained `VPC` object passing `plottype` keyword argument let's you decide between `:scatter`, `:percentile` and `:interval` forms of VPC
  plot with the default being `:scatter` type.
  
  For most users the method used in quantile regression is not going to be of concern, but if you see large run times switching `qreg_method` to `IP(true)` should help in improving the
  performance with a tradeoff in the accuracy of the fitting.   
"""

vpc(fpm::FittedPumasModel, reps::Integer=499, qreg_method=IP(); 
    dv::Symbol = keys(fpm.data[1].observations)[1], 
    stratify_by = nothing,
    quantiles::NTuple{3,Float64}=(0.1, 0.5, 0.9),
    level::Real=0.95,
    ensemblealg=EnsembleSerial(),
    bandwidth=2,
    numstrats= stratify_by === nothing ? nothing : [4 for i in 1:length(stratify_by)]
    ) = vpc(fpm.model, fpm.data, coef(fpm), reps, qreg_method; 
            dv = dv, stratify_by = stratify_by, quantiles = quantiles, level = level, ensemblealg = ensemblealg, bandwidth = bandwidth, numstrats = numstrats)


@recipe function f(vpc::PopVPC;scatter=true)
    scatterlabel = ["Observed data", "Observation quantiles"]
    if scatter == true
        for (i,sub) in enumerate(vpc.data)
            @series begin
                obsnames --> [vpc.dv]
                seriestype --> :scatter
                markercolor --> :blue
                markeralpha --> 0.2
                label --> (i == 1 ? scatterlabel[1] : "")
                legend --> :outertop
                sub
            end
        end
    end
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
                    title --> "Stratified on " * string(["$(colnames[j]): $(data_quantile[1,colinds[j]])" for j in 1:length(colinds)]...)
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

@recipe function f(vpc::VPC; plottype=:scatter)
    if plottype == :scatter || plottype == :percentile
        scatterlabel = ["Observed data", "Observation quantiles", "Simulated quantiles"]
        if plottype == :scatter
            for (i,sub) in enumerate(vpc.popvpc.data)
                @series begin
                    obsnames --> [vpc.popvpc.dv]
                    seriestype --> :scatter
                    markercolor --> :blue
                    markeralpha --> 0.2
                    label --> (i == 1 ? scatterlabel[1] : "")
                    legend --> :outertop
                    sub
                end
            end
        else
            scatter --> false
            vpc.popvpc
        end
        if vpc.popvpc.stratify_by === nothing
            sim_quantiles = groupby(vpc.simulated_quantiles, :τ)
            for i in 1:3
                @series begin
                    seriescolor --> :black
                    label --> (i == 1 ? scatterlabel[3] : "")
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
                        title --> "Stratified on " * string(["$(colnames[j]): $(sim_quantile[1,colinds[j]])" for j in 1:length(colinds)]...)
                        seriescolor --> :black
                        legend --> :outertop
                        label --> ((i == 1 && pltno == 1) ? scatterlabel[3] : "")
                        df_sim_quantile[i][!,:time],df_sim_quantile[i][!,:middle]
                    end
                end
            end
        end
    elseif plottype == :interval
        if vpc.popvpc.stratify_by === nothing
            sim_quantiles = groupby(vpc.simulated_quantiles, :τ)
            for i in 1:3
                @series begin
                    ribbon --> [sim_quantiles[i][!,:middle] .- sim_quantiles[i][!,:lower],sim_quantiles[i][!,:upper] .- sim_quantiles[i][!,:middle]]
                    fillalpha --> 0.2
                    seriescolor --> :black
                    legend --> :outertop
                    label --> (i == 1 ? "Simulated quantiles" : "")
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
                        ribbon --> [df_sim_quantile[i][!,:middle] .- df_sim_quantile[i][!,:lower],df_sim_quantile[i][!,:upper] .- df_sim_quantile[i][!,:middle]]
                        fillalpha --> 0.2
                        title --> "Stratified on " * string(["$(colnames[j]): $(sim_quantile[1,colinds[j]])" for j in 1:length(colinds)]...)
                        seriescolor --> :black
                        legend --> :outertop
                        label --> ((i == 1 && pltno == 1) ? "Simulated quantiles" : "")
                        df_sim_quantile[i][!,:time],df_sim_quantile[i][!,:middle]
                    end
                end
            end
        end
        scatter --> false
        vpc.popvpc
    end
end
