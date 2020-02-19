using DataStructures: OrderedDict, OrderedSet
using MacroTools
using ModelingToolkit

islinenum(x) = x isa LineNumberNode
function nt_expr(set, prefix=nothing)
  if isempty(set)
    return :(NamedTuple())
  else
    t = :(())
    for p in set
      sym = prefix === nothing ? p : Symbol(prefix, p)
      push!(t.args, :($p = $sym))
    end
    return t
  end
end
function nt_expr(dict::AbstractDict, prefix=nothing)
  if isempty(dict)
    return :(NamedTuple())
  else
    t = :(())
    for (p,d) in pairs(dict)
      sym = prefix === nothing ? d : Symbol(prefix, d)
      push!(t.args, :($p = $sym))
    end
    return t
  end
end

function var_def(tupvar, indvars)
  quote
    if $tupvar != nothing
      if $tupvar isa NamedTuple
        # Allow for NamedTuples to be in different order
        $(Expr(:block, [:($(esc(v)) = $tupvar.$v) for v in indvars]...))
      else
        $(Expr(:tuple, (esc(v) for v in indvars)...)) = $tupvar
      end
    end
  end
end
var_def(tupvar, inddict::AbstractDict) = var_def(tupvar, keys(inddict))

# This is used in derived. It creates vectors for all variables referenced in
# @pre. This is too expensive for constant covariates, so we have a fast path there
# that creates singleton pre variables that will still broadcast.
function timed_var_def(tupvar, indvars_pre, timevar)
  quote
    indvars_times = $tupvar.($timevar)
    $(Expr(:block, [:($(esc(v)) = map(t->t[$(QuoteNode(v))], indvars_times)) for v in indvars_pre]...))
  end
end
function extract_params!(vars, params, exprs)
  # should be called on an expression wrapped in a @param
  # expr can be:
  #  a block
  #  a constant expression
  #    a = 1
  #  a domain
  #    a ∈ RealDomain()
  #  a random variable
  #    a ~ Normal()
  if exprs isa Expr && exprs.head == :block
    _exprs = exprs.args
  else
    _exprs = (exprs,)
  end

  for expr in _exprs
    islinenum(expr) && continue
    @assert expr isa Expr
    if expr.head == :call && expr.args[1] in (:∈, :in)
      p = expr.args[2]
      p in vars && error("Variable $p already defined")
      push!(vars,p)
      params[p] = expr.args[3]
    elseif expr.head == :(=)
      p = expr.args[1]
      p in vars && error("Variable $p already defined")
      push!(vars,p)
      params[p] = :(ConstDomain($(expr.args[2])))
    elseif expr.head == :call && expr.args[1] == :~
      p = expr.args[2]
      p in vars && error("Variable $p already defined")
      push!(vars,p)
      params[p] = expr.args[3]
    else
      error("Invalid @param expression: $expr")
    end
  end
end

function param_obj(params)
  :(ParamSet($(esc(nt_expr(params)))))
end

function extract_randoms!(vars, randoms, exprs)
  # should be called on an expression wrapped in a @random
  # expr can be:
  #  a block
  #  a constant expression
  #    a = 1
  #  a random var
  #    a ~ Normal(0,1)

  if exprs isa Expr && exprs.head == :block
    _exprs = exprs.args
  else
    _exprs = (exprs,)
  end

  for expr in _exprs
    islinenum(expr) && continue
    @assert expr isa Expr
    if expr.head == :call && expr.args[1] == :~
      p = expr.args[2]
      p in vars && error("Variable $p already defined")
      push!(vars,p)
      randoms[p] = expr.args[3]
      # TODO support const expressions
      # elseif expr.head == :(=)
      #     p = expr.args[1]
      #     p in vars && error("Variable $p already defined")
      #     push!(vars,p)
      #     randoms[p] = :(ConstDomain($(expr.args[2])))
    else
      error("Invalid @random expression: $expr")
    end
  end
end

function random_obj(randoms, params)
  quote
    function (_param)
      $(var_def(:_param, params))
      ParamSet($(esc(nt_expr(randoms))))
    end
  end
end

function extract_syms!(vars, subvars, syms)
  if length(syms) == 1 && first(syms) isa Expr && first(syms).head == :block
    _syms = first(syms).args
  else
    _syms = syms
  end

  for p in _syms
    islinenum(p) && continue
    p in vars && error("Variable $p already defined")
    push!(subvars, p)
    push!(vars, p)
  end
end

function extract_pre!(vars, prevars, exprs)
  # should be called on an expression wrapped in a @pre
  # expr can be:
  #  a block
  #  an assignment
  #    a = 1

  if exprs isa Expr && exprs.head == :block
    _exprs = exprs.args
  else
    _exprs = (exprs,)
  end

  newexpr = Expr(:block)
  for expr in _exprs
    MacroTools.prewalk(expr) do ex
      if ex isa Expr && ((iseq = ex.head == :(=)) || ex.head == :(:=)) && length(ex.args) == 2
        p = ex.args[1]
        p in vars && error("Variable $p already defined")
        if iseq
          push!(vars, p)
          push!(prevars, p)
        end
        push!(newexpr.args, :($p = $(ex.args[2])))
      else
        ex
      end
    end
  end
  newexpr
end

_keys(x) = x
_keys(x::AbstractDict) = keys(x)

# This function is called in @model to construct the function that returns
# a function to evaluate the pre block for a subject given parameters
function pre_obj(preexpr, prevars, params, randoms, covariates)
  quote
    # This function is called when defining a differential equations problem
    function (_param::NamedTuple, _random::NamedTuple, _subject::Subject)
      # pre is evaluated at t. All covariates are available in `covar`.
      function pre(t)
        covar = _subject.tvcov(t)
        $(Expr(:escape, :t)) = t
        $(Expr(:block, [:($(esc(v)) = covar.$v) for v in _keys(covariates)]...))
        $(Expr(:block, [:($(esc(v)) = _param.$v) for v in _keys(params)]...))
        $(Expr(:block, [:($(esc(v)) = _random.$v) for v in _keys(randoms)]...))
        $(esc(preexpr))
        $(esc(nt_expr(prevars)))
      end
    end
  end
end


function extract_parameter_calls!(expr::Expr,prevars,callvars)
  if expr.head == :call && expr.args[1] ∈ prevars
    push!(callvars,expr.args[1])
  end
  extract_parameter_calls!.(expr.args,(prevars,),(callvars,))
end
extract_parameter_calls!(expr,prevars,callvars) = nothing

function extract_dynamics!(vars, odevars, prevars, callvars, ode_init, expr::Expr, eqs)
  if expr.head == :block
    for ex in expr.args
      islinenum(ex) && continue
      extract_dynamics!(vars, odevars, prevars, callvars, ode_init, ex, eqs)
    end
  elseif expr.head == :(=) || expr.head == :(:=)
    dp = expr.args[1]
    isder = dp isa Expr && dp.head == Symbol('\'')
    #dp in vars && error("Variable $dp already defined")
    (isder && length(dp.args) == 1) ||
      error("Invalid variable $dp: must be in the form of X'")
    p = dp.args[1]
    lhs = :(___D($p))
    push!(eqs.args, :($lhs ~ $(expr.args[2])))
    extract_parameter_calls!(expr.args[2],prevars,callvars)
    if p ∈ keys(ode_init)
      push!(odevars, p)
    elseif p ∉ vars
      push!(vars,p)
      push!(odevars, p)
      ode_init[p] = 0.0
    end
  else
    error("Invalid @dynamics expression: $expr")
  end
  return true # isstatic
end

# used for pre-defined analytical systems
function extract_dynamics!(vars, odevars, prevars, callvars, ode_init, sym::Symbol, eqs)
  obj = getproperty(@__MODULE__,sym)
  if obj isa Type && obj <: ExplicitModel # explict model
    for p in varnames(obj)
      if p in keys(ode_init)
        push!(odevars, p)
      else
        if p ∉ vars
          push!(vars,p)
          push!(odevars, p)
        end
        ode_init[p] = 0
      end
    end
    return true # isstatic
  else
    # we assume they are defined in @init
    for p in keys(ode_init)
      push!(odevars, p)
    end
    return false # isstatic
  end
end

function init_obj(ode_init,odevars,prevars,isstatic)
  if isstatic
    vecexpr = []
    for p in odevars
      push!(vecexpr, ode_init[p])
    end
    uType = SLArray{Tuple{length(odevars)},(odevars...,)}
    typeexpr = :($uType())
    append!(typeexpr.args,vecexpr)
    quote
      function (_pre,t)
        # since _pre is a function, we will unpack prevars (on the left hand side)
        # from a call to _pre(t) (on the right hand side)
        $(var_def(:(_pre(t)), prevars))
        $(esc(typeexpr))
      end
    end
  else
    vecexpr = :([])
    for p in odevars
      push!(vecexpr.args, ode_init[p])
    end
    quote
      function (_pre,t)
        # since _pre is a function, we will unpack prevars (on the left hand side)
        # from a call to _pre(t) (on the right hand side)
        $(var_def(:(_pre(t)), prevars))
        $(esc(vecexpr))
      end
    end
  end
end

function dynamics_obj(odeexpr::Expr, pre, odevars, callvars, bvars, eqs, isstatic)
  odeexpr == :() && return nothing

  # dvars and params are Operations (does params still have to now that pre is "pre-evaluated" in the wrapper function?)
  dvars = []
  params = []
  mteqs = []
  fname = gensym(:PumasDiffEqFunction)
  jname = gensym(:PumasJacobianFunction)
  Wname = gensym(:PumasWFactFunction)
  W_tname = gensym(:PumasW_tFactFunction)
  funcname = gensym(:PumasODEFunction)
  diffeq = :(ODEProblem{false}($funcname,nothing,nothing,nothing))

  # DVar - create array of dynamic variables
  # Combines symbols (v's), the Variable constructor and
  # the differential `D` to create dynamic variables of
  # the type ModelingToolkit.Operation.
  t = Variable(:t; known = true)()
  D = Differential(t)
  for v in odevars
    push!(dvars, Variable(v)(t))
  end

  # Param
  for p in pre
    if p ∈ callvars
      push!(params, Variable(p; known=true))
    else
      push!(params, Variable(p; known=true)())
    end
  end

  for eq in eqs.args
    lhsvar = D(dvars[findfirst(x->x.op.name == eq.args[2].args[2],dvars)])
    rhseq = eq.args[3]
    push!(mteqs,lhsvar ~ convert_rhs_to_Expression(rhseq,bvars,dvars,params,t))
  end

  f_ex = ModelingToolkit.generate_function(ODESystem(mteqs),dvars,params)[1]
  J_ex = ModelingToolkit.generate_jacobian(ODESystem(mteqs),dvars,params)[1]
  if length(eqs.args) < 4
    W_exs = ModelingToolkit.generate_factorized_W(ODESystem(mteqs),dvars,params,simplify=false)
    W_ex = W_exs[1][1]
    W_t_ex = W_exs[2][1]
  else
    W_ex = :nothing
    W_t_ex = :nothing
  end

  quote
    let
      $fname = $f_ex
      $jname = $J_ex
      $Wname = $W_ex
      $W_tname = $W_t_ex
      $funcname = ODEFunction($fname,jac=$jname,Wfact=$Wname,Wfact_t=$W_tname)
      $diffeq
    end
  end
end

function convert_rhs_to_Expression(s::Symbol, bvars, dvars, params, t)
  s == t.op.name && return t
  i = findfirst(x->x.op.name == s,dvars)
  i !== nothing && return dvars[i]
  i = findfirst(params) do x
    if x isa Operation
      x.op.name == s
    else # For CL(t)
      x.name == s
    end
  end
  i !== nothing && return params[i]

  # handle vars expression by substitution
  for i in 1:(length(bvars.args)÷2)
    if s == bvars.args[2*i].args[1]
      return convert_rhs_to_Expression(bvars.args[2*i].args[2].args[3],bvars,dvars,params,t)
    end
  end

  error("Unknown symbol $(string(s)) detected")
end
convert_rhs_to_Expression(x::Number, bvars, dvars, params, t) = ModelingToolkit.Constant(x)

function convert_rhs_to_Expression(ex,bvars,dvars,params,t)
  ex.head === :if && (ex = Expr(:call, ifelse, ex.args...))

  i = findfirst(x->x.op.name == ex.args[1],dvars)
  j = findfirst(params) do x
    if x isa Operation
      x.op.name == ex.args[1]
    else # For CL(t)
      x.name == ex.args[1]
    end
  end

  if i === j === nothing
    op = getproperty(@__MODULE__,ex.args[1])
    args = convert_rhs_to_Expression.(ex.args[2:end],(bvars,),(dvars,),(params,),t)
    return Operation(op, args)
  else # ex is a call, like CL(t)
    if i !== nothing
      var = dvars[i]
    elseif j !== nothing
      var = params[j]
    end
    return var(convert_rhs_to_Expression.(ex.args[2:end],(bvars,),(dvars,),(params,),t)...)
  end
end

function dynamics_obj(odename::Symbol, pre, odevars, callvars, bvars, eqs, isstatic)
  quote
    ($odename isa Type && $odename <: ExplicitModel) ? $odename() : $odename
  end
end

function extract_defs!(vars, defsdict, exprs)
  # should be called on an expression wrapped in a @derived
  # expr can be:
  #  a block
  #  an assignment
  #    a = 1

  if exprs isa Expr && exprs.head == :block
    _exprs = exprs.args
  else
    _exprs = (exprs,)
  end

  for expr in _exprs
    islinenum(expr) && continue
    @assert expr isa Expr
    if expr.head == :(=)
      p = expr.args[1]
      p in vars && error("Variable $p already defined")
      push!(vars,p)
      defsdict[p] = expr.args[2]
    else
      error("Invalid expression: $expr")
    end
  end
end

function extract_randvars!(vars, randvars, postexpr, expr)
  @assert expr isa Expr
  if expr.head == :block
    for ex in expr.args
      islinenum(ex) && continue
      extract_randvars!(vars, randvars, postexpr, ex)
    end
  elseif ((iseq = expr.head == :(=)) || expr.head == :(:=)) && length(expr.args) == 2
    ps = expr.args[1]
    istuple = ps isa Expr && ps.head == :tuple
    ps = istuple ? ps.args : [ps]
    for (i,p) in enumerate(ps)
      if p ∉ vars && iseq
        push!(vars,p)
        push!(randvars,p)
      end
      push!(postexpr.args, istuple ? :($p = $(expr.args[2])[$i]) : :($p = $(expr.args[2])))
    end
  elseif expr isa Expr && expr.head == :call && expr.args[1] == :~ && length(expr.args) == 3
    p = expr.args[2]
    if p ∉ vars
      push!(vars,p)
      push!(randvars,p)
    end
    push!(postexpr.args, :($p = $(expr.args[3])))
  else
    error("Invalid expression: $expr")
  end
end

function bvar_def(collection, indvars)
  quote
    if $collection isa DataFrame
      $(Expr(:block, [:($(esc(v)) = $collection.$v) for v in indvars]...))
    else eltype($collection) <: SArray
      $(Expr(:block, [:($(esc(v)) = map(x -> x[$i], $collection)) for (i,v) in enumerate(indvars)]...))
      #else
      #$(Expr(:block, [:($(esc(v)) = map(x -> x.$v, $collection)) for v in indvars]...))
    end
  end
end

function solvars_def(collection, odevars)
  quote
    $(Expr(:block, [:($(esc(v)) = $collection[$i,:]) for (i,v) in enumerate(odevars)]...))
  end
end

function derived_obj(derivedexpr, derivedvars, pre, odevars, params, randoms)
  quote
    function (_pre,_sol,_obstimes,_subject,_param, _random)
      $(esc(:events)) = _subject.events
      $(esc(:t)) = _obstimes
      $(var_def(:_param, params))
      $(var_def(:_random, randoms))

      # Unpack all solution variables such that concentrations etc can be used
      # with pre-vars (see below) to compute concentrations, etc.
      if _sol != nothing
        if typeof(_sol) <: PKPDAnalyticalSolution
          _solarr = _sol(_obstimes)
        else
          _solarr = _sol
        end
        $(solvars_def(:(_solarr), odevars))
      end
      # timed_var_def evaluates pre at _obstimes with `_pre.(_obstimes)` and
      # unpacks all pre-variables as vectors. This way all timevarying things
      # are evaluated and ready for use, say
      # @derived begin
      #   cp = @. (Central / v)
      #   dv ~ @. Normal(cp, sqrt(cp^2*σ_prop))
      # end
      # Where `v` is from `pre`, but the use in the at-derived block implies
      # that we want to use v at all obstime (in the denominator of the fraction
      # for cp that depends also on Central at obstimes, though this is unpacked
      # from the solution, not pre)
      if _subject.tvcov isa ConstantCovar
        $(var_def(:(_pre(0.0)), pre))
      else
        $(timed_var_def(:_pre, pre, :_obstimes))
      end
      $(esc(derivedexpr))
      $(esc(nt_expr(derivedvars)))
    end
  end
end

function observed_obj(observedexpr, observedvars, pre, odevars, derivedvars)
  quote
    function (_pre,_sol,_obstimes,_samples,_subject)
      $(esc(:events)) = _subject.events
      $(esc(:t)) = _obstimes
      if _sol != nothing
        if typeof(_sol) <: PKPDAnalyticalSolution
          _solarr = _sol(_obstimes)
        else
          _solarr = _sol
        end
        $(solvars_def(:(_solarr), odevars))
      end
      if _subject.tvcov isa ConstantCovar
        $(var_def(:(_pre(0.0)), pre))
      else
        $(timed_var_def(:_pre, pre, :_obstimes))
      end
      $(var_def(:_samples, derivedvars))
      $(esc(observedexpr))
      $(esc(nt_expr(observedvars)))
    end
  end
end

function broadcasted_vars(vars)
  foreach(vars.args) do ex
    isa(ex, Expr) || return
    ex.args[2] = :(@. $(ex.args[2]))
  end
  return vars
end

function add_vars(ex::Expr, vars)
  args = ex.head === :block ? ex.args : [ex]
  return Expr(:block, [vars.args; args]...)
end

to_ncasubj(name, t, events) = NCASubject(name, t, dose=map(ev->convert(NCADose, ev), events), clean=false, check=false)

macro nca(name)
  esc(:(Pumas.to_ncasubj($name, t, events)))
end

macro nca(names...)
  ex = Expr(:tuple)
  ex.args = [:(Pumas.to_ncasubj($name, t, events)) for name in names]
  esc(ex)
end

macro model(expr)

  vars = Set{Symbol}([:t]) # t is the only reserved symbol
  params = OrderedDict{Symbol, Any}()
  randoms = OrderedDict{Symbol, Any}()
  covariates = OrderedSet{Symbol}()
  prevars  = OrderedSet{Symbol}()
  ode_init  = OrderedDict{Symbol, Any}()
  odevars  = OrderedSet{Symbol}()
  preexpr = :()
  derivedvars = OrderedSet{Symbol}()
  eqs = Expr(:vect)
  derivedvars = OrderedSet{Symbol}()
  derivedexpr = Expr(:block)
  observedvars = OrderedSet{Symbol}()
  observedexpr = Expr(:block)
  bvars = :(begin end)
  callvars  = OrderedSet{Symbol}()
  local vars, params, randoms, covariates, prevars, preexpr, odeexpr, odevars
  local ode_init, eqs, derivedexpr, derivedvars, observedvars, observedexpr
  local isstatic, bvars, callvars

  isstatic = true
  odeexpr = :()
  lnndict = Dict{Symbol,LineNumberNode}()

  for ex ∈ expr.args
    islinenum(ex) && continue
    (isa(ex, Expr) && ex.head === :macrocall) || throw(ArgumentError("expected macro call, got $ex"))

    if ex.args[1] == Symbol("@param")
      extract_params!(vars, params, ex.args[3])
    elseif ex.args[1] == Symbol("@random")
      extract_randoms!(vars, randoms, ex.args[3])
    elseif ex.args[1] == Symbol("@covariates")
      extract_syms!(vars, covariates, ex.args[3:end])
    elseif ex.args[1] == Symbol("@pre")
      preexpr = extract_pre!(vars,prevars,ex.args[3])
    elseif ex.args[1] == Symbol("@vars")
      bvars = broadcasted_vars(ex.args[3])
    elseif ex.args[1] == Symbol("@init")
      extract_defs!(vars,ode_init, add_vars(ex.args[3], bvars))
    elseif ex.args[1] == Symbol("@dynamics")
      isstatic = extract_dynamics!(vars, odevars, prevars, callvars, ode_init, ex.args[3], eqs)
      odeexpr = ex.args[3]
    elseif ex.args[1] == Symbol("@derived")
      extract_randvars!(vars, derivedvars, derivedexpr, add_vars(ex.args[3], bvars))
      observedvars = copy(derivedvars)
    elseif ex.args[1] == Symbol("@observed")
      extract_randvars!(vars, observedvars, observedexpr, add_vars(ex.args[3], bvars))
    else
      throw(ArgumentError("Invalid macro $(ex.args[1])"))
    end
    #return nothing
  end

  ex = quote
    x = PumasModel(
    $(param_obj(params)),
    $(random_obj(randoms,params)),
    $(pre_obj(preexpr,prevars,params,randoms,covariates)),
    $(init_obj(ode_init,odevars,prevars,isstatic)),
    $(dynamics_obj(odeexpr,prevars,odevars,callvars,bvars,eqs,isstatic)),
    $(derived_obj(derivedexpr,derivedvars,prevars,odevars,params,randoms)),
    $(observed_obj(observedexpr,observedvars,prevars,odevars,derivedvars)))
    function Base.show(io::IO, ::typeof(x))
      println(io,"PumasModel")
      println(io,"  Parameters: ",$(join(keys(params),", ")))
      println(io,"  Random effects: ",$(join(keys(randoms),", ")))
      println(io,"  Covariates: ",$(join(covariates,", ")))
      println(io,"  Dynamical variables: ",$(join(odevars,", ")))
      println(io,"  Derived: ",$(join(derivedvars,", ")))
      println(io,"  Observed: ",$(join(observedvars,", ")))
    end
    x
  end
  ex.args[1] = __source__
  ex
end
