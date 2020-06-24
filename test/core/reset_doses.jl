using Pumas
model = @model begin
    @param begin
        θCl ∈ RealDomain(lower=0)
        θV ∈ RealDomain(lower=0)
        θKa ∈ RealDomain(lower=0)
    end

    @random begin
        η ~ MvNormal(Diagonal([0.0,0.0]))
    end

    @pre begin
        CL = θCl
        Vc = θV
        Ka = θKa
    end

    @dynamics Depots1Central1

    @derived begin
        cpDepot = Depot
        cp = @. Normal(Depot,1e-12)
    end
end

param = (θCl=0.1, θV=1.0, θKa=0.11)
analytic = (exp(6*[-param.θKa 0; param.θKa -param.θCl/param.θV])*[100, 0])[1]
df = DataFrame(
    id = ones(7),
    time = [0.0,0.0,6.0,12.0,6.0,12.0,6.0],
    evid = [1,0,0,4,0,4,0],
    cp   = [missing,analytic,analytic,missing,analytic,missing,analytic],
    amt  = [100,missing,missing,100,missing,100,missing],
    cmt  = ones(Int,7),
)
sub = read_pumas(df,dvs=[:cp])
@test sub[1].time == [0,6,18,30]
simsub = simobs(model, sub, param, obstimes=0:1:30)
@test simsub[1][:cpDepot][7] ≈ analytic
@test simsub[1][:cpDepot][19] ≈ analytic
@test simsub[1][:cpDepot][31] ≈ analytic
