using Pumas
using CSV

@testset "BOV" begin
################ Data importing ##########################
pkdata = CSV.read(example_data("event_data/bov1"), missingstring=".")
pkdata[!,:OCC] .= pkdata.OCC1 .+ 2 .* pkdata.OCC2 .+ 3 .* pkdata.OCC3

################### data mapping ########################
#read_pumas
data_PK_left = read_pumas(pkdata,
                    cvs =   [:CRCL, :OCC],
                    dvs =   [:DV],
                    id  =   :ID,
                    amt =   :AMT,
                    rate =   :RATE,
                    time =   :TIME,
                    evid =   :EVID,
                    cmt = :CMT,
                    cvs_direction=:left)
data_PK = read_pumas(pkdata,
                    cvs =   [:CRCL, :OCC],
                    dvs =   [:DV],
                    id  =   :ID,
                    amt =   :AMT,
                    rate =   :RATE,
                    time =   :TIME,
                    evid =   :EVID,
                    cmt = :CMT,
                    cvs_direction=:right)

# we have observations at 1.267 and 1.283, so they should disagree about
# the value of OCC > 1.267, so we evaluate at prevfloat(1.283) and 1.283
@test data_PK_left[1].covariates(1.267).OCC == 1
@test data_PK[1].covariates(1.267).OCC == 1
@test data_PK_left[1].covariates(prevfloat(1.283)).OCC == 1
@test data_PK[1].covariates(prevfloat(1.283)).OCC == 2
@test data_PK_left[1].covariates(1.283).OCC == 2
@test data_PK[1].covariates(1.283).OCC == 2

###### IV infusion + 2 separate IV boluses ################
################# model 2 #################################
# 3 cmt PK base model, comb.err, eta_CL,V, CRCL/100_CL#####
EACA_PKmodel = @model begin

  @param   begin
      TVCL     ∈ RealDomain(lower = 0.0, init = 7.0)
      TVVc     ∈ RealDomain(lower = 0.0, init = 6.0)
      TVQfast  ∈ RealDomain(lower = 0.0, init = 34.0)
      TVVpfast ∈ RealDomain(lower = 0.0, init = 7.0)
      TVQslow  ∈ RealDomain(lower = 0.0, init = 6.0)
      TVVpslow ∈ RealDomain(lower = 0.0, init = 21.0)
      ΩκCL     ∈ RealDomain(lower=0.0, init=0.04)
      ΩκV      ∈ RealDomain(lower=0.0, init=0.04)
      Ω        ∈ PDiagDomain(2)
      σ²_prop  ∈ RealDomain(lower = 0, init = 0.01)
      σ²_add   ∈ RealDomain(lower = 0, init = 50)
  end

  @random begin
      η ~ MvNormal(Ω)
      κCL ~ MvNormal([ΩκCL,ΩκCL,ΩκCL])
      κV ~ MvNormal([ΩκV,ΩκV,ΩκV])
  end

  @covariates CRCL OCC

  @pre begin
      CL     = TVCL * (CRCL/100) * exp(η[1] + κCL[OCC])
      Vc     = TVVc * exp(η[2]+ κV[OCC])
      Qfast  = TVQfast
      Vpfast = TVVpfast
      Qslow  = TVQslow
      Vpslow = TVVpslow
  end

  @dynamics begin
      Central' = -(CL/Vc)*Central + (Qfast/Vpfast)*Periph1 - (Qfast/Vc)*Central + (Qslow/Vpslow)*Periph2 - (Qslow/Vc)*Central
      Periph1' = (Qfast/Vc)*Central - (Qfast/Vpfast)*Periph1
      Periph2' = (Qslow/Vc)*Central - (Qslow/Vpslow)*Periph2
  end

  @derived begin
      CP = @. Central / Vc
      DV = @. Normal(CP, sqrt(CP^2*σ²_prop + σ²_add))
  end

end

results_comb_err_wo_iov = fit(EACA_PKmodel,
                    data_PK,
                    init_param(EACA_PKmodel),
                    Pumas.FOCEI(),
                  optimize_fn=Pumas.DefaultOptimizeFN(show_trace=true, extended_trace=false))

@test deviance(results_comb_err_wo_iov) ≈ 1864.8081364379095 # verified to match nonmem to the fourth decimal

#Infer_results_comb_wo_iov = infer(results_comb_err_wo_iov)
end