
using Pkg
Pkg.add("Pumas")


using Pumas


@param begin
  θ ∈ VectorDomain(12)
end


using LinearAlgebra
@random begin
  η ~ MvNormal(Matrix{Float64}(I, 11, 11))
end


@pre begin
    Ka1     = θ[1]
    CL      = θ[2]*exp(η[1])
    Vc      = θ[3]*exp(η[2])
    Q       = θ[4]*exp(η[3])
    Vp      = θ[5]*exp(η[4])
    Kin     = θ[6]*exp(η[5])
    Kout    = θ[7]*exp(η[6])
    IC50    = θ[8]*exp(η[7])
    IMAX    = θ[9]*exp(η[8])
    γ       = θ[10]*exp(η[9])
    Vmax    = θ[11]*exp(η[10])
    Km      = θ[12]*exp(η[11])
    Resp0   = θ[6]/θ[7]
end


@init begin
    Resp = Resp0
end


@dynamics begin
    Ev1'    = -Ka1*Ev1
    Cent'   =  Ka1*Ev1 - (CL+Vmax/(Km+(Cent/Vc))+Q)*(Cent/Vc)  + Q*(Periph/Vp)
    Periph' =  Q*(Cent/Vc)  - Q*(Periph/Vp)
    Resp'   =  Kin*(1-(IMAX*(Cent/Vc)^γ/(IC50^γ+(Cent/Vc)^γ)))  - Kout*Resp
end


@derived begin
    ev1    = Ev1
    cp     = Cent / θ[3]
    periph = Periph
    resp   = Resp
end


using LinearAlgebra
model = @model begin

    @param begin
      θ ∈ VectorDomain(12)
    end

    @random begin
      η ~ MvNormal(Matrix{Float64}(I, 11, 11))
    end

    @pre begin
        Ka1     = θ[1]
        CL      = θ[2]*exp(η[1])
        Vc      = θ[3]*exp(η[2])
        Q       = θ[4]*exp(η[3])
        Vp      = θ[5]*exp(η[4])
        Kin     = θ[6]*exp(η[5])
        Kout    = θ[7]*exp(η[6])
        IC50    = θ[8]*exp(η[7])
        IMAX    = θ[9]*exp(η[8])
        γ       = θ[10]*exp(η[9])
        Vmax    = θ[11]*exp(η[10])
        Km      = θ[12]*exp(η[11])
        Resp0   = θ[6]/θ[7]
    end

    @init begin
        Resp = Resp0
    end

    @dynamics begin
        Ev1'    = -Ka1*Ev1
        Cent'   =  Ka1*Ev1 - (CL+Vmax/(Km+(Cent/Vc))+Q)*(Cent/Vc)  + Q*(Periph/Vp)
        Periph' =  Q*(Cent/Vc)  - Q*(Periph/Vp)
        Resp'   =  Kin*(1-(IMAX*(Cent/Vc)^γ/(IC50^γ+(Cent/Vc)^γ)))  - Kout*Resp
    end

    @derived begin
        ev1    = Ev1
        cp     = Cent / θ[3]
        periph = Periph
        resp   = Resp
    end
end


DosageRegimen(15, time=0)


regimen = DosageRegimen([15,15,15,15], time=[0,4,8,12])


subject = Subject(id=1,evs=regimen)


fixeffs = (θ = [
          1, # Ka1  Absorption rate constant 1 (1/time)
          1, # CL   Clearance (volume/time)
          20, # Vc   Central volume (volume)
          2, # Q    Inter-compartmental clearance (volume/time)
          10, # Vp   Peripheral volume of distribution (volume)
          10, # Kin  Response in rate constant (1/time)
          2, # Kout Response out rate constant (1/time)
          2, # IC50 Concentration for 50% of max inhibition (mass/volume)
          1, # IMAX Maximum inhibition
          1, # γ    Emax model sigmoidicity
          0, # Vmax Maximum reaction velocity (mass/time)
          2  # Km   Michaelis constant (mass/volume)
          ],)

sim = simobs(model, subject, fixeffs)


using Plots
plot(sim)


plot(sim,
     color=2,thickness_scaling=1.5,
     legend=false, lw=2)


randeffs = (η = rand(11),)
sim = simobs(model, subject, fixeffs, randeffs)
plot(sim)


sim = simobs(model, subject, fixeffs, randeffs, obstimes = 0:0.1:19)
plot(sim)


sim[:cp]


df = DataFrame(sim)
first(df,6) # Print only the first 6: the DataFrame is huge!


plot(sim.times,sim[:ev1])


using PumasTutorials
PumasTutorials.tutorial_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

