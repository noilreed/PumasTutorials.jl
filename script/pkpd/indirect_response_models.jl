
using Pumas, LinearAlgebra, Plots


irm1 = @model begin
    @param begin
        θ ∈ VectorDomain(6)
        Ω ∈  PDiagDomain(5)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Ka      = θ[1]
        CL      = θ[2]*exp(η[1])
        Vc      = θ[3]*exp(η[2])
        Kin     = θ[4]*exp(η[3])
        Kout    = θ[5]*exp(η[4])
        IC50    = θ[6]*exp(η[5])
        IMAX    = 1
    end

    @init begin
        Resp = Kin/Kout
    end

    @vars begin
        cp = Cent/Vc
        inhibition = 1-(IMAX*cp/(IC50+cp))
    end

    @dynamics begin
        Gut'    = -Ka*Gut
        Cent'   =  Ka*Gut - (CL/Vc)*Cent
        Resp'   =  Kin*inhibition - Kout*Resp
    end

    @derived begin
        cp     = Cent / Vc
        resp   = Resp
    end
end


param = (θ = [
          0.5, # Ka  Absorption rate constant 1 (1/time)
          5, # CL   Clearance (volume/time)
          30, # Vc   Central volume (volume)
          10, # Kin  Response in rate constant (1/time)
          0.02, # Kout Response out rate constant (1/time)
          10, # IC50 Concentration for 50% of max inhibition (mass/volume)
          1, # IMAX Maximum inhibition
          ],
    Ω = Diagonal([0.04,0.04,0.04,0.04,0.04]));


regimen1 = DosageRegimen(1500, time=0,cmt=1)
subject1 = Subject(id=1,evs=regimen1)


sim = simobs(irm1,subject1,param,obstimes=0:0.1:100)
plot(sim, obsnames=[:cp,:resp])


irm2 = @model begin
    @param begin
        θ ∈ VectorDomain(6)
        Ω ∈  PDiagDomain(5)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Ka      = θ[1]
        CL      = θ[2]*exp(η[1])
        Vc      = θ[3]*exp(η[2])
        Kin     = θ[4]*exp(η[3])
        Kout    = θ[5]*exp(η[4])
        IC50    = θ[6]*exp(η[5])
        IMAX    = 1
    end

    @init begin
        Resp = Kin/Kout
    end

    @vars begin
        cp = Cent/Vc
        inhibition = 1-(IMAX*cp/(IC50+cp))
    end

    @dynamics begin
        Gut'    = -Ka*Gut
        Cent'   =  Ka*Gut - (CL/Vc)*Cent
        Resp'   =  Kin - Kout*inhibition*Resp
    end

    @derived begin
        cp     = Cent / Vc
        resp   = Resp
    end
end


sim = simobs(irm2,subject1,param,obstimes=0:0.1:100)
plot(sim, obsnames=[:cp,:resp])


irm3 = @model begin
    @param begin
        θ ∈ VectorDomain(6)
        Ω ∈  PDiagDomain(5)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Ka      = θ[1]
        CL      = θ[2]*exp(η[1])
        Vc      = θ[3]*exp(η[2])
        Kin     = θ[4]*exp(η[3])
        Kout    = θ[5]*exp(η[4])
        EC50    = θ[6]*exp(η[5])
        EMAX    = 1
    end

    @init begin
        Resp = Kin/Kout
    end

    @vars begin
        cp = Cent/Vc
        stimulation = 1 + (EMAX*cp/(EC50+cp))
    end

    @dynamics begin
        Gut'    = -Ka*Gut
        Cent'   =  Ka*Gut - (CL/Vc)*Cent
        Resp'   =  Kin*stimulation - Kout*Resp
    end

    @derived begin
        cp     = Cent / Vc
        resp   = Resp
    end
end


sim = simobs(irm3,subject1,param,obstimes=0:0.1:100)
plot(sim, obsnames=[:cp,:resp])


irm4 = @model begin
    @param begin
        θ ∈ VectorDomain(6)
        Ω ∈  PDiagDomain(5)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Ka      = θ[1]
        CL      = θ[2]*exp(η[1])
        Vc      = θ[3]*exp(η[2])
        Kin     = θ[4]*exp(η[3])
        Kout    = θ[5]*exp(η[4])
        EC50    = θ[6]*exp(η[5])
        EMAX    = 1
    end

    @init begin
        Resp = Kin/Kout
    end

    @vars begin
        cp = Cent/Vc
        stimulation = 1 + (EMAX*cp/(EC50+cp))
    end

    @dynamics begin
        Gut'    = -Ka*Gut
        Cent'   =  Ka*Gut - (CL/Vc)*Cent
        Resp'   =  Kin - Kout*stimulation*Resp
    end

    @derived begin
        cp     = Cent / Vc
        resp   = Resp
    end
end


sim = simobs(irm4,subject1,param,obstimes=0:0.1:100)
plot(sim, obsnames=[:cp,:resp])


using PumasTutorials
PumasTutorials.tutorial_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

