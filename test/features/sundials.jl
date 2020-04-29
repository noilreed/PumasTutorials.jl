using Pumas, LinearAlgebra, Sundials, DiffEqDevTools, Test

model = @model begin
    @param begin
        Fup ∈ RealDomain(init = 0.42)
        fumic ∈ RealDomain(init = 0.711)
        WEIGHT ∈ RealDomain(init = 73)
        MPPGL ∈ RealDomain(init = 30.3)
        MPPGI ∈ RealDomain(init = 0)
        C_OUTPUT ∈ RealDomain(init = 6.5)
        VmaxH ∈ RealDomain(init = 40)
        VmaxG ∈ RealDomain(init = 40)
        KmH ∈ RealDomain(init = 9.3)
        KmG ∈ RealDomain(init = 9.3)
        bp ∈ RealDomain(init = 1)
        kpad ∈ RealDomain(init = 9.89)
        kpbo ∈ RealDomain(init = 7.91)
        kpbr ∈ RealDomain(init = 7.35)
        kpgu ∈ RealDomain(init = 5.82)
        kphe ∈ RealDomain(init = 1.95)
        kpki ∈ RealDomain(init = 2.9)
        kpli ∈ RealDomain(init = 4.66)
        kplu ∈ RealDomain(init = 0.83)
        kpmu ∈ RealDomain(init = 2.94)
        kpsp ∈ RealDomain(init = 2.96)
        kpre ∈ RealDomain(init = 4)
        MW ∈ RealDomain(init = 349.317)
        logP ∈ RealDomain(init = 2.56)
        s_lumen ∈ RealDomain(init = 0.39*1000)
        L ∈ RealDomain(init = 280)
        d ∈ RealDomain(init = 2.5)
        PF ∈ RealDomain(init = 1.57)
        VF ∈ RealDomain(init = 6.5)
        MF ∈ RealDomain(init = 13)
        ITT ∈ RealDomain(init = 3.32)
        A ∈ RealDomain(init = 7440)
        B ∈ RealDomain(init = 1e7)
        alpha ∈ RealDomain(init = 0.6)
        beta ∈ RealDomain(init = 4.395)
        fabs ∈ RealDomain(init = 1)
        fdis ∈ RealDomain(init = 1)
        fperm ∈ RealDomain(init = 1)
        vad ∈ RealDomain(init = 18.2)
        vbo ∈ RealDomain(init =10.5)
        vbr ∈ RealDomain(init =1.45)
        vguWall ∈ RealDomain(init =0.65)
        vgulumen ∈ RealDomain(init =0.35)
        vhe ∈ RealDomain(init =0.33)
        vki ∈ RealDomain(init =0.31)
        vli ∈ RealDomain(init =1.8)
        vlu ∈ RealDomain(init =0.5)
        vmu ∈ RealDomain(init =29)
        vsp ∈ RealDomain(init =0.15)
        vbl ∈ RealDomain(init =5.6)
        FQad ∈ RealDomain(init = 0.05)
        FQbo ∈ RealDomain(init = 0.05)
        FQbr ∈ RealDomain(init = 0.12)
        FQgu ∈ RealDomain(init = 0.16)
        FQhe ∈ RealDomain(init = 0.04)
        FQki ∈ RealDomain(init = 0.19)
        FQli ∈ RealDomain(init = 0.255)
        FQmu ∈ RealDomain(init = 0.17)
        FQsp ∈ RealDomain(init = 0.03)
    end
    @pre begin
        Vgu = vguWall + vgulumen
        Vve = 0.705*vbl
        Var = 0.295*vbl
        Vre = WEIGHT - (vli+vki+vsp+vhe+vlu+vbo+vbr+vmu+vad+vguWall+vbl)
        CO = C_OUTPUT*60
        Qad = FQad*CO
        Qbo = FQbo*CO
        Qbr = FQbr*CO
        Qgu = FQgu*CO
        Qhe = FQhe*CO
        Qki = FQki*CO
        Qli = FQli*CO
        Qmu = FQmu*CO
        Qsp = FQsp*CO
        Qha = Qli - (Qgu+Qsp)
        Qtot = Qli+Qki+Qbo+Qhe+Qmu+Qad+Qbr
        Qre = CO - Qtot
        Qlu = CO
        Vgulumen = vgulumen
        S_lumen = s_lumen
        VguWall = vguWall
        Kpgu = kpgu
        BP = bp
        Vad = vad
        Kpad = kpad
        Vbr = vbr
        Kpbr = kpbr
        Vhe = vhe
        Kphe = kphe
        Vki = vki
        Kpki = kpki
        fup = Fup
        Vsp = vsp
        Kpsp = kpsp
        Vli = vli
        Kpli = kpli
        Vlu = vlu
        Kplu = kplu
        Kpmu = kpmu
        Kpre = kpre
        Vmu = vmu
        Vbl = vbl
        Vbo = vbo
        Kpbo = kpbo
        SA_abs = pi*L*d*PF*VF*MF*1e-4
        SA_basal = pi*L*d*PF*VF*1e-4
        MA = 10^logP
        MW_eff = MW - (3*17)
        Peff = fperm*A*(((MW_eff^(-alpha-beta))*MA)/((MW_eff^(-alpha)) + B*(MW_eff^(-beta))*MA) * 1e-2 * 3600)
        kd = fdis*Peff*SA_abs*1000/vgulumen
        ka = fabs*Peff*SA_basal*1000/VguWall
        kt = 1/ITT
        scale_factor_H = MPPGL*Vli*1000
        scale_factor_G = MPPGI*VguWall*1000
        CLintHep = ((VmaxH/KmH)*scale_factor_H*60*1e-6)/fumic
        CLintGut = ((VmaxG/KmG)*scale_factor_G*60*1e-6)/fumic
        #CLintHep = CLintHep/fumic
        #CLintGut = CLintGut/fumic
        CLrenal = 0.096
        f = 1
    end
    @dynamics begin
        GUTLUMEN' = -kd*Vgulumen*(f*(GUTLUMEN/Vgulumen) + (1-f)*S_lumen) -
            kt*GUTLUMEN
        GUTWALL' = kd*Vgulumen*(f*(GUTLUMEN/Vgulumen) + (1-f)*S_lumen) -
            ka*GUTWALL - CLintGut*(GUTWALL/VguWall)
        GUT' = ka*GUTWALL + Qgu*((ART/Var) - (GUT/VguWall)/(Kpgu/BP))
        ADIPOSE' = Qad*((ART/Var) - (ADIPOSE/Vad)/(Kpad/BP))
        BRAIN' = Qbr*((ART/Var) - (BRAIN/Vbr)/(Kpbr/BP))
        HEART' = Qhe*((ART/Var) - (HEART/Vhe)/(Kphe/BP))
        KIDNEY' = Qki*((ART/Var) - (KIDNEY/Vki)/(Kpki/BP)) -
            CLrenal*(((KIDNEY/Vki)*fup)/(Kpki/BP))
        LIVER' = Qgu*((GUT/VguWall)/(Kpgu/BP)) + Qsp*((SPLEEN/Vsp)/(Kpsp/BP)) +
            Qha*(ART/Var) - Qli*((LIVER/Vli)/(Kpli/BP)) -
            CLintHep*(((LIVER/Vli)*fup)/(Kpli/BP))
        LUNG' = Qlu*((VEN/Vve) - (LUNG/Vlu)/(Kplu/BP))
        MUSCLE' = Qmu*((ART/Var) - (MUSCLE/Vmu)/(Kpmu/BP))
        SPLEEN' = Qsp*((ART/Var) - (SPLEEN/Vsp)/(Kpsp/BP))
        BONE' = Qbo*((ART/Var) - (BONE/Vbo)/(Kpbo/BP))
        REST' = Qre*((ART/Var) - (REST/Vre)/(Kpre/BP))
        VEN' = Qad*((ADIPOSE/Vad)/(Kpad/BP)) + Qbr*((BRAIN/Vbr)/(Kpbr/BP)) +
            Qhe*((HEART/Vhe)/(Kphe/BP)) + Qki*((KIDNEY/Vki)/(Kpki/BP)) +
            Qli*((LIVER/Vli)/(Kpli/BP)) + Qmu*((MUSCLE/Vmu)/(Kpmu/BP)) +
            Qbo*((BONE/Vbo)/(Kpbo/BP)) + Qre*((REST/Vre)/(Kpre/BP)) -
            Qlu*(VEN/Vve)
        ART' = Qlu*((LUNG/Vlu)/(Kplu/BP) - (ART/Var))
    end
    @derived begin
        gutlumen = GUTLUMEN
        gutwall = GUTWALL
        gut = GUT
        adipose = ADIPOSE
        brain = BRAIN
        heart = HEART
        bone = BONE
        kidney = KIDNEY
        liver = LIVER
        lung = LUNG
        muscle = MUSCLE
        spleen = SPLEEN
        rest = REST
        art = ART
        ven = VEN
        Cvenn = VEN./Vve
        nca := @nca Cvenn
        # auc =  NCA.auc(nca, interval=(0,300))
        cmax = NCA.cmax(nca)
    end
end

pset = ParamSet((
    Fup = RealDomain(init = 0.42),
    fumic = RealDomain(init = 0.711),
    WEIGHT = RealDomain(init = 73),
    MPPGL = RealDomain(init = 30.3),
    MPPGI = RealDomain(init = 0),
    C_OUTPUT = RealDomain(init = 6.5),
    VmaxH = RealDomain(init = 40),
    VmaxG = RealDomain(init = 40),
    KmH = RealDomain(init = 9.3),
    KmG = RealDomain(init = 9.3),
    bp = RealDomain(init = 1),
    kpad = RealDomain(init = 9.89),
    kpbo = RealDomain(init = 7.91),
    kpbr = RealDomain(init = 7.35),
    kpgu = RealDomain(init = 5.82),
    kphe = RealDomain(init = 1.95),
    kpki = RealDomain(init = 2.9),
    kpli = RealDomain(init = 4.66),
    kplu = RealDomain(init = 0.83),
    kpmu = RealDomain(init = 2.94),
    kpsp = RealDomain(init = 2.96),
    kpre = RealDomain(init = 4),
    MW = RealDomain(init = 349.317),
    logP = RealDomain(init = 2.56),
    s_lumen = RealDomain(init = 0.39*1000),
    L = RealDomain(init = 280),
    d = RealDomain(init = 2.5),
    PF = RealDomain(init = 1.57),
    VF = RealDomain(init = 6.5),
    MF = RealDomain(init = 13),
    ITT = RealDomain(init = 3.32),
    A = RealDomain(init = 7440),
    B = RealDomain(init = 1e7),
    alpha = RealDomain(init = 0.6),
    beta = RealDomain(init = 4.395),
    fabs = RealDomain(init = 1),
    fdis = RealDomain(init = 1),
    fperm = RealDomain(init = 1),
    vad = RealDomain(init = 18.2),
    vbo = RealDomain(init =10.5),
    vbr = RealDomain(init =1.45),
    vguWall = RealDomain(init =0.65),
    vgulumen = RealDomain(init =0.35),
    vhe = RealDomain(init =0.33),
    vki = RealDomain(init =0.31),
    vli = RealDomain(init =1.8),
    vlu = RealDomain(init =0.5),
    vmu = RealDomain(init =29),
    vsp = RealDomain(init =0.15),
    vbl = RealDomain(init =5.6),
    FQad = RealDomain(init = 0.05),
    FQbo = RealDomain(init = 0.05),
    FQbr = RealDomain(init = 0.12),
    FQgu = RealDomain(init = 0.16),
    FQhe = RealDomain(init = 0.04),
    FQki = RealDomain(init = 0.19),
    FQli = RealDomain(init = 0.255),
    FQmu = RealDomain(init = 0.17),
    FQsp = RealDomain(init = 0.03)))

rfx_f(p) = ParamSet(())

function col_f(p,randeffs,subject)
    function f(t=nothing)
        CO = p.C_OUTPUT*60
        Qad = p.FQad*CO
        Qbo = p.FQbo*CO
        Qbr = p.FQbr*CO
        Qgu = p.FQgu*CO
        Qhe = p.FQhe*CO
        Qsp = p.FQsp*CO
        Qki = p.FQki*CO
        Qli = p.FQli*CO
        Qmu = p.FQmu*CO
        Qtot = Qli+Qki+Qbo+Qhe+Qmu+Qad+Qbr
        MW_eff = p.MW - (3*17)
        MA = 10^p.logP
        Peff = p.fperm*p.A*(((MW_eff^(-p.alpha-p.beta))*MA)/((MW_eff^(-p.alpha)) + p.B*(MW_eff^(-p.beta))*MA) * 1e-2 * 3600)
        SA_abs = pi*p.L*p.d*p.PF*p.VF*p.MF*1e-4
        SA_basal = pi*p.L*p.d*p.PF*p.VF*1e-4
        VguWall = p.vguWall
        scale_factor_H = p.MPPGL*p.vli*1000
        scale_factor_G = p.MPPGI*VguWall*1000
        (Vgu = p.vguWall + p.vgulumen,
        Vve = 0.705*p.vbl,
        Var = 0.295*p.vbl,
        Vre = p.WEIGHT - (p.vli+p.vki+p.vsp+p.vhe+p.vlu+p.vbo+p.vbr+p.vmu+p.vad+p.vguWall+p.vbl),
        CO = CO,
        Qad = Qad,
        Qbo = Qbo,
        Qbr = Qbr,
        Qgu = Qgu,
        Qhe = Qhe,
        Qki = Qki,
        Qli = Qli,
        Qmu = Qmu,
        Qsp = Qsp,
        Qha = Qli - (Qgu+Qsp),
        Qtot = Qtot,
        Qre = CO - Qtot,
        Qlu = CO,
        Vgulumen = p.vgulumen,
        S_lumen = p.s_lumen,
        VguWall = p.vguWall,
        Kpgu = p.kpgu,
        BP = p.bp,
        Vad = p.vad,
        Kpad = p.kpad,
        Vbr = p.vbr,
        Kpbr = p.kpbr,
        Vhe = p.vhe,
        Kphe = p.kphe,
        Vki = p.vki,
        Kpki = p.kpki,
        fup = p.Fup,
        Vsp = p.vsp,
        Kpsp = p.kpsp,
        Vli = p.vli,
        Kpli = p.kpli,
        Vlu = p.vlu,
        Kplu = p.kplu,
        Kpmu = p.kpmu,
        Kpre = p.kpre,
        Vmu = p.vmu,
        Vbl = p.vbl,
        Vbo = p.vbo,
        Kpbo = p.kpbo,
        SA_abs = SA_abs,
        SA_basal = SA_basal,
        MA = MA,
        MW_eff = MW_eff,
        Peff = Peff,
        kd = p.fdis*Peff*SA_abs*1000/p.vgulumen,
        ka = p.fabs*Peff*SA_basal*1000/VguWall,
        kt = 1/p.ITT,
        scale_factor_H = scale_factor_H,
        scale_factor_G = scale_factor_G,
        CLintHep = ((p.VmaxH/p.KmH)*scale_factor_H*60*1e-6)/p.fumic,
        CLintGut = ((p.VmaxG/p.KmG)*scale_factor_G*60*1e-6)/p.fumic,
        #CLintHep = CLintHep/fumic
        #CLintGut = CLintGut/fumic
        CLrenal = 0.096,
        f = 1)
    end
end

function init_f_iip(col,t0)
    zeros(15)
end

function pbpk_f_iip(du,u,p,t)
    @inbounds begin
        GUTLUMEN = u[1]
        GUTWALL = u[2]
        GUT = u[3]
        ADIPOSE = u[4]
        BRAIN = u[5]
        HEART = u[6]
        KIDNEY = u[7]
        LIVER = u[8]
        LUNG = u[9]
        MUSCLE = u[10]
        SPLEEN = u[11]
        BONE = u[12]
        REST = u[13]
        VEN = u[14]
        ART = u[15]
        du[1] = -p.kd*p.Vgulumen*(p.f*(GUTLUMEN/p.Vgulumen) + (1-p.f)*p.S_lumen) -
            p.kt*GUTLUMEN
        du[2] = p.kd*p.Vgulumen*(p.f*(GUTLUMEN/p.Vgulumen) + (1-p.f)*p.S_lumen) -
            p.ka*GUTWALL - p.CLintGut*(GUTWALL/p.VguWall)
        du[3] = p.ka*GUTWALL + p.Qgu*((ART/p.Var) - (GUT/p.VguWall)/(p.Kpgu/p.BP))
        du[4] = p.Qad*((ART/p.Var) - (ADIPOSE/p.Vad)/(p.Kpad/p.BP))
        du[5] = p.Qbr*((ART/p.Var) - (BRAIN/p.Vbr)/(p.Kpbr/p.BP))
        du[6] = p.Qhe*((ART/p.Var) - (HEART/p.Vhe)/(p.Kphe/p.BP))
        du[7] = p.Qki*((ART/p.Var) - (KIDNEY/p.Vki)/(p.Kpki/p.BP)) -
            p.CLrenal*(((KIDNEY/p.Vki)*p.fup)/(p.Kpki/p.BP))
        du[8] = p.Qgu*((GUT/p.VguWall)/(p.Kpgu/p.BP)) + p.Qsp*((SPLEEN/p.Vsp)/(p.Kpsp/p.BP)) +
            p.Qha*(ART/p.Var) - p.Qli*((LIVER/p.Vli)/(p.Kpli/p.BP)) -
            p.CLintHep*(((LIVER/p.Vli)*p.fup)/(p.Kpli/p.BP))
        du[9] = p.Qlu*((VEN/p.Vve) - (LUNG/p.Vlu)/(p.Kplu/p.BP))
        du[10] = p.Qmu*((ART/p.Var) - (MUSCLE/p.Vmu)/(p.Kpmu/p.BP))
        du[11] = p.Qsp*((ART/p.Var) - (SPLEEN/p.Vsp)/(p.Kpsp/p.BP))
        du[12] = p.Qbo*((ART/p.Var) - (BONE/p.Vbo)/(p.Kpbo/p.BP))
        du[13] = p.Qre*((ART/p.Var) - (REST/p.Vre)/(p.Kpre/p.BP))
        du[14] = p.Qad*((ADIPOSE/p.Vad)/(p.Kpad/p.BP)) + p.Qbr*((BRAIN/p.Vbr)/(p.Kpbr/p.BP)) +
            p.Qhe*((HEART/p.Vhe)/(p.Kphe/p.BP)) + p.Qki*((KIDNEY/p.Vki)/(p.Kpki/p.BP)) +
            p.Qli*((LIVER/p.Vli)/(p.Kpli/p.BP)) + p.Qmu*((MUSCLE/p.Vmu)/(p.Kpmu/p.BP)) +
            p.Qbo*((BONE/p.Vbo)/(p.Kpbo/p.BP)) + p.Qre*((REST/p.Vre)/(p.Kpre/p.BP)) -
            p.Qlu*(VEN/p.Vve)
        du[15] = p.Qlu*((LUNG/p.Vlu)/(p.Kplu/p.BP) - (ART/p.Var))
    end
end

p = (Fup = 0.42, fumic = 0.711, WEIGHT = 73, MPPGL = 30.3, MPPGI = 0,
    C_OUTPUT = 6.5, VmaxH = 40, VmaxG = 40, KmH = 9.3, KmG = 9.3, bp = 1,
    kpad = 9.89, kpbo = 7.91, kpbr = 7.35, kpgu = 5.82, kphe = 1.95, kpki = 2.9,
    kpli = 4.66, kplu = 0.83, kpmu = 2.94, kpsp = 2.96, kpre = 4, MW = 349.317,
    logP = 2.56, s_lumen = 0.39*1000, L = 280, d = 2.5, PF = 1.57, VF = 6.5,
    MF = 13, ITT = 3.32, A = 7440, B = 1e7, alpha = 0.6, beta = 4.395, fabs = 1,
    fdis = 1, fperm = 1, vad = 18.2, vbo = 10.5, vbr = 1.45, vguWall = 0.65,
    vgulumen = 0.35, vhe = 0.33, vki = 0.31, vli = 1.8, vlu = 0.5, vmu = 29,
    vsp = 0.15, vbl = 5.6, FQad = 0.05, FQbo = 0.05, FQbr = 0.12, FQgu = 0.16,
    FQhe = 0.04, FQki = 0.19, FQli = 0.255, FQmu = 0.17, FQsp = 0.03)

#Weight = 73 kg, regimen = 4mg/kg q12h IV
regimen = DosageRegimen(292, time = 0, addl=13, ii=12, cmt=14, rate = 292, ss = 1)
sub1 = Subject(id=1,evs=regimen)
prob = ODEProblem(pbpk_f_iip,nothing,nothing,nothing)

function derived_f(col,sol,obstimes,subject,  param, randeffs)
    (dv=sol(obstimes;idxs=2),)
end

pbpk_iip = PumasModel(pset,rfx_f,col_f,init_f_iip,prob,derived_f)

regimen_s = DosageRegimen(200, time = 0, rate=2, addl=12, ii=6, cmt=1, ss = 1)

sub_s = Subject(id=1,evs=regimen_s)

function generate_population(events,nsubs=4)
      pop = Population(map(i -> Subject(id=i,evs=events),1:nsubs))
    return pop
end

sub_p = generate_population(regimen_s)[1]
lowtolsol = solve(pbpk_iip, sub_p, p, alg=Rodas5(), abstol=1e-14, reltol=1e-14)
appx = TestSolution(lowtolsol)

sol_cvode = solve(pbpk_iip, sub_p, p, alg=CVODE_BDF(), abstol=1e-12, reltol=1e-12)
@test DiffEqDevTools.appxtrue(sol_cvode,appx).errors[:l2] < 1e-8

dsl_switch = solve(model, sub_p, p, alg=Rodas5())
@test dsl_switch.u[1] isa Pumas.LArray
