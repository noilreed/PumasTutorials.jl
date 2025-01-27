
using Dates


using Pumas, CairoMakie, PumasPlots, GlobalSensitivity


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
        FQad ∈ RealDomain(lower = 0.0, init = 0.05, upper = 1.0) #add bounds to parameters for estimation
        FQbo ∈ RealDomain(lower = 0.0, init = 0.05, upper = 1.0)
        FQbr ∈ RealDomain(lower = 0.0, init = 0.12, upper = 1.0)
        FQgu ∈ RealDomain(lower = 0.0, init = 0.16, upper = 1.0)
        FQhe ∈ RealDomain(lower = 0.0, init = 0.04, upper = 1.0)
        FQki ∈ RealDomain(lower = 0.0, init = 0.19, upper = 1.0)
        FQli ∈ RealDomain(lower = 0.0, init = 0.255, upper = 1.0)
        FQmu ∈ RealDomain(lower = 0.0, init = 0.17, upper = 1.0)
        FQsp ∈ RealDomain(lower = 0.0, init = 0.03, upper = 1.0)
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
        Cvenn = VEN./Vve
        cp ~ @. Normal(Cvenn, 0.1) #for estimation
    end
end


regimen_s = DosageRegimen(200, time=0, addl=13, ii=12, cmt=1, ss=1)
sub_s = Subject(id=1, events=regimen_s)


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


simdata = simobs(model, [sub_s], p)
sim_plot(model, simdata, observations=[:Cvenn])


data = read_pumas(DataFrame(simdata), observations = [:cp])
ft = fit(model, data, p, Pumas.NaivePooled(),
    constantcoef = (
        Fup = 0.42, fumic = 0.711, WEIGHT = 73, MPPGL = 30.3, MPPGI = 0,
        C_OUTPUT = 6.5, VmaxH = 40, VmaxG = 40, KmH = 9.3, KmG = 9.3, bp = 1,
        kpad = 9.89, kpbo = 7.91, kpbr = 7.35, kpgu = 5.82, kphe = 1.95,
        kpki = 2.9, kpli = 4.66, kplu = 0.83, kpmu = 2.94, kpsp = 2.96,
        kpre = 4, MW = 349.317, logP = 2.56, s_lumen = 0.39*1000, L = 280,
        d = 2.5, PF = 1.57, VF = 6.5, MF = 13, ITT = 3.32, A = 7440, B = 1e7,
        alpha = 0.6, beta = 4.395, fabs = 1, fdis = 1, fperm = 1, vad = 18.2,
        vbo = 10.5, vbr = 1.45, vguWall = 0.65, vgulumen = 0.35, vhe = 0.33,
        vki = 0.31, vli = 1.8, vlu = 0.5, vmu = 29, vsp = 0.15, vbl = 5.6),
    ensemblealg=EnsembleThreads())


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
        FQad ∈ RealDomain(lower = 0.0, init = 0.05, upper = 1.0)
        FQbo ∈ RealDomain(lower = 0.0, init = 0.05, upper = 1.0)
        FQbr ∈ RealDomain(lower = 0.0, init = 0.12, upper = 1.0)
        FQgu ∈ RealDomain(lower = 0.0, init = 0.16, upper = 1.0)
        FQhe ∈ RealDomain(lower = 0.0, init = 0.04, upper = 1.0)
        FQki ∈ RealDomain(lower = 0.0, init = 0.19, upper = 1.0)
        FQli ∈ RealDomain(lower = 0.0, init = 0.255, upper = 1.0)
        FQmu ∈ RealDomain(lower = 0.0, init = 0.17, upper = 1.0)
        FQsp ∈ RealDomain(lower = 0.0, init = 0.03, upper = 1.0)
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
        Cvenn = VEN./Vve
        #capturing NCA metrics for evaluations
        nca := @nca Cvenn
        auc =  last(NCA.auc(nca))
        cmax = last(NCA.cmax(nca))
    end
end


p_range_low = (fperm=1/3, s_lumen=390/3, ITT = 3.32/3, MPPGI=1.44/3, )

p_range_high = (fperm=1*3, s_lumen=390*3, ITT = 3.32*3, MPPGI=1.44*3, )


regimen_s = DosageRegimen(200, time=0, addl=13, ii=12, cmt=1, ss=1, route = Pumas.NCA.IVInfusion)
sub_s = Subject(id=1, events=regimen_s)
sobol_ = Pumas.gsa(model, sub_s, p, GlobalSensitivity.Sobol(), [:cmax,:auc], p_range_low,p_range_high, N=1000, obstimes=0.0:1.0:30.0)


keys_ = keys(p_range_low)
cmax_s1 = [sobol_.first_order[1,:][key] for key in keys_]
cmax_st = [sobol_.total_order[1,:][key] for key in keys_]

fig = Figure(resolution = (1200, 800))
plot_cmax_s1 = scatter(fig[1,1], 1:4, cmax_s1, axis = (yticks = 0:1, xticks = (1:4, [string.(keys_)...]), label = "First Order", title="Cmax"))
plot_cmax_st = scatter(fig[1,2], 1:4, cmax_st, axis = (yticks = 0:1, xticks = (1:4, [string.(keys_)...]), label = "Total Order"), marker=:utriangle)

auc_s1 = [sobol_.first_order[2,:][key] for key in keys_]
auc_st = [sobol_.total_order[2,:][key] for key in keys_]

plot_auc_s1 = scatter(fig[2,1], 1:4, auc_s1, axis = (yticks = 0:1, xticks = (1:4, [string.(keys_)...]), label = "First Order", title="AUC"))
plot_auc_st = scatter(fig[2,2], 1:4, auc_st, axis = (yticks = 0:1, xticks = (1:4, [string.(keys_)...]), label = "Total Order"), marker=:utriangle)
display(fig)


eFAST_ = Pumas.gsa(model, sub_s, p, GlobalSensitivity.eFAST(), [:cmax,:auc], p_range_low, p_range_high, n=1000, obstimes=0.0:1.0:30.0)


keys_ = keys(p_range_low)
cmax_s1 = [eFAST_.first_order[1,:][key] for key in keys_]
cmax_st = [eFAST_.total_order[1,:][key] for key in keys_]

fig = Figure(resolution = (1200,800))
plot_cmax_s1 = scatter(fig[1,1], 1:4, cmax_s1, axis = (yticks = 0:1, xticks = (1:4, [string.(keys_)...]), label = "First Order", title="Cmax"))
plot_cmax_st = scatter(fig[1,2], 1:4, cmax_st, axis = (yticks = 0:1, xticks = (1:4, [string.(keys_)...]), label = "Total Order"), marker=:utriangle)

auc_s1 = [eFAST_.first_order[2,:][key] for key in keys_]
auc_st = [eFAST_.total_order[2,:][key] for key in keys_]

plot_auc_s1 = scatter(fig[2,1], 1:4, auc_s1, axis = (yticks = 0:1, xticks = (1:4, [string.(keys_)...]), label = "First Order", title="AUC"))
plot_auc_st = scatter(fig[2,2], 1:4, auc_st, axis = (yticks = 0:1, xticks = (1:4, [string.(keys_)...]), label = "Total Order"), marker=:utriangle)
display(fig)

