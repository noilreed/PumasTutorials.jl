libraries <- c("ggplot2",
               "knitr")
               
suppressMessages(sapply(libraries, require, character.only=T))


## Style Template for Plotting


#Create style template

base_theme_obs<-function()
{ theme(strip.text.x = element_text(size = 14, face = "bold")) +
    theme(axis.title.x = element_text(face="bold", size=16)) +
    theme(axis.text.x = element_text( face="bold",size=12, colour='black')) +
    theme(axis.title.y = element_text(face="bold", size=16)) +
    theme(axis.text.y = element_text( face="bold",size=12, colour='black')) +
    theme(legend.text = element_text(size = 14)) +
    theme(legend.title = element_text(size = 14, face = "bold")) +
    theme(panel.background = element_rect(fill='white', colour='black'))+
  theme(plot.title = element_text(face="bold", size=18))
  }


col=c("blue","red","lightblue","darkgreen","lightgreen","orange")
col2=c("darkblue","cyan3")
col3=c("darkblue","cyan3","black")
col4=c("darkblue","cyan3","lightgreen","darkgreen")
col5=c("black","darkblue","cyan3","lightgreen","darkgreen")
col6=c("darkblue","cyan3","lightgreen","darkgreen","blue","lightblue")
col7=c("darkred","red","orange")


# Power simulation function 
# power_sim <- function(n, dose, CL, logit_param, alpha = 0.05, l = 1000) {
  # n:           number of subjects in each dose group (vector or scalar)
  # dose:        dose groups (vector or scalar)
  # CL:          vector containting the median  and coeffcient of variation (as a proportion) clearance (CL) parameters  
  # logit_param: vector containing the intercept (B0) and slope (B1) parameters of the logistic regression model
  # alpha:       type-I error rate (defaul = 0.05)
  # l:           number of study replications (default = 1000)  
  
  n = seq(from = 10, to = 150, by = 5)
  dose = c(1,2)
  CL = c(1,0.25)
  logit_param = c(-1.5, 1.0)
  alpha = 0.05
  l = 3
  
  m <- length(dose)     # Number of doses
  q <- length(n)        # Number of dose group sizes
  CL_median <- CL[1]    # Median clearance 
  CL_CV <- CL[2]        # C.V. of clearance as a proportion
  B0 <- logit_param[1]  # Logistic regression model intercept parameter
  B1 <- logit_param[2]  # Logistic regression model slope parameter
  power <- rep(NA,q)    # Initialize the vector that contains the power at each specified n
  
  for (i in 1:q) {      # Loop to determine the power at each specified n
    significant <- rep(NA,l) # Initialize the vector that contains the inidicator if a significant effect was determined for each study replication at n[i]
    for (j in 1:l) {         # Loop to simulate the study replicate j with sample size n[i] X m
      AUC <- array(dim = c(n[i],m))        # Initialize n[i] X m array of area under the curve (AUC)
      p <- array(dim = c(n[i],m))          # Initialize n[i] X m array of probabilities (p)
      binom_resp <- array(dim = c(n[i],m)) # Initialize n[i] X m array of binomial responses (binom_resp)
      for (k in 1:m) {                    # Loop to simulate the dose group k with sample size n[i]
        AUC[,k] <- dose[k]/rlnorm(n = n[i], meanlog = log(CL_median), sdlog = sqrt(log(CL_CV^2+1)))  # Simulate individual subject AUCs in dose group k (log-normal distribution of CL assumed)
        p[,k]   <- 1/(1+exp(-(B0+B1*AUC[,k])))                                                       # Simulate individual subject probability of response at each AUC in dose group k
        binom_resp[,k] <- rbinom(n = n[i], size = 1, prob = p[,k])                                   # Simulate individual subject responses at each AUC in dose group k
      }
      dataset <- data.frame(cbind(AUC = as.vector(AUC),response = as.vector(binom_resp)))  # Create analysis ready data-set n[i] X m in individual subject AUCs and corresponding responses across the m doses
      model_fit <- glm(response ~ AUC, family = binomial, data = dataset)                  # Fit a logistic-regression model to the analysis ready data-set of n[i] X m for the jth replicate
      p_value <- coef(summary(model_fit))[2,4]                                             # Extract the p-value for the estimated slope term from the fittted a logistic-regression model for the jth study replicate
      if (p_value <= alpha) {      # Loop to deteremine if a significant slope term (at alpha significance level) for the jth study replicate of n[i] X m study design
        significant[j] <- 1
      } else {
        significant[j] <- 0
      }
    } 
    power[i] <- mean(significant)  # Calculate the power to detect a significant slope term of n[i] X m study design
  }
  results <- data.frame(cbind(n = n, power = power, CL_CV = CL_CV)) # Create output data frame of the results with power at each specified n per dose group
  results
#}

# Basic power simulation


#comparing the E-R method to classical power
sim_0 <- power_sim(n = seq(from = 10, to = 150, by = 5), dose = c(1,2), 
                         CL = c(1,0.25), logit_param = c(-1.5, 1.0), alpha = 0.05, l = 10000)
 sim_0$Scenario <-100


#varying Operating Characteristic
#slopes
sim_1 <- power_sim(n = seq(from = 10, to = 150, by = 5), dose = c(1,2), 
                         CL = c(1,0.25), logit_param = c(-1.5, 0.5), alpha = 0.05, l = 1000)                         
sim_2 <- power_sim(n = seq(from = 10, to = 150, by = 5), dose = c(1,2), 
                         CL = c(1,0.25), logit_param = c(-1.5, 1), alpha = 0.05, l = 1000)
sim_3 <- power_sim(n = seq(from = 10, to = 150, by = 5), dose = c(1,2), 
                         CL = c(1,0.25), logit_param = c(-1.5, 2), alpha = 0.05, l = 1000)
sim_1$Scenario <-1
sim_2$Scenario <-2
sim_3$Scenario <-3                      

#intercepts                       
sim_4 <- power_sim(n = seq(from = 10, to = 150, by = 5), dose = c(1,2), 
                         CL = c(1,0.25), logit_param = c(-3, 1), alpha = 0.05, l = 1000)   
sim_5 <- power_sim(n = seq(from = 10, to = 150, by = 5), dose = c(1,2), 
                         CL = c(1,0.25), logit_param = c(-1.5, 1), alpha = 0.05, l = 1000)   
sim_6 <- power_sim(n = seq(from = 10, to = 150, by = 5), dose = c(1,2), 
                         CL = c(1,0.25), logit_param = c(-0.5, 1), alpha = 0.05, l = 1000)   
sim_4$Scenario <-4
sim_5$Scenario <-5
sim_6$Scenario <-6                         


# number of doses                         
sim_7 <-power_sim(n = seq(from = 10, to = 200, by = 5), dose = c(2), 
                         CL = c(1,0.25), logit_param = c(-1.5, 1), alpha = 0.05, l = 1000)
sim_8 <- power_sim(n = seq(from = 10, to = 100, by = 5), dose = c(1,2), 
                         CL = c(1,0.25), logit_param = c(-1.5, 1), alpha = 0.05, l = 1000)
sim_9 <- power_sim(n = seq(from = 10, to = 75, by = 5), dose = c(1,2,3), 
                         CL = c(1,0.25), logit_param = c(-1.5, 1), alpha = 0.05, l = 1000)


sim_7$Scenario <-7
sim_8$Scenario <-8
sim_9$Scenario <-9


# dose ranges                         
sim_10 <-power_sim(n = seq(from = 10, to = 150, by = 5), dose = c(1.5,3), 
                         CL = c(1,0.25), logit_param = c(-1.5, 1), alpha = 0.05, l = 1000)
sim_11 <- power_sim(n = seq(from = 10, to = 150, by = 5), dose = c(0.5,3.5), 
                         CL = c(1,0.25), logit_param = c(-1.5, 1), alpha = 0.05, l = 1000)


sim_10$Scenario <-10
sim_11$Scenario <-11


#different %CV's
sim_12 <-power_sim(n = seq(from = 5, to = 150, by = 5), dose = c(1,2), 
                         CL = c(1,0.1), logit_param = c(-1.5, 1), alpha = 0.05, l = 1000)
sim_13 <-power_sim(n = seq(from = 5, to = 150, by = 5), dose = c(1,2), 
                         CL = c(1,0.4), logit_param = c(-1.5, 1), alpha = 0.05, l = 1000)

sim_12$Scenario <-12
sim_13$Scenario <-13

# Case Study: Ixazomib Dose Ranging
# Scenario 1: Comparing one dose two dose groups (3 mg vs 4 mg) group (4 mg) to the response for the lower dose group (3 mg) with the conventional power calculation

sim_20 <- power_sim(n = seq(from = 10, to = 300, by = 5), dose = c(3000,4000), 
                         CL = c(33300,0.423), logit_param = c(-2.0328, 15.1229), alpha = 0.05, l = 1000)  
                         
#Scenario 2 : number of dose group
sim_21 <- power_sim(n = seq(from = 10, to = 300, by = 5), dose = c(3000,4000,5000), 
                         CL = c(33300,0.423), logit_param = c(-2.0328, 15.1229), alpha = 0.05, l = 1000) 
                         
#Scenario 3 : %CV

sim_22 <- power_sim(n = seq(from = 10, to = 300, by = 5), dose = c(3000,4000), 
                         CL = c(33300,0.10), logit_param = c(-2.0328, 15.1229), alpha = 0.05, l = 1000) 


 sim_20$Scenario <-20  
 sim_21$Scenario <-21  
 sim_22$Scenario <-22


# Power simulation from "Traditional" Power calculation

 power_sim_ref <- function(n, p0, p1, alpha){
   # n:           number of subjects in each dose group (vector or scalar)
   #p0:          Expected proportion of responders in the reference group (eg Placebo or reference treatment) 
   #p1:          Expected proportion of responders in the treatment group
   #alpha:     Type I error rate
 
   q <- length(n)        # N per dose group
   power <- rep(NA,q)    # Initialize the vector that contains the power at each specified n
   z <- rep(NA,q)    # Initialize the vector that contains the power at each specified n
   
   
 for (i in 1:q) {      # Loop to determine the power at each specified n
   
   z[i] <- ( p0- p1)/sqrt(p0*(1- p0)/ n[i]/1+ (p1*(1- p1)/ n[i]))
      power[i] <- pnorm(z[i]-qnorm(1- alpha/2))+pnorm(- z[i]-qnorm(1- alpha/2))  # Calculate the power of n[i] 
}
   results <- data.frame(cbind(n = n, power = power)) # Create output data frame of the results with power at each specified n per dose group
   results
}



# Operating Factors: Conventional Power Simulation
 sim_ref <- power_sim_ref(n = seq(from = 5, to = 150, by = 5), p0=0.38, p1=0.62, alpha = 0.05)
 sim_ref$Scenario <-0
 sim_ref$CL_CV <-0
 
# Case Study: Ixazomib Dose Ranging - Conventional Power
 sim_ref2 <- power_sim_ref(n = seq(from = 5, to = 600, by = 5), p0=0.37, p1=0.44, alpha = 0.05)
 sim_ref2$Scenario <-0
 sim_ref2$CL_CV <-0


#--------------
# define labeling

#comparing the E-R method to classical power (Figure 2 A)
PlotData0 <- rbind(sim_0, sim_ref)
PlotData0$Scenario <- factor(PlotData0$Scenario, levels=c('0','100'),labels=c('Conventional Power','Exposure-Response'))

#slopes (Figure 2 B)
PlotData1 <-rbind(sim_1,sim_2,sim_3)
PlotData1$Scenario <- factor(PlotData1$Scenario, levels=c('1','2','3'),labels=c('\u03b21=0.5 mL/ug','\u03b21=1 mL/ug','\u03b21=2 mL/ug'))

#intercepts (Figure 2 C)
PlotData2 <-rbind(sim_4,sim_5,sim_6)
PlotData2$Scenario <- factor(PlotData2$Scenario, levels=c('4','5','6'),labels=c('\u03b20=-3','\u03b20=-1.5','\u03b20=-0.5'))

# number of doses (Figure 2 D)
PlotData3 <-rbind(sim_7,sim_8,sim_9)
## in order to have total n on x axis for fair comparison
PlotData3$n_tot <- ifelse(PlotData3$Scenario==8, PlotData3$n*2, ifelse(PlotData3$Scenario==9, PlotData3$n*3, PlotData3$n))
PlotData3$Scenario <- factor(PlotData3$Scenario, levels=c('7','8','9'),labels=c('1 dose (2 mg)', '2 doses (1 and 2 mg)', '3 doses (1, 2, and 3 mg)'))  


#dose ranges (Figure 2 E)
PlotData4 <-rbind(sim_0,sim_10,sim_11)
PlotData4$Scenario <- factor(PlotData4$Scenario, levels=c('100','10','11'),labels=c('1 and 2 mg','1.5 and 3 mg','0.5 and 3.5 mg'))


#different %CV's (Figure 2 D)
PlotData5 <-rbind(sim_12,sim_0,sim_13, sim_ref)
PlotData5$Scenario <- factor(PlotData5$Scenario, levels=c('13','100','12','0'),labels=c('40 % CV','25 % CV','10 % CV','Conventional Power'))

# Case Study: Ixazomib Dose Ranging
# Scenario 1: Comparing one dose two dose groups (3 mg vs 4 mg) group (4 mg) to the response for the lower dose group (3 mg) with the conventional power calculation
PlotData6 <-rbind(sim_ref2,sim_20)
PlotData6$Scenario <- factor(PlotData6$Scenario, levels=c('0','20'),labels=c('Conventional Power','3 mg vs 4 mg'))

#Scenario 2: Comparing two dose groups (3 mg and 4 mg) and three dose groups (3 mg vs 4 mg vs 5 mg)
PlotData7 <-rbind(sim_20,sim_21)
## in order to have total n on x axis for fair comparison
PlotData7$n_tot <- ifelse(PlotData7$Scenario==20, PlotData7$n*2, ifelse(PlotData7$Scenario==21, PlotData7$n*3, PlotData7$n))
PlotData7$Scenario <- factor(PlotData7$Scenario, levels=c('20','21'),labels=c('2 doses: 3 and 4 mg', '3 doses: 3, 4 and 5 mg'))

# Scenario 3: %CV
PlotData8 <-rbind(sim_20,sim_22)
PlotData8$Scenario <- factor(PlotData8$Scenario, levels=c('20','22'),labels=c('42.3 % CV', '10 % CV'))


###Power Curve Plots  


#comparing the E-R method to classical power (Figure 2 A)
ggplot(PlotData0, aes(x=n, y=power, group=factor(Scenario), color=factor(Scenario)))+
   geom_line(size=1.5)+
  base_theme_obs()+
  theme( legend.position=c(0.6,0.3))+
  scale_colour_manual("Scenario",values=col6)+
  geom_hline(aes(yintercept=0.80),linetype="dashed",size=1)+
  scale_y_continuous(limits=c(0,1.01),breaks=seq(0, 1, 0.20))+
  scale_x_continuous(limits=c(0,150),breaks=seq(0, 150, 25)) + 
    theme(plot.title = element_text(size = 16, face = "bold"))+
xlab("N per Dose Group") +
  ylab("Power (%)")

#slopes (Figure 2 B)
ggplot(PlotData1, aes(x=n, y=power, group=factor(Scenario), color=factor(Scenario)))+
   geom_line(size=1.5)+
  base_theme_obs()+
  theme( legend.position=c(0.6,0.3))+
  scale_colour_manual("Scenario",values=col6)+
  geom_hline(aes(yintercept=0.80),linetype="dashed",size=1)+
  scale_y_continuous(limits=c(0,1.01),breaks=seq(0, 1, 0.20))+
  scale_x_continuous(limits=c(0,150),breaks=seq(0, 150, 25)) + 
    theme(plot.title = element_text(size = 16, face = "bold"))+
xlab("N per Dose Group") +
  ylab("Power (%)")

#intercepts (Figure 2 C)
ggplot(PlotData2, aes(x=n, y=power, group=factor(Scenario), color=factor(Scenario)))+
   geom_line(size=1.5)+
  base_theme_obs()+
  theme( legend.position=c(0.6,0.3))+
  scale_colour_manual("Scenario",values=col7)+
  geom_hline(aes(yintercept=0.80),linetype="dashed",size=1)+
  scale_y_continuous(limits=c(0,1.01),breaks=seq(0, 1, 0.20))+
  scale_x_continuous(limits=c(0,150),breaks=seq(0, 150, 25)) + 
    theme(plot.title = element_text(size = 16, face = "bold"))+
xlab("N per Dose Group") +
  ylab("Power (%)")

#number of doses (Figure 2 D)
ggplot(PlotData3, aes(x=n_tot, y=power, group=factor(Scenario), color=factor(Scenario)))+
   geom_line(size=1.5)+
  base_theme_obs()+
  theme( legend.position=c(0.7,0.2))+
  scale_colour_manual("Scenario",values=col6)+
  geom_hline(aes(yintercept=0.80),linetype="dashed",size=1)+
  scale_y_continuous(limits=c(0,1.01),breaks=seq(0, 1, 0.20))+
  scale_x_continuous(limits=c(0,200),breaks=seq(0, 200, 25)) + 
    theme(plot.title = element_text(size = 16, face = "bold"))+
xlab("Total N") +
  ylab("Power (%)")

#dose ranges (Figure 2 E)
ggplot(PlotData4, aes(x=n, y=power, group=factor(Scenario), color=factor(Scenario)))+
   geom_line(size=1.5)+
  base_theme_obs()+
  theme( legend.position=c(0.6,0.2))+
  scale_colour_manual("Scenario",values=col7)+
  geom_hline(aes(yintercept=0.80),linetype="dashed",size=1)+
  scale_y_continuous(limits=c(0,1.01),breaks=seq(0, 1, 0.20))+
  scale_x_continuous(limits=c(0,150),breaks=seq(0, 150, 25)) + 
    theme(plot.title = element_text(size = 16, face = "bold"))+
xlab("N per Dose Group") +
  ylab("Power (%)")


#different %CV's (Figure 2 D)
ggplot(PlotData5, aes(x=n, y=power, group=factor(Scenario), color=factor(Scenario)))+
   geom_line(size=1.5)+
  base_theme_obs()+
  theme( legend.position=c(0.6,0.2))+
  scale_colour_manual("Scenario",values=col6)+
  geom_hline(aes(yintercept=0.80),linetype="dashed",size=1)+
  scale_y_continuous(limits=c(0,1.01),breaks=seq(0, 1, 0.20))+
  scale_x_continuous(limits=c(0,150),breaks=seq(0, 150, 25)) + 
    theme(plot.title = element_text(size = 16, face = "bold"))+
xlab("N per Dose Group") +
  ylab("Power (%)")

# Case Study: Ixazomib Dose Ranging
#Scenario 1
ggplot(PlotData6, aes(x=n, y=power, group=factor(Scenario), color=factor(Scenario)))+
   geom_line(size=1.5)+
  base_theme_obs()+
  theme( legend.position=c(0.6,0.2))+
  scale_colour_manual("Scenario",values=col6)+
  geom_hline(aes(yintercept=0.80),linetype="dashed",size=1)+
  scale_y_continuous(limits=c(0,1.01),breaks=seq(0, 1, 0.20))+
  scale_x_continuous(limits=c(0,600),breaks=seq(0, 600, 50)) + 
    theme(plot.title = element_text(size = 16, face = "bold"))+
xlab("N per Dose Group") +
  ylab("Power (%)")

#Scenario 2
ggplot(PlotData7, aes(x=n_tot, y=power, group=factor(Scenario), color=factor(Scenario)))+
   geom_line(size=1.5)+
  base_theme_obs()+
  theme( legend.position=c(0.6,0.2))+
  scale_colour_manual("Scenario",values=col7)+
  geom_hline(aes(yintercept=0.80),linetype="dashed",size=1)+
  scale_y_continuous(limits=c(0,1.01),breaks=seq(0, 1, 0.20))+
  scale_x_continuous(limits=c(0,200),breaks=seq(0, 200, 25)) + 
    theme(plot.title = element_text(size = 16, face = "bold"))+
xlab("Total N") +
  ylab("Power (%)")

#Scenario 3
ggplot(PlotData8, aes(x=n, y=power, group=factor(Scenario), color=factor(Scenario)))+
   geom_line(size=1.5)+
  base_theme_obs()+
  theme( legend.position=c(0.6,0.2))+
  scale_colour_manual("Scenario",values=col6)+
  geom_hline(aes(yintercept=0.80),linetype="dashed",size=1)+
  scale_y_continuous(limits=c(0,1.01),breaks=seq(0, 1, 0.20))+
  scale_x_continuous(limits=c(0,300),breaks=seq(0, 300, 50)) + 
    theme(plot.title = element_text(size = 16, face = "bold"))+
xlab("N per Dose Group") +
  ylab("Power (%)")


