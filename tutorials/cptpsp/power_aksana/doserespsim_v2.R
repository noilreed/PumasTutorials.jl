library(tidyverse)
library(broom)
library(furrr)
library(tictoc)
# Power simulation function 
# power_sim <- function(n, dose, CL, logit_param, alpha = 0.05, l = 1000) {
# n:           number of subjects in each dose group (vector or scalar)
# dose:        dose groups (vector or scalar)
# CL:          vector containting the median  and coeffcient of variation (as a proportion) clearance (CL) parameters  
# logit_param: vector containing the intercept (B0) and slope (B1) parameters of the logistic regression model
# alpha:       type-I error rate (defaul = 0.05)
# l:           number of study replications (default = 1000) 
# relation between DAS28 and ACR20
# given the doses of 1:2 and different ED50's and the EC50's, the power of detecting DR between dose and exposure as a metric
# DR is based on precision criteria of confidence interval
# co-primary DR analysis should pass efficacy endpoint
# how far are the doses from ED50 (ratio of doses to ED50)


combs = list(slopes = c(0.5,1,2),
             intercept = c(-3,-1.5,-0.5),
             CL_CV = c(0.1,0.4),
             CL = c(1),
             alpha = c(0.05))

doses = c("c(2)", "c(1,2)","c(1.5,3)","c(0.5,3.5)","c(1,2,3)")

params = cross_df(combs) %>% 
  mutate(parset = 0:(n()-1))

params = map2_df(.x = seq_len(length(doses)), .y = doses, function(.ndose, .doses){
  params %>% 
    mutate(dose = .doses,
           doseset = .ndose)
})

n = seq(from = 10, to = 150, by = 5)

scenarios = map_df(n, function(.df){
  params %>% 
    mutate(npergrp = .df)
}) %>% 
  group_by(npergrp,doseset,parset) %>% 
  nest


f_pmap_list <- function(df){
  pmap(as.list(df),list)
}

s4 <- f_pmap_list(scenarios)

plan(multiprocess)
tic()
res <- future_map_dfr(s4, function(.list){
  
  #s5 <- s4[[88]]
  s5 <- .list
  reps <- 10
  parameters = s5[["data"]]
  .doses = eval(parse(text = parameters$dose))
  npergrp = s5[["npergrp"]]
  CL_median <- parameters[["CL"]]    # Median clearance 
  CL_CV <- parameters[["CL_CV"]]        # C.V. of clearance as a proportion
  B0 <- parameters[["intercept"]]  # Logistic regression model intercept parameter
  B1 <- parameters[["slopes"]]  # Logistic regression model slope parameter
  alpha = parameters[["alpha"]]
  
  sig <-   seq_along(1:reps) %>% 
    map_dfr(function(.rep){
      
      sim_response <- .doses %>% map_dfr(function(.dose){
        data.frame(npergrp = npergrp,
                   CL = rlnorm(npergrp,meanlog = log(CL_median), sdlog = sqrt(log(CL_CV^2+1))),
                   dose = .dose) %>% 
          mutate(AUC = dose/CL) %>% 
          mutate(probs = 1/(1+exp(-(B0+B1*AUC))), # Simulate individual subject probability of response at each AUC in dose group k
                 response = rbinom(n = npergrp, size = 1, prob = probs))
      })
      
      fit <- glm(response ~ AUC, family = binomial, data = sim_response)
      
      
      model_res <- data.frame(npergrp = npergrp,
                              pval=coef(summary(fit))[2,4]) %>%
        mutate(significant = ifelse(pval<=alpha,1,0),
               nrep = .rep,
               CL_median = CL_median, CL_CV = CL_CV, doses = parameters$dose) %>% 
        left_join(tidy(fit) %>% 
                    mutate(rse = std.error/estimate) %>% 
                    select(term,estimate,rse) %>% 
                    gather(type,val,estimate:rse) %>% 
                    spread(term,val)
                    rename(B0hat = `(Intercept)`, B1hat = AUC) %>% 
                    mutate(nrep = .rep) %>% 
                    mutate(B0dev = 100*(B0/B0hat), 
                           B1dev = 100*(B1/B1hat)))
      
    })
  pow <- sig %>%
    mutate(power = round(mean(significant),2),
           doseset = s5$doseset, parset = s5$parset) %>% 
    select(npergrp, power, doseset, parset) %>% 
    distinct(npergrp,.keep_all = T)
  
})
toc()

result = res %>% 
  left_join(scenarios %>% unnest)
saveRDS(result, "./data/simresults.Rds")

sims <- readRDS("./data/simresults.Rds")
sims <- sims %>% distinct(parset,slopes,intercept,CL_CV) %>% 
  arrange(CL_CV,slopes,desc(intercept)) %>% 
  mutate(groups = factor(1:n())) %>% left_join(sims)%>% 
  mutate(dose = fct_reorder(dose,doseset))

View(sims)
p1 = sims %>% 
  filter(CL_CV == 0.1, doseset%in%c(2,3,4)) %>% 
  ggplot(aes(x=npergrp, y=power, color=factor(dose)))+
  geom_line(size=1.4)+
  facet_wrap(slopes~intercept, labeller = labeller(slopes = label_both, intercept = label_both))+
  theme_bw()+
  scale_x_continuous(breaks=seq(from = 10, to = 150, by = 10))+
  scale_color_discrete("Dose Groups")+
  PKPDmisc::base_theme(axis_title_x = 18,axis_text_x = 16)+
  theme(axis.text.x = element_text(angle=90),legend.position = "top")+
  labs(x = "Number of Subjects Per Group",
       y = "Power",
       title = "Power curves based on exposure response analysis")

p4 = sims %>% 
  filter(CL_CV == 0.4, doseset%in%c(2,3,4)) %>% 
  ggplot(aes(x=npergrp, y=power, color=factor(dose)))+
  geom_line(size=1.4)+
  facet_wrap(slopes~intercept, labeller = labeller(slopes = label_both, intercept = label_both))+
  theme_bw()+
  scale_x_continuous(breaks=seq(from = 10, to = 150, by = 10))+
  scale_color_discrete("Dose Groups")+
  PKPDmisc::base_theme(axis_title_x = 18,axis_text_x = 16)+
  theme(axis.text.x = element_text(angle=90),legend.position = "top")+
  labs(x = "Number of Subjects Per Group",
       y = "Power",
       title = "Power curves based on exposure response analysis")

png("./reports/drsimp10cv.png",res=300,width = 14, height = 10, units="in")
p1
dev.off()
png("./reports/drsimp40cv.png",res=300,width = 14, height = 10, units="in")
p4
dev.off()

sims %>% 
  filter(CL_CV == 0.1, doseset%in%c(2,3,4)) %>% 
  ggplot(aes(x=dose, y=power, color=interaction(slopes,intercept)))+
  geom_line(size=2)+
  #facet_wrap(slopes~intercept+npergrp, labeller = labeller(slopes = label_both, intercept = label_both))+
  theme_bw()+
  #scale_x_continuous(breaks=seq(from = 10, to = 150, by = 10))+
  #scale_color_discrete("Dose Groups")+
  PKPDmisc::base_theme(axis_title_x = 18,axis_text_x = 16)+
  theme(axis.text.x = element_text(angle=90),legend.position = "top")+
  labs(x = "Number of Subjects Per Group",
       y = "Power",
       title = "Power curves based on exposure response analysis")

