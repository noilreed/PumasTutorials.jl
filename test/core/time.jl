using Pumas, Test

choose_covariates() = (isPM = rand((1, 0)), Wt = [rand(55:80), rand(90:110)])

covariates_time = (isPM=[1], Wt=[1.0, 4.0])

md =  DosageRegimen(100,ii=24,addl=6, cmt=1)

sub1 = Subject(id=1,covariates=choose_covariates(), covariates_time=covariates_time)

sub2 = Subject(id=1,time=[0.0],covariates=choose_covariates(), covariates_time=covariates_time)

sub3 = Subject(id=1,events=md,covariates=choose_covariates(), covariates_time=covariates_time)

sub4 = Subject(id=1,covariates=(isPM=1, Wt=0.4))

sub5 = Subject(id=1,time=[0.0],covariates=covariates=(isPM=1, Wt=0.4))

sub6 = Subject(id=1,events=md,covariates=covariates=(isPM=1, Wt=0.4))
init_central = 4.0
model = @model begin
                @covariates isPM Wt

                @pre begin
                  CL = 1.0
                  Vc = 2.0
                end
                @init begin
                  Central=init_central
                end
                @dynamics Central1
                @observed begin 
                    central=Central
                    end
             
            end


              model_diffeq = @model begin
                @covariates isPM Wt

                @pre begin
                  CL = 1.0
                  Vc = 2.0
                end
                @init begin
                  Central=4.0
                end
                @dynamics begin
                   Central' = -CL/Vc*Central
                  end
              @observed begin 
       central=Central
       end
       end


       
sol013 = solve(model_diffeq, sub1, NamedTuple(); saveat=[0.0, 1.0, 3.0])
val013 = map(i->getproperty(sol013.u[i, 1], :Central), 1:3)
@test val013[1] == init_central
sol03 = solve(model_diffeq, sub1, NamedTuple(); saveat=[0.0,3.0])
val03 = map(i->getproperty(sol03.u[i, 1], :Central), [1,2])
@test all(val03 .== val013[[1,3]])
sol13 = solve(model_diffeq, sub1, NamedTuple(); saveat=[1.0, 3.0])
val13 = map(i->getproperty(sol13.u[i, 1], :Central), 1:2)
@test all(val13 .== val013[[2,3]])

simobs013 = simobs(model_diffeq, sub1, NamedTuple(); obstimes=[0.0,1.0,3.0]).observations
@test all(simobs013.central .== val013[1:3])
simobs03 = simobs(model_diffeq, sub1, NamedTuple(); obstimes=[0.0,3.0]).observations
@test all(simobs03.central .== val013[[1,3]])
simobs13 = simobs(model_diffeq, sub1, NamedTuple(); obstimes=[1.0,3.0]).observations
@test all(simobs13.central .== val013[2:3])


