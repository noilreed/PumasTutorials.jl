using Pumas, Pumas.IVIVC, Test

using Random
Random.seed!(1)

vitro_data = @test_nowarn read_vitro(Pumas.example_data("ivivc_test_data/vitro_data"));
uir_data = @test_nowarn read_uir(Pumas.example_data("ivivc_test_data/uir_data"))
vivo_data = @test_nowarn read_vivo(Pumas.example_data("ivivc_test_data/vivo_data"));
ivivc_model(x, p) = @. p[1] * x + p[2]

p = rand(4)  # remember first two entries are always Tscale and Tshift,
             # so total size of this vector is 2 (or 1 if only scale or shift) + number of params in model
lb = fill(-Inf, 4)
ub = fill(Inf, 4)

model = @test_nowarn IVIVCModel(vitro_data, uir_data, vivo_data, vitro_model=:emax,
                                                    time_scale = true,
                                                    time_shift = true,
                                                    p = p, lb = lb, ub = ub,
                                                    uir_frac=1.0, deconvo_method=:wn, ivivc_model=ivivc_model);

sol = @test_nowarn predict_vivo(model, "medium")
time = vivo_data[1]["medium"].time
conc = vivo_data[1]["medium"].conc
cmax_pe, auc_pe = @test_nowarn percentage_prediction_error(time, conc, sol.t, sol.u)
@test cmax_pe ≈ 0.6097238263353497
@test auc_pe  ≈ 0.6511465890043491

# other Fdiss models

model = @test_nowarn IVIVCModel(vitro_data, uir_data, vivo_data, vitro_model=:weibull,
                                                    time_scale = true,
                                                    time_shift = true,
                                                    p = p, lb = lb, ub = ub,
                                                    uir_frac=1.0, deconvo_method=:wn, ivivc_model=ivivc_model);

sol = @test_nowarn predict_vivo(model, "medium")
time = vivo_data[1]["medium"].time
conc = vivo_data[1]["medium"].conc
cmax_pe, auc_pe = @test_nowarn percentage_prediction_error(time, conc, sol.t, sol.u)
@test cmax_pe  ≈ 3.4637819216318197
@test auc_pe  ≈ 5.213585111751991
