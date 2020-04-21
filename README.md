# PumasTutorials.jl

PumasTutorials.jl holds PDFs, webpages, and interactive Jupyter notebooks
showing how to do pharmaceutical modeling and simulation with Pumas.jl.

## Table of Contents

- Introduction
  - [Introduction to Pumas](https://tutorials.pumas.ai/html/introduction/introduction.html)
  - [Generating and Simulating Populations](https://tutorials.pumas.ai/html/introduction/simulating_populations.html)
  - [Introduction to Noncompartmental Analysis (NCA)](https://tutorials.pumas.ai/html/nca/basic_nca.html)
- Workshop Materials
  - [Workshop Exercises](https://tutorials.pumas.ai/html/exercises/workshop_exercises.html)
  - [Workshop Exercise Solutions](https://tutorials.pumas.ai/html/exercises/workshop_solutions.html)
  - [Workshop Exercise Fitting](https://tutorials.pumas.ai/html/exercises/fitting.html)
- Models
  - [Absorption models](https://tutorials.pumas.ai/html/discrete/absorption_models.html)
  - [Discrete Response Models](https://tutorials.pumas.ai/html/discrete/discrete_response_models.html)
  - [Indirect Response Models](https://tutorials.pumas.ai/html/pkpd/indirect_response_models.html)

# Developer Documentation

## Contributing

First of all, make sure that your current directory is `PumasTutorials`. All
of the files are generated from the Weave.jl files in the `tutorials` folder.
To run the generation process, do for example:

```julia
using Pkg, PumasTutorials
cd(joinpath(dirname(pathof(PumasTutorials)), ".."))
Pkg.pkg"activate ."
Pkg.pkg"instantiate"
PumasTutorials.weave_file("introduction","introduction.jmd")
```

To generate all of the notebooks, do:

```julia
PumasTutorials.weave_all()
```

If you add new tutorials which require new packages, simply updating your local
environment will change the project and manifest files. When this occurs, the
updated environment files should be included in the PR.
