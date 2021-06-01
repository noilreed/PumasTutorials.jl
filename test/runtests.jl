using PumasTutorials, Pkg
Pkg.activate("../tutorials")
PumasTutorials.weave_file(".", "test.jmd")
