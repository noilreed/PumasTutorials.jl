module PumasTutorials

using Weave, Pkg, InteractiveUtils, IJulia

repo_directory = joinpath(@__DIR__,"..")

function weave_file(folder, file, build_list=(:script, :html, :pdf, :notebook))
  println("File: $file")
  tmp = joinpath(repo_directory, "tutorials", folder,file)
  args = Dict{Symbol,String}(:folder => folder, :file => file)
  if :script ∈ build_list
    println("Building Script")
    dir = joinpath(repo_directory, "script", folder)
    isdir(dir) || mkdir(dir)
    tangle(tmp; out_path=dir)
  end
  if :html ∈ build_list
    println("Building HTML")
    dir = joinpath(repo_directory, "html", folder)
    isdir(dir) || mkdir(dir)
    weave(tmp, doctype="md2html", out_path=dir, args=args)
  end
  if :pdf ∈ build_list
    println("Building PDF")
    dir = joinpath(repo_directory, "pdf", folder)
    isdir(dir) || mkdir(dir)
    weave(tmp, doctype="md2pdf", out_path=dir, args=args)
  end
  if :github ∈ build_list
    println("Building Github Markdown")
    dir = joinpath(repo_directory, "markdown", folder)
    isdir(dir) || mkdir(dir)
    weave(tmp,doctype="github", out_path=dir, args=args)
  end
  if :notebook ∈ build_list
    println("Building Notebook")
    dir = joinpath(repo_directory, "notebook", folder)
    isdir(dir) || mkdir(dir)
    Weave.convert_doc(tmp,joinpath(dir, file[1:end-4]*".ipynb"))
  end
end

function weave_all(build_list=(:script, :html, :pdf, :notebook))
  for folder in readdir(joinpath(repo_directory, "tutorials"))
    folder == "test.jmd" && continue
    weave_folder(folder, build_list)
  end
end

function weave_folder(folder, build_list=(:script, :html, :pdf, :notebook))
  for file in readdir(joinpath(repo_directory, "tutorials", folder))
    println("Building $(joinpath(folder, file)))")
    try
      weave_file(folder, file, build_list)
    catch
    end
  end
end

tutorial_data(folder,file) = joinpath(joinpath(@__DIR__, ".."),"$folder/"*file*".csv")

function tutorial_footer(folder=nothing, file=nothing)
    display("text/markdown", """
    ## Appendix

     These tutorials are part of the PumasTutorials.jl repository, found at: <https://github.com/PumasAI/PumasTutorials.jl>
    """)
    if folder !== nothing && file !== nothing
        display("text/markdown", """
        To locally run this tutorial, do the following commands:
        ```
        using PumasTutorials
        PumasTutorials.weave_file("$folder","$file")
        ```
        """)
    end
    display("text/markdown", "Computer Information:")
    vinfo = sprint(InteractiveUtils.versioninfo)
    display("text/markdown",  """
    ```
    $(vinfo)
    ```
    """)

    # ctx = Pkg.API.Context()
    # pkgs = Pkg.Display.status(Pkg.API.Context(), use_as_api=true);

    # display("text/markdown", """
    # Package Information:
    # """)

    # md = ""
    # md *= "```\nStatus `$(ctx.env.project_file)`\n"

    # for pkg in pkgs
    #     md *= "[$(pkg.uuid)] $(pkg.name) $(pkg.old.ver)\n"
    # end
    # md *= "```"
    # display("text/markdown", md)
end

function open_notebooks()
  Base.eval(Main, Meta.parse("import IJulia"))
  path = joinpath(repo_directory, "notebook")
  IJulia.notebook(; dir=path)
end

end
