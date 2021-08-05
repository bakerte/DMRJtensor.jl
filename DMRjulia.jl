#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v1.0
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

println("_________  _________ _       _ _")
println("|     \\  \\/  || ___ (_)     | (_)")
println("| | | | .  . || |_/ /_ _   _| |_  __ _ ")
println("| | | | |\\/| ||    /| | | | | | |/ _` |")
println("| |/ /| |  | || |\\ \\| | |_| | | | (_| |")
println("|___/ \\_|  |_/\\_| \\_| |\\__,_|_|_|\\__,_|")
println("                   _/ |                ")
println("                  |__/                 ")
println("version 1.0")
println("(made for julia v1.5.4+ (March 11, 2021), see included license)")
println()
println("Code: https://github.com/bakerte/dmrjulia ")
println()
println("Documentation: T.E. Baker, S. Desrosiers, M. Tremblay, M.P. Thompson \"Méthodes de calcul avec réseaux de tenseurs en physique\" Canadian Journal of Physics 99, 4 (2021)")
println("                 [ibid. \"Basic tensor network computations in physics\" https://arxiv.org/abs/1911.11566]")
println("          and  T.E. Baker, M.P. Thompson \"\" arxiv: 2106.XXXX (basics of the library)")
println("          and  T.E. Baker, A. Foley \"\" arxiv: 2106.XXXX (computations with Abelian quantum numbers)")
#println("          and  (algorithms for DMRG)")
#println("          and  (advanced tensor network algorithms)")
#println("          and  (classical tensor network algorithms)")
println("Funding for this program is graciously provided by:")
println("   + Institut quantique (Université de Sherbrooke)")
println("   + Département de physique, Université de Sherbrooke")
println("   + Canada First Research Excellence Fund (CFREF)")
println("   + Institut Transdisciplinaire d'Information Quantique (INTRIQ)")
println("   + US-UK Fulbright Commission (Bureau of Education and Cultural Affairs from the United States Department of State)")
println("   + Department of Physics, University of York")

import LinearAlgebra

println("Running the julia kernel with --check-bounds=no can decrease runtimes by 20%")
juliathreads = Threads.nthreads()
println("julia threads: ",juliathreads,"    (modify with 'export JULIA_NUM_THREADS=')")
LinearAlgebra.BLAS.set_num_threads(juliathreads)
#println("BLAS threads (set in DMRjulia.jl): ",juliathreads)
#println("BLAS threads: ",ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ()))
println()

module DMRjulia
import LinearAlgebra

  filepath = "lib/"

  include(join([filepath,"tensor.jl"]))
  export tensor

  include(join([filepath,"QN.jl"]))
  export QN
  include(join([filepath,"Qtensor.jl"]))
  export Qtensor
  include(join([filepath,"Qtask.jl"]))
  export Qtask

  include(join([filepath,"contractions.jl"]))
  export contractions
  include(join([filepath,"decompositions.jl"]))
  export decompositions

  include(join([filepath,"Opt.jl"]))
  export Opt
  include(join([filepath,"MPutil.jl"]))
  export MPutil
  include(join([filepath,"MPmaker.jl"]))
  export MPmaker



  include(join([filepath,"tensornetwork.jl"]))
  export tensornetwork

  filepath = "algs/"

  include(join([filepath,"optimizeMPS.jl"]))
  export optimizeMPS


  include(join([filepath,"Krylov.jl"]))
  export Krylov




  include(join([filepath,"DMRG.jl"]))
  export DMRG

end

using .DMRjulia
using .tensor
using .QN
using .Qtensor
using .Qtask
using .contractions
using .decompositions
using .Opt
using .MPutil
using .MPmaker


using .Krylov

using .optimizeMPS



using .DMRG
