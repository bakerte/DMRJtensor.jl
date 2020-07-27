#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.1
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.0) or (v1.5)
#

println("_________  _________ _       _ _")
println("|     \\  \\/  || ___ (_)     | (_)")
println("| | | | .  . || |_/ /_ _   _| |_  __ _ ")
println("| | | | |\\/| ||    /| | | | | | |/ _` |")
println("| |/ /| |  | || |\\ \\| | |_| | | | (_| |")
println("|___/ \\_|  |_/\\_| \\_| |\\__,_|_|_|\\__,_|")
println("                   _/ |                ")
println("                  |__/                 ")
println("version 0.1")
println("(made for julia v1.1.1+ (May 16, 2019), see included license)")
println("Documentation: http://arxiv.org/abs/1911.11566 (introduction article)",)
println("          and <<coming soon!>>",)
println("Code: https://github.com/bakerte/dmrjulia",)
println("Funding for this program is graciously provided by:")
println("   + Institut quantique (Université de Sherbrooke)")
println("   + Canada First Research Excellence Fund (CFREF)")

import LinearAlgebra

juliathreads = Threads.nthreads()
println("julia threads: ",juliathreads)
LinearAlgebra.BLAS.set_num_threads(juliathreads)
println("BLAS threads (set in DMRjulia.jl): ",juliathreads)
#println("BLAS threads: ",ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ()))
println()
println("Notes: v1.1.1 is faster (x2-3) than v1.2/1.3 (regression on parallelization: https://github.com/JuliaLang/julia/issues/32701)")
println("  ---> Fixed! as of v1.5rc1 (June 26, 2020)")
println("       Running the julia kernel with --check-bounds=no can decrease runtimes by 20%")
println()

println("This version (v0.1) computes the ground state, more methods are being documented and will be released soon")
println("Timeline for full release of the library:")
println("  + August-September 2020: Full documentation of ground-state")
println("  + August-November 2020: Full release of time and other methods for the MPS")
println("  + ~January 2021: More advanced solvers")
println("Troubleshooting? Email: thomas.baker@usherbrooke.ca ")

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
