#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8.3
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1) or (v1.5)
#

module DMRJtensor
import LinearAlgebra
import Printf
import Serialization
import Distributed

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
println("Funding for this program is graciously provided by:")
println("   + Institut quantique (Université de Sherbrooke)")
println("   + Département de physique, Université de Sherbrooke")
println("   + Canada First Research Excellence Fund (CFREF)")
println("   + Institut Transdisciplinaire d'Information Quantique (INTRIQ)")
println("   + US-UK Fulbright Commission (Bureau of Education and Cultural Affairs from the United States Department of State)")
println("   + Department of Physics, University of York")
println()
println("Tip:")
println("Running the julia kernel with --check-bounds=no can decrease runtimes by 20%")
juliathreads = Threads.nthreads()
println("julia threads: ",juliathreads,"    (modify with 'export JULIA_NUM_THREADS=')")

  libpath = "../lib/"

  include(libpath*"tensor.jl")
  include(libpath*"Opt.jl")

  include(libpath*"QN.jl")
  include(libpath*"Qtensor.jl")

  include(libpath*"contractions.jl")
  include(libpath*"decompositions.jl")

  include(libpath*"MPtask.jl")


  algpath = "../algs/"

  include(algpath*"optimizeMPS.jl")
  include(algpath*"Krylov.jl")
  include(algpath*"DMRG.jl")

end

#using .DMRJtensor
