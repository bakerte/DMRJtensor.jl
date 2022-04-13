#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#



"""
DMRjulia  (version 0.8)\n
(made for julia v1.5.4+ (March 11, 2021), see included license)

Code: https://github.com/bakerte/DMRjulia.jl

Documentation: T.E. Baker, S. Desrosiers, M. Tremblay, M.P. Thompson "Méthodes de calcul avec réseaux de tenseurs en physique" Canadian Journal of Physics 99, 4 (2021)\n
                 [ibid. "Basic tensor network computations in physics" https://arxiv.org/abs/1911.11566 p. 19]\n
          and  T.E. Baker, M.P. Thompson "Build your own tensor network library: DMRjulia I. Basic library for the density matrix renormalization group" arxiv: 2109.03120\n
          and  T.E. Baker, A. Foley "" arxiv: 2106.XXXX (computations with quantum numbers)\n
Funding for this program is graciously provided by:
   + Institut quantique (Université de Sherbrooke)
   + Département de physique, Université de Sherbrooke
   + Canada First Research Excellence Fund (CFREF)
   + Institut Transdisciplinaire d'Information Quantique (INTRIQ)
   + US-UK Fulbright Commission (Bureau of Education and Cultural Affairs from the United States Department of State)
   + Department of Physics, University of York
   
Running the julia kernel with --check-bounds=no can decrease runtimes by 20%

"""
module DMRJtensor
import LinearAlgebra
import Printf
import Serialization



println("_________  _________ _       _ _")
println("|     \\  \\/  || ___ (_)     | (_)")
println("| | | | .  . || |_/ /_ _   _| |_  __ _ ")
println("| | | | |\\/| ||    /| | | | | | |/ _` |")
println("| |/ /| |  | || |\\ \\| | |_| | | | (_| |")
println("|___/ \\_|  |_/\\_| \\_| |\\__,_|_|_|\\__,_|")
println("                   _/ |                ")
println("                  |__/                 ")
println("version 0.8")
println("(made for julia v1.1.1+, see included license)")
println()
println("Code: https://github.com/bakerte/DMRJtensor.jl")
println()
println("Introduction: T.E. Baker, S. Desrosiers, M. Tremblay, M.P. Thompson \"Méthodes de calcul avec réseaux de tenseurs en physique\" Canadian Journal of Physics 99, 4 (2021)")
println("                 [ibid. \"Basic tensor network computations in physics\" arxiv: 1911.11566 p. 19]")
println("Documentation: T.E. Baker, et. al.\"DMRjulia: Tensor recipes for entanglement renormalization computations\" arxiv: 2111.14530")
println("          and  T.E. Baker, M.P. Thompson \"Build your own tensor network library: DMRjulia I. Basic library for the density matrix renormalization group\" arxiv: 2109.03120")
println("          and  T.E. Baker, A. Foley \"\" arxiv: 2106.XXXX (computations with quantum numbers)")
println("Funding for this program is graciously provided by:")
println("   + Institut quantique (Université de Sherbrooke)")
println("   + Département de physique, Université de Sherbrooke")
println("   + Canada First Research Excellence Fund (CFREF)")
println("   + Institut Transdisciplinaire d'Information Quantique (INTRIQ)")
println("   + US-UK Fulbright Commission (Bureau of Education and Cultural Affairs from the United States Department of State)")
println("   + Department of Physics, University of York")

println("Running the julia kernel with --check-bounds=no can decrease runtimes by 20%")
juliathreads = Threads.nthreads()
println("julia threads: ",juliathreads,"    (modify with 'export JULIA_NUM_THREADS=')")
LinearAlgebra.BLAS.set_num_threads(juliathreads)
println()




const DMRjulia = DMRJtensor
export DMRjulia

libpath = "../lib/"

#Basic tensors
include(libpath*"tensor.jl")

#Quantum number preserving tensors
include(libpath*"QN.jl")
include(libpath*"Qtensor.jl")

#Linear algebra routines
include(libpath*"libalg.jl")

#Contraction routines
include(libpath*"contractions.jl")

#Decomposition methods
include(libpath*"decompositions.jl")

#Lanczos methods
include(libpath*"Krylov.jl")

#MPO/MPS
include(libpath*"MPtask.jl")
include(libpath*"MPmeas.jl")
include(libpath*"largeMP.jl")
include(libpath*"autoMPO.jl")

#basic models (Hubbard, Heisenberg, t-J)
include(libpath*"models.jl")

#General tensor networks
include(libpath*"tensornetwork.jl")



algpath = "../algs/"

include(algpath*"optimizeMPS.jl")
include(algpath*"DMRG.jl")

end

#using .DMRJtensor
