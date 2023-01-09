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
DMRjulia  (version 0.9.6)\n
(made for julia v1.5.4+ (March 11, 2021), see included license)

Code: https://github.com/bakerte/DMRjulia.jl

Documentation: T.E. Baker, S. Desrosiers, M. Tremblay, M.P. Thompson "Méthodes de calcul avec réseaux de tenseurs en physique" Canadian Journal of Physics 99, 4 (2021)\n
                 [ibid. "Basic tensor network computations in physics" https://arxiv.org/abs/1911.11566 p. 19]\n
          and  T.E. Baker, M.P. Thompson "Build your own tensor network library: DMRjulia I. Basic library for the density matrix renormalization group" arxiv: 2109.03120\n
Funding for this program is graciously provided by:
   + Institut quantique (Université de Sherbrooke)
   + Département de physique, Université de Sherbrooke
   + Canada First Research Excellence Fund (CFREF)
   + Institut Transdisciplinaire d'Information Quantique (INTRIQ)
   + US-UK Fulbright Commission (Bureau of Education and Cultural Affairs from the United States Department of State)
   + Department of Physics, University of York
   + Canada Research Chair in Quantum Computing for Modeling of Molecules and Materials
   + Department of Physics & Astronomy, University of Victoria
   + Department of Chemistry, University of Victoria
   
Running the julia kernel with --check-bounds=no can decrease runtimes by 20%

"""
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
println("version 0.9.6")
println("(made for julia v1.8.3+, see included license)")
println()
println("Code: https://github.com/bakerte/DMRJtensor.jl")
println()
println("Introduction: T.E. Baker, S. Desrosiers, M. Tremblay, M.P. Thompson \"Méthodes de calcul avec réseaux de tenseurs en physique\" Canadian Journal of Physics 99, 4 (2021)")
println("                 [ibid. \"Basic tensor network computations in physics\" arxiv: 1911.11566 p. 20]")
println("Documentation: T.E. Baker, et. al.\"DMRjulia: Tensor recipes for entanglement renormalization computations\" arxiv: 2111.14530")
println("          and  T.E. Baker, M.P. Thompson \"Build your own tensor network library: DMRjulia I. Basic library for the density matrix renormalization group\" arxiv: 2109.03120")
#println("          and  T.E. Baker, A. Foley \"\" arxiv: 2106.XXXX (computations with quantum numbers)")
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
println("   + Canada Research Chair in Quantum Computing for Modeling of Molecules and Materials")
println("   + Department of Physics & Astronomy, University of Victoria")
println("   + Department of Chemistry, University of Victoria")

println("Running the julia kernel with --check-bounds=no can decrease runtimes by 20%")
juliathreads = Threads.nthreads()
println("julia threads: ",juliathreads,"    (modify with 'export JULIA_NUM_THREADS=')")
println("julia processes: ",Distributed.nprocs(),"    (modify with 'Distributed' package commands, `addprocs()` or `julia -p #`)")
LinearAlgebra.BLAS.set_num_threads(juliathreads)
#println("BLAS threads (set in DMRjulia.jl): ",juliathreads)
#println("BLAS threads: ",ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ()))
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

#MPO/MPS
include(libpath*"MPtask.jl")
include(libpath*"MPmeas.jl")
include(libpath*"MPlarge.jl")
include(libpath*"MPOauto.jl")

#basic models (Hubbard, Heisenberg, t-J)
include(libpath*"models.jl")

#General tensor networks
include(libpath*"tensornetwork.jl")


algpath = "../algs/"

include(algpath*"optimizeMPS.jl")
include(algpath*"DMRG.jl")

#Lanczos methods
include(algpath*"Krylov.jl")

end

#using .DMRJtensor
