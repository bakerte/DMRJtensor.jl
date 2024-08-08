###############################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                               v1.0
#
###############################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.0+)
#



"""
DMRjulia  (version 0.12.0)\n
(made for julia v1.10.4+ (July 6, 2024), see included license)

Code: https://github.com/bakerte/DMRJtensor.jl

Documentation: T.E. Baker, S. Desrosiers, M. Tremblay, M.P. Thompson "Méthodes de calcul avec réseaux de tenseurs en physique" Canadian Journal of Physics 99, 4 (2021)\n
                 [ibid. "Basic tensor network computations in physics" https://arxiv.org/abs/1911.11566 p. 19]\n
          and  T.E. Baker and M.P. Thompson "Build your own tensor network library: DMRjulia I. Basic library for the density matrix renormalization group" arxiv: 2109.03120\n
Funding for this program is graciously provided by:
   + Institut quantique (Université de Sherbrooke)
   + Département de physique, Université de Sherbrooke
   + Canada First Research Excellence Fund (CFREF)
   + Institut Transdisciplinaire d'Information Quantique (INTRIQ)
   + US-UK Fulbright Commission (Bureau of Education and Cultural Affairs from the United States Department of State)
   + Department of Physics, University of York
   + Canada Research Chair in Quantum Computing for Modelling of Molecules and Materials
   + Department of Physics & Astronomy, University of Victoria
   + Department of Chemistry, University of Victoria
   + Faculty of Science, University of Victoria
   + National Science and Engineering Research Council (NSERC)
"""
module DMRJtensor

const DMRjulia = DMRJtensor
export DMRjulia

const libdir = @__DIR__

const libpath = libdir*"/../lib/"

files = ["imports.jl","banner.jl","types.jl","models.jl","tensorhelper.jl","network.jl"]
for w = 1:length(files)
  include(libpath*files[w])
end


files = ["mult.jl","div.jl","star.jl","slash.jl","conj.jl","copy.jl","eltype.jl","emptyTensor.jl","get_tensors.jl","getindex.jl","lastindex.jl","length.jl","setindex.jl","size.jl","tensordisc.jl"]
subdir = "methods/"
for w = 1:length(files)
  include(libpath*subdir*files[w])
end


files = ["MPS.jl","randMPS.jl","makeMPS.jl","makeqMPS.jl","fullpsi.jl","nameMPS.jl","largeMPS.jl"]
subdir = "MPS/"
for w = 1:length(files)
  include(libpath*subdir*files[w])
end


files = ["MPO.jl","viewbulkMPO.jl","makeMPO.jl","makeqMPO.jl","fullmpo.jl","nameMPO.jl","largeMPO.jl","autoMPO.jl"]
subdir = "MPO/"
for w = 1:length(files)
  include(libpath*subdir*files[w])
end


files = ["environment.jl","largeEnv.jl"]
subdir = "Env/"
for w = 1:length(files)
  include(libpath*subdir*files[w])
end


files = ["correlationmatrix.jl","applyMPO.jl","boundaryMove.jl","correlation.jl","expect.jl","localizeMPO.jl","Lupdate.jl","makeEnv.jl","move.jl","moveL.jl","moveR.jl","penalty.jl","Rupdate.jl","localF.jl","transfermatrix.jl"]
subdir = "measure/"
for w = 1:length(files)
  include(libpath*subdir*files[w])
end


const algpath = libdir*"/../algs/"

include(algpath*"optimizeMPS.jl")
include(algpath*"DMRG.jl")

include(algpath*"time.jl")

const testpath = libdir*"/../test/"
include(testpath*"alltest.jl")

end

#using .DMRJtensor
