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
    mpo = nameMPO(A)

Assigns names to MPO `A` to make an `mpo` with `nametens`

See also: [`nameMPS`](@ref)
"""
function nameMPO(mpo::MPO)
  W = typeof(mpo[1])
  TNmpo = Array{directedtens{W},1}(undef,length(mpo))


  intens = reshape(mpo[1],size(mpo[1],2),size(mpo[1],3),size(mpo[1],4),merge=true)
  TNmpo[1] = directedtens(intens,["p","p","h1"],[-1,1,0])

  for i = 2:length(mpo)-1
    TNmpo[i] = directedtens(mpo[i],["h$(i-1)","p","p","h$i"],[0,-1,1,0])
  end

  Ns = length(mpo)
  intens = reshape(mpo[Ns],size(mpo[Ns],1),size(mpo[Ns],2),size(mpo[Ns],3),merge=true)
  TNmpo[Ns] = directedtens(intens,["h$(Ns-1)","p","p"],[0,-1,1])
  return MPO(TNmpo)
end
export nameMPO