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
    mps = nameMPS(A)

Assigns names to MPS `A` to create a network of named tensors `mps`

See also: [`nameMPO`](@ref)
"""
function nameMPS(psi::MPS;oc::Integer=psi.oc)
  W = typeof(psi[1])
  TNmps = Array{directedtens{W},1}(undef,length(psi))

  intens = reshape(psi[1],size(psi[1],2),size(psi[1],3),merge=true)
  TNmps[1] = directedtens(intens,["p","l1"],[1,0])

  for i = 2:length(TNmps)-1
    x = directedtens(psi[i],["l$(i-1)","p","l$i"],[0,1,0])
    TNmps[i] = x
  end

  Ns = length(psi)
  intens = reshape(psi[Ns],size(psi[Ns],1),size(psi[Ns],2),merge=true)
  TNmps[Ns] = directedtens(intens,["l$(Ns-1)","p"],[0,1])

  return MPS(TNmps,oc=oc)
end
export nameMPS