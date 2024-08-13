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
    H = fullH(mpo)

Generates the full Hamiltonian from an MPO (memory providing); assumes lower left triagular form
"""
function fullH(mpo::MPO)
  Ns = length(mpo)
  fullH = mpo[1]
  for p = 2:Ns
    fullH = contract(fullH,ndims(fullH),mpo[p],1)
  end
  dualinds = [i+1 for i = 2:2:2Ns]
  ketinds = [i+1 for i = 1:2:2Ns]
  finalinds = vcat([1],ketinds,dualinds,[ndims(fullH)])
  pfullH = permutedims(fullH,finalinds)

  size1 = size(pfullH,1)
  size2 = prod(a->size(fullH,a),ketinds)
  size3 = prod(a->size(fullH,a),dualinds)
  size4 = size(pfullH,ndims(pfullH))

  rpfullH = reshape!(pfullH,size1,size2,size3,size4)
  return rpfullH[size(rpfullH,1),:,:,1]
end
export fullH

fullmpo = fullH
export fullmpo

"""
    D,U,truncerr,mag = eigen(H)

Computes eigenvalue decomposition of an input MPO `H` that is contracted into the Hamiltonian tensor (will give a fault if the resulting array is too large); useful for small systems

#Inputs:
+ `A`: Any `TensType` in the library

#Outputs:
+ `D`: A diagonal matrix containing eigenvalues
+ `U`: A unitary matrix (U*D*U' is A)
+ `truncerr`: total truncation error (L-1 norm)
+ `mag`: magnitude of the output tensor
"""
function eigen(A::MPO)
  return eigen(fullH(A))
end