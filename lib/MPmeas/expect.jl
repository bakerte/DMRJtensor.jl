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
    expval = expect(dualpsi,psi,H[,Lbound=,Rbound=,order=])

evaluate <`dualpsi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`); `dualpsi` is conjugated inside of the algorithm; output is an expectation value `expval`

See also: [`overlap`](@ref)
"""
function expect(dualpsi::MPS,psi::MPS,H::MPO...;Lbound::TensType=defaultBoundary(psi[1]),Rbound::TensType=Lbound,order::intvecType=intType[])
  Ns = length(psi)
  nMPOs = size(H,1)
  nLsize = nMPOs+2
  nRsize = nLsize+1

  move!(dualpsi,psi.oc)

  Lenv,Renv = makeEnds(dualpsi,psi,H...,Lbound=Lbound,Rbound=Rbound)

  for i = length(psi):-1:1
    Renv = ccontract(dualpsi[i],3,Renv,nLsize)
    for j = 1:nMPOs
      Renv = contract(H[j][i],(3,4),Renv,(2,nRsize))
    end
    Renv = contract(psi[i],(2,3),Renv,(2,nRsize))
  end

  if order == intType[]
    permvec = ntuple(i->ndims(Lenv)-i+1,ndims(Lenv))
    modLenv = permutedims(Lenv,permvec)
  else
    modLenv = permutedims(Lenv,order)
  end

  return contract(modLenv,Renv)
end

"""
    expval = expect(psi,H[,Lbound=,Rbound=,order=])

evaluate <`psi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`); output is an expectation value `expval`

See also: [`overlap`](@ref)
"""
function expect(psi::MPS,H::MPO...;Lbound::TensType=defaultBoundary(psi[1]),Rbound::TensType=Lbound,order::intvecType=intType[])
  return expect(psi,psi,H...,Lbound=Lbound,Rbound=Rbound,order=order)
end
export expect