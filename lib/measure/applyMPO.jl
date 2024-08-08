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

#       +---------------------------------------+
#>------+       measurement operations          +---------<
#       +---------------------------------------+

"""
    psi = applyOps!(psi,sites,Op[,trail=ones(1,1)])

Applies operator `Op` (any `TensType`) in-place to the MPS `psi` at sites `sites`, a vector of integers. A trailing operator `trail` can be applied if not the default.
"""
function applyOps!(psi::MPS,sites::Array{W,1},Op::TensType;trail::TensType=ones(1,1)) where W <: Integer
  def_trail = ones(1,1)
  @inbounds for i = 1:length(sites)
    p = sites[i]
    psi[p] = contract([2,1,3],Op,2,psi[p],2)
    if trail != def_trail
      @inbounds for j = 1:p-1
        psi[j] = contract([2,1,3],trail,2,psi[j],2)
      end
    end
  end
  return psi
end
export applyOps!

"""
    newpsi = applyOps(psi,sites,Op[,trail=ones(1,1)])

Applies operator `Op` (any `TensType`) to the MPS `psi` at sites `sites`, a vector of integers. A trailing operator `trail` can be applied if not the default.
"""
function applyOps(psi::MPS,sites::Array{W,1},Op::TensType;trail::TensType=ones(1,1)) where W <: Integer
  cpsi = copy(psi)
  return applyOps!(cpsi,sites,Op,trail=trail)
end
export applyOps

"""
    newpsi = applyMPO(psi,H[,m=1,cutoff=0.])

Applies MPO (`H`) to an MPS (`psi`) and truncates to maximum bond dimension `m` and cutoff `cutoff`
"""
function applyMPO(psi::MPS,H::MPO;m::Integer=0,cutoff::Float64=0.)
  if m == 0
    m = maximum([size(psi[i],ndims(psi[i])) for i = 1:length(psi)])
  end

  thissize = length(psi)
  newpsi = [contract([1,3,4,2,5],psi[i],2,H[i],2) for i = 1:thissize]

  finalpsi = Array{typeof(psi[1]),1}(undef,thissize)
  finalpsi[thissize] = reshape!(newpsi[thissize],[[1],[2],[3],[4,5]],merge=true)

  for i = thissize:-1:2
    currTens = finalpsi[i]
    newsize = size(currTens)
    
    temp = reshape!(currTens,[[1,2],[3,4]])
    U,D,V = svd(temp,m = m,cutoff=cutoff)
    finalpsi[i] = reshape!(V,size(D,1),newsize[3],newsize[4],merge=true)
    tempT = contract(U,2,D,1)
    
    finalpsi[i-1] = contract(newpsi[i-1],(4,5),tempT,1)
  end
  finalpsi[1] = reshape!(finalpsi[1],[[1,2],[3],[4]],merge=true)
  return MPS(finalpsi)
end

"""
    newpsi = applyMPO(psi,H...[,m=1,cutoff=0.])

Applies MPOs (`H`) to an MPS (`psi`) and truncates to maximum bond dimension `m` and cutoff `cutoff`. Not recommended except for small problems since bond dimension is not truncated uniformly.
"""
function applyMPO(psi::MPS,H::MPO...;m::Integer=0,cutoff::Float64=0.)
  newpsi = psi
  for w = 1:length(H)
    newpsi = applyMPO(newpsi,H[w],m=m,cutoff=cutoff)
  end
  return newpsi
end
export applyMPO