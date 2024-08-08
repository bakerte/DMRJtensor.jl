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
    movecenter!(psi,newoc[,m=,cutoff=,minm=,Lfct=,Rfct=])

movement function to move `psi` to a new site, `newoc` with `Lfct` and `Rfct`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`move!`](@ref) [`move`](@ref)
"""
 function movecenter!(psi::MPS,pos::Integer;cutoff::Float64=1E-12,m::Integer=0,minm::Integer=0,Lfct::Function=moveR,Rfct::Function=moveL)
  if m == 0
    m = maximum([maximum(size(psi[i])) for i = 1:length(psi)])
  end
  while psi.oc != pos
    if psi.oc < pos
      iL = psi.oc
      iR = psi.oc+1
      psi[iL],psi[iR],D,truncerr = Lfct(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm,fast=true)
      psi.oc = iR
    else
      iL = psi.oc-1
      iR = psi.oc
      psi[iL],psi[iR],D,truncerr = Rfct(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm,fast=true)
      psi.oc = iL
    end
  end
  nothing
end

"""
    move!(psi,newoc[,m=,cutoff=,minm=])

in-place move orthgononality center of `psi` to a new site, `newoc`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`move`](@ref)
"""
 function move!(mps::MPS,pos::Integer;m::Integer=0,cutoff::Float64=0.,minm::Integer=0)
  movecenter!(mps,pos,cutoff=cutoff,m=m,minm=minm)
  nothing
end
export move!

"""
    move(psi,newoc[,m=,cutoff=,minm=])

same as `move!` but makes a copy of `psi`

See also: [`move!`](@ref)
"""
 function move(mps::MPS,pos::Integer;m::Integer=0,cutoff::Float64=1E-12,minm::Integer=0)
  newmps = copy(mps)
  movecenter!(newmps,pos,cutoff=cutoff,m=m,minm=minm)
  return newmps
end
export move

"""
    newpsi,D,V = leftnormalize(psi)

Creates a left-normalized MPS `newpsi` from `psi` and returns the external tensors `D` and `V`
"""
function leftnormalize(psi::MPS)
  newpsi = move(psi,length(psi))
  U,D,V = svd(psi[end],[[1,2],[3]])
  newpsi[end] = U
  newpsi.oc = length(psi)+1
  return newpsi,D,V
end
export leftnormalize

"""
    psi,D,V = leftnormalize!(psi)

Creates a left-normalized MPS in-place from `psi` and returns the external tensors `D` and `V`
"""
function leftnormalize!(psi::MPS)
  move!(psi,length(psi))
  U,D,V = svd(psi[end],[[1,2],[3]])
  psi[end] = U
  psi.oc = length(psi)+1
  return psi,D,V
end
export leftnormalize!

"""
    U,D,newpsi = rightnormalize(psi)

Creates a right-normalized MPS `newpsi` from `psi` and returns the external tensors `U` and `D`
"""
function rightnormalize(psi::MPS)
  newpsi = move(psi,1)
  U,D,V = svd(psi[1],[[1],[2,3]])
  newpsi[1] = V
  newpsi.oc = 0
  return U,D,newpsi
end
export rightnormalize

"""
    U,D,psi = rightnormalize!(psi)

Creates a right-normalized MPS in-place from `psi` and returns the external tensors `U` and `D`
"""
function rightnormalize!(psi::MPS)
  move!(psi,1)
  U,D,V = svd(psi[1],[[1],[2,3]])
  psi[1] = V
  psi.oc = 0
  return U,D,psi
end
export rightnormalize!