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


"""
    D,truncerr = moveL(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center left one space, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error. Modifies `Rpsi`

See also: [`moveL!`](@ref)
"""
 function moveL!(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                fast::Bool=true,lqfct::Function=lq!,svdfct::Function=svd)
  if (min(size(Rpsi,1),size(Rpsi,2)*size(Rpsi,3)) <= m || m == 0) && !isapprox(cutoff,0.) && fast
    modU,Rtens = lqfct(Rpsi,[[1],[2,3]])

    UD = (condition ? getindex!(modU,1:size(Lpsi,3),:) : modU)
    D = UD
    truncerr = 0.
  else
    U,D,Rtens,truncerr,sumD = svdfct(Rpsi,[[1],[2,3]],cutoff=cutoff,m=m,minm=minm,mag=mag)
    modU = (condition ? getindex!(U,1:size(Lpsi,3),:) : U)
    UD = dmul!(modU,D) #contract(modU,(2,),D,(1,))
  end
  Ltens = contract(Lpsi,3,UD,1)
  return Ltens,Rtens,D,truncerr
end

"""
    D,truncerr = moveL(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center left one space, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error. Does not modify `Lpsi` and `Rpsi`

See also: [`moveL!`](@ref)
"""
 function moveL(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
  fast::Bool=true,lqfct::Function=lq,svdfct::Function=svd)
  return moveL!(Lpsi,Rpsi,cutoff=cutoff,m=m,minm=minm,condition=condition,mag=mag,lqfct=lq,svdfct=svd)
end
export moveL

"""
    moveL!(psi[,cutoff=,m=,minm=,condition=])

acts in-place to move `psi`'s orthogonality center left one space, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`moveL`](@ref)
"""
 function moveL!(psi::MPS;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                  mag::Number=0.)
  iR = psi.oc
  psi[iR-1],psi[iR],D,truncerr = moveL!(psi[iR-1],psi[iR],cutoff=cutoff,m=m,minm=minm,condition=condition,mag=mag)
  psi.oc -= 1
  return D,truncerr
end
export moveL!

"""
    D,truncerr = moveR!(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center right one space, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error Modifies `Lpsi`

See also: [`moveR!`](@ref)
"""
 function moveR!(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                fast::Bool=true,qrfct::Function=qr!,svdfct::Function=svd)
  if (min(size(Lpsi,1)*size(Lpsi,2),size(Lpsi,3)) <= m || m == 0) && !isapprox(cutoff,0.) && fast
    Ltens,modV = qrfct(Lpsi,[[1,2],[3]])

    DV = (condition ? getindex!(modV,:,1:size(Rpsi,1)) : modV)
    D = DV
    truncerr = 0.
  else

    Ltens,D,V,truncerr,sumD = svdfct(Lpsi,[[1,2],[3]],cutoff=cutoff,m=m,minm=minm,mag=mag)

    modV = (condition ? getindex!(V,:,1:size(Rpsi,1)) : V)
    DV = dmul!(D,modV)
  end
  Rtens = contract(DV,2,Rpsi,1)
  return Ltens,Rtens,D,truncerr
end

"""
    D,truncerr = moveR(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center right one space, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error

See also: [`moveR!`](@ref)
"""
 function moveR(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                fast::Bool=true,qrfct::Function=qr!,svdfct::Function=svd)
  return moveR!(Lpsi,Rpsi,cutoff=cutoff,m=m,minm=minm,condition=condition,mag=mag,qrfct=qr,svdfct=svd)
end
export moveR

"""
    D,truncerr = moveR!(psi[,cutoff=,m=,minm=,condition=])

acts in-place to move `psi`'s orthogonality center right one space, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`moveR`](@ref)
"""
 function moveR!(psi::MPS;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                  mag::Number=0.)
  iL = psi.oc
  psi[iL],psi[iL+1],D,truncerr = moveR!(psi[iL],psi[iL+1],cutoff=cutoff,m=m,minm=minm,condition=condition,mag=mag)
  psi.oc += 1
  return D,truncerr
end
export moveR!
