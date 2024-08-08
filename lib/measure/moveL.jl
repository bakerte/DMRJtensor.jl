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