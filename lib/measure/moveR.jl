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
#>------+  move! orthogonality center in MPS    +---------<
#       +---------------------------------------+

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
