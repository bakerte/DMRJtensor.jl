#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.1
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.0) or (v1.5)
#

"""
    Module: Krylov

Methods of solving a reduced site Hamiltonian with a Krylov expansion (ex: Lanczos)
"""
module Krylov
using ..tensor
using ..QN
using ..Qtensor
using ..Qtask
using ..MPutil
using ..contractions
using ..decompositions
import LinearAlgebra

  """
      compute_alpha(updatefct,alpha,k,currpsi,ops,Lenv,Renv)

  Computes alpha variable for current iteration of Lanczos; loads `k` element of `alpha` and returns H*psi with environments applied

  #Arguments:
  +`updatefct::Function`: Function to construct reduced site H*psi (arguments: `currpsi`,`ops`,`Lenv`,`Renv`)
  +`alpha::Array{W,1}`: vector containing alpha coefficients of Lanczos
  +`k::Integer`: current iteration of Lanczos
  +`currpsi::TensType`: current wavefunction on 1 or 2 (or more) sites
  +`ops::TensType`: mpo corresponding to currpsi on 1 or 2 (or more) sites
  +`Lenv::TensType`: left environment for reduced site H*psi representation
  +`Renv::TensType`: right environment for reduced site H*psi representation
  """
  function compute_alpha(updatefct::Function,alpha::Array{W,1},k::Integer,currpsi::TensType,ops::TensType,
                        Lenv::TensType,Renv::TensType) where {W <: Number, T <: TensType}
    Hpsi = updatefct(currpsi,ops,Lenv,Renv)
    alpha[k] = real(ccontract(currpsi,Hpsi))
    return Hpsi
  end

  """
      lanczos_iterate(updatefct::Function,k::Integer,savepsi::Array{T,1},alpha::Array{W,1},beta::Array{W,1},Hpsi::TensType,
                      currpsi::TensType,ops::TensType,Lenv::TensType,Renv::TensType)

  Apply one step of Lanczos and returns prev psi, currpsi, Hpsi for the next iteration of Lanczos

  #Arguments:
  +`updatefct::Function`: Function to construct reduced site H*psi (arguments: `currpsi`,`ops`,`Lenv`,`Renv`)
  +`k::Integer`: current iteration of Lanczos
  +`savepsi::Array{T,1}': saves H*psi for later determination of ground state
  +`alpha::Array{W,1}`: vector containing alpha coefficients of Lanczos
  +`beta::Array{W,1}': vector containing beta coefficients of Lanczos
  +`Hpsi::TensType': MPO applied to wavefunction tensors and environment (current wavefunction in Lanczos iteration)
  +`currpsi::TensType`: current wavefunction on 1 or 2 (or more) sites
  +`ops::TensType`: mpo corresponding to currpsi on 1 or 2 (or more) sites
  +`Lenv::TensType`: left environment for reduced site H*psi representation
  +`Renv::TensType`: right environment for reduced site H*psi representation
  """
  function lanczos_iterate(updatefct::Function,k::Integer,savepsi::Array{T,1},alpha::Array{W,1},beta::Array{W,1},Hpsi::TensType,
                      currpsi::TensType,ops::TensType,Lenv::TensType,Renv::TensType) where {W <: Number, T <: TensType}
    Hpsi = div!(Hpsi, beta[k - 1])
    savepsi[k] = Hpsi

    prevpsi,currpsi = currpsi,Hpsi
    Hpsi = compute_alpha(updatefct,alpha,k,currpsi,ops,Lenv,Renv)
    return prevpsi,currpsi,Hpsi
  end

  function psiholder(A::Qtens{W,Q},maxiter::Integer) where {W <: Number, Q <: Qnum}
    return Array{Qtens{W,Q},1}(undef,maxiter)
  end

  function psiholder(A::Array{W,3},maxiter::Integer) where W <: Number
    return Array{Array{W,3},1}(undef,maxiter)
  end

  """
      lanczos(updatefct,AA,ops,Lenv,Renv[,maxiter=,betatest=])

  Lanczos algorithm 

  #Arguments:
  +`updatefct::Function`: Function to construct reduced site H*psi (arguments: `currpsi`,`ops`,`Lenv`,`Renv`)
  +`AA::TensType`: wavefunction tensors on reduced site contract together
  +`ops::TensType`: operators on reduced site contract together
  +`Lenv::TensType`: left environment for reduced site H*psi representation
  +`Renv::TensType`: right environment for reduced site H*psi representation
  +`maxiter::Integer`: maximum number of lanczos iterations
  +`betatest::Number`: tolerance for beta coefficient before cutting off
  """
  function lanczos(updatefct::Function, AA::K, ops::TensType, Lenv::Z, Renv::Z;
                    maxiter::Integer = 2,betatest::Number = 1E-6) where {Z <: TensType,K <: Union{Qtens{W,Q},Array{W,4},Array{W,3},tens{W}}} where {W <: Number, Q <: Qnum} #::Tuple{Array{Float64,1},Array{Float64,1},Array{W,1},Number} where W <: Union{Qtens{R,Q},Array{R,3}} where {R <: Number, Q <: Qnum})
    alpha = Array{Float64,1}(undef, maxiter)
    beta = Array{Float64,1}(undef, maxiter - 1)

    currpsi = div!(AA, norm(AA))
    prevpsi = currpsi

    Hpsi = compute_alpha(updatefct,alpha,1,currpsi,ops,Lenv,Renv)
    Hpsi = sub!(Hpsi, currpsi, alpha[1])

    savepsi = Array{K,1}(undef, maxiter)
    savepsi[1] = currpsi

    k = 2
    beta[k - 1] = sqrt(ccontract(Hpsi, Hpsi))
    while k < maxiter && beta[k - 1] > betatest
      prevpsi,currpsi,Hpsi = lanczos_iterate(updatefct,k,savepsi,alpha,beta,Hpsi,currpsi,ops,Lenv,Renv)

      Hpsi = sub!(Hpsi,currpsi,alpha[k])
      Hpsi = sub!(Hpsi,prevpsi,beta[k-1])
      
      k += 1
      beta[k - 1] = sqrt(real(ccontract(Hpsi, Hpsi)))
    end
    if beta[k - 1] <= betatest
      retK = k-1
    else
      prevpsi,currpsi,Hpsi = lanczos_iterate(updatefct,k,savepsi,alpha,beta,Hpsi,currpsi,ops,Lenv,Renv)
      retK = k
    end

    return alpha, beta, savepsi, retK
  end
  export lanczos

  """
      krylov(updatefct,AA,ops,Lenv,Renv[,maxiter=,retnum=,betatest=])

  Krylov subspace expansion

  #Arguments:
  +`updatefct::Function`: Function to construct reduced site H*psi (arguments: `currpsi`,`ops`,`Lenv`,`Renv`)
  +`AA::TensType`: wavefunction tensors on reduced site contract together
  +`ops::TensType`: operators on reduced site contract together
  +`Lenv::TensType`: left environment for reduced site H*psi representation
  +`Renv::TensType`: right environment for reduced site H*psi representation
  +`maxiter::Integer`: maximum number of lanczos iterations
  +`retnum::Integer`: number of excitations to return
  +`betatest::Number`: tolerance for beta coefficient before cutting off
  """
  function krylov(updatefct::Function, AA::TensType, ops::TensType, Lenv::TensType, Renv::TensType;maxiter::Integer = 2,retnum::Integer = 1,
                  betatest::Number = 1E-6)#::Tuple{Array{R,1},Array{Float64,1}} where R <: Union{Qtens{W,Q},Array{W,3}} where {W <: Number, Q <: Qnum}))

    alpha, beta, savepsi, sumsize = lanczos(updatefct, AA, ops, Lenv, Renv, maxiter = maxiter, betatest = betatest)

    if sumsize != maxiter
      if sumsize != 1
        AB = LinearAlgebra.SymTridiagonal(alpha[1:sumsize], beta[1:sumsize - 1])
      else
        AB = Float64[alpha[1] for i = 1:1, j=1:1]
      end
    else
      AB = LinearAlgebra.SymTridiagonal(alpha, beta)
    end

    D, U = LinearAlgebra.eigen(AB)::LinearAlgebra.Eigen{Float64,Float64,Array{Float64,2},Array{Float64,1}}

    retsize = min(size(U, 2), retnum)
    retpsi = Array{typeof(savepsi[1]),1}(undef, retsize)::Array{R,1} where R <: Union{Qtens{W,Q},Array{W,3},Array{W,4},tens{W}} where {W <: Number, Q <: Qnum}
    for i = 1:retsize
      retpsi[i] = savepsi[1] * conj(U[1,i])
      for k = 2:sumsize
        retpsi[i] = add!(retpsi[i], savepsi[k], conj(U[k,i])) # do the rest of the sum.
      end
    end
    return retpsi, D
  end
  export krylov

end
