#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8.3
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1) or (v1.5)
#

"""
    Module: Krylov

Methods of solving a reduced site Hamiltonian with a Krylov expansion (ex: Lanczos)
"""
#=
module Krylov
#using ..shuffle
using ..tensor
using ..QN
using ..Qtensor
using ..Qtask
using ..MPutil
using ..contractions
using ..decompositions
import LinearAlgebra
=#


function makeHpsi(H::TensType,psi::TensType)
  return contract(H,2,psi,1)
end

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
  function compute_alpha(updatefct::Function,alpha::Array{W,1},k::Integer,
                         Lenv::TensType,Renv::TensType,psiops::TensType...) where {W <: Number, T <: TensType}
    currpsi,Hpsi = updatefct(Lenv,Renv,psiops...)
    alpha[k] = real(ccontract(currpsi,Hpsi))
    return Hpsi
  end

  function compute_beta(beta::Array{W,1},k::Integer,savepsi::Array{R,1},Hpsi::TensType,psiops::TensType...) where {W <: Number, R <: TensType}
    beta[k-1] = norm(Hpsi) #sqrt(real(ccontract(Hpsi, Hpsi)))
    Hpsi = div!(Hpsi, beta[k-1])
    savepsi[k] = Hpsi
    psiops = (Hpsi,Base.tail(psiops)...)
    return Hpsi,psiops
  end
#=
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
                           Lenv::TensType,Renv::TensType,psiops::TensType...) where {W <: Number, T <: TensType}
#    Hpsi = div!(Hpsi, beta[k - 1])
#    savepsi[k] = Hpsi

#    prevpsi,currpsi = savepsi[k-1],Hpsi
    Hpsi = compute_alpha(updatefct,alpha,k,Lenv,Renv,psiops...)
    return Hpsi
  end
=#
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
  function lanczos(psiops::TensType...;maxiter::Integer = 2,betatest::Number = 1E-10,
                   updatefct::Function=makeHpsi, Lenv::Z=[0], Renv::Z=[0]) where {Z <: TensType,K <: Union{Qtens{W,Q},Array{W,4},Array{W,3},tens{W}}} where {W <: Number, Q <: Qnum}
    alpha = Array{Float64,1}(undef, maxiter)
    beta = Array{Float64,1}(undef, maxiter - 1)

    Hpsi = compute_alpha(updatefct,alpha,1,Lenv,Renv,psiops...)

    currpsi = psiops[1]
    savepsi = Array{typeof(currpsi),1}(undef, maxiter)
    savepsi[1] = currpsi

    Hpsi = sub!(Hpsi, currpsi, alpha[1])

    k = 2
    Hpsi,psiops = compute_beta(beta,k,savepsi,Hpsi,psiops...)
    betabool = beta[k - 1] > betatest
    while k < maxiter && betabool

      Hpsi = compute_alpha(updatefct,alpha,k,Lenv,Renv,psiops...)

      currpsi = savepsi[k]
      prevpsi = savepsi[k-1]

      Hpsi = tensorcombination((1,-alpha[k],-beta[k-1]),Hpsi,currpsi,prevpsi)
      k += 1
      Hpsi,psiops = compute_beta(beta,k,savepsi,Hpsi,psiops...)
      betabool = beta[k - 1] > betatest
    end
    if betabool
      Hpsi = compute_alpha(updatefct,alpha,k,Lenv,Renv,psiops...)
      retK = k
    else
      retK = k-1
    end

    return alpha, beta, savepsi, retK
  end
  export lanczos
#=
  """
      lanczos_coefficients(updatefct,AA,ops,Lenv,Renv[,maxiter=,retnum=,betatest=])

  Lanczos recursion call that outputs alpha coefficients, beta coefficients, the Krylov basis, and size of the Ky

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
  function lanczos_coefficients(updatefct::Function, Lenv::TensType, Renv::TensType, psiops::TensType...;maxiter::Integer = 2,retnum::Integer = 1,betatest::Number = 1E-10)
    alpha, beta, savepsi, sumsize = lanczos(updatefct, Lenv, Renv, psiops..., maxiter = maxiter, betatest = betatest)
    return alpha, beta, savepsi, sumsize
  end
=#
  """
      krylov(updatefct,Lenv,Renv,AA,ops...[,maxiter=,retnum=,betatest=])

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
  function krylov(psiops::TensType...;maxiter::Integer = 2,retnum::Integer = 1,
                  betatest::Number = 1E-10,lanczosfct::Function=lanczos,updatefct::Function=makeHpsi, Lenv::TensType=[0], Renv::TensType = [0])#::Tuple{Array{R,1},Array{Float64,1}} where R <: Union{Qtens{W,Q},Array{W,3}} where {W <: Number, Q <: Qnum}))

    alpha, beta, savepsi, sumsize = lanczosfct(psiops..., maxiter = maxiter, betatest = betatest, updatefct = updatefct, Lenv = Lenv, Renv = Renv)

    if sumsize != maxiter
      if sumsize != 1
        M = LinearAlgebra.SymTridiagonal(alpha[1:sumsize], beta[1:sumsize - 1])
      else
        M = Array{Float64,2}(undef,1,1)
        M[1,1] = alpha[1]
      end
    else
      M = LinearAlgebra.SymTridiagonal(alpha, beta)
    end

    D, U = LinearAlgebra.eigen(M)
    energies = [real(D[i]) for i = 1:size(D,1)]::Array{Float64,1}

    retsize = min(size(U, 2), retnum)
    retpsi = Array{typeof(savepsi[1]),1}(undef, retsize)
    for i = 1:retsize
      retpsi[i] = savepsi[1] * conj(U[1,i])
      for k = 2:sumsize          
        retpsi[i] = add!(retpsi[i], savepsi[k], conj(U[k,i])) # do the rest of the sum.
      end
    end

    return retpsi, energies, alpha, beta
  end
  export krylov
  
#end
