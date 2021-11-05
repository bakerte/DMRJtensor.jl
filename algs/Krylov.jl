#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
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

"""
  makeHpsi(Lenv,Renv,psiops)

Applies Hamiltonian to the wavefunction for exact diagonalization. The first two arguments are left- and right-environments of the reduced Hamiltonian site. The `psiops` tuple must have the current wavefunction as the first entry

Defining any other function here must also have `psiops` receive the same first input of the current wavefunction.

See also: [`krylov`](@ref)
"""
function makeHpsi(Lenv::G,Renv::G,psiops::TensType...) where G <: TensType
  psi = psiops[1]
  H = psiops[2]
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
                        Lenv::G,Renv::G,psiops::TensType...) where {W <: Number, G <: TensType}
  Hpsi = updatefct(Lenv,Renv,psiops...)
  currpsi = psiops[1]
  alpha[k] = real(ccontract(currpsi,Hpsi))
  return Hpsi
end

"""
  compute_beta(beta,k,savepsi,Hpsi,psiops)

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
function compute_beta(beta::Array{W,1},k::Integer,savepsi::Array{R,1},Hpsi::R,psiops::R...) where {W <: Number, R <: TensType}
  beta[k-1] = norm(Hpsi) #sqrt(real(ccontract(Hpsi, Hpsi)))
  Hpsi = div!(Hpsi, beta[k-1])
  savepsi[k] = Hpsi
  psiops = (Hpsi,Base.tail(psiops)...)
  return psiops
end

"""
    lanczos(updatefct,AA,ops,Lenv,Renv[,maxiter=2,betatest=1E-6])

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
function lanczos(psiops::TensType...;maxiter::Integer = 2,betatest::Number = 1E-6,
                  updatefct::Function=makeHpsi, Lenv::R=[0], Renv::R=[0]) where R <: TensType
  alpha = Array{Float64,1}(undef, maxiter)
  beta = Array{Float64,1}(undef, maxiter - 1)

  Hpsi = compute_alpha(updatefct,alpha,1,Lenv,Renv,psiops...)

  currpsi = psiops[1]
  savepsi = Array{typeof(currpsi),1}(undef, maxiter)
  savepsi[1] = currpsi

  Hpsi = sub!(Hpsi, currpsi, alpha[1])

  k = 2
  psiops = compute_beta(beta,k,savepsi,Hpsi,psiops...)

  betabool = beta[k - 1] > betatest
  while k < maxiter && betabool

    Hpsi = compute_alpha(updatefct,alpha,k,Lenv,Renv,psiops...)

    currpsi = savepsi[k]
    prevpsi = savepsi[k-1]

    Hpsi = tensorcombination((1,-alpha[k],-beta[k-1]),Hpsi,currpsi,prevpsi)
    k += 1
    psiops = compute_beta(beta,k,savepsi,Hpsi,psiops...)
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

"""
    krylov(updatefct,Lenv,Renv,AA,ops...[,maxiter=2,retnum=1,betatest=1E-6])

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
                betatest::Number = 1E-6,lanczosfct::Function=lanczos,updatefct::Function=makeHpsi, Lenv::R=[0], Renv::R = [0]) where R <: TensType

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
  if !(eltype(D) <: Real)
    energies = [real(D[i]) for i = 1:size(D,1)]::Array{Float64,1}
  else
    energies = D
  end

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
