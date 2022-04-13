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
@inline function makeHpsi(Lenv::TensType,Renv::TensType,psi::TensType,bigHam::TensType)
  return contract(bigHam,(2,),psi,(1,))
end

@inline function addpsi!(i::intType,savepsi::Array{P,1},nextpsi::P) where P <: TensType
  if i > length(savepsi)
    push!(savepsi,nextpsi)
  else
    savepsi[i] = nextpsi
  end
  nothing
end


@inline function constructM(alpha::Array{W,1},beta::Array{W,1}) where W <: Number
  return constructM(alpha,beta,length(alpha))
end

@inline function constructM(alpha::Array{W,1},beta::Array{W,1},p::intType) where W <: Number
  if p < length(alpha)
    M = LinearAlgebra.SymTridiagonal(alpha[1:p],beta[1:p-1])
  else
    M = LinearAlgebra.SymTridiagonal(alpha,beta)
  end
  return M
end

@inline function constructM(lastM::Union{W,TensType},alpha::Array{W,1},beta::Array{W,1},p::intType) where W <: Number
  return constructM(alpha,beta,p)
end

@inline function compute_alpha(i::intType,alpha::Array{W,1},savepsi::Array{P,1},Lenv::TensType,Renv::TensType,psi::TensType,psiops::TensType...;updatefct::Function=makeHpsi) where {N, W <: Number, P <: TensType}
  psi = psi
  bigHam = psiops[1]

  Hpsi = updatefct(Lenv,Renv,psi,bigHam)
  temp = real(dot(psi,Hpsi)) #ccontract(psi,convec,Hpsi,1)
  if i > length(alpha)
    push!(alpha,temp)
  else
    alpha[i] = temp
  end
  if i > length(savepsi)
    push!(savepsi,psi)
  else
    savepsi[i] = psi
  end
  return Hpsi
end
export constructM

@inline function compute_beta(i::intType,beta::Array{W,1},nextpsi::TensType,betatest::Real) where W <: Number
  temp = norm(nextpsi)
  if i > length(beta)
    push!(beta,temp)
  else
    beta[i] = temp
  end
  betabool = temp > betatest && !isnan(temp) && -Inf < temp < Inf
#  if betabool
    return div!(nextpsi,temp),betabool
#  else
#    return nextpsi,betabool
#  end
end

@inline function krylov_iterate(i::intType,maxiter::intType,alpha::Array{P,1},beta::Array{P,1},Hpsi::TensType,psi::TensType,prevpsi::TensType;betatest::Real=0.,reorth::Bool=false) where P <: Number
  G = eltype(Hpsi)
  if i > 1
    coeffs = (G(1),G(-alpha[i]),G(-beta[i-1]))
  else
    coeffs = (G(1),G(-alpha[i]))
  end
  nextpsi = tensorcombination!(coeffs,Hpsi,psi,prevpsi)
  newpsi,betabool = compute_beta(i,beta,nextpsi,betatest)
  if reorth
    overlap = ccontract(newpsi,psi)
    newpsi = sub!(newpsi,psi,overlap)
  end
  return newpsi,betabool
end

@inline function alphabeta(psi::TensType,maxiter::Integer)#;coefftype::DataType=W)
  alpha = Array{Float64,1}(undef,maxiter) #Array{eltype(psi),1}(undef,maxiter)
  beta = Array{Float64,1}(undef,maxiter) #Array{eltype(psi),1}(undef,maxiter)
  return alpha,beta
end

@inline function krylov(psi::TensType,psiops::TensType...;maxiter::Integer=20,converge::Bool= maxiter == 0,goal::Real=1E-10,ncvg::Integer=1,updatefct::Function=makeHpsi,Lenv::TensType=[0],Renv::TensType=Lenv,
                                      alphafct::Function=compute_alpha,krylov_iterate_fct::Function=krylov_iterate,betafct::Function=compute_beta,alphabetafct::Function=alphabeta,
                                      constructMfct::Function=constructM,normalizationfct::Function=norm!,reorth::Bool=false)
  psi = normalizationfct(psi)

  alpha,beta = alphabetafct(psi,maxiter)

  savepsi = Array{TensType,1}(undef,maxiter)
  prevpsi = psi

  i = 0
  G = eltype(psi)
  lastM = Array{G,2}(undef,0,0) #alpha[i]
  
  betabool = true
  notconverged = converge
  if converge
    lastD = G[G(1E100) for w = 1:ncvg]
  end
  while (i < maxiter || notconverged) && betabool
    i += 1
    Hpsi = alphafct(i,alpha,savepsi,Lenv,Renv,psi,psiops...,updatefct=updatefct)
    newpsi,betabool = krylov_iterate_fct(i,maxiter,alpha,beta,Hpsi,psi,prevpsi,reorth=reorth)

    if betabool
      prevpsi,psi = psi,newpsi

      if converge
        M = constructMfct(lastM,alpha,beta,i)
        D = eigvals(M)
        currEvals = min(length(D),length(lastD),ncvg)
        notconverged = sum(w->abs(D[w]-lastD[w]) >= goal,1:currEvals) > 0
        @inbounds @simd for w = 1:currEvals
          lastD[w] = D[w]
        end
        lastM = M
      end
    end
  end

  return alpha,beta,savepsi,i,lastM
end

@inline function lanczos(psiops::TensType...;maxiter::Integer = 0,retnum::Integer = maxiter==0 ? 100_000 : 1,cvgvals::Integer=1,goal::Float64=1E-12,
                                    converge::Bool=maxiter==0,betatest::Number = 1E-10,Lenv::TensType=[0], Renv::TensType = Lenv,
                                    krylovfct::Function=krylov,updatefct::Function=makeHpsi,
                                    alphafct::Function=compute_alpha,krylov_iterate_fct::Function=krylov_iterate,betafct::Function=compute_beta,
                                    alphabetafct::Function=alphabeta,constructMfct::Function=constructM,
                                    eigfct::Function=libeigen!,reorth::Bool=false) #LinearAlgebra.eigen)
  alpha,beta,savepsi,p,lastM = krylovfct(psiops...,maxiter=maxiter,converge=converge,goal=goal,updatefct=updatefct,Lenv=Lenv,Renv=Renv,alphafct=alphafct,krylov_iterate_fct=krylov_iterate_fct,betafct=betafct,alphabetafct=alphabetafct,constructMfct=constructMfct,reorth=reorth)

#  M = constructMfct(lastM,alpha,beta,p)
  D,U = eigfct(alpha,beta,p) #M)

  energies = D

  retsize = retnum == 0 ? p : min(p, retnum)
  retpsi = Array{eltype(savepsi),1}(undef, retsize)
  
  if p < length(savepsi) #&& retsize > 1
    savepsi = savepsi[1:p]
  end

  typepsi = eltype(savepsi[1])
  sametype = eltype(U) == eltype(savepsi[1])
  @inbounds for i = 1:retsize
    if sametype
      coeffs = ntuple(k->LinearAlgebra.adjoint(U[k,i]),p)
    else
      coeffs = ntuple(k->LinearAlgebra.adjoint(convert(typepsi,U[k,i])),p)
    end
#    if retsize == 1
#      retpsi[1] = tensorcombination(coeffs,savepsi...)
#    else
      retpsi[i] = tensorcombination!(coeffs,savepsi...)
#    end
  end

  return retpsi, energies, alpha, beta
end
export lanczos
