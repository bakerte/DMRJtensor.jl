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

function make2site(Lenv::TensType,Renv::TensType,psiL::TensType,psiR::TensType,mpoL::TensType,mpoR::TensType)
  Lpsi = contract(Lenv,3,psiL,1)
  LHpsi = contract(Lpsi,(2,3),mpoL,(1,2))
  LHpsipsi = contract(LHpsi,2,psiR,1)
  LHHpsipsi = contract(LHpsipsi,(3,4),mpoR,(1,2))
  return contract(LHHpsipsi,(3,5),Renv,(1,2))
end


function simplelanczos(Lenv::TensType,Renv::TensType,psiL::TensType,psiR::TensType,mpoL::TensType,mpoR::TensType;betatest::Float64 = 1E-10)
  Hpsi = make2site(Lenv,Renv,psiL,psiR,mpoL,mpoR)
  AA = contract(psiL,3,psiR,1)
#  AA = div!(AA,norm(AA))
  alpha1 = real(ccontract(AA,Hpsi))

  psi2 = sub!(Hpsi, AA, alpha1)
  beta1 = norm(psi2)
  if beta1 > betatest
    psi2 = div!(psi2, beta1)

    Hpsi2 = contract(Lenv,3,psi2,1)
    ops = contract(mpoL,4,mpoR,1)
    Hpsi2 = contract(Hpsi2,[2,3,4],ops,[1,2,4])
    Hpsi2 = contract(Hpsi2,[2,5],Renv,[1,2])

    alpha2 = real(ccontract(psi2,Hpsi2))
    M = Float64[alpha1 beta1; beta1 alpha2]
    D, U = eigen(M)
    energy = D[1,1]
    outAA = conj(U[1,1])*AA + conj(U[2,1])*psi2
  else
    energy = alpha1
    outAA = AA
  end
  return outAA,energy
end

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
    M = LinearAlgebra.SymTridiagonal(alpha[1:p],beta[1:p])
  else
    M = LinearAlgebra.SymTridiagonal(alpha,beta)
  end
  return M
end

@inline function constructM(lastM::Union{W,TensType},alpha::Array{W,1},beta::Array{W,1}) where W <: Number
  return constructM(alpha,beta,length(alpha))
end

@inline function constructM(lastM::Union{W,TensType},alpha::Array{W,1},beta::Array{W,1},p::intType) where W <: Number
  return constructM(alpha,beta,p)
end

@inline function compute_alpha(i::intType,alpha::Array{W,1},savepsi::Array{P,1},Lenv::TensType,Renv::TensType,psi::TensType,psiops::TensType...;updatefct::Function=makeHpsi) where {W <: Number, P <: TensType}

  checkHpsi = updatefct(makeArray(Lenv),makeArray(Renv),makeArray(psi),makeArray.(psiops)...)

  Hpsi = updatefct(Lenv,Renv,psi,psiops...)

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
  if betabool
    outpsi = div!(nextpsi,temp)
  else
    outpsi = nextpsi
  end
  return outpsi,betabool
end

@inline function krylov_iterate(i::intType,maxiter::intType,alpha::Array{P,1},beta::Array{P,1},savepsi::Array{X,1},prevpsi::TensType,psi::TensType,psiops::TensType...;updatefct::Function=makeHpsi,betatest::Real=0.,reorth::Bool=false,Lenv::TensType=[0],Renv::TensType=Lenv) where {P <: Number, X <: TensType}

  Hpsi = compute_alpha(i,alpha,savepsi,Lenv,Renv,psi,psiops...,updatefct=updatefct)

  if i < maxiter
    G = eltype(Hpsi)
    if i > 1
      coeffs = (G(1),G(-alpha[i]),G(-beta[i-1]))
    else
      coeffs = (G(1),G(-alpha[i]))
    end
    nextpsi = tensorcombination!(coeffs,Hpsi,psi,prevpsi)
  else
    nextpsi = Hpsi
  end
  if reorth
    for j = 1:i
      overlap = ccontract(nextpsi,savepsi[j])
      nextpsi = sub!(nextpsi,savepsi[j],overlap)
    end
  end
  newpsi,betabool = compute_beta(i,beta,nextpsi,betatest)

  return newpsi,betabool
end

@inline function makealpha(psi::TensType,maxiter::Integer)
  return Array{Float64,1}(undef,maxiter) #eltype(psi)
end

@inline function makebeta(psi::TensType,maxiter::Integer)
  return makealpha(psi,maxiter)
end

@inline function alphabeta(psi::TensType,maxiter::Integer)#;coefftype::DataType=W)
  return makealpha(psi,maxiter),makebeta(psi,maxiter)
end

@inline function krylov(psi::TensType,psiops::TensType...;maxiter::Integer=2,converge::Bool= maxiter == 0,goal::Real=1E-10,ncvg::Integer=1,updatefct::Function=makeHpsi,Lenv::TensType=[0],Renv::TensType=Lenv,start::Integer=0,
                                      alpha::Array{P,1}=makealpha(psi,maxiter),beta::Array{P,1}=makebeta(psi,maxiter),krylov_iterate_fct::Function=krylov_iterate,
                                      savepsi::Array{B,1}=Array{typeof(psi),1}(undef,maxiter),lastM::Array{G,2}=Array{eltype(psi),2}(undef,0,0),
                                      constructMfct::Function=constructM,normalizationfct::Function=norm!,reorth::Bool=false) where {G <: Number, P <: Union{TensType,Number}, B <: TensType}

  psi = normalizationfct(psi)

  prevpsi = start == 0 ? psi : savepsi[start]

  i = start
#  G = eltype(psi)
  
  betabool = true
  notconverged = converge
  if converge
    lastD = G[G(1E100) for w = 1:ncvg]
  end
  while (i < maxiter || notconverged) && betabool
    i += 1
    newpsi,betabool = krylov_iterate_fct(i,maxiter,alpha,beta,savepsi,prevpsi,psi,psiops...,updatefct=updatefct,reorth=reorth,Lenv=Lenv,Renv=Renv)

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
  return i,lastM
end

@inline function lanczos!(psi::TensType,psiops::TensType...;maxiter::Integer = 0,retnum::Integer = maxiter==0 ? 100_000 : 1,cvgvals::Integer=1,goal::Float64=1E-12,
                                    converge::Bool=maxiter==0,betatest::Number = 1E-10,Lenv::TensType=[0], Renv::TensType = Lenv,
                                    krylovfct::Function=krylov,updatefct::Function=makeHpsi,
                                    alpha::Array{P,1}=makealpha(psi,maxiter),beta::Array{P,1}=makebeta(psi,maxiter),
                                    savepsi::Array{B,1}=Array{typeof(psi),1}(undef,maxiter),lastM::Array{G,2}=Array{eltype(psi),2}(undef,0,0),
                                    krylov_iterate_fct::Function=krylov_iterate,start::Integer=0,
                                    constructMfct::Function=constructM,eigfct::Function=libeigen!,reorth::Bool=false) where {G <: Number, P <: Union{TensType,Number}, B <: TensType} #LinearAlgebra.eigen)
  p,lastM = krylovfct(psi,psiops...,maxiter=maxiter,converge=converge,goal=goal,updatefct=updatefct,Lenv=Lenv,Renv=Renv,krylov_iterate_fct=krylov_iterate_fct,alpha=alpha,beta=beta,constructMfct=constructMfct,reorth=reorth,savepsi=savepsi,lastM=lastM,start=start)

  energies,U = eigfct(alpha,beta,p)

  if p < length(savepsi)
    savepsi = savepsi[1:p]
  end

  if retnum == 0
    retpsi = savepsi
  else
    retsize = min(p, retnum)
    retpsi = Array{eltype(savepsi),1}(undef, retsize)

    typepsi = eltype(savepsi[1])
    sametype = eltype(U) == eltype(savepsi[1])
    if retsize == 1
      if sametype
        coeffs = ntuple(k->LinearAlgebra.adjoint(U[k,1]),p)
      else
        coeffs = ntuple(k->LinearAlgebra.adjoint(convert(typepsi,U[k,1])),p)
      end
      retpsi[1] = tensorcombination!(coeffs,savepsi...)
    else
      @inbounds for i = 1:retsize
        if sametype
          coeffs = ntuple(k->LinearAlgebra.adjoint(U[k,i]),p)
        else
          coeffs = ntuple(k->LinearAlgebra.adjoint(convert(typepsi,U[k,i])),p)
        end
        retpsi[i] = tensorcombination(coeffs,savepsi...)
      end
    end
  end
  return retpsi, energies
end
export lanczos

@inline function lanczos(psi::TensType,psiops::TensType...;maxiter::Integer = 0,retnum::Integer = maxiter==0 ? 100_000 : 1,cvgvals::Integer=1,goal::Float64=1E-12,
                                                            converge::Bool=maxiter==0,betatest::Number = 1E-10,Lenv::TensType=[0], Renv::TensType = Lenv,
                                                            krylovfct::Function=krylov,updatefct::Function=makeHpsi,
                                                            alpha::Array{P,1}=makealpha(psi,maxiter),beta::Array{P,1}=makebeta(psi,maxiter),
                                                            savepsi::Array{B,1}=Array{typeof(psi),1}(undef,maxiter),lastM::Array{G,2}=Array{eltype(psi),2}(undef,0,0),
                                                            krylov_iterate_fct::Function=krylov_iterate,start::Integer=0,
                                                            constructMfct::Function=constructM,eigfct::Function=libeigen!,reorth::Bool=false) where {G <: Number, P <: Union{TensType,Number}, B <: TensType} #LinearAlgebra.eigen)
  return lanczos!(psi,psiops...,maxiter=maxiter,retnum=retnum,cvgvals=cvgvals,goal=goal,converge=converge,betatest=betatest,
                                Lenv=Lenv,Renv=Renv,krylovfct=krylovfct,updatefct=updatefct,alpha=alpha,beta=beta,savepsi=savepsi,
                                lastM=lastM,krylov_iterate_fct=krylov_iterate_fct,start=start,constructMfct=constructMfct,
                                eigfct=libeigen,reorth=reorth)
end

