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

@inline function compute_alpha(i::intType,alpha::Array{W,1},savepsi::Array{P,1},Lenv::TensType,Renv::TensType,psi::TensType,psiops::TensType...;updatefct::Function=makeHpsi) where {N, W <: Number, P <: TensType}
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
  return div!(nextpsi,temp),betabool
end

@inline function krylov_iterate(i::intType,maxiter::intType,alpha::Array{P,1},beta::Array{P,1},savepsi::Array{X,1},psi::TensType,prevpsi::TensType,psiops::TensType...;updatefct::Function=makeHpsi,betatest::Real=0.,reorth::Bool=false,Lenv::TensType=[0],Renv::TensType=Lenv) where {P <: Number, X <: TensType}
  Hpsi = compute_alpha(i,alpha,savepsi,Lenv,Renv,psi,psiops...,updatefct=updatefct)

  G = eltype(Hpsi)
  if i > 1
    coeffs = (G(1),G(-alpha[i]),G(-beta[i-1]))
  else
    coeffs = (G(1),G(-alpha[i]))
  end
  nextpsi = tensorcombination!(coeffs,Hpsi,psi,prevpsi)
  newpsi,betabool = compute_beta(i,beta,nextpsi,betatest)
  if reorth
    for j = 1:i-2
      overlap = ccontract(newpsi,savepsi[j])
      newpsi = sub!(newpsi,savepsi[j],overlap)
    end
    s = Array{P,1}(undef,2)
    counter = 0
    for j = i-1:i
      counter += 1
      s[counter] = ccontract(newpsi,savepsi[j])
      newpsi = sub!(newpsi,savepsi[j],s[counter])
    end
    alpha[i] += s[2]
    beta[i] += s[1]
  end
  return newpsi,betabool
end

@inline function makealpha(psi::TensType,maxiter::Integer)
  return Array{eltype(psi),1}(undef,maxiter)
end

@inline function makebeta(psi::TensType,maxiter::Integer)
  return makealpha(psi,maxiter)
end

@inline function alphabeta(psi::TensType,maxiter::Integer)#;coefftype::DataType=W)
  return makealpha(psi,maxiter),makebeta(psi,maxiter)
end

@inline function krylov(psi::TensType,psiops::TensType...;maxiter::Integer=2,converge::Bool= maxiter == 0,goal::Real=1E-10,ncvg::Integer=1,updatefct::Function=makeHpsi,Lenv::TensType=[0],Renv::TensType=Lenv,start::Integer=0,
                                      alpha::Array=makealpha(psi,maxiter),beta::Array=makebeta(psi,maxiter),krylov_iterate_fct::Function=krylov_iterate,savepsi::Array=Array{typeof(psi),1}(undef,maxiter),lastM::Array=Array{eltype(psi),2}(undef,0,0),
                                      constructMfct::Function=constructM,normalizationfct::Function=norm!,reorth::Bool=false)
  psi = normalizationfct(psi)

  prevpsi = start == 0 ? psi : savepsi[start]

  i = start
  G = eltype(psi)
  
  betabool = true
  notconverged = converge
  if converge
    lastD = G[G(1E100) for w = 1:ncvg]
  end
  while (i < maxiter || notconverged) && betabool
    i += 1
    newpsi,betabool = krylov_iterate_fct(i,maxiter,alpha,beta,savepsi,psi,prevpsi,psiops...,updatefct=updatefct,reorth=reorth,Lenv=Lenv,Renv=Renv)

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

@inline function lanczos(psi::TensType,psiops::TensType...;maxiter::Integer = 0,retnum::Integer = maxiter==0 ? 100_000 : 1,cvgvals::Integer=1,goal::Float64=1E-12,
                                    converge::Bool=maxiter==0,betatest::Number = 1E-10,Lenv::TensType=[0], Renv::TensType = Lenv,
                                    krylovfct::Function=krylov,updatefct::Function=makeHpsi,
                                    alpha::Array=makealpha(psi,maxiter),beta::Array=makebeta(psi,maxiter),savepsi::Array=Array{typeof(psi),1}(undef,maxiter),lastM::Array=Array{eltype(psi),2}(undef,0,0),
                                    krylov_iterate_fct::Function=krylov_iterate,start::Integer=0,
                                    constructMfct::Function=constructM,eigfct::Function=libeigen!,reorth::Bool=false) #LinearAlgebra.eigen)
  p,lastM = krylovfct(psi,psiops...,maxiter=maxiter,converge=converge,goal=goal,updatefct=updatefct,Lenv=Lenv,Renv=Renv,krylov_iterate_fct=krylov_iterate_fct,alpha=alpha,beta=beta,constructMfct=constructMfct,reorth=reorth,savepsi=savepsi,lastM=lastM,start=start)

  D,U = eigfct(alpha,beta,p)

  energies = D

  retsize = retnum == 0 ? p : min(p, retnum)
  retpsi = Array{eltype(savepsi),1}(undef, retsize)
  
  if p < length(savepsi)
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
    retpsi[i] = tensorcombination!(coeffs,savepsi...)
  end

  return retpsi, energies
end
export lanczos

@inline function irlm(k::Integer,psi::TensType,psiops::TensType...;maxiter::Integer = 2*k+1,retnum::Integer = maxiter,cvgvals::Integer=1,goal::Float64=1E-12,
                        converge::Bool=maxiter==0,betatest::Number = 1E-10,Lenv::TensType=[0], Renv::TensType = Lenv,
                        krylovfct::Function=krylov,updatefct::Function=makeHpsi,
                        alpha::Array=makealpha(psi,maxiter),beta::Array=makebeta(psi,maxiter),savepsi::Array=Array{typeof(psi),1}(undef,maxiter),lastM::Array=Array{eltype(psi),2}(undef,0,0),
                        krylov_iterate_fct::Function=krylov_iterate,
                        constructMfct::Function=constructM,eigfct::Function=libeigen!,reorth::Bool=false) #LinearAlgebra.eigen)

  retpsi, energies = lanczos(psi,psiops...;maxiter = maxiter,retnum = maxiter,cvgvals=cvgvals,goal=goal,
                        converge=converge,betatest = betatest,Lenv=Lenv, Renv = Lenv,start=0,
                        krylovfct=krylovfct,updatefct=updatefct,alpha=alpha,beta=beta,
                        savepsi=savepsi,lastM=lastM,krylov_iterate_fct=krylov_iterate_fct,
                        constructMfct=constructMfct,eigfct=libeigen,reorth=reorth)
#println(energies)
g = 0
gmax = 100
while g < gmax && sum(k->abs(beta[k]),1:k) > 1E-10
  g += 1
#for g = 1:10

  mu,U = libeigen(alpha,beta)
  T = constructMfct(alpha,beta)

  println(mu)

  Q = LinearAlgebra.Diagonal(ones(eltype(psi),maxiter))
  for j = k:maxiter
    for x = 1:size(T,1)
      T[x,x] -= mu[j]
    end
    Qnew,R = LinearAlgebra.qr(T)
    T = R*Qnew
    for w = 1:size(T,1)
      T[w,w] += mu[j]
    end
    Q *= Qnew
  end

  for w = 1:size(T,1)
    alpha[w] = T[w,w]
  end
  for w = 1:size(T,1)-1
    beta[w] = T[w,w+1]
  end

  retpsi = Array{eltype(savepsi),1}(undef, length(savepsi))

  typepsi = eltype(savepsi[1])
  sametype = eltype(U) == eltype(savepsi[1])
  p = length(savepsi)
  @inbounds for i = 1:p
    if sametype
      coeffs = ntuple(k->LinearAlgebra.adjoint(Q[k,i]),p)
    else
      coeffs = ntuple(k->LinearAlgebra.adjoint(convert(typepsi,Q[k,i])),p)
    end
    retpsi[i] = tensorcombination!(coeffs,savepsi...)
  end


  for w = 1:length(retpsi)
    savepsi[w] = retpsi[w]
  end




rvec = savepsi[k-1] #tensorcombination!((Q[end,k],beta[k]),savepsi[end],savepsi[k]) #

savepsi, energies = lanczos(rvec,psiops...;maxiter = maxiter,retnum = maxiter,cvgvals=cvgvals,goal=goal,
                        converge=converge,betatest = betatest,Lenv=Lenv, Renv = Lenv, start = k,
                        krylovfct=krylovfct,updatefct=updatefct,alpha=alpha,beta=beta,
                        savepsi=savepsi,lastM=lastM,krylov_iterate_fct=krylov_iterate_fct,
                        constructMfct=constructMfct,eigfct=libeigen,reorth=true)
  println("energies ($g): ",energies)
  println(beta)
  println(sum(k->beta[k],1:k))
  println()
end

#println(energies)
#println(alpha)
#println(beta)
#=
println()
println("final:")
println()
=#
  return retpsi, energies
end
export irlms
