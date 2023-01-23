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
  Module: DMRG
  
Functions for the density matrix renormalization group
"""
#=
module DMRG
using ..tensor
using ..QN
using ..Qtensor
using ..Qtask
using ..MPutil
using ..MPmaker
using ..contractions
using ..decompositions
using ..Krylov
using ..optimizeMPS
=#

@inline function twosite_update(Lenv::TensType,Renv::TensType,tensors::TensType...)

  AA = tensors[1]
  psiL = tensors[2]
  psiR = tensors[3]
  mpoL = tensors[4]
  mpoR = tensors[5]

  ops = contract([1,3,5,6,2,4],mpoL,4,mpoR,1)

  Hpsi = contract(ops,(5,6),AA,(2,3))
  LHpsi = contract(Lenv,(2,3),Hpsi,(1,5))
  temp = contract(LHpsi,(5,4),Renv,(1,2))

  return temp
end
export twosite_update

import Base.string
#  import Printf
function string(vect::Array{W,1},start::Integer,stop::Integer;show::Integer=7) where W <: Number
  S = "$(Printf.@sprintf("%.5g",vect[start]))"
  incr = start > stop ? -1 : 1
  for y = start+incr:incr:stop
    S *= ", $(Printf.@sprintf("%.5g",vect[y]))"
  end
  if abs(stop-start)+1 > length(vect)
    S *= "..."
  end
  return S
end
#=
function load_storeD!(storeD::Array{W,1},Dmat::Array{W,1}) where {W <: Number}
  for a = 1:min(length(storeD),length(Dmat))
    storeD[a] = Dmat[a]
  end
  nothing
end
=#
function string(vect::Array{W,1}) where W <: Number
  return string(vect,1,length(vect))
end

@inline function SvNcheck!(i::Integer,j::Integer,D::TensType,Ns::Integer,params::TNparams;alpha::Bool=true,rev::Bool=false)
  if params.maxtrunc < params.truncerr
    params.maxtrunc = params.truncerr
  end
  if params.biggestm < size(D,1)
    params.biggestm = size(D,1)
  end
  Dmat = Float64[real(D[i,i]) for i = 1:min(size(D)...)]
  sizeDsq = length(Dmat)
  if params.allSvNbond
    SvNvec[i] = -sum(h->Dmat[h]^2*log(Dmat[h]^2),sizeDsq:-1:1)
    if i == params.startoc && !params.silent
      println("Entropies in chain: ",SvNvec)
    end
  elseif i == min(max(2,params.SvNbond+1),Ns) && j < 0
    if typeof(D) <: qarray
      sort!(Dmat,rev=true)
    end
    params.entropy = -sum(h->Dmat[h]^2*log(Dmat[h]^2),sizeDsq:-1:1)
    params.entropy = params.entropy
    ymax = min(sizeDsq,params.maxshowD)
    params.storeD = Dmat[1:ymax]
    if !params.silent
      print("SvN at center bond b=",params.SvNbond," = ",params.entropy)
      if alpha
        println(" (alpha = $(Printf.@sprintf("%.5f",params.noise)))")
      else
        println()
      end
      if rev
        println("Eigenvalues at center bond b=",params.SvNbond,": [",string(Dmat,length(Dmat),length(Dmat)-ymax+1),"]")
      else
        println("Singular values at center bond b=",params.SvNbond,": [",string(Dmat,1,ymax),"]")
      end
      println()
    end
  end
  nothing
end
#=
@inline function regular_SvNcheck!(i::Integer,j::Integer,D::TensType,Ns::Integer,params::TNparams;rev::Bool=false)
  return SvNcheck!(i,j,D,Ns,params,alpha=false,rev=rev)
end
=#
function setDMRG(psi::MPS,mpo::MPO,m::Integer,minm::Integer,Lenv::TensType,Renv::TensType,
                  halfsweep::Bool,alpha::Z,origj::Bool,allSvNbond::Bool,boundary::TensType...) where {Z <: Union{Float64,Array{Float64,1}}}
  zeroTens = typeof(psi[1])()
  if length(boundary) == 2
    Lbound = boundary[1]
    Rbound = boundary[2]
  elseif length(boundary) == 1
    if LRbounds == "L"
      Lbound = boundary[1]
      Rbound = [0]
    elseif LRbounds == "R"
      Lbound = [0]
      Rbound = boundary[1]
    else
      error("in DMRG...can't determine LRbounds parameter (only defined for \"L\" and \"R\")")
    end
  else
    if Lenv != zeroTens
      Lbound = Lenv[1]
    else
      Lbound = zeroTens
    end
    if Renv != zeroTens
      Rbound = Renv[length(Renv)]
    else
      Rbound = zeroTens
    end
  end
  if Lenv == zeroTens && Renv == zeroTens
    Lenv,Renv = makeEnv(psi,mpo,Lbound=Lbound,Rbound=Rbound)
  end

  SvN,lastSvN,maxtrunc,biggestm = [0.],[0.],[0.],[0]
  curr_alpha = alpha
  Ns = size(psi,1)
  timer = 0.

  Nsteps = ((Ns-1) * (halfsweep ? 1 : 2))
  # setup for the sweeping direction
  if psi.oc == 1
    j = 1
  elseif psi.oc == Ns #
    j = -1
  else
    j = origj ? 1 : -1 #if the oc is away from the edge, j could be externally specified.
  end

  if allSvNbond
    SvNvec = zeros(Ns)
  else
    SvNvec = zeros(0)
  end
  startoc = copy(psi.oc)

  return Lenv,Renv,SvN,lastSvN,maxtrunc,biggestm,Ns,curr_alpha,Nsteps,timer,j,SvNvec,startoc
end

@inline function Lexpand(A::TensType,ops::TensType,HL::TensType,alpha::Float64)
#  println("Lexpand:")
  #=@time =#Lenvpsi = contract(HL,(3,),A,(1,))
  #=@time =#Hpsi = contract((1,3,4,2),Lenvpsi,(2,3),ops,(1,2),alpha=alpha)
  #=@time =#expAA = reshape!(Hpsi,[[1],[2],[3,4]],merge=true)
  #=@time =#out = joinindex!(A,expAA,3)
#  println()
  return out
end

@inline function Rexpand(A::TensType,ops::TensType,HR::TensType,alpha::Float64)
#  println("Rexpand:")
  #=@time =#Renvpsi = contract(A,(3,),HR,(1,))
  #=@time =#Hpsi = contract((3,1,2,4),ops,(2,4),Renvpsi,(2,3),alpha=alpha)
  #=@time =#expAA = reshape!(Hpsi,[[1,2],[3],[4]],merge=true)
  #=@time =#out = joinindex!(A,expAA,1)
#  println()
  return out
end

const alpha_max = 1.
@inline function alpha_update(alpha::Float64,truncerr::Float64,cutoff::Float64,
                      lastEnergy::R,currenergy::S,noise_goal::Float64,
                      dalpha::Float64;avgE::W=currenergy)::Number where {R <: Number, S <: Number, W <: Number}
  trunc_check = truncerr <= max(cutoff,1E-12)
  errEst = abs(currenergy-lastEnergy) <= max(abs(currenergy*cutoff),1E-12)
  sitecond = abs((truncerr*avgE)/(lastEnergy/currenergy-1)) > noise_goal #recommendation in 3S paper
  if currenergy <= lastEnergy && (sitecond || errEst || trunc_check)
    alpha *= 1. - dalpha
  else
    alpha *= 1. + dalpha
  end
  return max(min(alpha,alpha_max),1E-42)
end
#=
function singlesiteOps(mpo::MPO,nsites::Integer)
  return mpo
end
=#
#=
function twositeOps(mpo::MPO,nsites::Integer)
  nbonds = length(mpo)-(nsites-1)
  ops = Array{typeof(mpo[1]),1}(undef,nbonds)
  #=Threads.@threads=# for i = 1:nbonds
    ops[i] = contract([1,3,5,6,2,4],mpo[i],ndims(mpo[i]),mpo[i+1],1)
    if typeof(ops[i]) <: qarray
      ops[i] = changeblock(ops[i],[1,2,3,4],[5,6])
    end
  end
  W = typeof(mpo)
  if W <: largeMPO
    newmpo = largeMPO(ops,label="twompo_")
  else
    newmpo = W(ops)
  end
  return newmpo
end
=#
#=
function NsiteOps(mpo::MPO,nsites::Integer)
  nbonds = length(mpo)-(nsites-1)
  ops = Array{typeof(mpo[1]),1}(undef,nbonds)
  for i = 1:nbonds

    ops[i] = contract(mpo[i],4,mpo[i+1],1)
    for a = 2:nsites-1
      ops[i] = contract(ops[i],ndims(ops[i]),mpo[i+a],1)
    end
    shufflevec = vcat([w for w = 1:2:ndims(ops[i])-1],[ndims(ops[i])],[w for w = 2:2:ndims(ops[i])-1])
    ops[i] = permutedims(ops[i],shufflevec)
    if typeof(ops[i]) <: qarray
      ops[i] = changeblock(ops[i],[w for w = 1:ndims(ops[i])-nsites],[w+(ndims(ops[i])-nsites) for w = 1:nsites])
    end
  end
  W = typeof(mpo)
  if W <: largeMPO
    newmpo = largeMPO(ops,label="$(nsites)mpo_")
  else
    newmpo = W(ops)
  end
  return newmpo
end
=#
@inline function Nsite_update(Lenv::TensType,Renv::TensType,psiops::TensType...)
  AA = psiops[1]
  ops = psiops[2]
  nsites = ndims(AA)-2
  
  AAinds = ntuple(i->i+1,nsites)
  opinds = ntuple(i->i+nsites+2,nsites)
  Hpsi = contract(ops,opinds,AA,AAinds)

  Hpsi_inds = (1,3+nsites)
  LHpsi = contract(Lenv,(2,3),Hpsi,Hpsi_inds)

  LHpsi_inds = (3+nsites,2+nsites)
  temp = contract(LHpsi,LHpsi_inds,Renv,(1,2))
  return temp
end
export Nsite_update

@inline function step3S(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number
#println(n," ",j," ",i," ",iL," ",iR)
  currops = mpo[i]
  #=@time =#AAvec,outEnergy = lanczos(psi[i],currops,maxiter=params.maxiter,updatefct=singlesite_update,Lenv=Lenv[i],Renv=Renv[i])
  noise = params.noise

  params.energy = outEnergy[1]
  alpha_condition = noise[1] > params.noise_incr

  minm = params.minm
  maxm = params.maxm
  cutoff = params.cutoff

  if j > 0
    #=@time =#tempL = (alpha_condition ? Lexpand(AAvec[1],currops,Lenv[iL],noise) : AAvec[1])
    #=@time =#psi[iL],psi[iR],D,truncerr = moveR(tempL,psi[iR],cutoff=cutoff,m=maxm,minm=minm,condition=alpha_condition)
  else
    #=@time =#tempR = (alpha_condition ? Rexpand(AAvec[1],currops,Renv[iR],noise) : AAvec[1])
    #=@time =#psi[iL],psi[iR],D,truncerr = moveL(psi[iL],tempR,cutoff=cutoff,m=maxm,minm=minm,condition=alpha_condition)
  end
  
  #=@time =#params.noise = alpha_update(noise,truncerr,params.cutoff,params.lastenergy,params.energy,params.noise_goal,params.noise_incr)
  params.truncerr = truncerr
  params.biggestm = max(params.biggestm,size(D,1))
#println()
  if !params.efficient
    SvNcheck!(i,j,D,length(psi),params)
  end
  nothing
end

@inline function twostep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                  psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number

#  println(n," ",j," ",i," ",iL," ",iR)

  if j > 0
    psi[iL] = div!(psi[iL],norm(psi[iL]))
  else
    psi[iR] = div!(psi[iR],norm(psi[iR]))
  end

  AA,energy = simplelanczos(Lenv[iL],Renv[iR],psi[iL],psi[iR],mpo[iL],mpo[iR])

  params.energy = energy
#  println("E = ",energy)


  checkU,checkD,checkV,truncerr = svd(makeArray(AA),[[1,2],[3,4]],m=params.maxm,minm=params.minm,cutoff=params.cutoff,mag=1.)

  U,D,V,truncerr = svd!(AA,[[1,2],[3,4]],m=params.maxm,minm=params.minm,cutoff=params.cutoff,mag=1.)
#=
  println()
  println("ultra check:")
  println(size(U)," ",size(D)," ",size(V))
  println(size(checkU)," ",size(checkD)," ",size(checkV))
  println(norm(makeArray(U)-checkU))
  println(norm(makeArray(D)-checkD))
  println(norm(makeArray(V)-checkV))
  println()

  println("eta = ",truncerr)
  println(sort([D[i,i] for i = 1:size(D,1)]))
  =#
  if j < 0
    psi[iL] = contract(U,(3,),D,(1,))
    psi[iR] = V
  else
    psi[iL] = U
    psi[iR] = contract(D,(2,),V,(1,))
  end
  params.truncerr = truncerr
  params.biggestm = max(params.biggestm,size(D,1))

  if !params.efficient
    SvNcheck!(i,j,D,length(psi),params,alpha=false)
  end
#  println()
  nothing
end

#=
function swapflux!(AA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  AA.flux,AA.QnumSum[end][1] = inv(AA.QnumSum[end][1]),inv(AA.flux)
  newQNs = Array{NTuple{2,Q},1}(undef,length(AA.T))
  for q = 1:length(AA.T)
    thisvec = [Q() for w = 1:2]
    for w = 1:2
      for a = 1:size(AA.ind[q][1],1)
        x = AA.currblock[w][a]
        y = AA.ind[q][w][a,1]+1
        thisvec[w] += getQnum(x,y,AA)
      end
    end
    newQNs[q] = (thisvec...,)
  end
  AA.Qblocksum = newQNs
  nothing
end

function swapflux!(AA::densTensType)
  nothing
end
=#

@inline function step2S(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number

#  println(n," ",i," ",j," ",iL," ",iR)
#  println(params.energy)

  if j > 0
    psi[iL] = div!(psi[iL],norm(psi[iL]))

#    println(norm(psi[iL]))

  else
    psi[iR] = div!(psi[iR],norm(psi[iR]))

#    println(norm(psi[iR]))

  end

  AA,energy = simplelanczos(Lenv[iL],Renv[iR],psi[iL],psi[iR],mpo[iL],mpo[iR])

  params.energy = energy

  noise = params.noise
  minm = params.minm
  maxm = params.maxm
  cutoff = params.cutoff

#  println(norm(AA))

  if j > 0
    rho = contractc(AA,(3,4),AA,(3,4))
  else
    rho = ccontract(AA,(1,2),AA,(1,2))
  end

#  println(norm(rho))

  if params.noise > 0.
    if j > 0
      temp = contract(Lenv[iL],(3,),AA,(1,))
      randrho = contractc(temp,(2,4,5),temp,(2,4,5))
    else
      temp = contract(AA,(4,),Renv[iR],(1,))
      randrho = ccontract(temp,(1,2,4),temp,(1,2,4))
    end
    rho = add!(rho,randrho,params.noise)
  end

#  println(norm(rho))
  
#  save_rho = deepcopy(rho)
  
#  println("symmetry version:")
  D,U,truncerr,sumD = eigen(rho,[[1,2],[3,4]],cutoff=params.cutoff,m=params.maxm,minm=params.minm,transpose = j < 0)

#  println(norm(D)," ",norm(U))

#  checkflux(D)

#  println(U)

#  checkflux(U)

  if j > 0
    psi[iL] = U
    psi[iR] = ccontract(U,(1,2),AA,(1,2))
  else

    psi[iL] = contractc(AA,(3,4),U,(2,3))
    psi[iR] = U
  end

#  println(norm(psi[iL])," ",norm(psi[iR]))

#  checkflux(psi[iL])
#  checkflux(psi[iR])
#  println()

  params.noise = alpha_update(noise,truncerr,params.cutoff,params.lastenergy,params.energy,params.noise_goal,params.noise_incr)

  params.truncerr = truncerr
  params.biggestm = max(params.biggestm,size(D,1))

  if !params.efficient
    SvNcheck!(i,j,D,length(psi),params,rev=true)
  end
#  println()
  nothing
end


#  import ..optimizeMPS.Nstep
@inline function dmrgNstep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                    psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number

  vecTens = [psi[a] for a = iL:iR]
  AA = vecTens[1]
  for a = 2:length(vecTens)
    AA = contract(AA,ndims(AA),vecTens[a],1)
  end

  newAA,outEnergy = lanczos(AA,mpo[iL],maxiter=params.maxiter,updatefct=Nsite_update,Lenv=Lenv[iL],Renv=Renv[iR])
  params.energy = outEnergy[1]

  AA = newAA[1]

  maxm = params.maxm
  minm = params.minm
  cutoff = params.cutoff


  truncerr = 0.
  D = 0
  if j > 0
    for w = iR:-1:iL+1
      U,D,psi[w],terr = svd(AA,[[i for i = 1:ndims(AA)-2],[ndims(AA)-1,ndims(AA)]],m=maxm,minm=minm,cutoff=cutoff)#,mag=1.)
      truncerr = max(truncerr,terr)
      params.biggestm = max(params.biggestm,size(D,1))
      if w == iL+1
        psi[iL] = U
        psi[iL+1] = contract(D,2,psi[w],1)
        psi.oc = iL+1
      else
        AA = contract(U,ndims(U),D,1)
      end
    end
  else
    for w = iL:iR-1
      psi[w],D,V,terr = svd(AA,[[1,2],[i for i = 3:ndims(AA)]],m=maxm,minm=minm,cutoff=cutoff)#,mag=1.)
      truncerr = max(truncerr,terr)
      params.biggestm = max(params.biggestm,size(D,1))
      if w == iR-1
        psi[iR-1] = contract(psi[w],3,D,1)
        psi[iR] = V
        psi.oc = iR-1
      else
        AA = contract(D,2,V,1)
      end
    end
  end
  params.truncerr = truncerr
  if !params.efficient
    SvNcheck!(i,j,D,length(psi),params)
  end
  nothing
end

function dmrg(psi::MPS,mpo::MPO;params::TNparams = algvars(psi),
                                m::Integer=params.maxm,minm::Integer=params.minm,
                                sweeps::Integer=params.sweeps,cutoff::Float64=params.cutoff,
                                silent::Bool=params.silent,goal::Float64=params.goal,
                                SvNbond::Integer=params.SvNbond,allSvNbond::Bool=params.allSvNbond,
                                nsites::Integer=params.nsites,efficient::Bool=params.efficient,
                                cvgE::Bool=params.cvgE,maxiter::Integer=params.maxiter,
                                mincr::Integer=params.mincr,fixD::Bool=params.fixD,Lbound::TensType=params.Lbound,Rbound::TensType=params.Rbound,
                                noise::P=params.noise,noise_goal::Float64=params.noise_goal,noise_incr::Float64=params.noise_incr,noise_decay::Float64=params.noise_decay,method::String="3S",shift::Bool=params.shift,
                                saveEnergy::AbstractArray=params.saveEnergy,halfsweep::Bool=params.halfsweep,Lenv::Env=params.Lenv,Renv::Env=params.Renv,origj::Bool=params.origj,maxshowD::Integer=params.maxshowD,
                                storeD::Array{W,1}=params.storeD,exnum::Integer=params.exnum) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  if params.load
    params.method = method
  end
  if params.method == "3S"
    return dmrg3S(psi,mpo,method=method,m=m,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,exnum=exnum,fixD=fixD,nsites=1,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
  elseif params.method == "Nsite" #|| nsites > 2
    return dmrg_Nsite(psi,mpo,method=method,m=m,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,exnum=exnum,fixD=fixD,nsites=nsites,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
  elseif params.method == "twosite"
    return dmrg_twosite(psi,mpo,method=method,m=m,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,exnum=exnum,fixD=fixD,nsites=2#=nsites=#,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
#    elseif method == "zero" || method == "0"
  elseif params.method == "2S"
    return dmrg2S(psi,mpo,method=method,m=m,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,exnum=exnum,fixD=fixD,nsites=2,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
  else
    error("DMRG method not defined")
  end
end
export dmrg

function dmrginformation(params::TNparams)
  if !params.silent
    println("Optimizing matrix product state...")
    println("  algorithm = ",params.method)
    println("  size of renormalized system = ",params.nsites)
    println("  minimum bond dimension = ",params.minm)
    println("  maximum bond dimension = ",params.maxm)
    println("  number of sweeps = ",params.sweeps)
    println("  Lanczos iterations = ",params.maxiter)
    println("  cutoff = ",params.cutoff)
    println("  converge in energy? ",params.cvgE," (otherwise, entropy)")
    println("  specified goal = ",params.goal)
    println("  initial noise parameter = ",params.noise)
    println("  noise increment = ",params.noise_incr)
    println("  fixing SvN values? ",params.fixD)
  end
  nothing
end

function dmrgcvg(n::Integer,timer::Number,
                 dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,psiLenv::Env,
                 psiRenv::Env,beta::Array{W,1},prevpsi::MPS...;params::TNparams=params())::Bool where W <: Number

  if !params.silent
    println("Sweep $n (back and forth): $(Printf.@sprintf("%.5f",timer)) sec")
    println("Largest truncation = $(params.maxtrunc), m = $(params.biggestm)")
    println("Energy at sweep $n is $(Printf.@sprintf("%.12f",params.energy))")
    println()
    flush(Base.stdout)
  end
  params.savebiggestm = max(params.savebiggestm,params.biggestm)
  params.biggestm = 0
  params.truncerr = params.maxtrunc
  params.maxtrunc = 0.

  if params.cvgE
    checkE = abs(params.energy[1] - params.lastenergy[1])
    if checkE < params.goal
      if !params.silent
        println("The energy for DMRG converged to a difference of $checkE after $n sweeps!")
        println("(...or it's stuck in a metastable state...)")
        println()
      end
      return true
    else
      params.lastenergy = copy(params.energy)
    end
  else
    checkSvN = abs(params.lastentropy - params.entropy)
    if checkSvN < params.goal
      if !params.silent
        println("The energy for DMRG converged entropy converged to a difference of $checkSvN after $n sweeps!")
        println("(...or it's stuck in a metastable state...)")
        println()
      end
      return true
    else
      params.lastentropy = copy(params.entropy)
    end
  end
  return false
end

#  import ..optimizeMPS.loadvars!
function dmrg3S(psi::MPS,mpo::MPO;params::TNparams = algvars(psi),
                                  m::Integer=params.maxm,minm::Integer=params.minm,
                                  sweeps::Integer=params.sweeps,cutoff::Float64=params.cutoff,
                                  silent::Bool=params.silent,goal::Float64=params.goal,
                                  SvNbond::Integer=params.SvNbond,allSvNbond::Bool=params.allSvNbond,
                                  nsites::Integer=params.nsites,efficient::Bool=params.efficient,
                                  cvgE::Bool=params.cvgE,maxiter::Integer=params.maxiter,
                                  mincr::Integer=params.mincr,fixD::Bool=params.fixD,Lbound::TensType=params.Lbound,Rbound::TensType=params.Rbound,
                                  noise::P=params.noise,noise_goal::Float64=params.noise_goal,noise_incr::Float64=params.noise_incr,noise_decay::Float64=params.noise_decay,method::String="3S",shift::Bool=params.shift,
                                  saveEnergy::AbstractArray=params.saveEnergy,halfsweep::Bool=params.halfsweep,Lenv::Env=params.Lenv,Renv::Env=params.Renv,origj::Bool=params.origj,maxshowD::Integer=params.maxshowD,
                                  storeD::Array{W,1}=params.storeD,exnum::Integer=params.exnum) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  loadvars!(params,"DMRG-"*method,minm,m,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,maxiter,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,stepfct=step3S,#=makeOps=singlesiteOps,=#cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg3S

#  import ..optimizeMPS.twositeOps
function dmrg_twosite(psi::MPS,mpo::MPO;params::TNparams = algvars(psi),
                                        m::Integer=params.maxm,minm::Integer=params.minm,
                                        sweeps::Integer=params.sweeps,cutoff::Float64=params.cutoff,
                                        silent::Bool=params.silent,goal::Float64=params.goal,
                                        SvNbond::Integer=params.SvNbond,allSvNbond::Bool=params.allSvNbond,
                                        nsites::Integer=2,#=params.nsites,=#efficient::Bool=params.efficient,
                                        cvgE::Bool=params.cvgE,maxiter::Integer=params.maxiter,
                                        mincr::Integer=params.mincr,fixD::Bool=params.fixD,Lbound::TensType=params.Lbound,Rbound::TensType=params.Rbound,
                                        noise::P=params.noise,noise_goal::Float64=params.noise_goal,noise_incr::Float64=params.noise_incr,noise_decay::Float64=params.noise_decay,method::String="twosite",shift::Bool=params.shift,
                                        saveEnergy::AbstractArray=params.saveEnergy,halfsweep::Bool=params.halfsweep,Lenv::Env=params.Lenv,Renv::Env=params.Renv,origj::Bool=params.origj,maxshowD::Integer=params.maxshowD,
                                        storeD::Array{W,1}=params.storeD,exnum::Integer=params.exnum) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  loadvars!(params,"DMRG-"*method,minm,m,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,maxiter,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,measfct=expect,stepfct=twostep,#=makeOps=twositeOps,=#cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg_twosite

#  import ..optimizeMPS.NsiteOps
function dmrg_Nsite(psi::MPS,mpo::MPO;m::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                  SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::Integer=2,efficient::Bool=false,params::TNparams = algvars(),
                  cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,fixD::Bool=false,Lbound::TensType=[0],Rbound::TensType=[0],
                  noise::P=1.0,noise_goal::Float64=0.3,noise_incr::Float64=params.noise_incr,noise_decay::Float64=params.noise_decay,method::String="Nsite",shift::Bool=params.shift,
                  saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::Env=params.Lenv,Renv::Env=params.Renv,origj::Bool=true,maxshowD::Integer=params.maxshowD,
                  storeD::Array{W,1}=params.storeD,alpha_decay::Float64=0.9,exnum::Integer=params.exnum) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  loadvars!(params,"DMRG-"*method*" (N=$nsites)",minm,m,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,maxiter,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,measfct=expect,stepfct=dmrgNstep,#=makeOps=NsiteOps,=#cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg_Nsite

function dmrg2S(psi::MPS,mpo::MPO;params::TNparams = algvars(psi),
                  m::Integer=params.maxm,minm::Integer=params.minm,
                  sweeps::Integer=params.sweeps,cutoff::Float64=params.cutoff,
                  silent::Bool=params.silent,goal::Float64=params.goal,
                  SvNbond::Integer=params.SvNbond,allSvNbond::Bool=params.allSvNbond,
                  nsites::Integer=params.nsites,efficient::Bool=params.efficient,
                  cvgE::Bool=params.cvgE,maxiter::Integer=params.maxiter,
                  mincr::Integer=params.mincr,fixD::Bool=params.fixD,Lbound::TensType=params.Lbound,Rbound::TensType=params.Rbound,
                  noise::P=params.noise,noise_goal::Float64=params.noise_goal,noise_incr::Float64=params.noise_incr,noise_decay::Float64=params.noise_decay,method::String="2S",shift::Bool=params.shift,
                  saveEnergy::AbstractArray=params.saveEnergy,halfsweep::Bool=params.halfsweep,Lenv::Env=params.Lenv,Renv::Env=params.Renv,origj::Bool=params.origj,maxshowD::Integer=params.maxshowD,
                  storeD::Array{W,1}=params.storeD,exnum::Integer=params.exnum) where {P <: Union{Number,Array{Float64,1}}, W <: Number}

  #swapflux!(psi[end])                  
  loadvars!(params,"DMRG-"*method,minm,m,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,maxiter,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,stepfct=step2S,#=makeOps=NsiteOps,=#cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg2S
#end
