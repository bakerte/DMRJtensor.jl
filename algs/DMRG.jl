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
  
Function for the density matrix renormalization group
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
const onetwo = (1,2)
const twothree = (2,3)

const threefour = (3,4)
const fivesix = (5,6)

const onefive = (1,5)
const fivefour = (5,4)


const onefour = (1,4)
const twofive = (2,5)

const twofourfive = (2,4,5)
const onetwofour = (1,2,4)

function twosite_update(Lenv::TensType,Renv::TensType,tensors::TensType...)
  AA = tensors[1]
  ops = tensors[2]
  Hpsi = contract(ops,fivesix,AA,twothree)
  LHpsi = contract(Lenv,twothree,Hpsi,onefive)
  temp = contract(LHpsi,fivefour,Renv,onetwo)
  return temp
end

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

function SvNcheck!(i::Integer,j::Integer,D::TensType,Ns::Integer,params::TNparams;alpha::Bool=true)
  if params.maxtrunc < params.truncerr
    params.maxtrunc = params.truncerr
  end
  if params.biggestm < size(D,1)
    params.biggestm = size(D,1)
  end
  Dmat = [searchindex(D,i,i) for i = 1:min(size(D)...)]
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
      println("Singular values at center bond b=",params.SvNbond,": [",string(Dmat,1,ymax),"]")
      println()
    end
  end
  nothing
end

function regular_SvNcheck!(i::Integer,j::Integer,D::TensType,Ns::Integer,params::TNparams)
  return SvNcheck!(i,j,D,Ns,params,alpha=false)
end

function setDMRG(psi::MPS,mpo::MPO,maxm::Integer,minm::Integer,Lenv::TensType,Renv::TensType,
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

function Lexpand(A::TensType,ops::TensType,HL::TensType,alpha::Float64)#;maxm::Integer=500)
  #=@time=# Hpsi = contract(ops,(2,),A,(2,))
  #=@time=# Hpsi = contract(HL,twothree,Hpsi,onefour,alpha=alpha)
  #=@time=# expAA = reshape!(Hpsi,[[1],[2],[3,4]],merge=true)
#=  if size(expAA,3) > 2*maxm
#    checkflux(expAA)
    expAA = expAA[:,:,1:2*maxm]
#    checkflux(expAA)
  end=#
  #=@time=# out = joinindex!(A,expAA,3)
#  println()
  return out
end

function Rexpand(A::TensType,ops::TensType,HR::TensType,alpha::Float64)#;maxm::Integer=500)
  #=@time=# Hpsi = contract(A,(2,),ops,(2,))
  #=@time=# Hpsi = contract(Hpsi,twofive,HR,onetwo,alpha=alpha)
  #=@time=# expAA = reshape!(Hpsi,[[1,2],[3],[4]],merge=true)
#=  if size(expAA,1) > 2*maxm
#    checkflux(expAA)
    expAA = expAA[1:2*maxm,:,:]
#    checkflux(expAA)
  end=#
  #=@time=# out = joinindex!(A,expAA,1)
#  println()
  return out
end

const alpha_max = 1.
function alpha_update(alpha::Float64,truncerr::Float64,cutoff::Float64,
                      lastEnergy::Union{R,Array{R,1}},infovals::Union{S,Array{S,1}},
                      noise_goal::Float64,dalpha::Float64;avgE::W=infovals[1])::Number where {R <: Number, S <: Number, W <: Number}
  trunc_check = truncerr <= max(cutoff,1E-12)
  errEst = abs(infovals[1]-lastEnergy[1]) <= max(abs(infovals[1]*cutoff),1E-12)
  sitecond = abs((truncerr*avgE)/(lastEnergy[1]/infovals[1]-1)) > noise_goal[1] #recommendation in 3S paper
  if infovals[1] <= lastEnergy[1] && (sitecond || errEst || trunc_check)
    alpha *= 1. - dalpha
  else
    alpha *= 1. + dalpha
  end
  return max(min(alpha,alpha_max),1E-42)
end

function singlesiteOps(mpo::MPO,params::TNparams)
  return mpo
end

function twositeOps(mpo::MPO,params::TNparams)
  nsites = params.nsites
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

function Nsite_update(Lenv::TensType,Renv::TensType,AA::TensType,ops::TensType)
  AAinds = ntuple(i->i-1,ndims(AA)-2)
  nelem = ndims(ops)-(ndims(ops)-length(AAinds)+1)
  opinds = ntuple(i->(ndims(ops)-length(AAinds)+1)+i,nelem)
  Hpsi = contract(ops,opinds,AA,AAinds)
  Hpsi_inds = ([1,1+ndims(AA)],)
  LHpsi = contract(Lenv,twothree,Hpsi,Hpsi_inds)
  LHpsi_inds = ([ndims(LHpsi),ndims(LHpsi)-1],)
  temp = contract(LHpsi,LHpsi_inds,Renv,onetwo)
  return temp
end

function step3S(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number
  currops = mpo[i]
#  println(n," ",i," ",j," ",iL," ",iR)
  #=@time=# psi[i] = div!(psi[i],norm(psi[i]))
  #=@time=# AAvec,outEnergy = krylov(psi[i],currops,maxiter=params.maxiter,updatefct=singlesite_update,Lenv=Lenv[i],Renv=Renv[i])
  noise = params.noise

  params.energy = outEnergy[1]
  alpha_condition = noise[1] > params.noise_incr

  minm = params.minm
  maxm = params.maxm
  cutoff = params.cutoff

#  println()
#  println("expansion function:")

  if j > 0
    tempL = (alpha_condition ? Lexpand(AAvec[1],currops,Lenv[iL],noise) : AAvec[1])
    psi[iL],psi[iR],D,truncerr = moveR(tempL,psi[iR],cutoff=cutoff,m=maxm,minm=minm,condition=alpha_condition)
  else
    tempR = (alpha_condition ? Rexpand(AAvec[1],currops,Renv[iR],noise) : AAvec[1])
    psi[iL],psi[iR],D,truncerr = moveL(psi[iL],tempR,cutoff=cutoff,m=maxm,minm=minm,condition=alpha_condition)
  end
#  println()

  params.noise = alpha_update(noise,truncerr,params.cutoff,params.lastenergy,params.energy,params.noise_goal,params.noise_incr)
  params.truncerr = truncerr
  params.biggestm = max(params.biggestm,size(D,1))

  if !params.efficient
    SvNcheck!(i,j,D,length(psi),params)
  end
  nothing
end

function twostep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                  psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number
#println(n," ",j," ",i," ",iL," ",iR)
  #=@time=# AA = contract(psi[iL],(3,),psi[iR],(1,))

  #=@time=# AA = div!(AA,norm(AA))
  #=@time=# newAA,outEnergy = krylov(AA,mpo[iL],maxiter=params.maxiter,updatefct=twosite_update,Lenv=Lenv[iL],Renv=Renv[iR])
  params.energy = outEnergy[1]

  #=@time=# U,D,V,truncerr = svd(newAA[1],[[1,2],[3,4]],m=params.maxm,minm=params.minm,cutoff=params.cutoff,mag=1.)
  
  if j < 0
    #=@time=# psi[iL] = contract(U,3,D,1)
    psi[iR] = V
  else
    psi[iL] = U
    #=@time=# psi[iR] = contract(D,2,V,1)
  end

#  println()

  params.truncerr = truncerr
  params.biggestm = max(params.biggestm,size(D,1))

  if !params.efficient
    regular_SvNcheck!(i,j,D,length(psi),params)
  end
  nothing
end


function step2S(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number


  AA = contract(psi[iL],ndims(psi[iL]),psi[iR],(1,))
  AAvec,outEnergy = krylov(AA,mpo[iL],maxiter=params.maxiter,updatefct=twosite_update,Lenv=Lenv[iL],Renv=Renv[iR])
  params.energy = outEnergy[1]

  m = params.maxm
  minm = params.minm
  cutoff = params.cutoff

  AA = AAvec[1]

#      sizeAA = size(AA)

    if j > 0
      rho = contractc(AA,threefour,AA,threefour)
    else
      rho = ccontract([3,4,1,2],AA,onetwo,AA,onetwo)
#        checkflux(AA)
      rho = conj(deepcopy(rho))
#        checkflux(AA)
    end

    if false #params.noise > 0.
      if j > 0
        temp = contract(Lenv[iL],(3,),AA,(1,))
        randrho = contractc(temp,twofourfive,temp,twofourfive)
      else
        temp = contract(AA,(4,),Renv[iR],(1,))
#          println(temp)
        randrho = ccontract(temp,onetwofour,temp,onetwofour)
#          println(size(randrho))
      end
      rho = add!(rho,randrho,params.noise)
    end

    D,U,truncerr,sumD = eigen(rho,[[1,2],[3,4]],cutoff=cutoff,m=m,minm=minm)
#=      if j < 0 && typeof(rho) <: Qtens
      U.QnumMat[3] = inv!.(U.QnumMat[3])
      U.flux = inv(U.flux)
      U = permutedims!(U,[3,1,2])
      checkflux(U)
    end=#
    if params.noise > 0.
      D = div!(D,sumD)
    end
#      println(size(AA)," ",size(U))
#      println(U)
    if j > 0
      psi[iL],psi[iR] = U,ccontract(U,onetwo,AA,onetwo)
    else
      println(U)
      println(AA)
      cU = conj(deepcopy(U))
      psi[iL] = contract(AA,threefour,cU,onetwo)
      newV = permutedims(U,[3,1,2])
      psi[iR] = conj!(deepcopy(newV))
    end
#      checkflux(psi[iL])
#      checkflux(psi[iR])
#      psi[iL] = unreshape!(newU,sizeAA[1],sizeAA[2],size(D,1))
#      psi[iR] = unreshape!(newV,size(D,1),sizeAA[3],sizeAA[4])

  if !params.efficient
    regular_SvNcheck!(i,j,D,length(psi),params)
  end
  nothing
end




#  import ..optimizeMPS.Nstep
function dmrgNstep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                    psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number

  vecTens = [psi[a] for a = iL:iR]
  AA = vecTens[1]
  for a = 2:length(vecTens)
    AA = contract(AA,ndims(AA),vecTens[a],1)
  end

  newAA,outEnergy = krylov(AA,mpo[iL],maxiter=params.maxiter,updatefct=Nsite_update,Lenv=Lenv[iL],Renv=Renv[iR])
  params.energy = outEnergy[1]

  AA = newAA[1]

  if j > 0
    for w = iR:-1:iL+1
      U,D,psi[w],truncerr = svd(AA,[[i for i = 1:ndims(AA)-2],[ndims(AA)-1,ndims(AA)]],m=maxm,minm=minm,cutoff=cutoff)#,mag=1.)
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
      psi[w],D,V,truncerr = svd(AA,[[1,2],[i for i = 3:ndims(AA)]],m=maxm,minm=minm,cutoff=cutoff)#,mag=1.)
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
  params.biggestm = max(params.biggestm,size(D,1))

  if !params.efficient
    regular_SvNcheck!(i,j,D,length(psi),params)
  end
  nothing
end

function dmrg(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                  SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::Integer=2,efficient::Bool=sweeps == 1,params::TNparams = algvars(),
                  cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,exnum::Integer=1,fixD::Bool=false,Lbound::TensType=[0],Rbound::TensType=[0],
                  noise::P=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Number=0.99,method::String="3S",shift::Bool=false,
                  saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::Env=[0],Renv::Env=[0],origj::Bool=true,maxshowD::Integer=8,storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,
                  partitions::Integer=Threads.nthreads()) where P <: Union{Number,Array{Float64,1}} where {B <: TensType, W <: Number}
  if params.load
    params.method = method
  end
  if params.method == "3S"
    return dmrg3S(psi,mpo,method=method,maxm=maxm,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,exnum=exnum,fixD=fixD,nsites=1,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
  elseif params.method == "Nsite" #|| nsites > 2
    return dmrg_Nsite(psi,mpo,method=method,maxm=maxm,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,exnum=exnum,fixD=fixD,nsites=nsites,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
  elseif params.method == "twosite"
    return dmrg_twosite(psi,mpo,method=method,maxm=maxm,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,exnum=exnum,fixD=fixD,nsites=nsites,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
#    elseif method == "zero" || method == "0"
  elseif params.method == "2S"
    return dmrg2S(psi,mpo,method=method,maxm=maxm,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
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
    println("  Krylov iterations = ",params.maxiter)
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
function dmrg3S(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                  SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::Integer=2,efficient::Bool=false,params::TNparams = algvars(),
                  cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,exnum::Integer=1,fixD::Bool=false,Lbound::TensType=[0],Rbound::TensType=[0],
                  noise::P=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Number=0.9,method::String="3S",shift::Bool=false,
                  saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::Env=[0],Renv::Env=[0],origj::Bool=true,maxshowD::Integer=8,storeD::Array{W,1}=[0.],
                  alpha_decay::Float64=0.9) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  loadvars!(params,"DMRG-"*method,minm,maxm,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,maxiter,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,stepfct=step3S,makeOps=singlesiteOps,cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg3S

#  import ..optimizeMPS.twositeOps
function dmrg_twosite(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                  SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::Integer=2,efficient::Bool=false,params::TNparams = algvars(),
                  cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,fixD::Bool=false,Lbound::TensType=[0],Rbound::TensType=[0],
                  noise::P=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Float64=0.9,method::String="3S",shift::Bool=false,
                  saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::Env=[0],Renv::Env=[0],origj::Bool=true,maxshowD::Integer=8,
                  storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,exnum::Integer=1) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  loadvars!(params,"DMRG-"*method,minm,maxm,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,maxiter,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,measfct=expect,stepfct=twostep,makeOps=twositeOps,cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg_twosite

#  import ..optimizeMPS.NsiteOps
function dmrg_Nsite(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                  SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::Integer=2,efficient::Bool=false,params::TNparams = algvars(),
                  cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,fixD::Bool=false,Lbound::TensType=[0],Rbound::TensType=[0],
                  noise::P=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Float64=0.9,method::String="Nsite",shift::Bool=false,
                  saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::Env=[0],Renv::Env=[0],origj::Bool=true,maxshowD::Integer=8,
                  storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,exnum::Integer=1) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  loadvars!(params,"DMRG-"*method*" (N=$nsites)",minm,maxm,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,maxiter,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,measfct=expect,stepfct=dmrgNstep,makeOps=NsiteOps,cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg_Nsite

function dmrg2S(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                  SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::Integer=2,efficient::Bool=false,params::TNparams = algvars(),
                  cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,fixD::Bool=false,Lbound::TensType=[0],Rbound::TensType=[0],
                  noise::P=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Float64=0.9,method::String="2S",shift::Bool=false,
                  saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::Env=[0],Renv::Env=[0],origj::Bool=true,maxshowD::Integer=8,
                  storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,exnum::Integer=1) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  loadvars!(params,"DMRG-"*method,minm,maxm,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,maxiter,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,stepfct=step2S,makeOps=NsiteOps,cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg2S
#end
