#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.9
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1+)
#
# Planned improvements:
#   v1.0: infinite, temperature, time (Lanczos, TDVP)
#

"""
    Module: DMRG
    
Function for the density matrix renormalization group
"""
#=
module DMRG
#using ..shuffle
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

const one = (1...,)
const two = (2...,)
const three = (3...,)
const four = (4...,)

const twofourfive = (2,4,5)
const onetwofour = (1,2,4)

  function twosite_update(Lenv::Z,Renv::Z,tensors::TensType...) where {X <: Union{Qtens{A,Q},Array{A,4},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,6},tens{B}},Z <: Union{Qtens{W,Q},Array{W,3},tens{W}}, G <: TensType} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
    AA = tensors[1]
    ops = tensors[2]
    Hpsi = contract(ops,fivesix,AA,twothree)
    LHpsi = contract(Lenv,twothree,Hpsi,onefive)
    temp = contract(LHpsi,fivefour,Renv,onetwo)
    currpsi = AA
    return currpsi,temp
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

  function SvNcheck!(i::Integer,j::Integer,D::W,Ns::Integer,
                     params::TNparams;alpha::Bool=true) where {W<: TensType}
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

  function regular_SvNcheck!(i::Integer,j::Integer,D::W,Ns::Integer,params::TNparams) where {W<: TensType}
    return SvNcheck!(i,j,D,Ns,params,alpha=false)
  end

  function setDMRG(psi::MPS,mpo::MPO,maxm::Integer,minm::Integer,Lenv::TensType,Renv::TensType,
                    halfsweep::Bool,alpha::Z,origj::Bool,allSvNbond::Bool,boundary::B...) where {Z <: Union{Float64,Array{Float64,1}},B <: TensType}
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
  
  function Lexpand(A::X,ops::Y,HL::Z,alpha::Float64) where {X <: Union{Qtens{M,Q},Array{M,3},tens{M}}, Y <: Union{Qtens{B,Q},Array{B,4},tens{B}},Z <: Union{Qtens{W,Q},Array{W,3},tens{W}}} where {M <: Number, B <: Number, W <: Number, Q <: Qnum}
    Hpsi = contract(ops,two,A,two) #::Union{Qtens{W,Q},Array{W,5}} where Q <: Qnum
    Hpsi = contract(HL,twothree,Hpsi,onefour,alpha=alpha) #::Union{Qtens{W,Q},Array{W,4}} where Q <: Qnum
    expAA = reshape!(Hpsi,[[1],[2],[3,4]],merge=true) #::Union{Qtens{W,Q},Array{W,3}} where Q <: Qnum
    return joinindex!(A,expAA,3) #::Union{Qtens{W,Q},Array{W,3}} where Q <: Qnum
  end
  
  function Rexpand(A::X,ops::Y,HR::Z,alpha::Float64) where {X <: Union{Qtens{M,Q},Array{M,3},tens{M}}, Y <: Union{Qtens{B,Q},Array{B,4},tens{B}},Z <: Union{Qtens{W,Q},Array{W,3},tens{W}}} where {M <: Number, B <: Number, W <: Number, Q <: Qnum}
    Hpsi = contract(A,two,ops,two) #::Union{Qtens{W,Q},Array{W,5}} where Q <: Qnum
    Hpsi = contract(Hpsi,twofive,HR,onetwo,alpha=alpha) #::Union{Qtens{W,Q},Array{W,4}} where Q <: Qnum
    expAA = reshape!(Hpsi,[[1,2],[3],[4]],merge=true) #::Union{Qtens{W,Q},Array{W,3}} where Q <: Qnum
    return joinindex!(A,expAA,1) #::Union{Qtens{W,Q},Array{W,3}} where Q <: Qnum
  end
  
  const alpha_max = 1.
  function alpha_update(alpha::Float64,truncerr::Float64,cutoff::Float64,
                        lastEnergy::Union{R,Array{R,1}},infovals::Union{S,Array{S,1}},
                        noise_goal::Float64,dalpha::Float64;avgE::W=infovals[1])::Number where {R <: Number, S <: Number, W <: Number}
    trunc_check = truncerr <= max(cutoff,1E-12)
    errEst = abs(infovals[1]-lastEnergy[1]) <= max(abs(infovals[1]*cutoff),1E-12)
    sitecond = abs((truncerr*avgE)/(lastEnergy[1]/infovals[1]-1)) > noise_goal[1] #recommendation in 3S paper
    if sitecond || errEst || trunc_check
      alpha *= 1. - dalpha
    else
      alpha *= 1. + dalpha
    end
    return max(min(alpha,alpha_max),1E-42)
  end

  function singlesiteOps(mpo::MPO,params::TNparams)
    return mpo
  end

  function twositeOps(mpo::W,params::TNparams) where W <: MPO
    nsites = params.nsites
    nbonds = length(mpo)-(nsites-1)
    ops = Array{typeof(mpo[1]),1}(undef,nbonds)
    #=Threads.@threads=# for i = 1:nbonds
      ops[i] = contract([1,3,5,6,2,4],mpo[i],ndims(mpo[i]),mpo[i+1],1)
      if typeof(ops[i]) <: qarray
        ops[i] = changeblock(ops[i],[1,2,3,4],[5,6])
      end
    end
    if W <: largeMPO
      newmpo = largeMPO(ops,label="twompo_")
    else
      newmpo = W(ops)
    end
    return newmpo
  end

  function Nsite_update(Lenv::Z,Renv::Z,AA::X,ops::Y) where {X <: Union{Qtens{A,Q},Array{A,4},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,6},tens{B}},Z <: Union{Qtens{W,Q},Array{W,3},tens{W}}} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
    AAinds = ([i for i = 2:ndims(AA)-1]...,)
    opinds = ([i for i = (ndims(ops)-length(AAinds)+1):ndims(ops)]...,)
    Hpsi = contract(ops,opinds,AA,AAinds) #::Union{Qtens{W,Q},Array{W,6}} where Q <: Qnum
    Hpsi_inds = ([1,1+ndims(AA)]...,)
    LHpsi = contract(Lenv,twothree,Hpsi,Hpsi_inds) #::Union{Qtens{W,Q},Array{W,5}} where Q <: Qnum
    LHpsi_inds = ([ndims(LHpsi),ndims(LHpsi)-1]...,)
    temp = contract(LHpsi,LHpsi_inds,Renv,onetwo) #::Union{Qtens{W,Q},Array{W,4}} where Q <: Qnum
    return temp
  end

#  import ..optimizeMPS.singlesite_update
  function step3S(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::R,Renv::S,
                  psiLenv::Z,psiRenv::G,beta::Array{Y,1},prevpsi::MPS...;
                  params::TNparams=params()) where {Z <: Union{AbstractArray,envType}, G <: Union{AbstractArray,envType}, R <: Union{AbstractArray,envType}, S <: Union{AbstractArray,envType}, M <: Union{qarray,Array{J,4},tens}} where {X <: Number, W <: Number, J <: Number, Q <: Qnum, P <: Number, Y <: Number}

    currops = mpo[i]
    psi[i] = div!(psi[i],norm(psi[i]))
    AAvec,outEnergy = krylov(singlesite_update,Lenv[i],Renv[i],psi[i],currops,maxiter=params.maxiter)
    noise = params.noise

    params.energy = outEnergy[1]
    alpha_condition = noise[1] > params.noise_incr

    minm = params.minm
    maxm = params.maxm
    cutoff = params.cutoff

    if j > 0
      tempL = (alpha_condition ? Lexpand(AAvec[1],currops,Lenv[iL],noise) : AAvec[1])
      psi[iL],psi[iR],D,truncerr = moveR(tempL,psi[iR],cutoff=cutoff,m=maxm,minm=minm,condition=alpha_condition)
    else
      tempR = (alpha_condition ? Rexpand(AAvec[1],currops,Renv[iR],noise) : AAvec[1])
      psi[iL],psi[iR],D,truncerr = moveL(psi[iL],tempR,cutoff=cutoff,m=maxm,minm=minm,condition=alpha_condition)
    end

    params.noise = alpha_update(noise,truncerr,params.cutoff,params.lastenergy,params.energy,params.noise_goal,params.noise_incr)
    params.truncerr = truncerr
    params.biggestm = max(params.biggestm,size(D,1))

    if !params.efficient
      SvNcheck!(i,j,D,length(psi),params)
    end
    nothing
  end

  function twostep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::R,Renv::R,
                    psiLenv::Z,psiRenv::Z,beta::Array{Y,1},prevpsi::MPS...;
                    params::TNparams=params()) where {Z <: Union{AbstractArray,envType}, R <: Union{AbstractArray,envType}, M <: Union{qarray,Array{J,4},tens}} where {X <: Number, W <: Number, J <: Number, Q <: Qnum, P <: Number, Y <: Number}

    AA = contract(psi[iL],three,psi[iR],one)

    AA = div!(AA,norm(AA))
    newAA,outEnergy = krylov(twosite_update,Lenv[iL],Renv[iR],AA,mpo[iL],maxiter=params.maxiter)
    params.energy = outEnergy[1]

    U,D,V,truncerr = svd(newAA[1],[[1,2],[3,4]],m=params.maxm,minm=params.minm,cutoff=params.cutoff,mag=1.)

    if j < 0
      psi[iL] = contract(U,3,D,1)
      psi[iR] = V
    else
      psi[iL] = U
      psi[iR] = contract(D,2,V,1)
    end

    params.truncerr = truncerr
    params.biggestm = max(params.biggestm,size(D,1))

    if !params.efficient
      regular_SvNcheck!(i,j,D,length(psi),params)
    end
    nothing
  end


  function step2S(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::R,Renv::R,
                  psiLenv::Z,psiRenv::Z,beta::Array{Y,1},prevpsi::MPS...;
                  params::TNparams=params()) where {Z <: Union{AbstractArray,envType}, R <: Union{AbstractArray,envType}, M <: Union{qarray,Array{J,4},tens}} where {X <: Number, W <: Number, J <: Number, Q <: Qnum, P <: Number, Y <: Number}


    AA = contract(psi[iL],ndims(psi[iL]),psi[iR],one)
    AAvec,outEnergy = krylov(twosite_update,Lenv[iL],Renv[iR],AA,mpo[iL],maxiter=params.maxiter)
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
          temp = contract(Lenv[iL],three,AA,one)
          randrho = contractc(temp,twofourfive,temp,twofourfive)
        else
          temp = contract(AA,four,Renv[iR],one)
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
  function dmrgNstep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::R,Renv::R,
                      psiLenv::Z,psiRenv::Z,beta::Array{Y,1},prevpsi::MPS...;
                      params::TNparams=params()) where {Z <: Union{AbstractArray,envType}, R <: Union{AbstractArray,envType}, M <: Union{qarray,Array{J,4},tens}} where {X <: Number, W <: Number, J <: Number, Q <: Qnum, P <: Number, Y <: Number}

    vecTens = [psi[a] for a = iL:iR]
    AA = vecTens[1]
    for a = 2:length(vecTens)
      AA = contract(AA,ndims(AA),vecTens[a],1)
    end

    newAA,outEnergy = krylov(Nsite_update,Lenv[iL],Renv[iR],AA,mpo[iL],maxiter=params.maxiter)
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

  function dmrgcvg(n::Integer,timer::Number,#biggestm::Array{P,1},maxtrunc::Array{X,1},lastenergy::Union{Z,Array{Z,1}},
                          dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Y,Renv::Y,psiLenv::B,
                          psiRenv::B,beta::Array{W,1},prevpsi::MPS...;params::TNparams=params())::Bool where {Y <: Union{envType,AbstractArray}, B <: Union{envType,AbstractArray}, Z <: Number, W <: Number, P <: Number, R <: MPS, X <: Number}

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
                    alpha_decay::Float64=0.9) where P <: Union{Number,Array{Float64,1}} where {B <: TensType, W <: Number}
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
                    storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,exnum::Integer=1) where P <: Union{Number,Array{Float64,1}} where {B <: TensType, W <: Number}
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
                    storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,exnum::Integer=1) where P <: Union{Number,Array{Float64,1}} where {B <: TensType, W <: Number}
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
                    storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,exnum::Integer=1) where P <: Union{Number,Array{Float64,1}} where {B <: TensType, W <: Number}
    loadvars!(params,"DMRG-"*method,minm,maxm,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,maxiter,fixD,nsites,
        noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
    return optmps(psi,psi,mpo,[1.],params=params,stepfct=step2S,makeOps=NsiteOps,cvgfct=dmrgcvg,displayfct=dmrginformation)
  end
  export dmrg2S
#end
