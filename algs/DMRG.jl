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
    Module: DMRG
    
Function for the density matrix renormalization group
"""
module DMRG
using ..tensor
using ..QN
using ..Qtensor
using ..Qtask
using ..MPutil
using ..contractions
using ..decompositions
using ..Krylov
using ..optimizeMPS

  function twosite_update(AA::X,ops::Y,Lenv::Z,Renv::Z) where {X <: Union{Qtens{A,Q},Array{A,4},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,6},tens{B}},Z <: Union{Qtens{W,Q},Array{W,3},tens{W}}} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
    Hpsi = contract(ops,[5,6],AA,[2,3])
    LHpsi = contract(Lenv,[2,3],Hpsi,[1,5])
    temp = contract(LHpsi,[5,4],Renv,[1,2])
    return temp
  end

  import Base.string
  import Printf
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

  function load_storeD!(storeD::Array{W,1},Dmat::Array{W,1}) where {W <: Number}
    for a = 1:min(length(storeD),length(Dmat))
      storeD[a] = Dmat[a]
    end
    nothing
  end

  function string(vect::Array{W,1}) where W <: Number
    return string(vect,1,length(vect))
  end

  function SvNcheck!(i::Integer,j::Integer,D::W,truncerr::Float64,
                     params::TNparams;alpha::Bool=true) where {W<: TensType}

    if params.maxtrunc < truncerr
      params.maxtrunc = truncerr
    end
    if params.biggestm < size(D,1)
      params.biggestm = size(D,1)
    end
    Dmat = [searchindex(D,i,i) for i = 1:size(D,1)]
    sizeDsq = length(Dmat)
    if params.allSvNbond
      SvNvec[i] = -sum(h->Dmat[h]^2*log(Dmat[h]^2),sizeDsq:-1:1)
      if i == params.startoc && !params.silent
        println("Entropies in chain: ",SvNvec)
      end
    elseif i == max(2,params.SvNbond+1) && j < 0
      if typeof(D) <: qarray
        sort!(Dmat,rev=true)
      end
      SvN = -sum(h->Dmat[h]^2*log(Dmat[h]^2),sizeDsq:-1:1)
      params.entropy = SvN
      if !params.silent
        print("SvN at center bond b=",params.SvNbond," = ",SvN)
        if alpha
          println(" (alpha = $(Printf.@sprintf("%.5f",params.noise)))")
        else
          println()
        end
        ymax = min(sizeDsq,params.maxshowD)
        load_storeD!(params.storeD,Dmat)
        println("Singular values at center bond b=",params.SvNbond,": [",string(Dmat,1,ymax),"]")
        println()
      end
    end
    nothing
  end

  function regular_SvNcheck!(i::Integer,j::Integer,D::W,truncerr::Float64,params::TNparams) where {W<: TensType}
    return SvNcheck!(i,j,D,truncerr,params,alpha=false)
  end

  function setDMRG(psi::MPS,mpo::MPO,maxm::Integer,minm::Integer,Lenv::TensType,Renv::TensType,
                    halfsweep::Bool,alpha::Z,origj::Bool,allSvNbond::Bool,boundary::B...) where {Z <: Union{Float64,Array{Float64,1}},B <: TensType}

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
      if Lenv != [0]
        Lbound = Lenv[1]
      else
        Lbound = [0]
      end
      if Renv != [0]
        Rbound = Renv[length(Renv)]
      else
        Rbound = [0]
      end
    end
    if Lenv == [0] && Renv == [0]
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
    Hpsi = contract(ops,[2],A,[2])
    Hpsi = contract(HL,[2,3],Hpsi,[1,4],alpha=alpha)
    expAA = reshape!(Hpsi,size(Hpsi,1),size(Hpsi,2),size(Hpsi,3)*size(Hpsi,4),merge=true)
    return concat!(A,expAA,3)
  end
  
  function Rexpand(A::X,ops::Y,HR::Z,alpha::Float64) where {X <: Union{Qtens{M,Q},Array{M,3},tens{M}}, Y <: Union{Qtens{B,Q},Array{B,4},tens{B}},Z <: Union{Qtens{W,Q},Array{W,3},tens{W}}} where {M <: Number, B <: Number, W <: Number, Q <: Qnum}
    Hpsi = contract(A,[2],ops,[2])
    Hpsi = contract(Hpsi,[2,5],HR,[1,2],alpha=alpha)
    expAA = reshape!(Hpsi,size(Hpsi,1)*size(Hpsi,2),size(Hpsi,3),size(Hpsi,4),merge=true)
    return concat!(A,expAA,1)
  end
  
  const alpha_max = 1.
  function alpha_update(alpha::Float64,truncerr::Float64,cutoff::Float64,
                        lastEnergy::Union{R,Array{R,1}},infovals::Union{S,Array{S,1}},
                        noise_goal::Float64,dalpha::Float64;avgE::W=1.)::Number where {R <: Number, S <: Number, W <: Number}
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

  function twositeOps(mpo::MPO,params::TNparams)
    nsites = params.nsites
    if typeof(mpo[1]) <: qarray
      mpoType =  qarray
    elseif typeof(mpo[1]) <: denstens
      mpoType = denstens
    else
      mpoType = Array{eltype(mpo[1]),6}
    end
    nbonds = length(mpo)-(nsites-1)
    ops = Array{mpoType,1}(undef,nbonds)
    for i = 1:nbonds
      ops[i] = contract([1,3,5,6,2,4],mpo[i],ndims(mpo[i]),mpo[i+1],1)
    end
    return MPO(ops)
  end

  function Nsite_update(AA::X,ops::Y,Lenv::Z,Renv::Z) where {X <: Union{Qtens{A,Q},Array{A,4},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,6},tens{B}},Z <: Union{Qtens{W,Q},Array{W,3},tens{W}}} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
    AAinds = [i for i = 2:ndims(AA)-1]
    Hpsi = contract(ops,[i for i = (ndims(ops)-length(AAinds)+1):ndims(ops)],AA,AAinds)
    LHpsi = contract(Lenv,[2,3],Hpsi,[1,1+ndims(AA)])
    temp = contract(LHpsi,[ndims(LHpsi),ndims(LHpsi)-1],Renv,[1,2])
    return temp
  end

  import ..optimizeMPS.singlesite_update
  function step3S(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,
                  Lenv::AbstractArray,Renv::AbstractArray,
                  psiLenv::AbstractArray,psiRenv::AbstractArray,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where {Z <: Union{qarray,Array{W,3},tens}, M <: Union{qarray,Array{J,4},tens}} where {X <: Number, W <: Number, J <: Number, Q <: Qnum, P <: Number, Y <: Number}

    AAvec,outEnergy = krylov(singlesite_update,psi[i],mpo[i],Lenv[i],Renv[i],maxiter=params.maxiter)

    noise = params.noise

    params.energy = outEnergy[1]
    alpha_condition = noise > params.noise_incr

    minm = params.minm
    maxm = params.maxm
    cutoff = params.cutoff

    if j > 0
      psi[iL] = (alpha_condition ? Lexpand(AAvec[1],mpo[iL],Lenv[iL],noise) : AAvec[1])::Union{Qtens{W,Q},Array{W,3},tens{W}} where Q <: Qnum where W <: Number
      psi[iL],psi[iR],D,truncerr = moveR(psi[iL],psi[iR],cutoff=cutoff,m=maxm,minm=minm,condition=alpha_condition)
    else
      psi[iR] = (alpha_condition ? Rexpand(AAvec[1],mpo[iR],Renv[iR],noise) : AAvec[1])::Union{Qtens{W,Q},Array{W,3},tens{W}} where Q <: Qnum where W <: Number
      psi[iL],psi[iR],D,truncerr = moveL(psi[iL],psi[iR],cutoff=cutoff,m=maxm,minm=minm,condition=alpha_condition)
    end

    params.noise = alpha_update(noise,truncerr,params.cutoff,params.lastenergy,params.energy,params.noise_goal,params.noise_incr)
    params.truncerr = truncerr
    params.biggestm = max(params.biggestm,size(D,1))

    if !params.efficient
      SvNcheck!(i,j,D,truncerr,params)
    end
    nothing
  end

  function twostep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,
                    Lenv::AbstractArray,Renv::AbstractArray,
                    psiLenv::AbstractArray,psiRenv::AbstractArray,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where {Z <: Union{qarray,Array{W,3},tens}, M <: Union{qarray,Array{J,4},tens}} where {X <: Number, W <: Number, J <: Number, Q <: Qnum, P <: Number, Y <: Number}

    AA = contract(psi[iL],3,psi[iR],1)
    newAA,outEnergy = krylov(twosite_update,AA,mpo[iL],Lenv[iL],Renv[iR],maxiter=params.maxiter)

    params.energy = outEnergy[1]
    U,D,V,truncerr = svd(newAA[1],[[1,2],[3,4]],m=params.maxm,minm=params.minm,cutoff=params.cutoff,mag=1.)

    if j < 0
      psi[iL] = contract(U,3,D,1)
      psi[iR] = V
    else
      psi[iL] = U
      psi[iR] = contract(D,2,V,1)
    end

    if !params.efficient
      regular_SvNcheck!(i,j,D,truncerr,params)
    end
    nothing
  end




  function step2S(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,
                  Lenv::AbstractArray,Renv::AbstractArray,
                  psiLenv::AbstractArray,psiRenv::AbstractArray,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where {Z <: Union{qarray,Array{W,3},tens}, M <: Union{qarray,Array{J,4},tens}} where {X <: Number, W <: Number, J <: Number, Q <: Qnum, P <: Number, Y <: Number}

    AA = contract(psi[iL],ndims(psi[iL]),psi[iR],1)
    AAvec,outEnergy = krylov(twosite_update,AA,ops[iL],Lenv[iL],Renv[iR],maxiter=maxiter)
    infovals[1] = outEnergy[1]

    m = params.maxm
    minm = params.minm
    cutoff = params.cutoff

    AA = AAvec[1]

      if j > 0
        rho = contractc(AA,[3,4],AA,[3,4])
      else
        rho = ccontract([3,4,1,2],AA,[1,2],AA,[1,2])
      end

      if false #alpha > 0.
        if j > 0
          temp = contract(Lenv[iL],3,AA,1)
          randrho = contractc(temp,[2,4,5],temp,[2,4,5])
        else
          temp = contract(AA,4,Renv[iR],1)
          randrho = ccontract(temp,[1,2,4],temp,[1,2,4])
        end
        rho = add!(rho,randrho,alpha)
      end

      D,U,truncerr,sumD = eigen(rho,[[1,2],[3,4]],cutoff=cutoff,m=m,minm=minm)
      if alpha > 0.
        D = div!(D,sumD)
      end
      if j > 0
        psi[iL],psi[iR] = U,ccontract(U,[1,2],AA,[1,2])
      else
        psi[iL],newV = contractc(AA,[3,4],U,[1,2]),permutedims(U,[3,1,2])
        psi[iR] = conj!(newV)
      end
    if !params.efficient
      regular_SvNcheck!(i,j,D,truncerr,params)
    end
    nothing
  end




  import ..optimizeMPS.Nstep
  function dmrgNstep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,
                    Lenv::Array{Z,1},Renv::Array{Z,1},
                    psiLenv::AbstractArray,psiRenv::AbstractArray,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where {Z <: Union{qarray,Array{W,3},tens}, M <: Union{qarray,Array{J,4},tens}} where {X <: Number, W <: Number, J <: Number, Q <: Qnum, P <: Number, Y <: Number}

    Nstep(n,j,i,iL,iR,dualpsi,psi,ops,Lenv,Renv,psiLenv,psiRenv,beta,prevpsi...,params=params)
    if !params.efficient
      regular_SvNcheck!(i,j,D,truncerr,params)
    end
    nothing
  end

  function dmrg(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                    SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::intType=2,efficient::Bool=false,
                    cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,fixD::Bool=false,Lbound::AbstractArray=[0],Rbound::AbstractArray=[0],
                    noise::Number=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Float64=0.9,method::String="3S",shift::Bool=false,
                    saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::TensType=[0],Renv::TensType=[0],origj::Bool=true,maxshowD::Integer=8,storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,
                    partitions::Integer=Threads.nthreads()) where {B <: TensType, W <: Number}
    if method == "3S"
      return dmrg3S(psi,mpo,method=method,maxm=maxm,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,
                    SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,fixD=fixD,nsites=1,
                    noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                    halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
    elseif method == "Nsite" || nsites > 2
      return dmrg_Nsite(psi,mpo,method=method,maxm=maxm,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,
                    SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,fixD=fixD,nsites=nsites,
                    noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                    halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
    elseif method == "twosite"
      return dmrg_twosite(psi,mpo,method=method,maxm=maxm,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,
                    SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,fixD=fixD,nsites=nsites,
                    noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                    halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
    elseif method == "2S"
      return dmrg2S(psi,mpo,method=method,maxm=maxm,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,
                    SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,fixD=fixD,nsites=2,
                    noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                    halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
    else
      error("DMRG method not defined")
    end
  end
  export dmrg

  function dmrginformation(params::TNparams)
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

  function dmrgcvg(n::Integer,timer::Number,
                          dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::AbstractArray,Renv::AbstractArray,psiLenv::AbstractArray,
                          psiRenv::AbstractArray,beta::Array{W,1},prevpsi::MPS...;params::TNparams=params())::Bool where {Z <: Number, W <: Number, P <: Number, R <: MPS, X <: Number}

    if !params.silent
      println("Sweep $n (back and forth): $(Printf.@sprintf("%.5f",timer)) sec")
      println("Largest truncation = $(params.maxtrunc), m = $(params.biggestm)")
      println("Energy at sweep $n is $(Printf.@sprintf("%.12f",params.energy))")
      println()
      flush(Base.stdout)
    end
    params.biggestm = 0
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

  function dmrg3S(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                    SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::intType=2,efficient::Bool=false,
                    cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,fixD::Bool=false,Lbound::AbstractArray=[0],Rbound::AbstractArray=[0],
                    noise::Number=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Float64=0.9,method::String="3S",shift::Bool=false,
                    saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::TensType=[0],Renv::TensType=[0],origj::Bool=true,maxshowD::Integer=8,storeD::Array{W,1}=[0.],
                    alpha_decay::Float64=0.9) where {B <: TensType, W <: Number}

    params = algvars()
    params.method = "DMRG-"*method
    params.minm = minm
    params.maxm = maxm
    params.sweeps = sweeps
    params.cutoff = cutoff
    params.silent = silent
    params.goal = goal
    params.startnoise = noise
    params.SvNbond = SvNbond
    params.allSvNbond = allSvNbond
    params.efficient = efficient
    params.cvgE = cvgE
    params.maxiter = maxiter
    params.fixD = fixD
    params.nsites = nsites
    params.noise = noise
    params.noise_decay = noise_decay
    params.noise_goal = noise_goal
    params.noise_incr = noise_incr
    params.saveEnergy = saveEnergy
    params.halfsweep = halfsweep
    params.Lbound = Lbound
    params.Rbound = Rbound
    params.Lenv = Lenv
    params.Renv = Renv
    params.origj = origj
    params.maxshowD = maxshowD
    params.storeD = storeD
    return optmps(psi,psi,mpo,[1.],params=params,stepfct=step3S,makeOps=singlesiteOps,cvgfct=dmrgcvg)
  end
  export dmrg3S

  function dmrg_twosite(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                    SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::intType=2,efficient::Bool=false,
                    cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,fixD::Bool=false,Lbound::AbstractArray=[0],Rbound::AbstractArray=[0],
                    noise::Number=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Float64=0.9,method::String="3S",shift::Bool=false,
                    saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::TensType=[0],Renv::TensType=[0],origj::Bool=true,maxshowD::Integer=8,storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,
                    partitions::Integer=Threads.nthreads()) where {B <: TensType, W <: Number}

    params = algvars()
    params.method = "DMRG-"*method
    params.minm = minm
    params.maxm = maxm
    params.sweeps = sweeps
    params.cutoff = cutoff
    params.silent = silent
    params.goal = goal
    params.startnoise = noise
    params.SvNbond = SvNbond
    params.allSvNbond = allSvNbond
    params.efficient = efficient
    params.cvgE = cvgE
    params.maxiter = maxiter
    params.fixD = fixD
    params.nsites = nsites
    params.noise = noise
    params.noise_decay = noise_decay
    params.noise_goal = noise_goal
    params.noise_incr = noise_incr
    params.saveEnergy = saveEnergy
    params.halfsweep = halfsweep
    params.Lbound = Lbound
    params.Rbound = Rbound
    params.Lenv = Lenv
    params.Renv = Renv
    params.startoc = psi.oc
    params.origj = origj
    params.maxshowD = maxshowD
    params.storeD = storeD

    return optmps(psi,psi,mpo,[1.],params=params,measfct=expect,stepfct=twostep,
                    makeOps=twositeOps,cvgfct=dmrgcvg)
  end
  export dmrg_twosite

  import ..optimizeMPS.NsiteOps
  function dmrg_Nsite(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                    SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::intType=2,efficient::Bool=false,
                    cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,fixD::Bool=false,Lbound::AbstractArray=[0],Rbound::AbstractArray=[0],
                    noise::Number=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Float64=0.9,method::String="Nsite",shift::Bool=false,
                    saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::TensType=[0],Renv::TensType=[0],origj::Bool=true,maxshowD::Integer=8,storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,
                    partitions::Integer=Threads.nthreads()) where {B <: TensType, W <: Number}

    params = algvars()
    params.method = "DMRG-"*method*" (N=$nsites)"
    params.minm = minm
    params.maxm = maxm
    params.sweeps = sweeps
    params.cutoff = cutoff
    params.silent = silent
    params.goal = goal
    params.startnoise = noise
    params.SvNbond = SvNbond
    params.allSvNbond = allSvNbond
    params.efficient = efficient
    params.cvgE = cvgE
    params.maxiter = maxiter
    params.fixD = fixD
    params.nsites = nsites
    params.noise = noise
    params.noise_decay = noise_decay
    params.noise_goal = noise_goal
    params.noise_incr = noise_incr
    params.saveEnergy = saveEnergy
    params.halfsweep = halfsweep
    params.Lbound = Lbound
    params.Rbound = Rbound
    params.Lenv = Lenv
    params.Renv = Renv
    params.startoc = psi.oc
    params.origj = origj
    params.maxshowD = maxshowD
    params.storeD = storeD

    return optmps(psi,psi,mpo,[1.],params=params,measfct=expect,stepfct=dmrgNstep,
                    makeOps=NsiteOps,cvgfct=dmrgcvg)
  end
  export dmrg_Nsite

  function dmrg2S(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                    SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::intType=2,efficient::Bool=false,
                    cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,fixD::Bool=false,Lbound::AbstractArray=[0],Rbound::AbstractArray=[0],
                    noise::Number=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Float64=0.9,method::String="2S",shift::Bool=false,
                    saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::TensType=[0],Renv::TensType=[0],origj::Bool=true,maxshowD::Integer=8,storeD::Array{W,1}=[0.],
                    alpha_decay::Float64=0.9) where {B <: TensType, W <: Number}

    params = algvars()
    params.method = "DMRG-"*method
    params.minm = minm
    params.maxm = maxm
    params.sweeps = sweeps
    params.cutoff = cutoff
    params.silent = silent
    params.goal = goal
    params.startnoise = noise
    params.SvNbond = SvNbond
    params.allSvNbond = allSvNbond
    params.efficient = efficient
    params.cvgE = cvgE
    params.maxiter = maxiter
    params.fixD = fixD
    params.nsites = nsites
    params.noise = noise
    params.noise_decay = noise_decay
    params.noise_goal = noise_goal
    params.noise_incr = noise_incr
    params.saveEnergy = saveEnergy
    params.halfsweep = halfsweep
    params.Lbound = Lbound
    params.Rbound = Rbound
    params.Lenv = Lenv
    params.Renv = Renv
    params.origj = origj
    params.maxshowD = maxshowD
    params.storeD = storeD
    return optmps(psi,psi,mpo,[1.],params=params,stepfct=step2S,makeOps=singlesiteOps,cvgfct=dmrgcvg)
  end
  export dmrg2S
end
