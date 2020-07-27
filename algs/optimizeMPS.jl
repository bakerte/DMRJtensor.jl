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

module optimizeMPS
using ..tensor
using ..QN
using ..Qtensor
using ..Qtask
using ..MPutil
using ..contractions
using ..decompositions

  """
      TNparams

  parameters of a tensor network calculation
  """
  abstract type TNparams end
  export TNparams

  """
      algvars

  Struct to hold variables for the MPS optimization function and others
  """
  mutable struct algvars{W <: Number,Q <: Qnum} <: TNparams
    method::String
    nsites::intType
    minm::intType
    maxm::intType
    cutoff::Float64

    sweeps::intType
    halfsweep::Bool

    maxiter::intType
    cvgE::Bool
    goal::W
    startnoise::Float64
    noise::Union{Array{Float64,1},Float64}
    noise_goal::Float64
    noise_decay::Float64
    noise_incr::Float64

    shift::Bool
    fixD::Bool

    startoc::intType
    origj::Bool

    maxshowD::intType

    storeD::Array{Float64,1}
    saveEnergy::Array{W,1}

    energy::Union{Array{W,1},W}
    lastenergy::Union{Array{W,1},W}

    entropy::Union{Array{Float64,1},Float64}
    lastentropy::Union{Array{Float64,1},Float64}

    truncerr::Float64
    maxtrunc::Float64
    biggestm::intType

    SvNbond::intType
    allSvNbond::Bool

    Lbound::TensType
    Rbound::TensType
    Lenv::AbstractArray  #Union{envType,Array{Array{Int64,1},1}}
    Renv::AbstractArray  #Union{envType,Array{Array{Int64,1},1}}

    efficient::Bool
    silent::Bool
  end


  export algvars

  function optinformation(params::TNparams)
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
  end

  function setEnv(dualpsi::MPS,psi::MPS,mpo::MPO,params::TNparams;measfct::Function=expect) where {Z <: Union{Float64,Array{Float64,1}},B <: TensType}

    if params.Lenv == [0] && params.Renv == [0]
      params.Lenv,params.Renv = makeEnv(dualpsi,psi,mpo,Lbound=params.Lbound,Rbound=params.Rbound)
    end

    SvN,lastSvN,maxtrunc,biggestm = [0.],[0.],[0.],[0]
    params.lastentropy = copy(params.entropy)
    Ns = length(psi)
    timer = 0.

    Nsteps = Ns-1
    if !params.halfsweep
      Nsteps *= 2
    end

    # setup for the sweeping direction
    if psi.oc == 1
      j = 1
    elseif psi.oc == Ns #
      j = -1
    else
      j = params.origj ? 1 : -1 #if the oc is away from the edge, j could be externally specified.
    end

    if params.allSvNbond
      SvNvec = zeros(Ns)
    else
      SvNvec = zeros(0)
    end
    startoc = copy(psi.oc)

    if isapprox(sum(params.energy),0.) && !params.efficient
      params.energy = measfct(dualpsi,psi,mpo,Lbound=params.Lenv[1],Rbound=params.Renv[end])
    end
    params.lastenergy = copy(params.energy)

    if !params.cvgE && params.allSvNbond &&length(psi) != length(params.entropy)
      params.entropy = zeros(length(psi))
    end
    params.lastentropy = copy(params.entropy)

    return params.Lenv,params.Renv,SvN,lastSvN,params.maxm,maxtrunc,biggestm,Ns,Nsteps,timer,j,SvNvec,startoc
  end


  function Cpsipsi(AA::X,Lenv::W,Renv::W) where {X <: Union{qarray,denstens,AbstractArray},W <: Union{qarray,denstens,AbstractArray}}
    LHpsi = contract(Lenv,2,AA,1)
    return contract(LHpsi,3,Renv,1)
  end

  function randLexpand(A::W,expalpha::Float64,maxdim::Int64;bonddimfrac::Integer=5,mindim::Integer=10) where W <: Union{denstens,AbstractArray}
    checkmaxdim = max(maxdim-size(A,3),size(A,3))
    expAA = expalpha * rand(size(A,1),size(A,2),min(max(cld(size(A,3),bonddimfrac),mindim),checkmaxdim))
    return concat(A,expAA,3)
  end

  function randRexpand(A::W,expalpha::Float64,maxdim::Int64;bonddimfrac::Integer=5,mindim::Integer=10) where W <: Union{denstens,AbstractArray}
    checkmaxdim = max(maxdim-size(A,1),size(A,1))
    expAA = expalpha * rand(min(max(cld(size(A,1),bonddimfrac),mindim),checkmaxdim),size(A,2),size(A,3))
    return concat(A,expAA,1)
  end


  function randLexpand(A::qarray,alpha::Float64,maxdim::Int64;bonddimfrac::Integer=5,mindim::Integer=10)
    mindim = min(mindim,size(A,3))
    QnumL = [ A.flux + inv(q1) + inv(q2) for q1 in A.QnumMat[1] for q2 in A.QnumMat[2]]
    checkmaxdim = max(maxdim-size(A,3),size(A,3))
    dim = min(max(cld(size(A,3),bonddimfrac),mindim),checkmaxdim,length(QnumL))
    newQ = [QnumL[rand(1:length(QnumL),1)...] for a = 1:dim]
    if length(newQ) > 0
      Qlabels = [A.QnumMat[1:2]...,newQ]
      addsize = length(newQ)
      Hpsi = rand(Qlabels)
      A = concat(A,Hpsi,3)
    end
    return A
  end

  function randRexpand(B::qarray,alpha::Float64,maxdim::Int64;bonddimfrac::Integer=5,mindim::Integer=10)::qarray where {W <: qarray,X <: qarray,Y <: qarray}
    mindim = min(mindim,size(B,1))
    QnumL = [ B.flux + inv(q1) + inv(q2) for q1 in B.QnumMat[2] for q2 in B.QnumMat[3]]
    checkmaxdim = max(maxdim-size(B,3),size(B,3))
    dim = min(max(cld(size(B,1),bonddimfrac),mindim),checkmaxdim,length(QnumL))
    newQ = [QnumL[rand(1:length(QnumL),1)...] for a = 1:dim]
    if length(newQ) > 0
      Qlabels = [newQ,B.QnumMat[2:3]...]
      Hpsi = rand(Qlabels)
      B = concat(B,Hpsi,1)
    end
    return B
  end

  function NsiteOps(mpo::MPO,params::TNparams)
    nsites = params.nsites
    if nsites == 1
      out_mpo = mpo
    else
      if typeof(mpo[1]) <: qarray
        mpoType =  qarray
      elseif typeof(mpo[1]) <: denstens
        mpoType = denstens
      else
        mpoType = Array{eltype(mpo[1]),6}
      end
      nbonds = length(mpo)-nsites+1
      ops = Array{mpoType,1}(undef,nbonds)
      neworder = vcat([i for i = 1:2:2*nsites+1],[2*nsites+2],[i for i = 2:2:2*nsites])
      for i = 1:nbonds
        ops[i] = mpo[i]
        for b = 2:nsites
          ops[i] = contract(ops[i],ndims(ops[i]),mpo[i+b-1],1)
        end
        ops[i] = permutedims(ops[i],neworder)
      end
      out_mpo = MPO(ops)
    end
    return out_mpo
  end

  function singlesite_update(AA::X,ops::Y,Lenv::Z,Renv::Z) where {X <: Union{Qtens{A,Q},Array{A,3},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,4},tens{B}},Z <: Union{Qtens{W,Q},Array{W,3},tens{W}}} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
    Hpsi = contract(ops,[2],AA,[2]) #::Union{Qtens{W,Q},Array{W,5}} where Q <: Qnum
    LHpsi = contract(Lenv,[2,3],Hpsi,[1,4]) #::Union{Qtens{W,Q},Array{W,4}} where Q <: Qnum
    return contract(LHpsi,[4,3],Renv,[1,2]) #::Union{Qtens{W,Q},Array{W,3}} where Q <: Qnum
  end

  function algvars()
    algvars{Float64,U1}("",
                        1,
                        2,
                        2,
                        0.,
                        0,
                        false, #halfsweep
                        2, #maxiter
                        true,
                        0., #goal
                        0., #startnoise
                        0., #noise
                        0.3, #noise_goal
                        0.9, #noise_decay
                        0.01, #noise_incr
                        false, #shift
                        false, #fixD

                        1, #startoc
                        true, #origj
                        8, #maxshowD
                        [0. for i = 1:8], #storeD
                        [0.], #saveEnergy

                        0., #energy
                        0., #lastenergy

                        0., #entropy
                        0., #lastentropy
                        0., #truncerr
                        0.,
                        0, #biggestm
                        1, #SvNbond
                        false, #allSvNbond
                        [0], #Lbound
                        [0], #Rbound
                        [0], #Lenv
                        [0], #Renv

                        false, #efficient
                        false #silent
                        )
  end

  function Nstep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,
                    Lenv::Array{Z,1},Renv::Array{Z,1},
                    psiLenv::AbstractArray,psiRenv::AbstractArray,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where {Z <: Union{qarray,Array{W,3},tens}, M <: Union{qarray,Array{J,4},tens}} where {X <: Number, W <: Number, J <: Number, Q <: Qnum, P <: Number, Y <: Number}
    AA = psi[iL]
    for a = iL+1:iR
      AA = contract(AA,ndims(AA),psi[a],1)
    end

    maxm = params.maxm
    minm = params.minm
    cutoff = params.cutoff

    D = AA + rand(AA)
    D /= norm(D)
    
    truncerr = 0.
    if j > 0
      for w = iR:-1:iL+1
        U,D,psi[w],truncerr = svd(D,[[i for i = 1:ndims(D)-2],[ndims(D)-1,ndims(D)]],m=maxm,minm=minm,cutoff=cutoff,mag=1.)
        if w == iL+1
          psi[iL] = U
          psi[iL+1] = contract(D,2,psi[iL+1],1)
        else
          D = contract(U,ndims(U),D,1)
        end
      end
    else
      for w = iL:iR-1
        psi[w],D,V,truncerr = svd(D,[[1,2],[i for i = 3:ndims(D)]],m=maxm,minm=minm,cutoff=cutoff,mag=1.)
        if w == iR-1
          psi[iR-1] = contract(psi[iR-1],3,D,1)
          psi[iR] = V
        else
          D = contract(D,2,V,1)
        end
      end
    end
    nothing
  end

  """
  infovals: stores current energy (skips initialization for it), truncation error, 
                          entanglement entropy on requested bond (default center), 
                          largest m value, noise parameter  <---switch to second slot
  """
  function optimize(dualpsi::MPS,psi::MPS,mpo::MPO,beta::Array{P,1},prevpsi::MPS...;
                    method::String="optimize",maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,
                    cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,infovals::Array{Float64,1}=zeros(5),
                    SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,efficient::Bool=false,
                    cvgE::Bool=true,maxiter::Integer=2,fixD::Bool=false,nsites::intType=1,
                    noise::Number=1.0,noise_decay::Float64=0.9,noise_goal::Float64=0.3,noise_incr::Float64=0.01,
                    saveEnergy::Array{R,1}=[0],halfsweep::Bool=false,Lbound::AbstractArray=[0],Rbound::AbstractArray=[0],
                    Lenv::AbstractArray=[0],Renv::AbstractArray=[0],origj::Bool=true,maxshowD::Integer=8,
                    storeD::Array{W,1}=[0.]) where {B <: TensType,W <: Number,R <: Number,P <: Number}
    params = algvars()
    params.method = method
    params.minm = minm
    params.maxm = maxm
    params.sweeps = sweeps
    params.cutoff = cutoff
    params.silent = silent
    params.goal = goal
    params.SvNbond = SvNbond
    params.allSvNbond = allSvNbond
    params.efficient = efficient
    params.cvgE = cvgE
    params.maxiter = maxiter
    params.fixD = fixD
    params.nsites = nsites
    params.startnoise = noise
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
    return optmps(dualpsi,psi,mpo,beta,prevpsi...,params=params)
  end
  export optimize




  function Noptimize(dualpsi::MPS,psi::MPS,mpo::MPO,beta::Array{P,1},prevpsi::MPS...;
                    method::String="optimize",maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,
                    cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,infovals::Array{Float64,1}=zeros(5),
                    SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,efficient::Bool=false,
                    cvgE::Bool=true,maxiter::Integer=2,fixD::Bool=false,nsites::intType=2,
                    noise::Number=1.0,noise_decay::Float64=0.9,noise_goal::Float64=0.3,noise_incr::Float64=0.01,
                    saveEnergy::Array{R,1}=[0],halfsweep::Bool=false,Lbound::AbstractArray=[0],Rbound::AbstractArray=[0],
                    Lenv::AbstractArray=[0],Renv::AbstractArray=[0],origj::Bool=true,maxshowD::Integer=8,
                    storeD::Array{W,1}=[0.]) where {B <: TensType,W <: Number,R <: Number,P <: Number}
    params = algvars()
    params.method = method
    params.minm = minm
    params.maxm = maxm
    params.sweeps = sweeps
    params.cutoff = cutoff
    params.silent = silent
    params.goal = goal
    params.SvNbond = SvNbond
    params.allSvNbond = allSvNbond
    params.efficient = efficient
    params.cvgE = cvgE
    params.maxiter = maxiter
    params.fixD = fixD
    params.nsites = nsites
    params.startnoise = noise
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
    return optmps(dualpsi,psi,mpo,beta,prevpsi...,params=params,stepfct=Nstep,makeOps=NsiteOps,cvgfct=nullcvg)
  end
  export Noptimize





  function optmps(dualpsi::MPS,psi::MPS,mpo::MPO,beta::Array{P,1},prevpsi::MPS...;params::TNparams=algvars(),
                    displayfct::Function=optinformation,Envfct::Function=setEnv,
                    measfct::Function=expect,stepfct::Function=optstep,
                    makeOps::Function=NsiteOps,cvgfct::Function=optcvg,
                    mover::Function=move!,boundarymover::Function=boundaryMove!) where {B <: TensType,W <: Number,R <: Number,P <: Number}

    if !params.silent && !params.efficient
      displayfct(params)
    end

    out = Envfct(dualpsi,psi,mpo,params,measfct=measfct)
    Lenv,Renv,SvN,lastSvN,m,maxtrunc,biggestm,Ns,Nsteps,timer,j,SvNvec,startoc = out

    #make operators...or identity
    ops = makeOps(mpo,params)
    
    range = max(2,params.nsites)-1

    mover(dualpsi,psi.oc)

    psiLenv = [Array{eltype(psi),1}(undef,Ns) for a = 1:length(prevpsi)]
    psiRenv = [Array{eltype(psi),1}(undef,Ns) for a = 1:length(prevpsi)]
    if length(prevpsi) > 0
      for b = 1:length(beta)
        mover(prevpsi[b],psi.oc)
        x,y = makePsiEnv(dualpsi,prevpsi[b])#!!!!!!
        for w = 1:prevpsi[b].oc
          psiLenv[b][w] = x[w]
        end
        for w = Ns:-1:prevpsi[b].oc
          psiRenv[b][w] = y[w]
        end
      end
    end

    totsweeps = params.halfsweep ? params.sweeps+1 : params.sweeps
    if length(psi) == 1
      for n = 1:totsweeps
        AAvec,outEnergy = krylov(singlesite_update,psi[1],mpo[1],Lenv[1],Renv[1],maxiter=maxiter)
        psi[1] = AAvec[1]
        params.energy = outEnergy[1]
      end
    else
      for n = 1:totsweeps
        if !params.silent
          timer = -time()
        end

        for ns in 1:Nsteps
          i = psi.oc
          
          if j > 0
            iL,iR = i,i+range
          else
            iL,iR = i-range,i
          end

          #D,truncerr = 
          stepfct(n,j,i,iL,iR,dualpsi,psi,ops,Lenv,Renv,psiLenv,psiRenv,beta,prevpsi...,params=params)

          if j > 0
            Lenv[i+1] = Lupdate(Lenv[i],dualpsi[i],psi[i],mpo[i])
            for b = 1:length(prevpsi)
              psiLenv[b][i+1] = Lupdate(psiLenv[b][i],dualpsi[i],prevpsi[b][i])
            end
          else
            Renv[i-1] = Rupdate(Renv[i],dualpsi[i],psi[i],mpo[i])
            for b = 1:length(prevpsi)
              psiRenv[b][i-1] = Rupdate(psiRenv[b][i],dualpsi[i],prevpsi[b][i])
            end
          end

          #update oc's position
          psi.oc += j
          ##reverse direction if we hit the edge
          if psi.oc == Ns - (range-1) && j > 0
            boundarymover(psi,Ns,Lenv,Renv,mpo)
            j *= -1
          elseif psi.oc == 1 + (range-1) && j < 0
            boundarymover(psi,1,Lenv,Renv,mpo)
            j *= -1
          end
          for b = 1:length(prevpsi)
            boundarymover(prevpsi[b],psi.oc,psiLenv[b],psiRenv[b])
          end
        end
        if !params.silent
          timer += time()
        end
        if !params.efficient
          breakbool = cvgfct(n,timer,dualpsi,psi,ops,Lenv,Renv,psiLenv,psiRenv,beta,prevpsi...,params=params)
          
          if breakbool
            break
          end
        end
      end
    end

    return params.energy
  end
  export optmps

  function makePsiEnv(dualpsi::MPS,psi::MPS)
    Ns = size(psi,1)
    if typeof(psi[1]) <: qarray
      thistype = qarray
    elseif typeof(psi[1]) <: denstens
      thistype = denstens
    elseif typeof(psi[1]) <: AbstractArray
      thistype = AbstractArray
    end
    Lenv = Array{thistype,1}(undef,Ns)
    Renv = Array{thistype,1}(undef,Ns)
    Lenv[1],Renv[end] = makeEnds(dualpsi,psi)
    for i = 1:dualpsi.oc-1
      Lenv[i+1] = Lupdate(Lenv[i],dualpsi[i],psi[i])
    end
    for i = Ns:-1:dualpsi.oc+1
      Renv[i-1] = Rupdate(Renv[i],dualpsi[i],psi[i])
    end
    return Lenv,Renv
  end

  function Hpsi(psi::MPS,a::intType,mpo::MPO,Lenv::AbstractArray,Renv::AbstractArray,
                psiLenv::AbstractArray,psiRenv::AbstractArray,beta::Array{W,1},prevpsi::MPS...) where W <: Number
    AAvec = singlesite_update(psi[a],mpo[a],Lenv[a],Renv[a])
    for b = 1:length(prevpsi)
      lpsi = Cpsipsi(prevpsi[b][a],psiLenv[b][a],psiRenv[b][a])
      AAvec = sub!(AAvec,lpsi,beta[b])
    end
    normAAvec = norm(AAvec)
    if normAAvec == 0
      AAvec = copy(psi[a])
      println("BIG WARNING")
    else
      AAvec = div!(AAvec,normAAvec)
    end
    return AAvec
  end

  function optstep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,
                    Lenv::AbstractArray,Renv::AbstractArray,
                    psiLenv::AbstractArray,psiRenv::AbstractArray,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params())where {Z <: Union{qarray,Array{W,3},tens}, M <: Union{qarray,Array{J,4},tens}} where {X <: Number, W <: Number, J <: Number, Q <: Qnum, P <: Number, Y <: Number}
    AAvec = Hpsi(psi,i,mpo,Lenv,Renv,psiLenv,psiRenv,beta,prevpsi...)
    dualpsi[i] = AAvec
    expalpha = params.noise
    exp_condition = expalpha > 0.

    maxm = params.maxm
    minm = params.minm
    cutoff = params.cutoff

    if exp_condition
      if j > 0
        dualpsi[i] = randLexpand(AAvec,expalpha,maxm)
      else
        dualpsi[i] = randRexpand(AAvec,expalpha,maxm)
      end
    else
      dualpsi[i] = AAvec
    end
    if j > 0
      dualpsi[iL],dualpsi[iR],D,truncerr = moveR(dualpsi[iL],dualpsi[iR],cutoff=cutoff,m=maxm,minm=minm,condition=exp_condition)
    else
      dualpsi[iL],dualpsi[iR],D,truncerr = moveL(dualpsi[iL],dualpsi[iR],cutoff=cutoff,m=maxm,minm=minm,condition=exp_condition)
    end

    dualpsi.oc += j

    return D,truncerr
  end
  export optstep

  function nullcvg(n::Integer,timer::Number,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::AbstractArray,Renv::AbstractArray,psiLenv::AbstractArray,
                    psiRenv::AbstractArray,beta::Array{W,1},prevpsi::MPS...;params::TNparams=params())::Bool where W <: Number
    println("sweep $n, time = $timer")
    return false
  end

  function optcvg(n::Integer,timer::Number,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::AbstractArray,Renv::AbstractArray,psiLenv::AbstractArray,
                  psiRenv::AbstractArray,beta::Array{W,1},prevpsi::MPS...;params::TNparams=params())::Bool where W <: Number

    i = dualpsi.oc
    tmp = singlesite_update(psi[i],mpo[i],Lenv[i],Renv[i])
    overlap = ccontract(tmp,dualpsi[i])
    for b = 1:length(prevpsi)
      tmp = Cpsipsi(prevpsi[b][i],psiLenv[b][i],psiRenv[b][i])
      overlap -= beta[b]*ccontract(tmp,dualpsi[i])
    end
    normpsi = ccontract(dualpsi[i])

    Error_val = (params.goal + real(normpsi) - 2*real(overlap))/real(normpsi)

    for a in dualpsi.A
      print(size(a))
    end
    println("Error at sweep $n: $Error_val")

    m = maximum([size(dualpsi[a],3) for a = 1:size(dualpsi,1)])
    if (m < params.maxm || true) && n%3 == 0
      params.noise = params.startnoise
    else
      params.noise = 0.
    end

    return false
  end
end
