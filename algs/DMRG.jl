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






function twosite_update(Lenv::TensType,Renv::TensType,tensors::TensType...)

  psiL = tensors[1]
  psiR = tensors[2]
  mpoL = tensors[3]
  mpoR = tensors[4]

  AA = psiL*psiR
  ops = contract([1,3,5,6,2,4],mpoL,4,mpoR,1)

  Hpsi = contract(ops,(5,6),AA,(2,3))
  LHpsi = contract(Lenv,(2,3),Hpsi,(1,5))
  temp = contract(LHpsi,(5,4),Renv,(1,2))

  return temp
end
#export twosite_update








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
  AA = div!(AA,norm(AA))
  alpha1 = real(ccontract(AA,Hpsi))

  psi2 = sub!(Hpsi, AA, alpha1)
  beta1 = norm(psi2)
  if beta1 > betatest
    psi2 = div!(psi2, beta1)

    Hpsi2 = contract(Lenv,3,psi2,1)
#    if true
      ops = contract(mpoL,4,mpoR,1)
      Hpsi2 = contract(Hpsi2,[2,3,4],ops,[1,2,4])
      Hpsi2 = contract(Hpsi2,[2,5],Renv,[1,2])
#=    else
      Hpsi2 = contract(Hpsi2,(2,3),mpoL,(1,2))
      Hpsi2 = contract(Hpsi2,(5,2),mpoR,(1,2))
      Hpsi2 = contract(Hpsi2,(2,5),Renv,(1,2))
    end
=#
    alpha2 = real(ccontract(psi2,Hpsi2))
    M = Float64[alpha1 beta1; beta1 alpha2]
    D, U = eigen(M)
    energy = D[1,1]
    outAA = tensorcombination!((conj(U[1,1]),conj(U[2,1])),AA,psi2) #conj(U[1,1])*AA + conj(U[2,1])*psi2
  else
    energy = alpha1
    outAA = AA
  end
  return energy,outAA
end


function makeNsite(Lenv::TensType,Renv::TensType,psivec::MPS,mpovec::MPO,iL::intType,iR::intType)

  Lpsi = contract(Lenv,3,psivec[iL],1)

  LHpsi = contract(Lpsi,(2,3),mpovec[iL],(1,2))
  for w = 1:iR-iL
    LHpsi = contract(LHpsi,w+1,psivec[iL+w],1)
    LHpsi = contract(LHpsi,(w+1,w+2),mpovec[iL+w],(1,2))
  end
  G = iR-iL+1
  return contract(LHpsi,(1+G,3+G),Renv,(1,2))
end

function simplelanczos(Lenv::TensType,Renv::TensType,psivec::MPS,mpovec::MPO,iL::intType,iR::intType;betatest::Float64 = 1E-10)
  Hpsi = makeNsite(Lenv,Renv,psivec,mpovec,iL,iR)
  AA = psivec[iL]
  G = length(psivec)
  for w = iL+1:iR
    AA = contract(AA,ndims(AA),psivec[w],1)
  end
  alpha1 = real(ccontract(AA,Hpsi))

  psi2 = sub!(Hpsi, AA, alpha1)
  beta1 = norm(psi2)
  if beta1 > betatest
    psi2 = div!(psi2, beta1)

    Hpsi2 = contract(Lenv,3,psi2,1)
#    if true
      ops = mpovec[iL]
      for w = iL+1:iR
        ops = contract(ops,ndims(ops),mpovec[w],1)
      end
      Hpsi2 = contract(Hpsi2,[p for p = 2:ndims(Hpsi2)-1],ops,vcat([1],[w for w = 2:2:ndims(Hpsi2)]))
      Hpsi2 = contract(Hpsi2,[2,ndims(Hpsi2)],Renv,[1,2])
#    else
#    end

    alpha2 = real(ccontract(psi2,Hpsi2))
    M = Float64[alpha1 beta1; beta1 alpha2]
    D, U = eigen(M)
    energy = D[1,1]
    outAA = tensorcombination!((conj(U[1,1]),conj(U[2,1])),AA,psi2)
  else
    energy = alpha1
    outAA = AA
  end
  return outAA,energy
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

function SvNcheck!(i::Integer,j::Integer,D::Union{TensType,diagonal},Ns::Integer,params::TNparams;alpha::Bool=true,rev::Bool=false)
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
function regular_SvNcheck!(i::Integer,j::Integer,D::TensType,Ns::Integer,params::TNparams;rev::Bool=false)
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
      Rbound = default_boundary
    elseif LRbounds == "R"
      Lbound = default_boundary
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

function Lexpand(A::TensType,ops::TensType,HL::TensType,alpha::Float64)
  Lenvpsi = contract(HL,(3,),A,(1,))
  Hpsi = contract((1,3,4,2),Lenvpsi,(2,3),ops,(1,2),alpha=alpha)
  expAA = reshape!(Hpsi,[[1],[2],[3,4]],merge=true)
  return joinindex!(A,expAA,3)
end

function Rexpand(A::TensType,ops::TensType,HR::TensType,alpha::Float64)
  Renvpsi = contract(A,(3,),HR,(1,))
  Hpsi = contract((3,1,2,4),ops,(2,4),Renvpsi,(2,3),alpha=alpha)
  expAA = reshape!(Hpsi,[[1,2],[3],[4]],merge=true)
  out = joinindex!(A,expAA,1)
  return out
end

const alpha_max = 1.
function alpha_update(alpha::Float64,truncerr::Float64,cutoff::Float64,
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
export alpha_update
#=
function Nsite_update(Lenv::TensType,Renv::TensType,psiops::TensType...)
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
=#

function step3S(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number
                
  currops = mpo[i]

  outEnergy,AAvec,alpha,beta,savepsi = lanczos(psi[i],currops,r=params.r,m=1,updatefct=singlesite_update,Lenv=Lenv[i],Renv=Renv[i])
  noise = params.noise

  AAvec = AAvec[:,:,:,1] #savepsi[1]

  params.energy = outEnergy[1]
  alpha_condition = noise[1] > params.noise_incr

  minm = params.minm
  maxm = params.maxm
  cutoff = params.cutoff

  if j > 0
    tempL = (alpha_condition ? Lexpand(AAvec,currops,Lenv[iL],noise) : AAvec)
    psi[iL],psi[iR],D,truncerr = moveR!(tempL,psi[iR],cutoff=cutoff,m=maxm,minm=minm,condition=alpha_condition)
  else
    tempR = (alpha_condition ? Rexpand(AAvec,currops,Renv[iR],noise) : AAvec)
    psi[iL],psi[iR],D,truncerr = moveL!(psi[iL],tempR,cutoff=cutoff,m=maxm,minm=minm,condition=alpha_condition)
  end

  params.noise = alpha_update(noise,truncerr,params.cutoff,params.lastenergy[1],params.energy[1],params.noise_goal,params.noise_incr)

  params.truncerr = truncerr
  params.biggestm = max(params.biggestm,size(D,1))

#  println()

  if !params.efficient
    SvNcheck!(i,j,D,length(psi),params)
  end
  nothing
end

function twostep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                  psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number

  if j > 0
    psi[iL] = div!(psi[iL],norm(psi[iL]))
  else
    psi[iR] = div!(psi[iR],norm(psi[iR]))
  end

  energy,AA = simplelanczos(Lenv[iL],Renv[iR],psi[iL],psi[iR],mpo[iL],mpo[iR])

#  inAA = psi[iL]*psi[iR]
  checkAA = twosite_update(Lenv[iL],Renv[iR],psi[iL],psi[iR],mpo[iL],mpo[iR])

  params.energy = energy

#  println(n," ",i," ",energy)

#=
  @time if j > 0
    res = ccontract(AA,(1,2,3),AA,(1,2,3))
    D,U = eigen(res)
  end
  =#

  U,D,V,truncerr = fsvd!(AA,[[1,2],[3,4]],m=params.maxm,minm=params.minm,cutoff=params.cutoff,mag=1.)

  if j < 0
    psi[iL] = U*D
    psi[iR] = V
  else
    psi[iL] = U
    psi[iR] = D*V
  end

  params.truncerr = truncerr
  params.biggestm = max(params.biggestm,size(D,1))

  if !params.efficient
    SvNcheck!(i,j,D,length(psi),params,alpha=false)
  end

  nothing
end

function step2S(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number

  if j > 0
    psi[iL] = div!(psi[iL],norm(psi[iL]))
  else
    psi[iR] = div!(psi[iR],norm(psi[iR]))
  end

  energy,AA = simplelanczos(Lenv[iL],Renv[iR],psi[iL],psi[iR],mpo[iL],mpo[iR])

  params.energy = energy

  noise = typeof(AA) <: qarray ? 0. : params.noise
  minm = params.minm
  maxm = params.maxm
  cutoff = params.cutoff

  if j > 0
    rho = contractc(AA,(3,4),AA,(3,4))
  else
    rho = ccontract(AA,(1,2),AA,(1,2))
  end

  #adds noise after first sweep to ensure large enough bond dimension on quantum number version. Some issues for spin-half, m=45 runs with quantum numbers (truncation of full tensor)
  if noise > 1E-12#&& n > 2
    if j > 0
      temp = contract(Lenv[iL],(3,),AA,(1,))
      randrho = contractc(temp,(2,4,5),temp,(2,4,5))
    else
      temp = contract(AA,(4,),Renv[iR],(1,))
      randrho = ccontract(temp,(1,2,4),temp,(1,2,4))
    end
    rho = add!(rho,randrho,params.noise)
#    if !isapprox(norm(prop_rho),0)
#      rho = prop_rho
#    end
  end



  println()
  println(n," ",j," ",i," ",iL," ",iR," ",norm(rho)," ",params.noise," ",energy)



#  if length(rho) == 0
#    rho = rand(rho)
#  end

#  println(size(rho))
#  println(norm(rho))



#  println(params.cutoff," ",params.maxm," ",params.minm)

#  prho = reshape(rho,[[1,2],[3,4]],merge=true)
#  println("check = ",norm(Array(prho)-Array(prho)'))

  D,U,truncerr,sumD = eigen(rho,[[1,2],[3,4]],cutoff=params.cutoff,m=params.maxm,minm=params.minm,transpose = j < 0)

#  println(norm(D))

#  if j < 0
#    checkAA = ccontract(U,1,D*U,1)
#  else
#    checkAA = contractc(U*D,ndims(U),U,ndims(U))
#  end
#  println(size(rho)," ",size(checkAA))
#  println("diff = ",norm(rho-checkAA))

#    pcheckAA = reshape(checkAA,[[1,2],[3,4]],merge=true)
#  println("check check = ",norm(Array(pcheckAA)-Array(pcheckAA)'))


  if j > 0
    psi[iL] = U
    psi[iR] = ccontract(U,(1,2),AA,(1,2))
  else

    psi[iL] = contractc(AA,(3,4),U,(2,3))
    psi[iR] = U
  end

  params.noise = alpha_update(noise,truncerr,params.cutoff,params.lastenergy[1],params.energy[1],params.noise_goal,params.noise_incr)

  params.truncerr = truncerr
  params.biggestm = max(params.biggestm,size(D,1))

  if !params.efficient
    SvNcheck!(i,j,D,length(psi),params,rev=true)
  end
  nothing
end


function dmrgNstep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,Lenv::Env,Renv::Env,
                    psiLenv::Env,psiRenv::Env,beta::Array{Y,1},prevpsi::MPS...;params::TNparams=params()) where Y <: Number

  if j > 0
    psi[iL] = div!(psi[iL],norm(psi[iL]))
  else
    psi[iR] = div!(psi[iR],norm(psi[iR]))
  end

  energy,AA = simplelanczos(Lenv[iL],Renv[iR],psi,mpo,iL,iR)
  params.energy = energy

  truncerr = 0.
  D = 0

  if j > 0
    for p = 1:iR-iL-1
      U,D,psi[iR-p+1],truncerr = svd!(AA,[[w for w = 1:ndims(AA)-2],[ndims(AA)-1,ndims(AA)]],m=params.maxm,minm=params.minm,cutoff=params.cutoff,mag=1.)
      AA = U*D
    end

    psi[iL],D,V,truncerr = svd!(AA,[[w for w = 1:ndims(AA)-2],[ndims(AA)-1,ndims(AA)]],m=params.maxm,minm=params.minm,cutoff=params.cutoff,mag=1.)

    psi[iL+1] = D*V

  else
    for p = 1:iR-iL-1
      psi[iL+p],D,V,truncerr = svd!(AA,[[1,2],[p for p = 3:ndims(AA)]],m=params.maxm,minm=params.minm,cutoff=params.cutoff,mag=1.)
      AA = D*V
    end

    U,D,psi[iR],truncerr = svd!(AA,[[1,2],[p for p = 3:ndims(AA)]],m=params.maxm,minm=params.minm,cutoff=params.cutoff,mag=1.)

    psi[iR-1] = U*D
  end

  params.truncerr = truncerr
  params.biggestm = max(params.biggestm,size(D,1))

  if !params.efficient
    SvNcheck!(i,j,D,length(psi),params,alpha=false)
  end
  nothing
end

function dmrg(psi::MPS,mpo::MPO;params::TNparams = algvars(psi),
                                m::Integer=params.maxm,minm::Integer=2,
                                sweeps::Integer=params.sweeps,cutoff::Float64=params.cutoff,
                                silent::Bool=params.silent,goal::Float64=params.goal,
                                SvNbond::Integer=params.SvNbond,allSvNbond::Bool=params.allSvNbond,
                                nsites::Integer=params.nsites,efficient::Bool=params.efficient,
                                g::Integer=params.g,
                                cvgE::Bool=params.cvgE,r::Integer=params.r,
                                mincr::Integer=params.mincr,fixD::Bool=params.fixD,Lbound::TensType=params.Lbound,Rbound::TensType=params.Rbound,
                                noise::P=params.noise,noise_goal::Float64=params.noise_goal,noise_incr::Float64=params.noise_incr,noise_decay::Float64=params.noise_decay,method::String="twosite",shift::Bool=params.shift,
                                saveEnergy::AbstractArray=params.saveEnergy,halfsweep::Bool=params.halfsweep,Lenv::Env=params.Lenv,Renv::Env=params.Renv,origj::Bool=params.origj,maxshowD::Integer=params.maxshowD,
                                storeD::Array{W,1}=params.storeD,exnum::Integer=params.exnum) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  if params.load
    params.method = method
  end
  if params.method == "3S"
    return dmrg3S(psi,mpo,method=method,m=m,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,r=r,exnum=exnum,fixD=fixD,nsites=1,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
  elseif params.method == "Nsite" #|| nsites > 2
    return dmrg_Nsite(psi,mpo,method=method,m=m,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,r=r,exnum=exnum,fixD=fixD,nsites=nsites,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
  elseif params.method == "twosite"
    return dmrg_twosite(psi,mpo,method=method,m=m,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,r=r,exnum=exnum,fixD=fixD,nsites=2#=nsites=#,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
#    elseif method == "zero" || method == "0"
  elseif params.method == "2S"
    return dmrg2S(psi,mpo,method=method,m=m,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,params=params,
                  SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,r=r,exnum=exnum,fixD=fixD,nsites=2,
                  noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                  halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD)
  else
    error("DMRG method ($method) not defined")
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
    println("  Lanczos iterations = ",params.r)
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
                                  cvgE::Bool=params.cvgE,
                                  r::Integer=params.r,
                                  g::Integer=params.g,
                                  mincr::Integer=params.mincr,fixD::Bool=params.fixD,Lbound::TensType=params.Lbound,Rbound::TensType=params.Rbound,
                                  noise::P=params.noise,noise_goal::Float64=params.noise_goal,noise_incr::Float64=params.noise_incr,noise_decay::Float64=params.noise_decay,method::String="3S",shift::Bool=params.shift,
                                  saveEnergy::AbstractArray=params.saveEnergy,halfsweep::Bool=params.halfsweep,Lenv::Env=params.Lenv,Renv::Env=params.Renv,origj::Bool=params.origj,maxshowD::Integer=params.maxshowD,
                                  storeD::Array{W,1}=params.storeD,exnum::Integer=params.exnum) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  loadvars!(params,"DMRG-"*method,minm,m,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,r,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,stepfct=step3S,cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg3S

#  import ..optimizeMPS.twositeOps
function dmrg_twosite(psi::MPS,mpo::MPO;params::TNparams = algvars(psi),
                                        m::Integer=params.maxm,minm::Integer=params.minm,
                                        sweeps::Integer=params.sweeps,cutoff::Float64=params.cutoff,
                                        silent::Bool=params.silent,goal::Float64=params.goal,
                                        SvNbond::Integer=params.SvNbond,allSvNbond::Bool=params.allSvNbond,
                                        r::Integer=params.r,
                                        g::Integer=params.g,
                                        nsites::Integer=2,#=params.nsites,=#efficient::Bool=params.efficient,
                                        cvgE::Bool=params.cvgE,
                                        mincr::Integer=params.mincr,fixD::Bool=params.fixD,Lbound::TensType=params.Lbound,Rbound::TensType=params.Rbound,
                                        noise::P=params.noise,noise_goal::Float64=params.noise_goal,noise_incr::Float64=params.noise_incr,noise_decay::Float64=params.noise_decay,method::String="twosite",shift::Bool=params.shift,
                                        saveEnergy::AbstractArray=params.saveEnergy,halfsweep::Bool=params.halfsweep,Lenv::Env=params.Lenv,Renv::Env=params.Renv,origj::Bool=params.origj,maxshowD::Integer=params.maxshowD,
                                        storeD::Array{W,1}=params.storeD,exnum::Integer=params.exnum) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  loadvars!(params,"DMRG-"*method,minm,m,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,r,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,measfct=startval,stepfct=twostep,#=makeOps=twositeOps,=#cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg_twosite

#  import ..optimizeMPS.NsiteOps
function dmrg_Nsite(psi::MPS,mpo::MPO;m::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                  SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::Integer=2,efficient::Bool=false,params::TNparams = algvars(psi),
                  cvgE::Bool=true,mincr::Integer=2,mperiod::Integer=0,fixD::Bool=false,Lbound::TensType=default_boundary,Rbound::TensType=default_boundary,
                  r::Integer=params.r,
                  g::Integer=param.g,
                  noise::P=1.0,noise_goal::Float64=0.3,noise_incr::Float64=params.noise_incr,noise_decay::Float64=params.noise_decay,method::String="Nsite",shift::Bool=params.shift,
                  saveEnergy::AbstractArray=[0.],halfsweep::Bool=false,Lenv::Env=params.Lenv,Renv::Env=params.Renv,origj::Bool=true,maxshowD::Integer=params.maxshowD,
                  storeD::Array{W,1}=params.storeD,alpha_decay::Float64=0.9,exnum::Integer=params.exnum) where {P <: Union{Number,Array{Float64,1}}, W <: Number}
  loadvars!(params,"DMRG-"*method*" (N=$nsites)",minm,m,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,r,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,measfct=startval,stepfct=dmrgNstep,#=makeOps=NsiteOps,=#cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg_Nsite

function dmrg2S(psi::MPS,mpo::MPO;params::TNparams = algvars(psi),
                  m::Integer=params.maxm,minm::Integer=params.minm,
                  sweeps::Integer=params.sweeps,cutoff::Float64=params.cutoff,
                  silent::Bool=params.silent,goal::Float64=params.goal,
                  SvNbond::Integer=params.SvNbond,allSvNbond::Bool=params.allSvNbond,
                  nsites::Integer=params.nsites,efficient::Bool=params.efficient,
                  cvgE::Bool=params.cvgE,
                  r::Integer=params.r,
                  g::Integer=param.g,
                  mincr::Integer=params.mincr,fixD::Bool=params.fixD,Lbound::TensType=params.Lbound,Rbound::TensType=params.Rbound,
                  noise::P=params.noise,noise_goal::Float64=params.noise_goal,noise_incr::Float64=params.noise_incr,noise_decay::Float64=params.noise_decay,method::String="2S",shift::Bool=params.shift,
                  saveEnergy::AbstractArray=params.saveEnergy,halfsweep::Bool=params.halfsweep,Lenv::Env=params.Lenv,Renv::Env=params.Renv,origj::Bool=params.origj,maxshowD::Integer=params.maxshowD,
                  storeD::Array{W,1}=params.storeD,exnum::Integer=params.exnum) where {P <: Union{Number,Array{Float64,1}}, W <: Number}

  if eltype(psi) <: qarray
    error("deprecated DMRG-2S for quantum numbers")
  end
  loadvars!(params,"DMRG-"*method,minm,m,sweeps,cutoff,silent,goal,SvNbond,allSvNbond,efficient,cvgE,r,fixD,nsites,
      noise,noise_decay,noise_goal,noise_incr,saveEnergy,halfsweep,Lbound,Rbound,Lenv,Renv,psi.oc,origj,maxshowD,storeD,exnum)
  return optmps(psi,psi,mpo,[1.],params=params,stepfct=step2S,#=makeOps=NsiteOps,=#cvgfct=dmrgcvg,displayfct=dmrginformation)
end
export dmrg2S
#end
