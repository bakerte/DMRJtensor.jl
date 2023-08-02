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

#=
module mpstime
#using ..shuffle
using ..tensor
using ..Qtensor
using ..Qtask
using ..MPutil
using ..QN
using ..Opt
using ..contractions
using ..decompositions
=#



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


@inline function joinedtwosite_update(Lenv::TensType,Renv::TensType,tensors::TensType...)

  AA = tensors[1]
  ops = tensors[2]

  Hpsi = contract(ops,(5,6),AA,(2,3))
  LHpsi = contract(Lenv,(2,3),Hpsi,(1,5))
  temp = contract(LHpsi,(5,4),Renv,(1,2))

  return temp
end


#           +---------------------------------+
#>----------| Time evolving block decimation  |----------<
#           +---------------------------------+

  const timeType = Union{ComplexF64,Float64}

  function makegates(H::MPO)
    retType = eltype(H[1])
    if typeof(H[1]) <: AbstractArray
      thistype = Array{retType,4}
    else
      thistype = denstens
    end
    gates = Array{thistype,1}(undef,length(H)-1)
    for i = 1:length(H)-1
      ops = contract(H.H[i],ndims(H.H[i]),H.H[i+1],1)
      if 1 < i < length(H)-1
        tops = ops[size(ops,1),:,:,:,:,1]
      elseif i == 1
        tops = ops[1,:,:,:,:,1]
      elseif i == length(H)-1
        tops = ops[size(ops,1),:,:,:,:,1]
      end
      gates[i] = permutedims(tops,[1,3,2,4])
    end
    return gates
  end
  export makegates

  function makeExpGates(H::MPO,prefactor::timeType)
    gates = makegates(H)
    retType = typeof(eltype(gates[1])(1)*prefactor)
    if typeof(H[1]) <: AbstractArray
      thistype = Array{retType,4}
    else
      thistype = tens{retType}
    end
    expgates = Array{thistype,1}(undef,length(gates))
    for j = 1:length(gates)
      sizegates = size(gates[j])
      Lsize = sizegates[1]*sizegates[2]
      Rsize = sizegates[3]*sizegates[4]
      ops = reshape(gates[j],Lsize,Rsize)
      if eltype(ops) <: retType
        ops = tensorcombination!((prefactor,),ops)
      else
        ops = tensorcombination!((prefactor,),convertTens(retType,ops))
      end
#      ops *= prefactor
      ops = exp(ops)
      expgates[j] = reshape(ops,sizegates[1],sizegates[2],sizegates[3],sizegates[4])
    end
    return expgates
  end

  function makeExpGates(Qlabels::Array{Array{Q,1},1},H::MPO,prefactor::G) where {Q <: Qnum, G <: timeType}
    expgates = makeExpGates(H,prefactor)
    W = elnumtype(prefactor,H,expgates...) #*elnumtype(prefactor)(1))
    qgates = Array{Qtens{W,Q},1}(undef, length(expgates))
    zeroQN = typeof(Qlabels[1][1])() # + inv(Qlabels[1][1])
    newQsizes = Array{intType,1}[[1],[2],[3],[4]]
    numQs = length(Qlabels)
    for i = 1:length(expgates)
      ind = (i-1) % numQs + 1
      indpone = i % numQs + 1
      newQnumMat = [inv.(Qlabels[ind]),inv.(Qlabels[indpone]),Qlabels[ind],Qlabels[indpone]]
      qgates[i] = Qtens(expgates[i],newQnumMat)
    end
    return qgates
  end
  export makeExpGates

  function qmakeExpGates(H::MPO,Qlabels::Array{Array{Q,1},1},prefactor::timeType) where Q <: Qnum
    return makeExpGates(Qlabels,H,prefactor)
  end

  function qmakeExpGates(H::MPO,Qlabels::Array{Q,1},prefactor::timeType) where Q <: Qnum
    return makeExpGates([Qlabels],H,prefactor)
  end
  export qmakeExpGates

  function tebd(psi::MPS,expgates::Array{R,1};cutoff::Float64=0.,m::Integer=0) where R <: TensType
    if eltype(expgates[1]) != eltype(psi[1])
      psi = MPS(eltype(expgates[1]),psi)
    end


    Ns = length(psi)
    if psi.oc == Ns
      j = -1
    else #if psi.oc == 1
      j = 1
    end



    for i = 1:Ns-1
      if j == 1
        iL,iR = psi.oc,psi.oc+1
      else
        iL,iR = psi.oc-1,psi.oc
      end
      AA = contract(psi[iL],3,psi[iR],1)
      AA = contract([1,3,4,2],AA,[2,3],expgates[i],[1,2])
      U,D,V = svd(AA,[[1,2],[3,4]],cutoff=cutoff,m=m)
      if j == 1
        psi[iL] = U 
        psi[iR] = contract(D,2,V,1)
        norm!(psi[iR])
      else
        psi[iR] = V
        psi[iL] = contract(U,3,D,1)
        norm!(psi[iL])
      end
      psi.oc += j

      if psi.oc == Ns || psi.oc == 1
        j *= -1
      end
    end
    return psi
  end





#=
    #forward
    move!(cpsi,1)
    for i = 1:length(cpsi)-1
      AA = contract(cpsi.A[i],3,cpsi.A[i+1],1)
      AA = contract([1,3,4,2],AA,[2,3],expgates[i],[1,2])
      cpsi[i],D,V = svd(AA,[[1,2],[3,4]],cutoff=cutoff,m=m)
      cpsi[i+1] = contract(D,2,V,1)
      cpsi[i+1] /= norm(cpsi[i+1])
    end
    #backward
    for i = length(cpsi)-1:-1:1
      AA = contract(cpsi.A[i],3,cpsi.A[i+1],1)
      AA = contract([1,3,4,2],AA,[2,3],expgates[i],[1,2])
      U,D,cpsi[i+1] = svd(AA,[[1,2],[3,4]],cutoff=cutoff,m=m)
      cpsi[i] = contract(U,3,D,1)
      cpsi[i] /= norm(cpsi[i])
    end
    return cpsi
  end
  =#
  export tebd



#           +---------------------------------------+
#>----------| Time dependent variational principle  |----------<
#           +---------------------------------------+







  function krylovTimeEvol(psiops::TensType...;prefactor::Number=1,lanczosfct::Function=lanczos,maxiter::Integer=2,updatefct::Function=makeHpsi,Lenv::TensType=eltype(psiops[1])[0],Renv::TensType=Lenv)

#    println(typeof(psiops)," ",typeof(psiops[1])," ",typeof(Lenv)," ",typeof(Renv))

    alpha = Array{Float64,1}(undef,maxiter)
    beta = Array{Float64,1}(undef,maxiter)

    psivec,energies = lanczosfct(psiops...,maxiter=maxiter,retnum=maxiter,updatefct=updatefct,Lenv=Lenv,Renv=Renv,alpha=alpha,beta=beta)
    M = LinearAlgebra.SymTridiagonal(alpha,beta)
    En,U = LinearAlgebra.eigen(M)
    expH = exp(M,prefactor)
    if isapprox(sum(w->expH[w,1],1:size(expH,1)),0) #|| isapprox(maximum(expH),Inf)
      expH = LinearAlgebra.I + LinearAlgebra.SymTridiagonal(alpha,beta)
    end
    coeffs = ntuple(w->convert(eltype(psivec[1]),expH[w,1]),length(psivec)) #works for any orthogonal basis, else (expH*overlaps)
    keepers = [!isapprox(coeffs[w],0) for w = 1:length(coeffs)]
    if sum(keepers) < length(coeffs)
      psivec = psivec[keepers]
      coeffs = coeffs[keepers]
    end
    newpsi = tensorcombination!(psivec...,alpha=coeffs)
    return newpsi,En[1]
  end
  export krylovTimeEvol

  function OC_update(Lenv::TensType,Renv::TensType,psiops::TensType...)
    D = psiops[1]
    LenvD = contract(Lenv,3,D,1)
    return contract(LenvD,(2,3),Renv,(2,1))
  end
  
  #  import ..optimizeMPS.singlesite_update
    function tdvp(psi::MPS,mpo::MPO;maxm::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),minm::Integer=2,cutoff::Real=0.,origj::Bool=true,maxiter::Integer=2,prefactor::Number=1.)

      timestep = prefactor/length(psi)
      
      dualpsi = psi
#      if Lenv == [0] && Renv == [0]
      Lenv,Renv = makeEnv(psi,psi,mpo)
#      end

      Ns = length(psi)
      if psi.oc == 1
        j = 1
      elseif psi.oc == Ns #
        j = -1
      else
        j = origj ? 1 : -1 #if the oc is away from the edge, j could be externally specified.
      end

      range = 0
      En = 0
  
      Nsteps = (Ns-1)
      for n = 1:Nsteps

        i = psi.oc

        if j > 0
          iL,iR = i,i+1
        else
          iL,iR = i-1,i
        end

        norm!(psi[i])
        if j > 0

          newpsi,En = krylovTimeEvol(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=singlesite_update,Lenv=Lenv[i],Renv=Renv[i])
          norm!(newpsi)

          psi[i],D,V = svd(newpsi,[[1,2],[3]],m=maxm,minm=minm,cutoff=cutoff)
          DV = contract(D,2,V,1)
  
          Lenv[iR] = Lupdate(Lenv[iL],psi[iL],psi[iL],mpo[iL])
          norm!(DV)
          nextDV,En = krylovTimeEvol(DV,prefactor=timestep,maxiter=maxiter,updatefct=OC_update,Lenv=Lenv[iR],Renv=Renv[iL])
  
          psi[iR] = contract(nextDV,2,psi[iR],1)
          norm!(psi[iR])
  
        else
  
          newpsi,En = krylovTimeEvol(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=singlesite_update,Lenv=Lenv[i],Renv=Renv[i])
          norm!(newpsi)

          U,D,psi[i] = svd(newpsi,[[1],[2,3]],m=maxm,minm=minm,cutoff=cutoff)
          UD = contract(U,2,D,1)
  
          Renv[iL] = Rupdate(Renv[iR],psi[iR],psi[iR],mpo[iR])
          norm!(UD)
          nextUD,En = krylovTimeEvol(UD,prefactor=timestep,maxiter=maxiter,updatefct=OC_update,Lenv=Lenv[iR],Renv=Renv[iL])
  
          psi[iL] = contract(psi[iL],3,nextUD,1)
          norm!(psi[iL])
  
        end
      
#        params.truncerr = truncerr
#        params.biggestm = max(params.biggestm,size(D,1))
  
        psi.oc += j
  
        if psi.oc == Ns - (range-1) && j > 0
          j *= -1
        elseif psi.oc == 1 + (range-1) && j < 0
          j *= -1
        end
      end
    
      return En
    end
    export tdvp


    function tdvp_twosite(psi::MPS,mpo::MPO;maxm::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),minm::Integer=2,cutoff::Real=0.,origj::Bool=true,maxiter::Integer=2,prefactor::Number=1.)

      timestep = prefactor/length(psi)
      
      dualpsi = psi
#      if Lenv == [0] && Renv == [0]
      Lenv,Renv = makeEnv(psi,psi,mpo)
#      end

      En = 0.
      range = 1
      ops = twositeOps(mpo,range+1)

      Ns = length(psi)
      if psi.oc == 1
        j = 1
      elseif psi.oc == Ns #
        j = -1
      else
        j = origj ? 1 : -1 #if the oc is away from the edge, j could be externally specified.
      end

  
      Nsteps = (Ns-1)
      for n = 1:Nsteps

        i = psi.oc

        if j > 0
          iL,iR = i,i+1
        else
          iL,iR = i-1,i
        end

        norm!(psi[i])
        AA = contract(psi[iL],3,psi[iR],1)
        if j > 0

          newpsi,En = krylovTimeEvol(AA,ops[iL],prefactor=timestep,maxiter=maxiter,updatefct=joinedtwosite_update,Lenv=Lenv[iL],Renv=Renv[iR])
          norm!(newpsi)

          psi[iL],D,V = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
          DV = contract(D,2,V,1)
  
          Lenv[iR] = Lupdate(Lenv[iL],psi[iL],psi[iL],mpo[iL])
          norm!(DV)

          psi[iR],En = krylovTimeEvol(DV,mpo[iR],prefactor=timestep,maxiter=maxiter,updatefct=singlesite_update,Lenv=Lenv[iR],Renv=Renv[iR])
          norm!(psi[iR])

        else
  
          newpsi,En = krylovTimeEvol(AA,ops[iL],prefactor=timestep,maxiter=maxiter,updatefct=joinedtwosite_update,Lenv=Lenv[iL],Renv=Renv[iR])
          norm!(newpsi)

          U,D,psi[iR] = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
          UD = contract(U,3,D,1)
  
          Renv[iL] = Rupdate(Renv[iR],psi[iR],psi[iR],mpo[iR])
          norm!(UD)
          
          psi[iL],En = krylovTimeEvol(UD,mpo[iL],prefactor=timestep,maxiter=maxiter,updatefct=singlesite_update,Lenv=Lenv[iL],Renv=Renv[iL])
          norm!(psi[iL])
  
        end
      
#        params.truncerr = truncerr
#        params.biggestm = max(params.biggestm,size(D,1))
  
        psi.oc += j
  
        if psi.oc == Ns - (range-1) && j > 0
          j *= -1
        elseif psi.oc == 1 + (range-1) && j < 0
          j *= -1
        end
      end
    
      return En
    end
    export tdvp_twosite


#           +------------------------------------+
#>----------| Krylov methods (local and global)  |----------<
#           +------------------------------------+

function localkrylov(psi::MPS,mpo::MPO;nrounds::intType=1)
  dualpsi = psi

  Lenv,Renv = makeEnv(dualpsi,psi,mpo)
  
  Ns = length(psi)
  if psi.oc == 1
    j = 1
  elseif psi.oc == Ns #
    j = -1
  else
    j = origj ? 1 : -1 #if the oc is away from the edge, j could be externally specified.
  end

  range = 2
  ops = twositeOps(mpo,range)

  Nsteps = (Ns-1)
  for n = 1:Nsteps

    i = psi.oc

    if j > 0
      iL,iR = i,i+1
    else
      iL,iR = i-1,i
    end

    psi[i] = div!(psi[i],norm(psi[i]))
    if j > 0
      if iR < Ns

        U,D,V = svd(psi[iL],[[1,2],[3]],m=maxm,minm=minm,cutoff=cutoff)
        P = copy(U)
        DV = contract(D,2,V,1)
        psi[iR] = contract(DV,2,psi[iR],1)

        AA = contract(U,3,psi[iR],1)

        newpsi = krylovTimeEvol(AA,ops[iL],prefactor=prefactor,maxiter=maxiter,updatefct=joinedtwosite_update,Lenv=Lenv[iL],Renv=Renv[iR])

        psi[iL],D,V = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
        Lenv[iR] = Lupdate(Lenv[iL],dualpsi[iL],psi[iL],mpo[iL])






        Pmod = ccontract(P,(1,2),psi[iL],(1,2))
        psi[iR] = contract(Pmod,2,psi[iR],1)


        psi[iR],psi[iR+1] = moveR(psi[iR],psi[iR+1],m=maxm,minm=minm,cutoff=cutoff)
      else
        AA = contract(psi[iL],3,psi[iR],1)
        newpsi = krylovTimeEvol(AA,ops[iL],prefactor=prefactor,maxiter=maxiter,updatefct=joinedtwosite_update,Lenv=Lenv[iL],Renv=Renv[iR])

        psi[iL],D,V = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
        Lenv[iR] = Lupdate(Lenv[iL],dualpsi[iL],psi[iL],mpo[iL])
        psi[iR] = contract(D,2,V,1)
      end

    else

      if iL > 1

        U,D,V = svd(psi[iR],[[1],[2,3]],m=maxm,minm=minm,cutoff=cutoff)
        P = copy(V)
        UD = contract(U,2,D,1)
        psi[iL] = contract(psi[iL],3,UD,1)

        AA = contract(psi[iR],3,V,1)

        newpsi = krylovTimeEvol(AA,ops[iL],prefactor=prefactor,maxiter=maxiter,updatefct=joinedtwosite_update,Lenv=Lenv[iL],Renv=Renv[iR])

        U,D,psi[iR] = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
        Renv[iL] = Rupdate(Renv[iR],dualpsi[iR],psi[iR],mpo[iR])






        Pmod = ccontract(P,(2,3),psi[iR],(2,3))
        psi[iL] = contract(psi[iL],3,Pmod,1)


        psi[iL-1],psi[iL] = moveL(psi[iL-1],psi[iL],m=maxm,minm=minm,cutoff=cutoff)
      else
        AA = contract(psi[iL],3,psi[iR],1)
        newpsi = krylovTimeEvol(AA,ops[iL],prefactor=prefactor,maxiter=maxiter,updatefct=joinedtwosite_update,Lenv=Lenv[iL],Renv=Renv[iR])

        U,D,psi[iR] = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
        Renv[iL] = Rupdate(Renv[iR],dualpsi[iR],psi[iR],mpo[iR])
        psi[iL] = contract(U,3,D,1)
      end

    end






    psi.oc += j
    
    if psi.oc == Ns - (range-1) && j > 0
      j *= -1
    elseif psi.oc == 1 + (range-1) && j < 0
      j *= -1
    end
  end
  nothing
end





  function globalkrylov(psi::MPS,mpo::MPO;nrounds::intType=1)
    retType = typeof(eltype(psi[1])(1) * eltype(mpo[1])(1))
    alpha = Array{retType,1}(undef,nrounds)
    beta = Array{retType,1}(undef,nrounds-1)
    alpha[1] = expect(psi,mpo,mpo)
    newpsi = Array{MPS,1}(undef,nrounds)
    newpsi[1] = psi
    for i = 1:nrounds-1
      beta[i] = sqrt(expect(newpsi[i]))

      #could possibly combine mpo with alpha[i] for efficiency...check this again
      if i > 1
        newpsi[i+1] = optimizer(psi,mpo,[alpha[i],beta[i]],[newpsi[i],newpsi[i-1]])
      else
        newpsi[i+1] = optimizer(psi,mpo,[alpha[i]],[newpsi[i]])
      end
      alpha[i+1] = expect(newpsi[i+1],mpo,mpo)
    end
    beta[end] = sqrt(expect(newpsi[end-1]))
    return alpha,beta
  end

#           +--------------------+
#>----------| W1 and W2 methods  |----------<
#           +--------------------+

#=
W1 is so bad that 1901.56... doesn't even comapre against it

deltaT = 0.001

function tMPO(i::Int64)
onsite(i::Int64) = (mu * Ndens + HubU * Nup * Ndn)*(i!=1 && i!= Ns ? 0.5 : 1)
        return [Id  O O O O O;
            Cup' O O O O O;
            Cup  O O O O O;
            Cdn' O O O O O;
            Cdn  O O O O O;
            onsite(i) -t*Cup*F conj(t)*Cup'*F -t*Cdn*F conj(t)*Cdn'*F Id]
    end

    tH = convert2MPO(tMPO,4,Ns)#,type=ComplexF64)

    dT = deltaT*im
    sdT = sqrt(dT)

    function checkZ1(i::Int64)
      onsite(i::Int64) = (mu * Ndens + HubU * Nup * Ndn)*(i!=1 && i!= Ns ? 0.5 : 1)

      return [Id + dT * onsite(i) -t*dT * Cup*F conj(t)*dT * Cup'*F -t*dT * Cdn*F conj(t)*dT * Cdn'*F;
              Cup' O O O O;
              Cup  O O O O;
              Cdn' O O O O;
              Cdn O O O O]
    end
    zmpo = convert2MPO(checkZ1,4,Ns,saveL=1)
    =#


#           +--------+
#>----------|  TDVP  |----------<
#           +--------+
#=
function center_update(psi::TensType,mpo::TensType,Lenv::TensType,Renv::TensType)
  CLenv = contract(Lenv,3,psi,1)
  return contract(CLenv,[4,2],Renv,[1,2])
end

  function tdvpstep(n::Integer,j::Integer,i::Integer,iL::Integer,iR::Integer,dualpsi::MPS,psi::MPS,mpo::MPO,
                  Lenv::AbstractArray,Renv::AbstractArray,infovals::Array{P,1},lastenergy::Array{R,1},
                  psiLenv::AbstractArray,psiRenv::AbstractArray,beta::Array{W,1},prevpsi::MPS...;exnum::Integer=1,
                  noise::Float64=0.,maxiter::Integer=2,noise_goal::Float64=0.3,noise_incr::Float64=0.01,cutoff::Float64=0.,
                  maxm::Integer=0,minm::Integer=2) where {W <: Number, P <: Number, R <: Number}




    newLenv = ccontract(psiLenv[1][i],2,Lenv[i],1)
    newRenv = contractc(Renv[i],3,psiRenv[1][i],1)

    newAAvec,infovals[1] = krylov(singlesite_update,psi[i],mpo[i],newLenv,newRenv,maxiter=maxiter)

    dualpsi[i] = newAAvec[1]

    if j > 0

      dualpsi[iL],D,V = svd(dualpsi[i],[[1,2],[3]],m=maxm,minm=minm,cutoff=cutoff)
      dualpsi[iR] = contract(V,2,dualpsi[iR],1)

      newRenv = contract(newRenv,3,V,1)

      interLenv = Lupdate(Lenv[i],dualpsi[i],psi[i],mpo[i])
      internewLenv = ccontract(psiLenv[1][i],2,Lenv[i],1)

      move!(psi[i],psi[i].oc+j)
      newCenter,infovals[1] = krylov(center_update,psi[i],mpo[i],internewLenv,newRenv,maxiter=maxiter)
      dualpsi[iR] = contract(newCenter,2,dualpsi[iR],1)
    else

      U,D,dualpsi[iR] = svd(dualpsi[i],[[1],[2,3]],m=maxm,minm=minm,cutoff=cutoff)
      dualpsi[iL] = contract(dualpsi[iL],3,U,1)

      newLenv = contract(U,1,newLenv,1)

      interRenv = Rupdate(Renv[i],dualpsi[i],psi[i],mpo[i])
      internewRenv = contractc(Renv[i],3,psiRenv[1][i],1)

      move!(psi[i],psi[i].oc+j)
      newCenter,infovals[1] = krylov(center_update,psi[i],mpo[i],newLenv,internewRenv,maxiter=maxiter)
      dualpsi[iR] = contract(newCenter,2,dualpsi[iR],1)
    end

    dualpsi.oc += j

    return D,truncerr
  end
  export optstep
  =#
#=
  function tdvp(psi::MPS,mpo::MPO;maxm::Integer=0,minm::Integer=2,sweeps::Integer=1,cutoff::Float64=0.,silent::Bool=false,goal::Float64=0.,
                    infovals::Array{Float64,1}=zeros(5),SvNbond::Integer=fld(length(psi),2),allSvNbond::Bool=false,nsites::intType=2,efficient::Bool=false,
                    cvgE::Bool=true,maxiter::Integer=2,mincr::Integer=2,mperiod::Integer=0,exnum::Integer=1,fixD::Bool=false,Lbound::AbstractArray=[0],Rbound::AbstractArray=[0],
                    noise::Number=1.0,noise_goal::Float64=0.3,noise_incr::Float64=0.01,noise_decay::Float64=0.9,method::String="TDVP",shift::Bool=false,
                    saveEnergy::AbstractArray=[0],halfsweep::Bool=false,Lenv::TensType=[0],Renv::TensType=[0],origj::Bool=true,maxshowD::Integer=8,storeD::Array{W,1}=[0.],alpha_decay::Float64=0.9,
                    partitions::Integer=Threads.nthreads()) where {B <: TensType, W <: Number}
    return optmps(psi,mpo,[1.],psi,method=method,maxm=maxm,minm=minm,sweeps=sweeps,cutoff=cutoff,silent=silent,goal=goal,infovals=infovals,
                    SvNbond=SvNbond,allSvNbond=allSvNbond,efficient=efficient,cvgE=cvgE,maxiter=maxiter,exnum=exnum,fixD=fixD,nsites=nsites,
                    noise=noise,noise_decay=noise_decay,noise_goal=noise_goal,noise_incr=noise_incr,shift=shift,saveEnergy=saveEnergy,
                    halfsweep=halfsweep,Lbound=Lbound,Rbound=Rbound,Lenv=Lenv,Renv=Renv,origj=origj,maxshowD=maxshowD,storeD=storeD,
                    #=displayfct=displayinformation,Envfct=setEnv,=#measfct=expect,stepfct=tdvpstep,SvNfct=SvNcheck!#=,
                    makeOps=twositeOps,SvNfct=SvNcheck!,cvgfct=standardcvg=#)
  end
  export tdvp
=#

  function globalkrylov(psi::MPS,mpo::MPO;nextpsi::MPS=copy(psi),nlevels::intType=1)
    retType = typeof(eltype(psi)(1)*eltype(mpo)(1)*eltype(nextpsi)(1))
    alpha = Array{retType,1}(undef,nlevels)
    beta = Array{retType,1}(undef,nlevels-1)
    psivec = Array{MPS,1}(undef,nlevels)
    psivec[1] = psi
    alpha[1] = expect(psi,mpo,mpo) #or is that c?

    #
    # make parameters
    #

    alphampo = mpo + alpha[1]
    optmps(nextpsi,psivec[1],alphampo;params=params) #nextpsi
    psivec[2] = copy(nextpsi)
    beta[1] = norm(psivec[2])
    for i = 3:nlevels-1
      alphampo = mpo + alpha[i-1]
      optmps(psivec[i+1],psivec[i],alphampo,beta[i-2],psivec[i-1]...;params=params)
      psivec[i+1] = copy(psivec[i])
      beta[i] = norm(psivec[i+1])
    end
    alphampo = mpo + alpha[end-1]
    optmps(psivec[end],psivec[end-1],alphampo,beta[i-2],psivec[end-2]...;params=params)
    beta[end] = norm(psivec[end])
    return alpha,beta,psivec
  end
  export globalkrylov

#end















function XOC_update(Lenv::TensType,Renv::TensType,psiops::TensType...)
  D = psiops[1]
  LenvD = contract(Lenv,3,D,1)
  return contract([1,3,2],LenvD,(2,3),Renv,(2,1))
end

function blockkrylovTimeEvol(psiops::TensType...;prefactor::Number=1,lanczosfct::Function=blocklanczos,maxiter::Integer=2,updatefct::Function=XOC_update,Lenv::TensType=eltype(psiops[1])[0],Renv::TensType=Lenv)
  alpha,beta,psivec,p = lanczosfct(psiops...,updatefct=updatefct,maxiter=maxiter,Lenv=Lenv,Renv=Renv)

  LanczosH = makeM(alpha,beta,p)
  expH = exp(LanczosH,prefactor)


  nex = size(psiops[1],ndims(psiops[1]))
  D,U,gs_psi = diagonalizeM(LanczosH,nex,psivec)#,qselect = qselect)
#  retpsi = contract(newpsi, ndims(newpsi), U, 1)



truncexpH = expH[:,1:nex]

retpsi = contract(psivec,ndims(psivec),truncexpH,1)

  return retpsi,[D[i,i] for i = 1:size(D,1)]
end
export blockkrylovTimeEvol

function block_tdvp(psi::MPS,mpo::MPO;maxm::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),minm::Integer=2,cutoff::Real=0.,origj::Bool=true,maxiter::Integer=2,prefactor::Number=1.)

  timestep = prefactor/length(psi)
  
  dualpsi = psi
#      if Lenv == [0] && Renv == [0]
  Lenv,Renv = makeEnv(dualpsi,psi,mpo)
#      end

  Ns = length(psi)
  if psi.oc == 1
    j = 1
  elseif psi.oc == Ns #
    j = -1
  else
    j = origj ? 1 : -1 #if the oc is away from the edge, j could be externally specified.
  end

  range = 1

  Nsteps = (Ns-1)
  for n = 1:Nsteps

    i = psi.oc

    if j > 0
      iL,iR = i,i+1
    else
      iL,iR = i-1,i
    end

#    norm!(psi[i])
    if j > 0

      newpsi,En = blockkrylovTimeEvol(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=ex3S_update,Lenv=Lenv[i],Renv=Renv[i])
      newpsi,D,V = normalization!(newpsi,size(newpsi),((1,2,3),(4,)))

      psi[i],D,V = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
      DV = contract(D,2,V,1)

      Lenv[iR] = Lupdate(Lenv[iL],dualpsi[iL],psi[iL],mpo[iL])
      DV,D,V = normalization!(DV,size(DV),((1,2),(3,)))
      nextDV,En = blockkrylovTimeEvol(DV,prefactor=timestep,maxiter=maxiter,updatefct=XOC_update,Lenv=Lenv[iR],Renv=Renv[iL])

      psi[iR] = contract([1,3,4,2],nextDV,2,psi[iR],1)
      psi[iR],D,V = normalization!(psi[iR],size(psi[iR]),((1,2,3),(4,)))

    else

      newpsi,En = blockkrylovTimeEvol(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=ex3S_update,Lenv=Lenv[i],Renv=Renv[i])
      newpsi,D,V = normalization!(newpsi,size(newpsi),((1,2,3),(4,)))

      U,D,psi[i] = svd(newpsi,[[1,4],[2,3]],m=maxm,minm=minm,cutoff=cutoff)
      UD = contract([1,3,2],U,3,D,1)

      Renv[iL] = Rupdate(Renv[iR],dualpsi[iR],psi[iR],mpo[iR])
      UD,D,V = normalization!(UD,size(UD),((1,2),(3,)))
      nextUD,En = blockkrylovTimeEvol(UD,prefactor=timestep,maxiter=maxiter,updatefct=XOC_update,Lenv=Lenv[iR],Renv=Renv[iL])

      psi[iL] = contract(psi[iL],3,nextUD,1)
      psi[iL],D,V = normalization!(psi[iL],size(psi[iL]),((1,2,3),(4,)))

    end
  
#        params.truncerr = truncerr
#        params.biggestm = max(params.biggestm,size(D,1))

    psi.oc += j

    if psi.oc == Ns - (range-1) && j > 0
      j *= -1
    elseif psi.oc == 1 + (range-1) && j < 0
      j *= -1
    end
  end

  nothing
end
export block_tdvp



function XOC_update_exfull(Lenv::TensType,Renv::TensType,psiops::TensType...)
  D = psiops[1]
  LenvD = contract(Lenv,3,D,1)
  return contract([1,3,2],LenvD,(2,3),Renv,(2,1))
end

function exblockkrylovTimeEvol(psiops::TensType...;prefactor::Number=1,lanczosfct::Function=blocklanczos,maxiter::Integer=2,updatefct::Function=XOC_update_exfull,Lenv::TensType=eltype(psiops[1])[0],Renv::TensType=Lenv)
  Hpsi = updatefct(Lenv,Renv,psiops...)
  contuple = ntuple(i->i,ndims(psiops[1])-1)
  LanczosH = ccontract(psiops[1],contuple,Hpsi,contuple)

  expH = exp(LanczosH,prefactor)


  nex = size(psiops[1],ndims(psiops[1]))
  D,U,gs_psi = diagonalizeM(LanczosH,nex,psiops[1])#,qselect = qselect)

  retpsi = contract(psiops[1],ndims(psiops[1]),expH,1)

  return retpsi,[D[i,i] for i = 1:size(D,1)]
end
export exblockkrylovTimeEvol

function block_tdvp_exfull(psi::MPS,mpo::MPO;maxm::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),minm::Integer=2,cutoff::Real=0.,origj::Bool=true,maxiter::Integer=2,prefactor::Number=1.)

  timestep = prefactor/length(psi)
  
  dualpsi = psi
#      if Lenv == [0] && Renv == [0]
  Lenv,Renv = makeEnv(dualpsi,psi,mpo)
#      end

  Ns = length(psi)
  if psi.oc == 1
    j = 1
  elseif psi.oc == Ns #
    j = -1
  else
    j = origj ? 1 : -1 #if the oc is away from the edge, j could be externally specified.
  end

  range = 1

  Nsteps = (Ns-1)
  for n = 1:Nsteps

    i = psi.oc

    if j > 0
      iL,iR = i,i+1
    else
      iL,iR = i-1,i
    end

#    norm!(psi[i])
    if j > 0

      newpsi,En = exblockkrylovTimeEvol(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=ex3S_update,Lenv=Lenv[i],Renv=Renv[i])
      newpsi,D,V = normalization!(newpsi,size(newpsi),((1,2,3),(4,)))

      psi[i],D,V = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
      DV = contract(D,2,V,1)

      Lenv[iR] = Lupdate(Lenv[iL],dualpsi[iL],psi[iL],mpo[iL])
      DV,D,V = normalization!(DV,size(DV),((1,2),(3,)))
      nextDV,En = exblockkrylovTimeEvol(DV,prefactor=timestep,maxiter=maxiter,updatefct=XOC_update_exfull,Lenv=Lenv[iR],Renv=Renv[iL])

      psi[iR] = contract([1,3,4,2],nextDV,2,psi[iR],1)
      psi[iR],D,V = normalization!(psi[iR],size(psi[iR]),((1,2,3),(4,)))

    else

      newpsi,En = exblockkrylovTimeEvol(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=ex3S_update,Lenv=Lenv[i],Renv=Renv[i])
      newpsi,D,V = normalization!(newpsi,size(newpsi),((1,2,3),(4,)))

      U,D,psi[i] = svd(newpsi,[[1,4],[2,3]],m=maxm,minm=minm,cutoff=cutoff)
      UD = contract([1,3,2],U,3,D,1)

      Renv[iL] = Rupdate(Renv[iR],dualpsi[iR],psi[iR],mpo[iR])
      UD,D,V = normalization!(UD,size(UD),((1,2),(3,)))
      nextUD,En = exblockkrylovTimeEvol(UD,prefactor=timestep,maxiter=maxiter,updatefct=XOC_update_exfull,Lenv=Lenv[iR],Renv=Renv[iL])

      psi[iL] = contract(psi[iL],3,nextUD,1)
      psi[iL],D,V = normalization!(psi[iL],size(psi[iL]),((1,2,3),(4,)))

    end
  
#        params.truncerr = truncerr
#        params.biggestm = max(params.biggestm,size(D,1))

    psi.oc += j

    if psi.oc == Ns - (range-1) && j > 0
      j *= -1
    elseif psi.oc == 1 + (range-1) && j < 0
      j *= -1
    end
  end

  nothing
end
export block_tdvp_exfull









#=

function exblockkrylovTimeEvol(psiops::TensType...;prefactor::Number=1,lanczosfct::Function=blocklanczos,maxiter::Integer=2,updatefct::Function=XOC_update,Lenv::TensType=eltype(psiops[1])[0],Renv::TensType=Lenv)

  Hpsi = ex3S_update(Lenv,Renv,psiops[1],psiops[2])
  LanczosH = ccontract(psiops[1],(1,2,3),Hpsi,(1,2,3))

  expH = exp(LanczosH,prefactor)


  nex = size(psiops[1],ndims(psiops[1]))
  D,U,gs_psi = diagonalizeM(LanczosH,nex,psiops[1])#,qselect = qselect)

  
  truncexpH = expH
  retpsi = contract(psiops[1],ndims(psiops[1]),truncexpH,1)

  return retpsi,[D[i,i] for i = 1:size(D,1)]
end
export exblockkrylovTimeEvol
=#
function block_tdvp_exbasis(psi::MPS,mpo::MPO;maxm::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),minm::Integer=2,cutoff::Real=0.,origj::Bool=true,maxiter::Integer=2,prefactor::Number=1.)

  timestep = prefactor/length(psi)*2
  
  dualpsi = psi
#      if Lenv == [0] && Renv == [0]
  Lenv,Renv = makeEnv(dualpsi,psi,mpo)
#      end

  Ns = length(psi)
  if psi.oc == 1
    j = 1
  elseif psi.oc == Ns #
    j = -1
  else
    j = origj ? 1 : -1 #if the oc is away from the edge, j could be externally specified.
  end

  range = 1

  Nsteps = (Ns-1)
  for n = 1:Nsteps

    i = psi.oc

    if j > 0
      iL,iR = i,i+1
    else
      iL,iR = i-1,i
    end

#    norm!(psi[i])
    if j > 0

      newpsi,En = exblockkrylovTimeEvol(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=ex3S_update,Lenv=Lenv[i],Renv=Renv[i])
      newpsi,D,V = normalization!(newpsi,size(newpsi),((1,2,3),(4,)))

      psi[iL],D,V = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
      DV = contract(D,2,V,1)
      psi[iR] = contract([1,3,4,2],DV,2,psi[iR],1)

      Lenv[iR] = Lupdate(Lenv[iL],dualpsi[iL],psi[iL],mpo[iL])

    else

      newpsi,En = exblockkrylovTimeEvol(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=ex3S_update,Lenv=Lenv[i],Renv=Renv[i])
      newpsi,D,V = normalization!(newpsi,size(newpsi),((1,2,3),(4,)))

      U,D,psi[iR] = svd(newpsi,[[1,4],[2,3]],m=maxm,minm=minm,cutoff=cutoff)
      UD = contract([1,3,2],U,3,D,1)
      psi[iL] = contract(psi[iL],3,UD,1)

      Renv[iL] = Rupdate(Renv[iR],dualpsi[iR],psi[iR],mpo[iR])

    end

    psi.oc += j

    if psi.oc == Ns - (range-1) && j > 0
      j *= -1
    elseif psi.oc == 1 + (range-1) && j < 0
      j *= -1
    end
  end

  nothing
end
export block_tdvp_exbasis



function block_tdvp_exLR(psi::MPS,mpo::MPO;maxm::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),minm::Integer=2,cutoff::Real=0.,origj::Bool=true,maxiter::Integer=2,prefactor::Number=1.)

  timestep = prefactor/length(psi)*2
  
  dualpsi = psi
#      if Lenv == [0] && Renv == [0]
  Lenv,Renv = makeEnv(dualpsi,psi,mpo)
#      end

  Ns = length(psi)
  if psi.oc == 1
    j = 1
  elseif psi.oc == Ns #
    j = -1
  else
    j = origj ? 1 : -1 #if the oc is away from the edge, j could be externally specified.
  end

  range = 1

  Nsteps = (Ns-1)
  for n = 1:Nsteps

    i = psi.oc

    if j > 0
      iL,iR = i,i+1
    else
      iL,iR = i-1,i
    end

#    norm!(psi[i])
    if j > 0

      UL,DL,VL = svd(psi[i],[[1,2],[3]])
      Lpsi = contract(UL,3,contract(DL,2,VL,1),1)

      UR,DR,VR = svd(psi[i],[[1],[2,3]])
      Rpsi = contract(contract(UR,2,DR,1),VR,1)

      psi[i] = Lpsi + Rpsi #tensorcombination!(Lpsi,Rpsi,alpha=(1/sqrt(2),1/sqrt(2)))
      psi[i],D,V = normalization!(psi[i],size(psi[i]),((1,2,3),(4,)))

      newpsi,En = exblockkrylovTimeEvol(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=ex3S_update,Lenv=Lenv[i],Renv=Renv[i])
      newpsi,D,V = normalization!(newpsi,size(newpsi),((1,2,3),(4,)))

      psi[iL],D,V = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
      DV = contract(D,2,V,1)
      psi[iR] = contract([1,3,4,2],DV,2,psi[iR],1)

      Lenv[iR] = Lupdate(Lenv[iL],dualpsi[iL],psi[iL],mpo[iL])

    else

      newpsi,En = exblockkrylovTimeEvol(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=ex3S_update,Lenv=Lenv[i],Renv=Renv[i])
      newpsi,D,V = normalization!(newpsi,size(newpsi),((1,2,3),(4,)))

      U,D,psi[iR] = svd(newpsi,[[1,4],[2,3]],m=maxm,minm=minm,cutoff=cutoff)
      UD = contract([1,3,2],U,3,D,1)
      psi[iL] = contract(psi[iL],3,UD,1)

      Renv[iL] = Rupdate(Renv[iR],dualpsi[iR],psi[iR],mpo[iR])

    end

    psi.oc += j

    if psi.oc == Ns - (range-1) && j > 0
      j *= -1
    elseif psi.oc == 1 + (range-1) && j < 0
      j *= -1
    end
  end

  nothing
end
export block_tdvp_exLR


























function XOC_update_noOC(Lenv::TensType,Renv::TensType,psiops::TensType...)
  D = psiops[1]
  LenvD = contract(Lenv,3,D,1)
  return contract([1,3,2],LenvD,(2,3),Renv,(2,1))
end

function blockkrylovTimeEvol_noOC(psiops::TensType...;prefactor::Number=1,lanczosfct::Function=blocklanczos,maxiter::Integer=2,updatefct::Function=XOC_update,Lenv::TensType=eltype(psiops[1])[0],Renv::TensType=Lenv)
  alpha,beta,psivec,p = lanczosfct(psiops...,updatefct=updatefct,maxiter=maxiter,Lenv=Lenv,Renv=Renv)

  LanczosH = makeM(alpha,beta,p)
  expH = exp(LanczosH,prefactor)


  nex = size(psiops[1],ndims(psiops[1]))
  D,U,gs_psi = diagonalizeM(LanczosH,nex,psivec)#,qselect = qselect)
#  retpsi = contract(newpsi, ndims(newpsi), U, 1)



truncexpH = expH[:,1:nex]

retpsi = contract(psivec,ndims(psivec),truncexpH,1)

  return retpsi,[D[i,i] for i = 1:size(D,1)]
end
export blockkrylovTimeEvol_noOC

function block_tdvp_noOC(psi::MPS,mpo::MPO;maxm::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),minm::Integer=2,cutoff::Real=0.,origj::Bool=true,maxiter::Integer=2,prefactor::Number=1.)

  timestep = prefactor/length(psi)*2
  
  dualpsi = psi
#      if Lenv == [0] && Renv == [0]
  Lenv,Renv = makeEnv(dualpsi,psi,mpo)
#      end

  Ns = length(psi)
  if psi.oc == 1
    j = 1
  elseif psi.oc == Ns #
    j = -1
  else
    j = origj ? 1 : -1 #if the oc is away from the edge, j could be externally specified.
  end

  range = 1

  Nsteps = (Ns-1)
  for n = 1:Nsteps

    i = psi.oc

    if j > 0
      iL,iR = i,i+1
    else
      iL,iR = i-1,i
    end

#    norm!(psi[i])
    if j > 0

      newpsi,En = blockkrylovTimeEvol_noOC(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=ex3S_update,Lenv=Lenv[i],Renv=Renv[i])
      newpsi,D,V = normalization!(newpsi,size(newpsi),((1,2,3),(4,)))

      psi[iL],D,V = svd(newpsi,[[1,2],[3,4]],m=maxm,minm=minm,cutoff=cutoff)
      DV = contract(D,2,V,1)
      psi[iR] = contract([1,3,4,2],DV,2,psi[iR],1)

      Lenv[iR] = Lupdate(Lenv[iL],dualpsi[iL],psi[iL],mpo[iL])

    else

      newpsi,En = blockkrylovTimeEvol_noOC(psi[i],mpo[i],prefactor=timestep,maxiter=maxiter,updatefct=ex3S_update,Lenv=Lenv[i],Renv=Renv[i])
      newpsi,D,V = normalization!(newpsi,size(newpsi),((1,2,3),(4,)))

      U,D,psi[iR] = svd(newpsi,[[1,4],[2,3]],m=maxm,minm=minm,cutoff=cutoff)
      UD = contract([1,3,2],U,3,D,1)
      psi[iL] = contract(psi[iL],3,UD,1)

      Renv[iL] = Rupdate(Renv[iR],dualpsi[iR],psi[iR],mpo[iR])

    end
  
#        params.truncerr = truncerr
#        params.biggestm = max(params.biggestm,size(D,1))

    psi.oc += j

    if psi.oc == Ns - (range-1) && j > 0
      j *= -1
    elseif psi.oc == 1 + (range-1) && j < 0
      j *= -1
    end
  end

  nothing
end
export block_tdvp_noOC

