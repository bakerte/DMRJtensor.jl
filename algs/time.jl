###############################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                               v1.0
#
###############################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.0+)
#


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
      ops = H[i]*H[i+1] #contract(H.H[i],ndims(H.H[i]),H.H[i+1],1)
#      if i > 1 #1 < i <= length(H)-1
        tops = ops[size(ops,1),:,:,:,:,1]
#      else #i == 1
#        tops = ops[1,:,:,:,:,1]
#      elseif i == length(H)-1
#        tops = ops[size(ops,1),:,:,:,:,1]
#      end
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
      ops *= prefactor
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
      AA = psi[iL]*psi[iR] #contract(psi[iL],3,psi[iR],1)
      AA = contract([1,3,4,2],AA,[2,3],expgates[i],[1,2])
      U,D,V = svd(AA,[[1,2],[3,4]],cutoff=cutoff,m=m)
      if j == 1
        psi[iL] = U 
        psi[iR] = D*V
      else
        psi[iL] = U*D
        psi[iR] = V
      end
      psi.oc += j

      norm!(psi[psi.oc])

      if psi.oc == Ns || psi.oc == 1
        j *= -1
      end
    end
    return psi
  end
  export tebd



#           +---------------------------------------+
#>----------| Time dependent variational principle  |----------<
#           +---------------------------------------+



function twositeOps(mpo::MPO,nsites::Integer)
  nbonds = length(mpo)-(nsites-1)
  ops = Array{typeof(mpo[1]),1}(undef,nbonds)
  #=Threads.@threads=# for i = 1:nbonds
    ops[i] = contract([1,3,5,6,2,4],mpo[i],ndims(mpo[i]),mpo[i+1],1)
    if typeof(ops[i]) <: qarray
      ops[i] = changeblock(ops[i],[1,2,3,4],[5,6])
    end
  end
  if typeof(mpo) <: largeMPO
    newmpo = largeMPO(ops,label="twompo_")
  else
    newmpo = MPO(ops)
  end
  return newmpo
end


@inline function joinedtwosite_update(Lenv::TensType,Renv::TensType,AA,tops)

  ops = tops[1]

  Hpsi = contract(ops,(5,6),AA,(2,3))
  LHpsi = contract(Lenv,(2,3),Hpsi,(1,5))
  temp = contract(LHpsi,(5,4),Renv,(1,2))

  return temp
end



  function krylovTimeEvol(psiops::TensType...;prefactor::Number=1,lanczosfct::Function=lanczos,r::Integer=2,updatefct::Function=makeHpsi,Lenv::TensType=eltype(psiops[1])[0],Renv::TensType=Lenv)

    energies,retpsi,alpha,beta,psivec = lanczosfct(psiops...,r=r,m=r,updatefct=updatefct,Lenv=Lenv,Renv=Renv)

    M = LinearAlgebra.SymTridiagonal(alpha,beta)
    En,U = LinearAlgebra.eigen(M)
    expH = exp(M,prefactor)
    if isapprox(sum(w->expH[w,1],1:size(expH,1)),0)
      expH = LinearAlgebra.I + LinearAlgebra.SymTridiagonal(alpha,beta)
    end
    coeffs = ntuple(w->convert(eltype(psivec[1]),expH[w,1]),length(psivec)) #works for any orthogonal basis, else (expH*overlaps)

    keepers = [!isapprox(coeffs[w],0) && isassigned(psivec,w) for w = 1:length(coeffs)]
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
  
    function tdvp(psi::MPS,mpo::MPO;m::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),minm::Integer=2,cutoff::Real=0.,origj::Bool=true,r::Integer=2,prefactor::Number=1.)

      timestep = prefactor/length(psi)/2
      
      dualpsi = psi
      Lenv,Renv = makeEnv(psi,psi,mpo)

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
  
      Nsteps = Ns #(Ns-1)
      for n = 1:Nsteps

        i = psi.oc

        if j > 0
          iL,iR = i,i+1
        else
          iL,iR = i-1,i
        end

        if j > 0

          newpsi,En = krylovTimeEvol(psi[i],mpo[i],prefactor=timestep,r=r,updatefct=singlesite_update,Lenv=Lenv[i],Renv=Renv[i])
          norm!(newpsi)

          psi[i],D,V = svd(newpsi,[[1,2],[3]],m=m,minm=minm,cutoff=cutoff)
          DV = D*V
          norm!(DV)
  
          Lenv[iR] = Lupdate(Lenv[iL],psi[iL],psi[iL],mpo[iL])

          nextDV,En = krylovTimeEvol(DV,prefactor=timestep,r=r,updatefct=OC_update,Lenv=Lenv[iR],Renv=Renv[iL])
  
          psi[iR] = contract(nextDV,2,psi[iR],1)
  
        else
  
          newpsi,En = krylovTimeEvol(psi[i],mpo[i],prefactor=timestep,r=r,updatefct=singlesite_update,Lenv=Lenv[i],Renv=Renv[i])
          norm!(newpsi)

          U,D,psi[i] = svd(newpsi,[[1],[2,3]],m=m,minm=minm,cutoff=cutoff)
          UD = U*D
          norm!(UD)
  
          Renv[iL] = Rupdate(Renv[iR],psi[iR],psi[iR],mpo[iR])
          nextUD,En = krylovTimeEvol(UD,prefactor=timestep,r=r,updatefct=OC_update,Lenv=Lenv[iR],Renv=Renv[iL])
  
          psi[iL] = contract(psi[iL],3,nextUD,1)
  
        end

        psi.oc += j

        norm!(psi[psi.oc])
  
        if psi.oc == Ns - (range-1) && j > 0
          j *= -1
        elseif psi.oc == 1 + (range-1) && j < 0
          j *= -1
        end
      end
    
      return En
    end
    export tdvp


    function tdvp_twosite(psi::MPS,mpo::MPO;m::Integer=maximum([size(psi[w],3) for w = 1:length(psi)]),minm::Integer=2,cutoff::Real=0.,origj::Bool=true,r::Integer=2,prefactor::Number=1.)

      timestep = prefactor/(length(psi)-1)/2 #divide by 2 from theory
      #divide by number of bonds because this is how many times each operator is applied
      
      dualpsi = psi
      Lenv,Renv = makeEnv(psi,psi,mpo)

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

        AA = psi[iL]*psi[iR]
        if j > 0

          newpsi,En = krylovTimeEvol(AA,ops[iL],prefactor=timestep,r=r,updatefct=joinedtwosite_update,Lenv=Lenv[iL],Renv=Renv[iR])
          norm!(newpsi)

          psi[iL],D,V = svd(newpsi,[[1,2],[3,4]],m=m,minm=minm,cutoff=cutoff)
          DV = D*V
  
          Lenv[iR] = Lupdate(Lenv[iL],psi[iL],psi[iL],mpo[iL])
          norm!(DV)

          psi[iR],En = krylovTimeEvol(DV,mpo[iR],prefactor=timestep,r=r,updatefct=singlesite_update,Lenv=Lenv[iR],Renv=Renv[iR])

        else
  
          newpsi,En = krylovTimeEvol(AA,ops[iL],prefactor=timestep,r=r,updatefct=joinedtwosite_update,Lenv=Lenv[iL],Renv=Renv[iR])
          norm!(newpsi)

          U,D,psi[iR] = svd(newpsi,[[1,2],[3,4]],m=m,minm=minm,cutoff=cutoff)
          UD = U*D
          norm!(UD)
  
          Renv[iL] = Rupdate(Renv[iR],psi[iR],psi[iR],mpo[iR])
          
          psi[iL],En = krylovTimeEvol(UD,mpo[iL],prefactor=timestep,r=r,updatefct=singlesite_update,Lenv=Lenv[iL],Renv=Renv[iL])
  
        end

        psi.oc += j

        norm!(psi[psi.oc])
  
        if psi.oc == Ns - (range-1) && j > 0
          j *= -1
        elseif psi.oc == 1 + (range-1) && j < 0
          j *= -1
        end
      end
    
      return En
    end
    export tdvp_twosite
