#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v1.0
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1+)
#

"""
    Module: MPmaker

Functions to generate MPSs and MPOs
"""
module MPmaker
using ..tensor
using ..QN
using ..Qtensor
using ..Qtask
using ..contractions
using ..decompositions
using ..MPutil

#       +---------------------------------------+
#>------+  move! orthogonality center in MPS    +---------<
#       +---------------------------------------+

  """
      moveR(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

  moves `psi`'s orthogonality center from `Lpsi` to `Rpsi`, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error

  See also: [`moveR!`](@ref)
  """
  function moveR(Lpsi::P,Rpsi::P;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                  recursive::Bool=false,fast::Bool=false) where P <: TensType where {R <: Number, Z <: Qnum}
    if (size(Lpsi,3) <= m) && !isapprox(cutoff,0.) && fast
      Ltens,modV = qr(Lpsi,[[1,2],[3]])

      DV = (condition ? getindex!(modV,:,1:size(Rpsi,1)) : modV)
      D = DV
      truncerr = 0.
    else
      Ltens,D,V,truncerr,sumD = svd(Lpsi,[[1,2],[3]],cutoff=cutoff,m=m,minm=minm,recursive=recursive,mag=mag)
      modV = (condition ? getindex!(V,:,1:size(Rpsi,1)) : V)
      DV = contract(D,[2],modV,[1])
    end
    Rtens = contract(DV,2,Rpsi,1)
    return Ltens,Rtens,D,truncerr
  end
  export moveR

  """
      moveL(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

  moves `psi`'s orthogonality center from `Rpsi` to `Lpsi`, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error

  See also: [`moveL!`](@ref)
  """
  function moveL(Lpsi::P,Rpsi::P;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                  recursive::Bool=false,fast::Bool=false) where P <: TensType  where {R <: Number, Z <: Qnum}
    if (size(Rpsi,1) <= m) && !isapprox(cutoff,0.) && fast
      modU,Rtens = lq(Rpsi,[[1],[2,3]])
      UD = (condition ? getindex!(modU,1:size(Lpsi,3),:) : modU)
      D = UD
      truncerr = 0.
    else
      U,D,Rtens,truncerr,sumD = svd(Rpsi,[[1],[2,3]],cutoff=cutoff,m=m,minm=minm,recursive=recursive,mag=mag)
      modU = (condition ? getindex!(U,1:size(Lpsi,3),:) : U)
      UD = contract(modU,[2],D,[1])
    end
    Ltens = contract(Lpsi,3,UD,1)
    return Ltens,Rtens,D,truncerr
  end
  export moveL

  """
      moveR!(psi,iL,iR[,cutoff=,m=,minm=,condition=])
  
  acts in-place to move `psi`'s orthogonality center from `iL` to `iR`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

  See also: [`moveR`](@ref)
  """
  function moveR!(psi::MPS,iL::Integer,iR::Integer;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                   recursive::Bool=false,mag::Number=0.)
    psi[iL],psi[iR],D,truncerr = moveR(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm,condition=condition,recursive=recursive,mag=mag)
    return D,truncerr
  end
  export moveR!

  """
      moveL!(psi,iL,iR[,cutoff=,m=,minm=,condition=])

  acts in-place to move `psi`'s orthogonality center from `iR` to `iL`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

  See also: [`moveL`](@ref)
  """
  function moveL!(psi::MPS,iL::Integer,iR::Integer;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                   recursive::Bool=false,mag::Number=0.)
    psi[iL],psi[iR],D,truncerr = moveL(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm,condition=condition,recursive=recursive,mag=mag)
    return D,truncerr
  end
  export moveL!

  """
      movecenter!(psi,newoc[,m=,cutoff=,minm=,Lfct=,Rfct=])

  movement function to move `psi` to a new site, `newoc` with `Lfct` and `Rfct`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

  See also: [`move!`](@ref) [`move`](@ref)
  """
  function movecenter!(psi::MPS,pos::Integer;cutoff::Float64=1E-14,m::Integer=0,minm::Integer=0,
                        Lfct::Function=moveR,Rfct::Function=moveL)
    if m == 0
      m = maximum([maximum(size(psi[i])) for i = 1:size(psi,1)])
    end
    while psi.oc != pos
      if psi.oc < pos
        iL = psi.oc
        iR = psi.oc+1
        psi[iL],psi[iR],D,truncerr = Lfct(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm,fast=true)
        psi.oc = iR
      else
        iL = psi.oc-1
        iR = psi.oc
        psi[iL],psi[iR],D,truncerr = Rfct(psi[iL],psi[iR],cutoff=cutoff,m=m,minm=minm,fast=true)
        psi.oc = iL
      end
    end
    nothing
  end

  """
      move!(psi,newoc[,m=,cutoff=,minm=])
  
  in-place move orthgononality center of `psi` to a new site, `newoc`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

  See also: [`move`](@ref)
  """
  function move!(mps::MPS,pos::intType;m::Integer=0,cutoff::Float64=1E-14,minm::Integer=0)
    movecenter!(mps,pos,cutoff=cutoff,m=m,minm=minm)
    nothing
  end
  export move!

  """
      move(psi,newoc[,m=,cutoff=,minm=])

  same as `move!` but makes a copy of `psi`

  See also: [`move!`](@ref)
  """
  function move(mps::MPS,pos::intType;m::Integer=0,cutoff::Float64=1E-14,minm::Integer=0)
    newmps = copy(mps)
    movecenter!(newmps,pos,cutoff=cutoff,m=m,minm=minm)
    return newmps
  end
  export move

  function leftnormalize(psi::MPS)
    newpsi = move(psi,length(psi))
    U,D,V = svd(psi[end],[[1,2],[3]])
    newpsi[end] = U
    newpsi.oc = 0
    return newpsi,D,V
  end
  export leftnormalize
  
  function leftnormalize!(psi::MPS)
    move!(psi,length(psi))
    U,D,V = svd(psi[end],[[1,2],[3]])
    psi[end] = U
    psi.oc = 0
    return D,V
  end
  export leftnormalize!
  
  function rightnormalize(psi::MPS)
    newpsi = move(psi,1)
    U,D,V = svd(psi[1],[[1],[2,3]])
    newpsi[1] = V
    newpsi.oc = 0
    return U,D,newpsi
  end
  export rightnormalize
  
  function rightnormalize!(psi::MPS)
    psi = move!(psi,1)
    U,D,V = svd(psi[1],[[1],[2,3]])
    psi[1] = V
    psi.oc = 0
    return U,D
  end
  export rightnormalize!

  """
      Lupdate(Lenv,dualpsi,psi,mpo)

  Updates left environment tensor `Lenv` with dual MPS `dualpsi`, MPS `psi`, and MPO `mpo`
  """
  function  Lupdate(Lenv::X,dualpsi::Y,psi::Y,mpo::Z...) where {X <: Union{Qtens{A,Q},Array{A,3},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,3},tens{B}},Z <: Union{Qtens{W,Q},Array{W,4},tens{W}}} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
    nMPOs = length(mpo)
    nLsize = nMPOs+2
    tempLenv = contractc(Lenv,1,dualpsi,1)
    for j = 1:nMPOs
      tempLenv = contract(tempLenv,[1,nLsize],mpo[j],[1,3])
    end
    return contract(tempLenv,[1,nLsize],psi,[1,2])
  end
  export Lupdate

  """
      Rupdate(Renv,dualpsi,psi,mpo)

  Updates right environment tensor `Renv` with dual MPS `dualpsi`, MPS `psi`, and MPO `mpo`
  """
  function  Rupdate(Renv::X,dualpsi::Y,psi::Y,mpo::Z...) where {X <: Union{Qtens{A,Q},Array{A,3},tens{A}}, Y <: Union{Qtens{B,Q},Array{B,3},tens{B}},Z <: Union{Qtens{W,Q},Array{W,4},tens{W}}} where {A <: Number, B <: Number, W <: Number, Q <: Qnum}
    nMPOs = length(mpo)
    nRsize = nMPOs+2
    tempRenv = contract(Renv,1,psi,3)
    for j = 1:nMPOs
      tempRenv = contract(tempRenv,[nRsize+1,1],mpo[j],[2,4])
    end
    return contractc(tempRenv,[nRsize+1,1],dualpsi,[2,3])
  end
  export Rupdate

  """
      Lupdate!(i,Lenv,psi,dualpsi,mpo)

  Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
  """
  function Lupdate!(i::Integer,Lenv::Env,dualpsi::MPS,psi::MPS,mpo::MPO...) where B <: TensType
    Lenv[i+1] = Lupdate(Lenv[i],dualpsi[i],psi[i],[mpo[a][i] for a = 1:length(mpo)]...)
    nothing
  end

  """
      Lupdate!(i,Lenv,psi,mpo)

  Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
  """
  function Lupdate!(i::Integer,Lenv::Env,psi::MPS,mpo::MPO...) where B <: TensType
    Lupdate!(i,Lenv,psi,psi,mpo...)
  end
  export Lupdate!

  """
      Rupdate!(i,Renv,dualpsi,psi,mpo)

  Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
  """
  function Rupdate!(i::Integer,Renv::Env,dualpsi::MPS,psi::MPS,mpo::MPO...) where B <: TensType
    Renv[i-1] = Rupdate(Renv[i],dualpsi[i],psi[i],[mpo[a][i] for a = 1:length(mpo)]...)
    nothing
  end

  """
      Rupdate!(i,Renv,psi,mpo)

  Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
  """
  function Rupdate!(i::Integer,Renv::Env,psi::MPS,mpo::MPO...) where B <: TensType
    Rupdate!(i,Renv,psi,psi,mpo...)
  end
  export Rupdate!

  """
      boundaryMove!([dualpsi,]psi,i,mpo,Lenv,Renv)

  Move orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

  See also: [`move!`](@ref)
  """
  function boundaryMove!(psi::MPS,i::Integer,Lenv::Env,
                          Renv::Env,mpo::MPO...;mover::Function=move!)
    origoc = psi.oc
    if origoc < i
      mover(psi,i)
      for w = origoc:i-1
        Lupdate!(w,Lenv,psi,mpo...)
      end
    elseif origoc > i
      mover(psi,i)
      for w = origoc:-1:i+1
        Rupdate!(w,Renv,psi,mpo...)
      end
    end
    nothing
  end

  function boundaryMove!(dualpsi::MPS,psi::MPS,i::Integer,Lenv::Env,
                          Renv::Env,mpo::MPO...;mover::Function=move!)
    origoc = psi.oc
    if origoc < i
      mover(psi,i)
      mover(dualpsi,i)
      for w = origoc:i-1
        Lenv[w+1] = Lupdate(Lenv[w],dualpsi[w],psi[w],[mpo[a][w] for a = 1:length(mpo)]...)
      end
    elseif origoc > i
      mover(psi,i)
      mover(dualpsi,i)
      for w = origoc:-1:i+1
        Renv[w-1] = Rupdate(Renv[w],dualpsi[w],psi[w],[mpo[a][w] for a = 1:length(mpo)]...)
      end
    end
    nothing
  end
  export boundaryMove!

  """
      boundaryMove([dualpsi,]psi,i,mpo,Lenv,Renv)

  Copies `psi` and moves orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

  See also: [`move!`](@ref)
  """
  function boundaryMove(dualpsi::MPS,psi::MPS,i::Integer,mpo::MPO,Lenv::Env,Renv::Env)
    newpsi = copy(psi)
    newdualpsi = copy(dualpsi)
    newLenv = copy(Lenv)
    newRenv = copy(Renv)
    boundaryMove!(newdualpsi,newpsi,i,mpo,newLenv,newRenv)
    return newdualpsi,newpsi,newLenv,newRenv
  end

  function boundaryMove(psi::MPS,i::Integer,mpo::MPO,Lenv::Env,Renv::Env)
    newpsi = copy(psi)
    newLenv = copy(Lenv)
    newRenv = copy(Renv)
    boundaryMove!(newpsi,newpsi,i,mpo,newLenv,newRenv)
    return newpsi,newLenv,newRenv
  end
  export boundaryMove

  """
      applyMPO(psi,H[,m=,cutoff=])
  
  Applies MPO (`H`) to an MPS (`psi`) and truncates to maximum bond dimension `m` and cutoff `cutoff`
  """
  function applyMPO(psi::MPS,H::MPO;m::intType=0,cutoff::Float64=0.)::MPS
    if m == 0
      m = maximum([size(psi[i],ndims(psi[i])) for i = 1:size(psi.A,1)])
    end

    thissize = size(psi,1)
    newpsi = [contract([1,3,4,2,5],psi[i],2,H[i],2) for i = 1:thissize]

    finalpsi = Array{typeof(psi[1]),1}(undef,thissize)
    finalpsi[thissize] = reshape!(newpsi[thissize],[[1],[2],[3],[4,5]],merge=true)

    for i = thissize:-1:2
      currTens = finalpsi[i]
      newsize = size(currTens)
      
      temp = reshape!(currTens,[[1,2],[3,4]])
      U,D,V = svd(temp,m = m,cutoff=cutoff)
      finalpsi[i] = reshape!(V,size(D,1),newsize[3],newsize[4],merge=true)
      tempT = contract(U,2,D,1)
      
      finalpsi[i-1] = contract(newpsi[i-1],[4,5],tempT,1)
    end
    finalpsi[1] = reshape!(finalpsi[1],[[1,2],[3],[4]],merge=true)
    return MPS(finalpsi,1)
  end
  export applyMPO

#       +---------------------------------------+
#>------+       measurement operations          +---------<
#       +---------------------------------------+

  """
      expect(dualpsi,psi,H[,Lbound=,Rbound=])

  evaluate <`dualpsi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`); `dualpsi` is conjugated inside of the algorithm

  See also: [`overlap`](@ref)
  """
  function expect(dualpsi::MPS,psi::MPS,H::MPO...;Lbound::TensType=typeof(psi[1])(),Rbound::TensType=Lbound,order::Array{Int64,1}=Int64[])
    Ns = size(psi,1)
    nMPOs = size(H,1)
    nLsize = nMPOs+2
    Lenv,Renv = makeEnds(dualpsi,psi,H...,Lbound=Lbound,Rbound=Rbound)

    for i = 1:length(psi)
      Lenv = contractc(Lenv,1,dualpsi[i],1)
      for j = 1:nMPOs
        Lenv = contract(Lenv,[1,nLsize],H[j][i],[1,3])
      end
      Lenv = contract(Lenv,[1,nLsize],psi[i],[1,2])
    end

    if order == Int64[]
      permvec = vcat([ndims(Renv)],[i for i = 2:ndims(Renv)-1],[1])
      modRenv = permutedims(Renv,permvec)
    else
      modRenv = Renv
    end
    
    return contract(Lenv,modRenv)
  end

  """
      expect(psi,H[,Lbound=,Rbound=])

  evaluate <`psi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`)

  See also: [`overlap`](@ref)
  """
  function expect(psi::MPS,H::MPO...;Lbound::TensType=typeof(psi[1])(),Rbound::TensType=Lbound)
    return expect(psi,psi,H...,Lbound=Lbound,Rbound=Rbound)
  end
  export expect

  """
      correlationmatrix(dualpsi,psi,Cc,Ca[,F,silent=])

  Compute the correlation funciton (example, <`dualpsi`|`Cc`_i . `Ca`_j|`psi`>) on all sites; can also specify a trail `F` for Fermion strings; `silent` toggles output of every line

  # Note:
  + More efficient than using `mpoterm`s
  + Use `mpoterm` and `applyMPO` for higher order correlation functions or write a new function
  """

  function correlationmatrix(dualpsi::MPS, psi::MPS, mCc::Q, mCa::R, F::S...;silent::Bool = true) where {Q <: TensType,R <: TensType,S <: TensType}
    if typeof(mCc) <: Array && eltype(psi) <: denstens
      Cc = tens(mCc)
    else
      Cc = mCc
    end
    if typeof(mCa) <: Array && eltype(psi) <: denstens
      Ca = tens(mCa)
    else
      Ca = mCa
    end
    rho = Array{eltype(psi[1]),2}(undef,size(psi,1),size(psi,1))
    onsite = contract(Cc,2,Ca,1)
    if size(F,1) != 0
      FCc = contract(Cc,2,F[1],1)
    else
      FCc = Cc
    end
    diffTensors = !(psi == dualpsi)
    for i = 1:size(psi,1)
      move!(psi,i)
      if diffTensors
        move!(dualpsi,i)
      end
      TopTerm = contract([1,3,2],psi[i],[2],onsite,[1])
      rho[i,i] = contractc(TopTerm,dualpsi[i])
    end
    for i = 1:size(psi,1)-1
      move!(psi,i)
      if diffTensors
        move!(dualpsi,i)
      end
      TopTerm = contract(psi[i],[2],FCc,[1])
      Lenv = contractc(TopTerm,[1,3],dualpsi[i],[1,2])
      for j = i+1:size(psi,1)
        Renv = contract(psi[j],[2],Ca,[1])
        Renv = contractc(Renv,[2,3],dualpsi[j],[3,2])
        DMElement = contract(Lenv,Renv)
        if j < size(psi,1)
          if size(F,1) != 0
            Lenv = contract(Lenv,1,psi[j],1)
            Lenv = contract(Lenv,2,F[1],1)
            Lenv = contractc(Lenv,[1,3],dualpsi[j],[1,2])
          else
            Lenv = contract(Lenv, 1, psi[j], 1)
            Lenv = contractc(Lenv, [1,2], dualpsi[j], [1,2])
          end
        end
        rho[i,j] = DMElement
        rho[j,i] = conj(DMElement)
        if !silent
          println("Printing element: ",i," , ",j," ",DMElement)
        end
      end
    end
    return rho
  end

  """
      correlationmatrix(psi,Cc,Ca[,F,silent=])

  Compute the correlation funciton (example, <`psi`|`Cc`_i . `Ca`_j|`psi`>) on all sites; can also specify a trail `F` for Fermion strings; `silent` toggles output of every line

  # Example:
  ```julia
  Cup, Cdn, Nup, Ndn, Ndens, F, O, Id = fermionOps()
  rho = correlationmatrix(psi,Cup',Cup,F) #density matrix
  ```
  """
  function correlationmatrix(psi::MPS, Cc::Q, Ca::R, F::S...;silent::Bool = true) where {Q <: TensType,R <: TensType,S <: TensType}
    return correlationmatrix(psi,psi,Cc,Ca,F...,silent=silent)
  end
  export correlationmatrix





  function operator_in_order!(pos::Array{G,1},sizes::Union{Array{G,1},Tuple}) where G <: intType
    w = length(pos)
    pos[w] += 1
    while w > 1 && pos[w] > sizes[w]
      w -= 1
      @inbounds pos[w] += 1
      @simd for x = w:length(pos)-1
        @inbounds pos[x+1] = pos[x]
      end
    end
    nothing
  end

  #heap algorithm for permutations (non-recursive)...
  function permutations(nelem::intType)
    vec = [i for i = 1:nelem]
    numvecs = factorial(nelem)
    storevecs = Array{Array{intType,1},1}(undef,numvecs)
    saveind = zeros(intType,nelem)
    i = 0
    counter = 1
    storevecs[1] = copy(vec)
    while i < nelem
      if saveind[i+1] < i
        if i % 2 == 0
          a,b = 0,i
        else
          a,b = saveind[i+1],i
        end
        vec[a+1],vec[b+1] = vec[b+1],vec[a+1]
        
        counter += 1
        storevecs[counter] = copy(vec)
  
        saveind[i+1] += 1
        i = 0
      else
        saveind[i+1] = 0
        i += 1
      end
    end
    return storevecs
  end

#  import Combinatorics

  function correlation(dualpsi::MPS, psi::MPS, inputoperators::S...;
                          trail::Tuple=(),sites::G=ntuple(i->1:length(psi),length(inputoperators))#=,
                          sites::NTuple{length(operators),UnitRange{intType}}=ntuple(i->1:length(psi),length(operators)),
                          sym::Bool=length(operators)==2=#) where {P <: TensType,Q <: TensType,R <: TensType,S <: Union{Array{TensType,1},TensType},G <: Union{genColType,Tuple}}

      operators = Array{Array{typeof(psi[1]),1},1}(undef,length(inputoperators))
      lengthops = Array{intType,1}(undef,length(operators))
      for k = 1:length(operators)
        if eltype(inputoperators[k]) <: TensType
          operators[k] = inputoperators[k]
          lengthops[k] = length(operators[k])
        else
          operators[k] = [inputoperators[k]]
          lengthops[k] = 1
        end
      end

      Ns = length(psi)
      maxOplength = maximum(lengthops)
#      Ncorrelations = Ns - maxOplength + 1
      retType = typeof(eltype(dualpsi[1])(1) * eltype(psi[1])(1) * prod(w->prod(a->eltype(operators[w][a])(1),1:lengthops[w]),1:length(operators)))

      base_sizes = [length(sites[i]) - (lengthops[i] - 1) for i = 1:length(sites)]

#      max_sizes = [sites[i][end] for i = 1:length(sites)]

      omega = Array{retType,length(operators)}(undef,base_sizes...)

      perm = permutations(length(operators))

      move!(psi,1)
      move!(dualpsi,1)

      Lenv,Renv = makeEnv(dualpsi,psi)
      for b = 1:length(Renv)
        Renv[b] = permutedims(Renv[b],[2,1])
      end

      isId = [true for r = 1:length(operators)]
      if length(trail) > 0
        for r = 1:length(isId)
          index = 0
          while isId[r]
            index += 1
            isId[r] = searchindex(bigtrail,index,index) == 1
          end
        end
      end

#      println(maxOplength)

      for i = 1:length(perm)

        order = perm[i]

        base_pos = ones(intType,length(operators)) #[sites[1][1] for i = 1:length(operators)]

        pos = [sites[1][1] for i = 1:length(operators)] #ones(intType,length(operators))
        prevpos = [sites[1][1] for i = 1:length(operators)] #ones(intType,length(operators))

        while sum(base_sizes - pos) >= 0
          startsite = 1
          while startsite < length(pos) && pos[startsite] == prevpos[startsite]
            startsite += 1
          end

          while startsite > 1 && pos[startsite-1] == prevpos[startsite]
            startsite -= 1
          end

          beginsite = prevpos[startsite]
          finalsite = pos[end] #min(pos[end], Ns - (maxOplength-1))

          thisLenv = Lenv[beginsite]

#          g = startsite
          for w = beginsite:finalsite
            newpsi = psi[w]
            for g = 1:length(pos)
              opdist = w - pos[g]
              if 0 <= opdist < lengthops[g]
                newpsi = contract([2,1,3],operators[order[g]][opdist + 1],2,newpsi,2)
              end
            end
#            while g <= length(pos) && w == pos[g]
#              newpsi = contract([2,1,3],operators[order[g]][1],2,newpsi,2)
#              g += 1
#            end
            for r = 1:length(pos)
              if  w < pos[r] && !isId[r]
                newpsi = contract([2,1,3],trail[r],2,newpsi,2)
              end
            end
            thisLenv = Lupdate(thisLenv,dualpsi[w],newpsi)
            if w < Ns
              Lenv[w+1] = thisLenv
            end
          end

          thisRenv = Renv[finalsite]
          res = contract(thisLenv,thisRenv)
          finalpos = pos[order]
          
          @inbounds omega[finalpos...] = res

          @simd for b = 1:length(pos)
            @inbounds prevpos[b] = pos[b]
          end
          operator_in_order!(base_pos,base_sizes)
          @simd for b = 1:length(pos)
            @inbounds pos[b] = sites[b][base_pos[b]]
          end
        end

      end
      return omega
    end
    export correlation

  #       +---------------------------------------+
  #>------+    Construction of boundary tensors   +---------<
  #       +---------------------------------------+

  #
  #Current environment convention is
  #     LEFT              RIGHT
  #   +--<-- 1          3 ---<--+
  #   |                         |
  #   |                         |
  #   +-->-- 2          2 --->--+
  #   |                         |
  #   |                         |
  #   +-->-- 3          1 --->--+
  # any MPOs in between have the same arrow conventions as 2

  """
      makeEnds(dualpsi,psi[,mpovec,Lbound=,Rbound=])

  Generates first and last environments for a given system of variable MPOs

  # Arguments:
  + `dualpsi::MPS`: dual MPS
  + `psi::MPS`: MPS
  + `mpovec::MPO`: MPOs
  + `Lbound::TensType`: left boundary
  + `Rbound::TensType`: right boundary
  """
  function makeEnds(dualpsi::K,psi::Z,mpovec::P...;Lbound::G=typeof(psi[1])(),Rbound::G=typeof(psi[1])()) where {G <: TensType, Z <: MPS, K <: MPS, P <: MPO}
    if abs(norm(Lbound)) <= 1E-13
      Lout = makeBoundary(dualpsi,psi,mpovec...)
    else
      Lout = copy(Lbound)
    end
    if abs(norm(Rbound)) <= 1E-13
      Rout = makeBoundary(dualpsi,psi,mpovec...,left=false)
    else
      Rout = copy(Rbound)
    end
    return Lout,Rout
  end

  function makeBoundaries(dualpsi::K,psi::Z,mpovec::P...;Lbound::G=typeof(psi[1])(),Rbound::G=typeof(psi[1])()) where {G <: TensType, Z <: MPS, K <: MPS, P <: MPO}
    return makeEnds(dualpsi,psi,mpovec...;Lbound=Lbound,Rbound=Rbound)
  end

  """
      makeEnds(psi[,mpovec,Lbound=,Rbound=])

  Generates first and last environment tensors for a given system of variable MPOs.  Same as other implementation but `dualpsi`=`psi`

  # Arguments:
  + `psi::MPS`: MPS
  + `mpovec::MPO`: MPOs
  + `Lbound::TensType`: left boundary
  + `Rbound::TensType`: right boundary
  """
  function makeEnds(psi::K,mpovec::P...;Lbound::TensType=typeof(psi[1])(),Rbound::TensType=typeof(psi[1])()) where {K <: MPS, P <: MPO}
    return makeEnds(psi,psi,mpovec...,Lbound=Lbound,Rbound=Rbound)
  end
  export makeEnds

  """
      makeEnv(dualpsi,psi,mpo[,Lbound=,Rbound=])

  Generates environment tensors for a MPS (`psi` and its dual `dualpsi`) with boundaries `Lbound` and `Rbound`
  """
  function makeEnv(dualpsi::K,psi::Z,mpo::P...;Lbound::TensType=typeof(psi[1])(),
            Rbound::TensType=typeof(psi[1])()) where B <: TensType where {G <: TensType, Z <: MPS, K <: MPS, P <: MPO}
    Ns = length(psi)
    numtype = elnumtype(dualpsi,psi,mpo...)
    C = psi[1]

    if typeof(psi) <: largeMPS || typeof(mpo) <: largeMPO
      Lenv,Renv = largeLRenv(numtype,Ns)
    else
      Lenv = environment(psi[1],Ns)
      Renv = environment(psi[1],Ns)
    end
    Lenv[1],Renv[Ns] = makeEnds(dualpsi,psi,mpo...,Lbound=Lbound,Rbound=Rbound)
    for i = Ns:-1:psi.oc+1
      Rupdate!(i,Renv,dualpsi,psi,mpo...)
    end
    for i = 1:psi.oc-1
      Lupdate!(i,Lenv,dualpsi,psi,mpo...)
    end
    return Lenv,Renv
  end

  """
      makeEnv(psi,mpo[,Lbound=,Rbound=])

  Generates environment tensors for a MPS (`psi`) with boundaries `Lbound` and `Rbound`
  """
  function makeEnv(psi::K,mpo::P;Lbound::TensType=[0],Rbound::TensType=[0]) where B <: TensType where {G <: TensType, Z <: MPS, K <: MPS, P <: MPO}
    return makeEnv(psi,psi,mpo,Lbound=Lbound,Rbound=Rbound)
  end
  export makeEnv





#       +---------------------------------------+
#>------+       Constructing MPO operators      +---------<
#       +---------------------------------------+

    #converts an array to an MPO so that it is instead of being represented in by an array,
    #it is represented by a tensor diagrammatically as
    #
    #       s2
    #       |
    # a1 -- W -- a2       =    W[a1,s1,s2,a2]
    #       |
    #       s1
    #
    #The original Hamiltonian matrix H in the DMRjulia.jl file is of the form
    #
    # H = [ W_11^s1s2  W_12^s1s2 W_13^s1s2 ....
    #       W_21^s1s2  W_22^s1s2 W_23^s1s2 ....
    #       W_31^s1s2  W_32^s1s2 W_33^s1s2 ....
    #       W_41^s1s2  W_42^s1s2 W_43^s1s2 ....]
    #where each W occupies the equivalent of (vars.qstates X vars.qstates) sub-matrices in
    #of the H matrix as recorded in each s1s2 pair.  These are each of the operators in H.

  """
      convert2MPO(H,physSize,Ns[,infinite=,lower=])

  Converts function or vector (`H`) to each of `Ns` MPO tensors; `physSize` can be a vector (one element for the physical index size on each site) or a number (uniform sites); `lower` signifies the input is the lower triangular form (default)

  # Example:

  ```julia
  spinmag = 0.5;Ns = 10
  Sx,Sy,Sz,Sp,Sm,O,Id = spinOps(spinmag)
  function H(i::Integer)
      return [Id O;
              Sz Id]
  end
  isingmpo = convert2MPO(H,size(Id,1),Ns)
  ```
  """
  function convert2MPO(H::AbstractArray,physSize::Array{Y,1},Ns::intType;
                       lower::Bool=true,regtens::Bool=false) where {X <: Integer, Y <: Integer}
    retType = typeof(prod(a->eltype(H[a])(1),1:Ns))
    finalMPO = Array{Array{retType,4},1}(undef,Ns)
    for i = 1:Ns
      thisH = lower ? H[i] : transpose(H[i])
      states = physSize[(i-1)%size(physSize,1) + 1]
      a1size = div(size(thisH,1),states) #represented in LEFT link indices
      a2size = div(size(thisH,2),states) #represented in RIGHT link indices
      P = eltype(thisH)

      currsize = [a1size,states,states,a2size]
      G = Array{P,4}(undef,currsize...)
      
      for m = 1:a2size
        for k = 1:states
          for j = 1:states
            @simd for l = 1:a1size
              G[l,j,k,m] = thisH[j + (l-1)*states, k + (m-1)*states]
            end
          end
        end
      end
      
      finalMPO[i] = G
    end
    if lower
      finalMPO[1] = finalMPO[1][end:end,:,:,:]
      finalMPO[end] = finalMPO[end][:,:,:,1:1]
    else
      finalMPO[1] = finalMPO[1][1:1,:,:,:]
      finalMPO[end] = finalMPO[end][:,:,:,end:end]
    end
    return MPO(finalMPO,regtens=regtens)
  end

  function convert2MPO(H::AbstractArray,physSize::Y,Ns::intType;
                       lower::Bool=true,regtens::Bool=false) where X <: Integer where Y <: Integer
    return convert2MPO(H,[physSize],Ns,lower=lower,regtens=regtens)
  end

  function convert2MPO(H::Function,physSize::Array{X,1},Ns::intType;
                       lower::Bool=true,regtens::Bool=false) where X <: Integer
    thisvec = [H(i) for i = 1:Ns]
    return convert2MPO(thisvec,physSize,Ns,lower=lower,regtens=regtens)
  end

  function convert2MPO(H::Function,physSize::intType,Ns::intType;
                       lower::Bool=true,regtens::Bool=false)
    return convert2MPO(H,[physSize],Ns,lower=lower,regtens=regtens)
  end
  export convert2MPO

  """
      makeMPS(vect,physInd,Ns[,oc=])

  generates an MPS from a single vector (i.e., from exact diagonalization) for `Ns` sites and `physInd` size physical index at orthogonality center `oc`
  """
  function makeMPS(vect::Array{W,1},physInd::Array{intType,1};Ns::intType=length(physInd),left2right::Bool=true,oc::intType=left2right ? Ns : 1,regtens::Bool=false) where W <: Number
    mps = Array{Array{W,3},1}(undef,Ns)
    # MPS building loop
    if left2right
      M = reshape(vect, physInd[1], div(length(vect),physInd[1]))
      Lindsize = 1 #current size of the left index
      for i=1:Ns-1
        U,DV = qr(M)
        temp = reshape(U,Lindsize,physInd[i],size(DV,1))
        mps[i] = temp

        Lindsize = size(DV,1)
        if i == Ns-1
          temp = unreshape(DV,Lindsize,physInd[i+1],1)
          mps[Ns] = temp
        else
          Rsize = cld(size(M,2),physInd[i+1]) #integer division, round up
          M = unreshape(DV,size(DV,1)*physInd[i+1],Rsize)
        end
      end
      finalmps = MPS(mps,Ns,regtens=regtens)
    else
      M = reshape(vect, div(length(vect),physInd[end]), physInd[end])
      Rindsize = 1 #current size of the right index
      for i=Ns:-1:2
        UD,V = lq(M)
        temp = reshape(V,size(UD,2),physInd[i],Rindsize)
        mps[i] = temp
        Rindsize = size(UD,2)
        if i == 2
          temp = unreshape(UD,1,physInd[i-1],Rindsize)
          mps[1] = temp
        else
          Rsize = cld(size(M,1),physInd[i-1]) #integer division, round up
          M = unreshape(UD,Rsize,size(UD,2)*physInd[i-1])
        end
      end
      finalmps = MPS(mps,1,regtens=regtens)
    end
    move!(finalmps,oc)
    return finalmps
  end

  function makeMPS(vect::W,physInd::Array{intType,1};Ns::intType=length(physInd),
                   left2right::Bool=true,oc::intType=left2right ? Ns : 1,regtens::Bool=false) where W <: denstens
    newvect = copy(vect.T)
    return makeMPS(newvect,physInd;Ns=Ns,oc=oc,left2right=left2right,regtens=regtens)
  end

  function makeMPS(vect::Array{W,1},physInd::intType;Ns::intType=convert(Int64,log(physInd,length(vect))),
                   left2right::Bool=true,oc::intType=left2right ? Ns : 1,regtens::Bool=false) where W <: Union{denstens,Number}
    vecPhysInd = [physInd for i = 1:Ns]
    return makeMPS(vect,vecPhysInd;Ns=Ns,oc=oc,left2right=left2right,regtens=regtens)
  end
  export makeMPS

  """
      fullH(mpo)

  Generates the full Hamiltonian from an MPO (memory providing); assumes lower left triagular form
  """
  function fullH(mpo::MPO)
    Ns = length(mpo)
    fullH = mpo[1]
    for p = 2:Ns
      fullH = contract(fullH,ndims(fullH),mpo[p],1)
    end
    dualinds = [i+1 for i = 2:2:2Ns]
    ketinds = [i+1 for i = 1:2:2Ns]
    finalinds = vcat([1],ketinds,dualinds,[ndims(fullH)])
    pfullH = permutedims(fullH,finalinds)

    size1 = size(pfullH,1)
    size2 = prod(a->size(fullH,a),ketinds)
    size3 = prod(a->size(fullH,a),dualinds)
    size4 = size(pfullH,ndims(pfullH))

    rpfullH = reshape(pfullH,size1,size2,size3,size4)
    return rpfullH[size(rpfullH,1),:,:,1]
  end
  export fullH



#       +---------------------------------------+
#>------+           convert to qMPS             +---------<
#       +---------------------------------------+
#=
  """
      makeQstore(numlegs,dualpsi,psi[,mpovec...,left=,rightind=])

  Generates quantum number for an environment tensor (helper function for making environments with Qtensors)
  
  # Arguments:
  + `numlegs::intType`: number of legs of the environment created
  + `dualpsi::MPS`: dual MPS
  + `psi::MPS`: MPS
  + `mpovec::MPO`: MPO tensor
  + `left::Bool`: toggle for left or right to get correct arrow convention
  + `rightind::intType`: number of the right link index
  """
  function makeQstore(numlegs::intType,dualpsi::MPS,psi::MPS,mpovec::MPO...;left::Bool=true,rightind::intType=0)
    LQNs = Array{typeof(psi[1].flux),1}(undef,numlegs)
    Ns = size(psi,1)
    site = left ? 1 : Ns
    for i = 1:numlegs
      if i == 1
        index = left ? 1 : ndims(dualpsi[Ns])
        LQNs[i] = copy(getQnum(index,1,dualpsi[site]))
      elseif i==numlegs
        if rightind == 0
          index = left ? 1 : ndims(dualpsi[Ns])
        else
          index = left ? 1 : rightind
        end
        LQNs[i] = copy(getQnum(index,1,psi[site]))
      else
        index = left ? 1 : ndims(mpovec[i-1][Ns])
        LQNs[i] = copy(getQnum(index,1,mpovec[i-1][site]))
      end
    end
    return LQNs
  end
  export Qstore
=#
    """
      makeBoundary(qind,newArrows[,retType=])

  makes a boundary tensor for an input from the quantum numbers `qind` and arrows `newArrows`; can also define type of resulting Qtensor `retType` (default `Float64`)

  #Note:
  +dense tensors are just ones(1,1,1,...)

  See also: [`makeEnds`](@ref)
  """
  function makeBoundary(dualpsi::K,psi::Z,mpovec::P...;left::Bool=true,rightind::intType=3) where {K <: MPS, Z <: MPS, P <: MPO}
    retType = elnumtype(dualpsi,psi,mpovec...)
    nrank = 2 + length(mpovec)
    boundary = ones(retType,ones(intType,nrank)...)
    if typeof(psi[1]) <: qarray

      Q = typeof(psi[1].flux)

      qind = Array{Q,1}(undef,nrank)
      Ns = length(psi)
      site = left ? 1 : Ns
      index = left ? 1 : rightind
      qind[1] = inv(getQnum(index,1,dualpsi[site]))
      qind[end] = copy(getQnum(index,1,psi[site]))
      for i = 1:length(mpovec)
        index = left ? 1 : ndims(mpovec[i][Ns])
        qind[i+1] = copy(getQnum(index,1,mpovec[i][site]))
      end

      thisQnumMat = Array{Array{Q,1},1}(undef,nrank)
      for j = 1:nrank
        qn = qind[j]
        thisQnumMat[j] = Q[qn]
      end
      return Qtens(boundary,thisQnumMat)
    else
      if typeof(psi[1]) <: denstens
        return tens(boundary)
      else
        return boundary
      end
    end
  end
  export makeBoundary

  """
      assignflux!(i,mps,QnumMat,storeVal)
  
  Assigns flux to the right link index on an MPS tensor

  # Arguments
  + `i::Int64`: current position
  + `mps::MPS`: MPS
  + `QnumMat::Array{Array{Qnum,1},1}`: quantum number matrix for the physical index
  + `storeVal::Array{T,1}`: maximum value found in MPS tensor, determine quantum number
  """
  function assignflux!(i::intType,mps::MPS,QnumMat::Array{Array{Q,1},1},storeVal::Array{T,1}) where {Q <: Qnum, T <: Number}
    for c = 1:size(mps[i],3)
      for b = 1:size(mps[i],2),a = 1:size(mps[i],1)
        absval = abs(mps[i][a,b,c])
        if absval > storeVal[c]
          storeVal[c] = absval
          QnumMat[3][c] = inv(QnumMat[1][a]+QnumMat[2][b])
        end
      end
    end
    nothing
  end

  """
      makeqMPS(mps,Qlabels[,arrows,newnorm=,setflux=,flux=,randomize=,override=])

  creates quantum number MPS from regular MPS according to `Qlabels`

  # Arguments
  + `mps::MPS`: dense MPS
  + `Qlabels::Array{Array{Qnum,1},1}`: quantum number labels on each physical index (assigns physical index labels mod size of this vector)
  + `arrows::Array{Bool,1}`: first entry of this tuple is the arrow convention on MPS tensors (default: [false,true,true])
  + `newnorm::Bool`: set new norm of the MPS tensor
  + `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
  + `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
  + `randomize::Bool`: randomize last tensor if flux forces a zero tensor
  + `warning::Bool`: Toggle warning message if last tensor has no values or is zero
  """
  function makeqMPS(mps::MPS,Qlabels::Array{Array{Q,1},1},arrows::Array{Bool,1}...;newnorm::Bool=true,setflux::Bool=false,
                    flux::Q=Q(),randomize::Bool=true,override::Bool=true,warning::Bool=true,lastfluxzero::Bool=false)::MPS where Q <: Qnum
    if newnorm
      if warning
        println("(makeqMPS: If converting from a non-QN MPS to a QN MPS, then beware that applying a 3S or adding noise in general to the MPS is not considered when calling makeqMPS)")
      end
      start_norm = expect(mps)
    end
    W = elnumtype(mps)
    QtensVec = Array{Qtens{W,Q},1}(undef, size(mps.A,1))
    thisflux  = Q()
    zeroQN = copy(thisflux)
    Ns = length(mps)
    storeQnumMat = Q[zeroQN]
    theseArrows = length(arrows) == 0 ? Bool[false,true,true] : arrows[1]
    for i = 1:Ns
      currSize = size(mps[i])
      QnumMat = Array{Q,1}[Array{Q,1}(undef,currSize[a]) for a = 1:ndims(mps[i])]

      QnumMat[1] = inv.(storeQnumMat)
      QnumMat[2] = copy(Qlabels[(i-1) % size(Qlabels,1) + 1])
      storeVal = zeros(eltype(mps[i]),size(mps[i],3))
      if i < Ns
        assignflux!(i,mps,QnumMat,storeVal)
      else
        if setflux
          QnumMat[3][1] = flux
        else
          assignflux!(i,mps,QnumMat,storeVal)
        end
      end
      storeQnumMat = QnumMat[3]
      optblocks = i <= mps.oc ? [[1,2],[3]] : [[1],[2,3]]
      QtensVec[i] = Qtens(mps[i],QnumMat,currblock=optblocks)

      if size(QtensVec[i].T,1) == 0 && randomize
        QtensVec[i] = rand(QtensVec[i])
        if size(QtensVec[i].T,1) == 0 && !override
          error("specified bad quantum number when making QN MPS...try a different quantum number")
        end
      end
    end
    finalMPS = MPS(QtensVec,mps.oc)

    thisnorm = expect(finalMPS)

    if newnorm
#      @assert(!isapprox(thisnorm,0.))
      finalMPS[mps.oc] *= sqrt(start_norm)/sqrt(thisnorm)
    end
    if lastfluxzero
      for q = 1:length(finalMPS[end].Qblocksum)
        finalMPS[end].Qblocksum[q][2] = inv(finalMPS[end].flux)
      end
    else
      Qnumber = finalMPS[end].QnumMat[3][1]
      finalMPS[end].flux,newQnum = inv(finalMPS[end].QnumSum[3][Qnumber]),inv(finalMPS[end].flux)
      finalMPS[end].QnumSum[3][1] = newQnum
    end

    for q = 1:length(finalMPS[end].Qblocksum)
      newQsum = Q()
      index = finalMPS[end].ind[q][2][:,1] .+ 1 #[x]
      pos = finalMPS[end].currblock[2]
      for y = 1:length(pos)
        Qnumber = finalMPS[end].QnumMat[pos[y]][index[y]]
        add!(newQsum,finalMPS[end].QnumSum[pos[y]][Qnumber])
      end
      finalMPS[end].Qblocksum[q][2] = newQsum
    end

    return finalMPS #sets initial orthogonality center at site 1 (hence the arrow definition above)
  end

  """
      makeqMPS(mps,Qlabels[,arrows,newnorm=,setflux=,flux=,randomize=,override=])

  creates quantum number MPS from regular MPS according to `Qlabels`

  # Arguments
  + `mps::MPS`: dense MPS
  + `Qlabels::Array{Qnum,1}`: quantum number labels on each physical index (uniform physical indices)
  + `arrows::Array{Bool,1}`: first entry of this tuple is the arrow convention on MPS tensors (default: [false,true,true])
  + `newnorm::Bool`: set new norm of the MPS tensor
  + `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
  + `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
  + `randomize::Bool`: randomize last tensor if flux forces a zero tensor
  + `warning::Bool`: Toggle warning message if last tensor has no values or is zero
  """
  function makeqMPS(mps::MPS,Qlabels::Array{Q,1},arrows::Array{Bool,1}...;newnorm::Bool=true,setflux::Bool=false,
    flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where Q <: Qnum
    return makeqMPS(mps,[Qlabels],arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
  end

  function makeqMPS(arr::Array,Qlabels::W,arrows::Array{Bool,1}...;oc::intType=1,newnorm::Bool=true,setflux::Bool=false,
                    flux::Q=Q(),randomize::Bool=true,override::Bool=true,warning::Bool=true,lastfluxzero::Bool=false)::MPS where W <: Union{Array{Array{Q,1},1},Array{Q,1}} where Q <: Qnum
    mps = MPS(arr,oc)
    makeqMPS(mps,Qlabels,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,warning=warning,lastfluxzero=lastfluxzero)
  end
  export makeqMPS

  #       +---------------------------------------+
  #>------+           convert to qMPO             +---------<
  #       +---------------------------------------+

  """
      makeqMPO(mpo,Qlabels[,arrows])

  Generates quantum number MPO from a dense Hamiltonian based on `Qlabels`

  # Arguments:
  + `mpo::MPO`: dense MPO
  + `Qlabels::Array{Array{Qnum,1},1}`: quantum numbers for physical indices (modulus size of vector)
  + `arrows::Array{Bool,1}`: arrow convention for quantum numbers (default: [false,false,true,true])
  """
  function makeqMPO(mpo::MPO,Qlabels::Array{Array{Q,1},1},arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::intType=1)::MPO where Q <: Qnum
    zeroQN = Q()
    Ns = infinite ? 3*unitcell*length(mpo) : length(mpo)
    W = elnumtype(mpo)
    QtensVec = Array{Qtens{W,Q},1}(undef, Ns)
    storeQnumMat = Q[zeroQN]
    theseArrows = length(arrows) == 0 ? Bool[false,false,true,true] : arrows[1]
    zeroQN = Q()
    tempQN = Q()
    for w = 1:Ns
      i = (w-1) % length(mpo) + 1
      QnumMat = Array{Q,1}[Array{Q,1}(undef,size(mpo[i],a)) for a = 1:ndims(mpo[i])]

      QnumMat[1] = inv.(storeQnumMat)
      theseQN = Qlabels[(i-1) % size(Qlabels,1) + 1]
      QnumMat[2] = inv.(theseQN)
      QnumMat[3] = theseQN
      storeVal = -ones(Float64,size(mpo[i],4))
      for d = 1:size(mpo[i],4)
        for c = 1:size(mpo[i],3), b = 1:size(mpo[i],2), a = 1:size(mpo[i],1)
          absval = abs(mpo[i][a,b,c,d])
          if absval > storeVal[d]
            storeVal[d] = absval
            copy!(tempQN,zeroQN)
            add!(tempQN,QnumMat[1][a])
            add!(tempQN,QnumMat[2][b])
            add!(tempQN,QnumMat[3][c])
            QnumMat[4][d] = inv(tempQN)
          end
        end
      end
      storeQnumMat = QnumMat[4]
      baseQtens = Qtens(QnumMat,currblock=[[1,2],[3,4]])
      QtensVec[i] = Qtens(mpo[i],baseQtens)
    end
    T = prod(a->eltype(mpo[a])(1),1:length(mpo))
    if infinite
      finalQtensVec = QtensVec[unitcell*length(mpo)+1:2*unitcell*length(mpo)]
    else
      finalQtensVec = QtensVec
    end
    finalMPO = MPO(typeof(T),finalQtensVec)
    return finalMPO
  end

  """
      makeqMPO(mpo,Qlabels[,arrows])

  Generates quantum number MPO from a dense Hamiltonian based on `Qlabels`

  # Arguments:
  + `mpo::MPO`: dense MPO
  + `Qlabels::Array{Qnum,1}`: quantum numbers for physical indices (uniform)
  + `arrows::Array{Bool,1}`: arrow convention for quantum numbers (default: [false,false,true,true])
  """
  function makeqMPO(mpo::MPO,Qlabels::Array{Q,1},arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::intType=1)::MPO where Q <: Qnum
    return makeqMPO(mpo,[Qlabels],arrows...,infinite=infinite,unitcell=unitcell)
  end

  function makeqMPO(arr::Array,Qlabels::W,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::intType=1)::MPO where W <: Union{Array{Array{Q,1},1},Array{Q,1}} where Q <: Qnum
    mpo = convert2MPO(arr,infinite=infinite)
    return makeqMPO(mpo,Qlabels,arrows...,infinite=infinite,unitcell=unitcell)
  end
  export makeqMPO


  #       +---------------------------------------+
  #>------+    Automatic determination of MPO     +---------<
  #       +---------------------------------------+

  """
      reorder!(C[,Ncols=])

  Reorders the `Ncols` columns of `C` according to the Fiedler vector reordering in place if site is not 0

  See also: [`reorder`](@ref)
  """
  function reorder!(C::Array{W,2};Ncols::Integer=2) where W <: Number
      sitevec = vcat(C[:,1],C[:,2])
      for w = 3:Ncols
        sitevec = vcat(sitevec,C[:,w])
      end
      Ns = maximum(sitevec)
      A = zeros(Int64,Ns,Ns) #adjacency matrix = neighbor table
      D = zeros(Int64,Ns) #degree matrix
      for i = 1:size(C,1)
        for x = 1:Ncols
          for w = x+1:Ncols
            xpos = C[i,x]
            ypos = C[i,w]
            if xpos != 0 && ypos != 0
              A[xpos,ypos] = 1
              D[xpos] += 1
              D[ypos] += 1
            end
          end
        end
      end
      L = D - A
      D,U = LinearAlgebra.eigen(L)
      fiedlervec = sortperm(U[:,2]) #lowest is all ones, so this is the first non-trivial one
      for i = 1:size(C,1)
        for w = 1:Ncols
          if C[i,w] != 0
            C[i,w] = fiedlervec[C[i,w]]
          end
        end
      end
      return C,fiedlervec #second eigenvector is the Fiedler vector
    end

    """
        reorder(C[,Ncols=])

    Reorders the `Ncols` columns of `C` according to the Fiedler vector reordering if site is not 0
  
    See also: [`reorder!`](@ref)
    """
    function reorder(C::Array{W,2};Ncols::Integer=2) where W <: Number
      P = copy(C)
      return reorder!(P,Ncols=Ncols)
    end

  """
      mpoterm(val,operator,ind,base,trail...)

  Creates an MPO from operators (`operator`) with a prefactor `val` on sites `ind`.  Must also provide a `base` function which are identity operators for each site.  `trail` operators can be defined (example: fermion string operators)

  # Example:
  ```julia
  Cup, Cdn, Nup, Ndn, Ndens, F, O, Id = fermionOps()
  base = [Id for i = 1:Ns]; #identity operators of the appropriate physical index size for each site
  CupCupdag = mpoterm(-1.,[Cup,Cup'],[1,2],base,F); #Cup_1 Cup^dag_2
  newpsi = applyMPO(psi,CupCupdag); #applies <Cup_1 Cup^dag_2> to psi
  expect(psi,newpsi);
  ```
  """
  function mpoterm(val::Number,operator::Array{W,1},ind::Array{intType,1},base::Array{X,1},trail::Y...)::MPO where {W <: AbstractArray, X <: AbstractArray, Y <: AbstractArray}
    opString = copy(base)
    for a = 1:size(ind,1)
      opString[ind[a]] = (a == 1 ? val : 1.)*copy(operator[a])
      if length(trail) > 0
        for b = 1:ind[a]-1
          opString[b] = contract(trail[1],2,opString[b],1)
        end
      end
    end
    return MPO(opString)
  end

  function mpoterm(operator::AbstractArray,ind::Array{intType,1},base::AbstractArray,trail::AbstractArray...)::MPO
    return mpoterm(1.,operator,ind,base,trail...)
  end
  export mpoterm

  """
      mpoterm(Qlabels,val,operator,ind,base,trail...)
  
  Same as `mpoterm` but converts to quantum number MPO with `Qlabels` on the physical indices

  See also: [`mpoterm`](@ref)
  """
  function mpoterm(Qlabels::Array{Array{Q,1},1},val::Number,operator::AbstractArray,ind::Array{intType,1},base::AbstractArray,trail::AbstractArray...)::MPO where Q <: Qnum
    return makeqMPO(mpoterm(val,operator,ind,base,trail...),Qlabels)
  end

  function mpoterm(Qlabels::Array{Array{Q,1},1},operator::AbstractArray,ind::Array{intType,1},base::AbstractArray,trail::AbstractArray...)::MPO where Q <: Qnum
    return mpoterm(Qlabels,1.,operator,ind,base,trail...)
  end
  export mpoterm

  import Base.*
  """
      mpo * mpo

  functionality for multiplying MPOs together; contracts physical indices together
  """
  function *(X::MPO,Y::MPO)
    return mult!(copy(X),Y)
  end

  import .Qtensor.mult!
  function mult!(X::MPO,Y::MPO)
    for i = 1:size(X,1)
      temp = contract([4,1,2,5,6,3],X[i],3,Y[i],2)
      X[i] = reshape(temp,size(temp,1)*size(temp,2),size(temp,3),size(temp,4),size(temp,5)*size(temp,6))
    end
    return deparallelize!(X)
  end

  import Base.+
  """
      A + B

  functionality for adding (similar to direct sum) of MPOs together; uses joinindex function to make a combined MPO

  note: deparallelizes after every addition

  See also: [`deparallelization`](@ref) [`add!`](@ref)
  """
  function +(X::MPO,Y::MPO)
    checktype = typeof(eltype(X[1])(1) * eltype(Y[1])(1))
    if checktype != eltype(X[1])
      S = MPO(checktype,copy(X))
    else
      S = copy(X)
    end
    return add!(S,Y)
  end

  """
      H + c

  Adds a constant `c` to a Hamiltonian `H` (commutative)
  """
  function +(H::MPO,c::Number;pos::intType=1)
    const_term = MPO([i == pos ? mult!(c,makeId(H[i],[2,3])) : makeId(H[i],[2,3]) for i = 1:length(H)])
    return copy(H) + const_term
  end

  function +(c::Number,H::MPO;pos::intType=1)
    return +(H,c,pos=pos)
  end
  
  import Base.-
  """
      H - c

  Adds a constant `c` to a Hamiltonian `H`
  """
  function -(H::MPO,c::Number;pos::intType=1)
    return +(H,-c,pos=pos)
  end

  import .QN.add!
  """
      add!(A,B)

  functionality for adding (similar to direct sum) of MPOs together and replacing `A`; uses joinindex function to make a combined MPO

  note: deparallelizes after every addition

  See also: [`deparallelization`](@ref) [`+`](@ref)
  """
  function add!(A::MPO,B::MPO;finiteBC::Bool=true)
    if finiteBC
      A[1] = joinindex!(4,A[1],B[1])
      for a = 2:size(A,1)-1
        A[a] = joinindex!([1,4],A[a],B[a])
      end
      A[end] = joinindex!(1,A[size(A,1)],B[size(A,1)])
    else
      for a = 1:size(A,1)
        A[a] = joinindex!([1,4],A[a],B[a])
      end
    end
    return deparallelize!(A)
  end

  """
      makeInvD(D)

  Finds nearest factor of 2 to the magnitude of `D`
  """
  function makeInvD(D::TensType)
    avgD = sum(D)
    avgD /= size(D,1)
    finaltwo = 2^(max(0,convert(intType,floor(log(2,avgD))))+1)
    return finaltwo
  end

  function pullvec(M::TensType,j::intType,left::Bool)
    if typeof(M) <: qarray
      if left
        temp = M[:,:,:,j]
        firstvec = reshape(temp,size(M,1),size(M,2),size(M,3),1)

        newqn = [inv(firstvec.flux)]
        firstvec.QnumMat[4] = newqn
        firstvec.QnumSum[4] = newqn
        firstvec.flux = typeof(M.flux)()
      else
        temp = M[j,:,:,:]
        firstvec = reshape(temp,size(M,2),size(M,3),size(M,4),1)

        newqn = [inv(firstvec.flux)]
        firstvec.QnumMat[4] = newqn
        firstvec.QnumSum[4] = newqn
        firstvec.flux = typeof(M.flux)()
        firstvec = permutedims!(firstvec,[4,1,2,3])
      end
    else
      if left
        firstvec = reshape!(M[:,:,:,j],size(M,1),size(M,2),size(M,3),1)
      else
        firstvec = reshape!(M[j,:,:,:],1,size(M,2),size(M,3),size(M,4))
      end
    end
    return firstvec
  end

  """
      deparallelize!(M[,left=])

  Deparallelizes a matrix-equivalent of a rank-4 tensor `M`; toggle the decomposition into the `left` or `right`
  """
  function deparallelize!(M::TensType;left::Bool=true,zero::Float64=0.) where W <: Number
    if left
      rsize = size(M,4)
      lsize = max(prod(a->size(M,a),1:3),rsize)
    else
      lsize = size(M,1)
      rsize = max(lsize,prod(a->size(M,a),2:4))
    end
    T = zeros(eltype(M),lsize,rsize) #maximum size for either left or right
    firstvec = pullvec(M,1,left)

    K = typeof(M)[firstvec]
    normK = eltype(M)[norm(K[1])]

    b = left ? 4 : 1
    for j = 1:size(M,b)
      thisvec = pullvec(M,j,left)

      mag_thisvec = norm(thisvec) # |A|

      condition = true
      i = 0
      while condition  && i < size(K,1)
        i += 1
        if left
          dot = ccontract(K[i],thisvec)#,left,!left)
        else
          dot = contractc(K[i],thisvec)
        end
#        dot = searchindex(temp,1,1)
        if isapprox(real(dot),mag_thisvec * normK[i]) && !isapprox(normK[i],0) #not sure why it would be zero...
          normres = mag_thisvec/normK[i]
          if left
            T[i,j] = normres
          else
            T[j,i] = normres
          end
          condition = false
        end
      end

      if condition && !(isapprox(mag_thisvec,0.))

        push!(K,thisvec)
        push!(normK,norm(K[end]))

        if left
          T[length(K),j] = 1.
        else
          T[j,length(K)] = 1.
        end
      end
    end

    QNflag = typeof(M) <: qarray
    if left
      newT = T[1:size(K,1),:]
      if size(K,1) == 1
        finalT = reshape(newT,1,size(T,2))
      else
        finalT = newT
      end

      newK = K[1]
      for a = 2:size(K,1)
        newK = joinindex!(newK,K[a],4)
      end

      if QNflag
        finalT = Qtens(finalT,[inv.(newK.QnumMat[4]),M.QnumMat[4]])
      end

      return newK,finalT
    else
      newT = T[:,1:size(K,1)]
      if size(K,1) == 1
        finalT = reshape(newT,size(T,1),1)
      else
        finalT = newT
      end

      newK = K[1]
      for a = 2:size(K,1)
        newK = joinindex!(newK,K[a],1)
      end

      if QNflag
        finalT = Qtens(finalT,[M.QnumMat[1],inv.(newK.QnumMat[1])])
      end

      return finalT,newK
    end
  end

  function deparallelize!(M::tens{W};left::Bool=true) where W <: Number
    X = reshape(M.T,size(M)...)
    out = deparallelize!(X,left=left)
    return tens(out[1]),tens(out[2])
  end

  """
      deparallelize!(W[,sweeps=])

  Applies `sweeps` to MPO (`W`) to compress the bond dimension
  """
  function deparallelize!(W::MPO;sweeps::intType=1)
    for n = 1:sweeps
      for i = 1:length(W)-1
        W[i],T = deparallelize!(W[i],left=true)
        W[i+1] = contract(T,2,W[i+1],1)
      end
      for i = length(W):-1:2
        T,W[i] = deparallelize!(W[i],left=false)
        W[i-1] = contract(W[i-1],4,T,1)
      end
    end
    return W
  end
  export deparallelize!

  """
      deparallelize!(W[,sweeps=])

  Deparallelize an array of MPOs (`W`) for `sweeps` passes; compressed MPO appears in first entry
  """
  function deparallelize!(W::Array{MPO,1};sweeps::Integer=1)
    nlevels = floor(intType,log(2,size(W,1)))
    active = Bool[true for i = 1:size(W,1)]
    if size(W,1) > 2
      for j = 1:nlevels
        currsize = fld(length(W),2^(j-1))
        let currsize = currsize, W = W, j = j
          #=Threads.@threads=# for i = 1:2^j:currsize
            iL = i
            iR = iL + 2^(j-1)
            if iR < currsize
              add!(W[iL],W[iR])
              W[iL] = deparallelize!(W[iL],sweeps=sweeps)
              active[iR] = false
            end
          end
        end
      end
      if sum(active) > 1
        deparallelize!(W[active],sweeps=sweeps)
      end
    end
    if size(W,1) == 2
      W[1] = add!(W[1],W[2])
    end
    return deparallelize!(W[1],sweeps=sweeps)
  end

  """
      deparallelize(W[,sweeps=])

  makes copy of W while deparallelizing

  See also: [`deparallelize!`](@ref)
  """
  function deparallelize(W::MPO;sweeps::intType=1)
    return deparallelize!(copy(W),sweeps=sweeps)
  end

  function deparallelize(W::T;sweeps::intType=1) where T <: Union{AbstractArray,qarray}
    return deparallelize!(copy(W),sweeps=sweeps)
  end
  export deparallelize

  """
      compressMPO!(W[,sweeps=,cutoff=,deltam=,minsweep=,nozeros=])
  
  compresses MPO (`W`; or several `M`) with SVD compression for `sweeps` sweeps, `cutoff` applied to the SVD, `deltam` target for teh bond dimension compression, and `nozeros` defaulted to true to eliminate all zeros in the SVD
  """
  const forwardshape = Array{intType,1}[intType[1,2,3],intType[4]]
  const backwardshape = Array{intType,1}[intType[1],intType[2,3,4]]
  function compressMPO!(W::MPO,M::MPO...;sweeps::intType=1000,cutoff::Float64=1E-16,
                        deltam::intType=0,minsweep::intType=1,nozeros::Bool=false)
    for a = 1:length(M)
      W = add!(W,M[a])
    end
    n = 0
    mchange = 1000
    lastmdiff = [size(W[i],4) for i = 1:size(W,1)-1]
    while (n < sweeps && mchange > deltam) || (n < minsweep)
      n += 1
      for i = 1:size(W,1)-1
        U,D,V = svd(W[i],forwardshape,cutoff=cutoff,nozeros=nozeros)
        scaleD = makeInvD(D)

        U = mult!(U,scaleD)
        W[i] = U

        scaleDV = contract(D,2,V,1,alpha=1/scaleD)
        W[i+1] = contract(scaleDV,2,W[i+1],1)
      end
      for i = size(W,1):-1:2
        U,D,V = svd(W[i],backwardshape,cutoff=cutoff,nozeros=nozeros)
        scaleD = makeInvD(D)
        
        V = mult!(V,scaleD)
        W[i] = V

        scaleUD = contract(U,2,D,1,alpha=1/scaleD)
        W[i-1] = contract(W[i-1],4,scaleUD,1)
      end
      thismdiff = intType[size(W[i],4) for i = 1:size(W,1)-1]
      mchange = sum(a->lastmdiff[a]-thismdiff[a],1:size(thismdiff,1))
      lastmdiff = copy(thismdiff)
    end
    return W
  end

  """
      compressMPO!(W[,sweeps=,cutoff=,deltam=,minsweep=,nozeros=])
  
  compresses an array of MPOs (`W`) in parallel with SVD compression for `sweeps` sweeps, `cutoff` applied to the SVD, `deltam` target for teh bond dimension compression, and `nozeros` defaulted to true to eliminate all zeros in the SVD
  """
  function compressMPO!(W::Array{MPO,1};sweeps::intType=1000,cutoff::Float64=1E-16,
          deltam::intType=0,minsweep::intType=1,nozeros::Bool=true) # where V <: Union{Array{MPO,1},SubArray{MPO,1,Array{MPO,1},Tuple{Array{intType,1}}}}
    nlevels = floor(intType,log(2,size(W,1)))
    active = Bool[true for i = 1:size(W,1)]
    if size(W,1) > 2
      for j = 1:nlevels
        currsize = fld(length(W),2^(j-1))
        let currsize = currsize, W = W, sweeps = sweeps, cutoff = cutoff, j = j, nozeros=nozeros
          #=Threads.@threads=# for i = 1:2^j:currsize
            iL = i
            iR = iL + 2^(j-1)
            if iR < currsize
              W[iL] = compressMPO!(W[iL],W[iR],sweeps=sweeps,cutoff=cutoff,nozeros=nozeros)
              active[iR] = false
            end
          end
        end
      end
      if sum(active) > 1
        compressMPO!(W[active],sweeps=sweeps,cutoff=cutoff,nozeros=nozeros)
      end
    end
    return size(W,1) == 2 ? compressMPO!(W[1],W[2],nozeros=nozeros) : compressMPO!(W[1],nozeros=nozeros)
  end
  export compressMPO!

  """
      compressMPO(W,sweeps=,cutoff=,deltam=,minsweep=,nozeros=])
  
  Same as `compressMPO!` but a copy is made of the original vector of MPOs

  See also: [`compressMPO!`](@ref)
  """
  function compressMPO(W::Array{MPO,1};sweeps::intType=1000,cutoff::Float64=1E-16,deltam::intType=0,minsweep::intType=1,nozeros::Bool=true)
    M = copy(W)
    return compressMPO!(M;sweeps=sweeps,cutoff=cutoff,deltam=deltam,minsweep=minsweep,nozeros=nozeros)
  end
  export compressMPO

#       +--------------------------------+
#>------+    Methods for excitations     +---------<
#       +--------------------------------+

  """
      penalty!(mpo,lambda,psi[,compress=])

  Adds penalty to Hamiltonian (`mpo`), H0, of the form H0 + `lambda` * |`psi`><`psi`|; toggle to compress resulting wavefunction

  See also: [`penalty`](@ref)
  """
  function penalty!(mpo::MPO,lambda::Float64,psi::MPS;compress::Bool=true)
    for i = 1:length(psi)
      QS = size(psi[i],2)
      R = eltype(mpo[i])
      temp_psi = reshape(psi[i],size(psi[i])...,1)
      if i == psi.oc
        term = contractc(temp_psi,4,temp_psi,4,alpha=lambda)
      else
        term = contractc(temp_psi,4,temp_psi,4)
      end
      bigrho = permutedims(term,[1,4,5,2,3,6])
      rho = reshape!(bigrho,size(bigrho,1)*size(bigrho,2),QS,QS,size(bigrho,5)*size(bigrho,6),merge=true)
      if i == 1
        mpo[i] = joinindex!(4,mpo[i],rho)
      elseif i == length(psi)
        mpo[i] = joinindex!(1,mpo[i],rho)
      else
        mpo[i] = joinindex!([1,4],mpo[i],rho)
      end
    end
    return compress ? compressMPO!(mpo) : mpo
  end
  export penalty!

  """
      penalty!(mpo,lambda,psi[,compress=])

  Same as `penalty!` but makes a copy of `mpo`

  See also: [`penalty!`](@ref)
    """
  function penalty(mpo::MPO,lambda::Float64,psi::MPS;compress::Bool=true)
    newmpo = copy(mpo)
    return penalty!(newmpo,lambda,psi,compress=compress)
  end
  export penalty

  """
    transfermatrix([dualpsi,]psi,i,j[,transfermat=])

  Forms the transfer matrix (an MPS tensor and its dual contracted along the physical index) between sites `i` and `j` (inclusive). If not specified, the `transfermat` field will initialize to the transfer matrix from the `i-1` site.  If not set otherwise, `dualpsi = psi`.

  The form of the transfer matrix must be is as follows (dual wavefunction tensor on top, conjugated)

  1 ------------ 3
         |
         |
         |
         |
  2 ------------ 4

  """
  function transfermatrix(dualpsi::MPS,psi::MPS,i::Integer,j::Integer;startsite::Integer=i,transfermat::TensType=ccontract([1,3,2,4],dualpsi[startsite],2,psi[startsite],2))
    for k = startsite+1:j
      transfermat = contractc(transfermat,3,dualpsi[k],1)
      transfermat = contract(transfermat,[3,4],psi[k],[1,2])
    end
    return transfermat
  end

  function transfermatrix(psi::MPS,i::Integer,j::Integer;startsite::Integer=i,transfermat::TensType=ccontract([1,3,2,4],psi[startsite],2,psi[startsite],2))
    return transfermatrix(psi,psi,i,j,transfermat=transfermat)
  end
end