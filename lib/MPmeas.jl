#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker and M.P. Thompson (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

#       +---------------------------------------+
#>------+       measurement operations          +---------<
#       +---------------------------------------+



"""
  psi = applyOps!(psi,sites,Op[,trail=ones(1,1)])

Applies operator `Op` (any `TensType`) in-place to the MPS `psi` at sites `sites`, a vector of integers. A trailing operator `trail` can be applied if not the default.
"""
function applyOps!(psi::MPS,sites::Array{W,1},Op::TensType;trail::TensType=ones(1,1)) where W <: Integer
  def_trail = ones(1,1)
  #=@inbounds=# for i = 1:length(sites)
    site = sites[i]
    p = site
    psi[p] = contract([2,1,3],Op,2,psi[p],2)
    if trail != def_trail
      #=@inbounds=# for j = 1:p-1
        psi[j] = contract([2,1,3],trail,2,psi[j],2)
      end
    end
  end
  return psi
end
export applyOps!

"""
  newpsi = applyOps(psi,sites,Op[,trail=ones(1,1)])

Applies operator `Op` (any `TensType`) to the MPS `psi` at sites `sites`, a vector of integers. A trailing operator `trail` can be applied if not the default.
"""
function applyOps(psi::MPS,sites::Array{W,1},Op::TensType;trail::TensType=ones(1,1)) where W <: Integer
  cpsi = copy(psi)
  return applyOps!(cpsi,sites,Op,trail=trail)
end
export applyOps

"""
    applyMPO(psi,H[,m=1,cutoff=0.])

Applies MPO (`H`) to an MPS (`psi`) and truncates to maximum bond dimension `m` and cutoff `cutoff`
"""
function applyMPO(psi::MPS,H::MPO;m::Integer=0,cutoff::Float64=0.)
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
    
    finalpsi[i-1] = contract(newpsi[i-1],(4,5),tempT,1)
  end
  finalpsi[1] = reshape!(finalpsi[1],[[1,2],[3],[4]],merge=true)
  return MPS(finalpsi)
end

"""
    applyMPO(psi,H...[,m=1,cutoff=0.])

Applies MPOs (`H`) to an MPS (`psi`) and truncates to maximum bond dimension `m` and cutoff `cutoff`. Not recommended except for small problems since bond dimension is not truncated uniformly.
"""
function applyMPO(psi::MPS,H::MPO...;m::Integer=0,cutoff::Float64=0.)
  newpsi = psi
  for a = 1:length(H)
    newpsi = applyMPO(newpsi,H[i],m=m,cutoff=cutoff)
  end
  return newpsi
end
export applyMPO


"""
    expect(dualpsi,psi,H[,Lbound=,Rbound=,order=])

evaluate <`dualpsi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`); `dualpsi` is conjugated inside of the algorithm

See also: [`overlap`](@ref)
"""
function expect(dualpsi::MPS,psi::MPS,H::MPO...;Lbound::TensType=typeof(psi[1])(),Rbound::TensType=Lbound,order::intvecType=intType[])
  Ns = size(psi,1)
  nMPOs = size(H,1)
  nLsize = nMPOs+2
  nRsize = nLsize+1
  Lenv,Renv = makeEnds(dualpsi,psi,H...,Lbound=Lbound,Rbound=Rbound)

  for i = length(psi):-1:1
    Renv = ccontract(dualpsi[i],3,Renv,nLsize)
    for j = 1:nMPOs
      Renv = contract(H[j][i],(3,4),Renv,(2,nRsize))
    end
    Renv = contract(psi[i],(2,3),Renv,(2,nRsize))
  end

  if order == intType[]
    permvec = [i for i = ndims(Lenv):-1:1] #vcat([ndims(Lenv)],[i for i = ndims(Lenv)-1:-1:2],[1])
    modLenv = permutedims(Lenv,permvec)
  else
    modLenv = permutedims(Lenv,order)
  end
  
  return contract(modLenv,Renv)
end

"""
    expect(psi,H[,Lbound=,Rbound=,order=])

evaluate <`psi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`)

See also: [`overlap`](@ref)
"""
function expect(psi::MPS,H::MPO...;Lbound::TensType=typeof(psi[1])(),Rbound::TensType=Lbound,order::intvecType=intType[])
  return expect(psi,psi,H...,Lbound=Lbound,Rbound=Rbound,order=order)
end
export expect

"""
    correlationmatrix(dualpsi,psi,Cc,Ca[,F,silent=])

Compute the correlation funciton (example, <`dualpsi`|`Cc`_i . `Ca`_j|`psi`>) on all sites; can also specify a trail `F` for Fermion strings; `silent` toggles output of every line

# Note:
+ More efficient than using `mpoterm`s
+ Use `mpoterm` and `applyMPO` for higher order correlation functions or write a new function
"""
function correlationmatrix(dualpsi::MPS, psi::MPS, Cc::TensType, Ca::TensType; trail=[])
  rho = Array{eltype(psi[1]),2}(undef,size(psi,1),size(psi,1))
  if trail != []
    FCc = contract(Cc,2,trail,1)
  else
    FCc = Cc
  end
  diffTensors = !(psi == dualpsi)
  onsite = contract(Cc,2,Ca,1)
  for i = 1:size(psi,1)
    move!(psi,i)
    if diffTensors
      move!(dualpsi,i)
    end
    TopTerm = contract([2,1,3],onsite,2,psi[i],2)
    rho[i,i] = contractc(TopTerm,dualpsi[i])
  end
  for i = 1:size(psi,1)-1
    move!(psi,i)
    if diffTensors
      move!(dualpsi,i)
    end
    TopTerm = contract(FCc,2,psi[i],2)
    Lenv = contractc(TopTerm,(2,1),dualpsi[i],(1,2))
    for j = i+1:size(psi,1)
      Renv = contract(Ca,2,psi[j],2)
      Renv = contractc(Renv,(1,3),dualpsi[j],(2,3))
      DMElement = contract(Lenv,Renv)
      if j < size(psi,1)
        if trail != []
          Lenv = contract(Lenv,1,psi[j],1)
          Lenv = contract(Lenv,2,trail,2)
          Lenv = contractc(Lenv,(1,3),dualpsi[j],(1,2))
        else
          Lenv = contract(Lenv, 1, psi[j], 1)
          Lenv = contractc(Lenv, (1,2), dualpsi[j], (1,2))
        end
      end
      rho[i,j] = DMElement
      rho[j,i] = conj(DMElement)
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
function correlationmatrix(psi::MPS, Cc::TensType, Ca::TensType; trail=[])
  return correlationmatrix(psi,psi,Cc,Ca,trail=trail)
end
export correlationmatrix































function localizeOp(psi::MPS,Oparray::Array{G,1},sites::Array{R,1};centerpsi::TensType=psi[psi.oc],order::Array{intType,1}=[1,2,3],trail::Tuple=()) where {G <: TensType, R <: Integer}

  #trail operations....
  isId = [isapprox(norm(trail[r])^2,size(trail[r],1)) for r = 1:length(trail)]
  if length(trail) > 0
    for r = 1:length(isId)
      index = 0
      #=@inbounds=# while isId[r]
        index += 1
        isId[r] = searchindex(trail[r],index,index) == 1
      end
    end
  end


  #needs to incorporate trail operators between operators.

#  Lenv = makeBoundary(psi,psi)
  minsite = minimum(sites)
  maxsite = maximum(sites)
  if minsite < psi.oc
    if length(isId) > 0 && sum(isId) != 0
      Lenv = makeBoundary(psi,psi)
      for w = minsite+1:psi.oc-1
        if w in sites
          p = findfirst(r->w==sites[r],1:length(sites))
          temp = contract([2,1,3],Oparray[p],2,psi[w],2)
        end
        temp = contract([2,1,3],trail[1],2,temp,2)
        Lenv = Lupdate(Lenv,psi[w],temp)
      end
    else
      p = findfirst(r->minsite==sites[r],1:length(sites))
      psiOp = contract(Oparray[p],2,psi[minsite],2)
      Lenv = ccontract(psiOp,(2,1),psi[minsite],(1,2))
    end
    for w = minsite+1:psi.oc-1
      if w in sites
        p = findfirst(r->w==sites[r],1:length(sites))
        temp = contract([2,1,3],Oparray[p],2,psi[w],2)
      else
        temp = psi[w]
      end
      Lenv = Lupdate(Lenv,psi[w],temp)
    end
  else
    Lenv = ccontract(psi[psi.oc-1],(1,2),psi[psi.oc-1],(1,2))
  end

  if psi.oc in sites
    p = findfirst(r->psi.oc==sites[r],1:length(sites))
#    cpsi = contract([2,1,3],Oparray[p],2,centerpsi,2)
    outOp = Oparray[p]
  else
    outOp = makeId(eltype(psi[1]),size(psi[psi.oc],2))
#    cpsi = centerpsi
  end

#  Renv = makeBoundary(psi,psi,left=false)
  if maxsite > psi.oc
    p = findfirst(r->maxsite==sites[r],1:length(sites))
    psiOp = contract(Oparray[p],2,psi[maxsite],2)
    Renv = contractc(psiOp,(1,3),psi[maxsite],(2,3))
    for w = maxsite-1:-1:psi.oc+1
      if w in sites
        p = findfirst(r->w==sites[r],1:length(sites))
        temp = contract([2,1,3],Oparray[p],2,psi[w],2)
      else
        temp = psi[w]
      end
      Renv = Rupdate(Renv,psi[w],temp)
    end
  else
    Renv = contractc(psi[psi.oc+1],(2,3),psi[psi.oc+1],(2,3))
  end
#  Lpsi = contract(Lenv,2,outOp,1)
  return Lenv,outOp,Renv #contract(order,Lpsi,3,Renv,1)
end

function localizeOp(psi::MPS,mpo::MPO...;centerpsi::TensType=psi[psi.oc],Hpsi::Function=singlesite_update) where {G <: TensType, R <: Integer}
  Lenv,Renv = makeEnv(psi,mpo...)
  nMPOs = length(mpo)
  nLsize = nMPOs+2
#  bundleMPS = ndims(centerpsi) == 4
  Hpsi = contract(Renv,1,centerpsi,3)
  LHpsi = contract(Lenv,2,mpo[1][psi.oc],1)
  for w = 2:nMPOs
    LHpsi = contract(LHpsi,(2,nMPOs+1+w),mpo[w][psi.oc],(1,2))
  end

  tup_Renv = ntuple(i->i,nMPOs)
  tup_Renv = (nMPOs+2,nMPOs+3,tup_Renv...)

  tup_Lenv = ntuple(i->3+i,nMPOs)
  tup_Lenv = (1,3,tup_Lenv...)

  return contract(LHpsi,tup_Lenv,HRenv,tup_Renv)
end
export localizeOp


























"""
  operator_in_order!(pos,sizes)

Increments elements of input vector `pos` with sizes of a tensor `sizes` such that `pos` all elements are ordered least to greatest.  For use in `correlation` function.
"""
@inline function operator_in_order!(pos::Array{G,1},sizes::intvecType) where G <: Integer
  w = length(pos)
  pos[w] += 1
  while w > 1 && pos[w] > sizes[w]
    w -= 1
    #=@inbounds=# pos[w] += 1
    @simd for x = w:length(pos)-1
      #=@inbounds=# pos[x+1] = pos[x]
    end
  end
  nothing
end

#heap algorithm for permutations (non-recursive)...
"""
  G = permutations(nelem)

Heap algorithm for finding permutations of `nelem` elements. Output `G` is an Vector of all permutations stored as vectors.  For example, a permutation of [1,2,3,4] is [2,1,3,4].
"""
function permutations(nelem::Integer)
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
export permutations

"""
correlation(psi,inputoperators...[,sites=ntuple(i->1,length(inputoperators)),trail=()])

Computes the general correlation on the input MPS `psi` with operators defined in the tuple `inputoperators`. The `sites` command specifies which sites to allow for the operators to be applied (i.e., some operators can skip some sites) and the `trail` option allows for a trailing operator like the Fermion sign operator.

See also: [`expect`](@ref) [`correlationmatrix`](@ref)
"""
function correlation(psi::MPS, inputoperators...;
                      sites::intvecType=ntuple(i->1:length(psi),length(inputoperators)),
                      trail::Tuple=()) #where S <: Union{Vector{AbstractMatrix{Float64}},TensType}

  return correlation(psi,psi,inputoperators...,sites=sites,trail=trail)
end
"""
  correlation(dualpsi,psi,inputoperators...[,sites=ntuple(i->1,length(inputoperators)),trail=()])

Computes the general correlation on the input MPS `psi` with operators defined in the tuple `inputoperators`. The `sites` command specifies which sites to allow for the operators to be applied (i.e., some operators can skip some sites) and the `trail` option allows for a trailing operator like the Fermion sign operator.

See also: [`expect`](@ref) [`correlationmatrix`](@ref)
"""
function correlation(dualpsi::MPS, psi::MPS, inputoperators...;
                      sites::intvecType=ntuple(i->1:length(psi) - (ndims(inputoperators[i]) == 1 ? length(inputoperators[i]) : 0),length(inputoperators)),
                      trail::Union{Tuple,TensType}=()) #where S <: Union{Vector{AbstractMatrix{Float64}},TensType}

  move!(psi,1)
  move!(dualpsi,1)

  if length(trail) != 0
    if typeof(trail) <: Tuple || ndims(trail) == 1
      subtrail = [trail[(w-1) % length(inputoperators) + 1] for w = 1:length(inputoperators)]
    elseif trail != ()
      subtrail = [trail for w = 1:length(inputoperators)]
    end
  end

  isId = [isapprox(norm(subtrail[r])^2,size(subtrail[r],1)) for r = 1:length(subtrail)]
  if length(subtrail) > 0
    for r = 1:length(isId)
      index = 0
      @inbounds while isId[r]
        index += 1
        isId[r] = searchindex(subtrail[r],index,index) == 1
      end
    end
  end

  Lenv,Renv = makeEnv(dualpsi,psi)

  savepsi = Array{typeof(psi[1]),1}(undef,length(psi))


  numops = length(inputoperators)
  operators = Array{Array{TensType,1},1}(undef,numops)
  lengthops = Array{intType,1}(undef,numops)
  for k = 1:numops
    if ndims(inputoperators[k]) == 1
      operators[k] = inputoperators[k]
      lengthops[k] = length(operators[k])
    else
      operators[k] = [inputoperators[k]]
      lengthops[k] = 1
    end
  end

  totalOPs = 1
  for w = 1:numops
    totalOPs *= length(sites[w])
  end




  temp = eltype(dualpsi[1])(1)
  temp *= eltype(psi[1])(1)
  for w = 1:numops
    @inbounds @simd for a = 1:lengthops[w]
      temp *= eltype(operators[w][a][1])(1)
    end
  end
  @inbounds @simd for r = 1:length(subtrail)
    temp *= eltype(subtrail[r])(1)
  end
  retType = typeof(temp)

  base_sizes = Array{intType,1}(undef,length(sites))
  @inbounds @simd for i = 1:length(sites)
    base_sizes[i] = length(sites[i])
  end


  pos = Array{intType,1}(undef,numops)

  base_pos = Array{intType,1}(undef,numops) #ones(intType,numops)

  omega = zeros(retType,base_sizes...)

  @inbounds @simd for w = 1:numops
    base_pos[w] = 1
  end

  while sum(w->base_pos[w]<=base_sizes[w],1:length(base_pos)) == length(base_sizes)

    @inbounds @simd for w = 1:length(psi)
      savepsi[w] = psi[w]
    end

    @inbounds @simd for w = 1:length(pos)
      pos[w] = sites[w][base_pos[w]]
    end

    maxopspos = 1

    @inbounds for g = numops:-1:2
      maxopspos = max(pos[g]+lengthops[g]-1,maxopspos)
      @inbounds for p = 1:lengthops[g]
        currsite = pos[g] + p-1
        savepsi[currsite] = contract([2,1,3],operators[g][p],2,savepsi[currsite],2)
      end
      if length(isId) > 0 && !isId[g]
        @inbounds for w = 1:pos[g]-1
          savepsi[w] = contract([2,1,3],subtrail[g],2,savepsi[w],2)
        end
      end
    end

    @inbounds for a = maxopspos:-1:2
      Renv[a-1] = Rupdate(Renv[a],dualpsi[a],savepsi[a])
    end

    @inbounds for y = 1:length(sites[1]) #w in sites[1]
      w = sites[1][y]
      thisLenv = Lenv[w]

      @inbounds for p = 1:lengthops[1]
        currsite = w + p-1

        newpsi = contract([2,1,3],operators[1][p],2,savepsi[currsite],2)
        thisLenv = Lupdate(thisLenv,dualpsi[currsite],newpsi)
      end
#=
      thisRenv = Renv[w+lengthops[1]-1] #
      res = contract(thisLenv,(1,2),thisRenv,(2,1))
      pos[1] = w
      omega[pos...] = res[1]
=#

thisRenv = permutedims(Renv[w+lengthops[1]-1],(2,1))
res = contract(thisLenv,thisRenv)
pos[1] = w
omega[pos...] = res

      if length(isId) > 0 && !isId[1]
        savepsi[w] = contract([2,1,3],subtrail[1],2,savepsi[w],2)
      end

      if w < sites[1][end]
        @inbounds for r = w:sites[1][y+1]
          Lenv[w+1] = Lupdate(Lenv[w],dualpsi[w],savepsi[w])
        end
      end
    end
    base_pos[1] = base_sizes[1]
    position_incrementer!(base_pos,base_sizes)
  end
  return omega
end
export correlation

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

There is no in-place version of this function

"""
function transfermatrix(dualpsi::MPS,psi::MPS,i::Integer,j::Integer;startsite::Integer=i,transfermat::TensType=ccontract([1,3,2,4],dualpsi[startsite],2,psi[startsite],2))
  for k = startsite+1:j
    transfermat = contractc(transfermat,3,dualpsi[k],1)
    transfermat = contract(transfermat,(3,4),psi[k],(1,2))
  end
  return transfermat
end

function transfermatrix(psi::MPS,i::Integer,j::Integer;startsite::Integer=i,transfermat::TensType=ccontract([1,3,2,4],psi[startsite],2,psi[startsite],2))
  return transfermatrix(psi,psi,i,j,transfermat=transfermat)
end
