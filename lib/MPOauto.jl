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


#       +---------------------------------------+
#>------+    Automatic determination of MPO     +---------<
#       +---------------------------------------+

"""
  MPOterm

Abstract type used to store terms of an MPO
"""
abstract type MPOterm end
export MPOterm

"""
  mpoterm{W}

Stores terms of an MPO
"""
struct mpoterm <: MPOterm #where W <: Tuple
  T::Vector{Any}
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
function mpoterm(val::Number,operator::Array{W,1},ind::Array{P,1},base::Array{X,1},trail::Y...)::MPO where {W <: densTensType, X <: densTensType, Y <: densTensType, P <: Integer}
  finalnum = typeof(val)(1)
  @inbounds @simd for w = 1:length(operator)
    finalnum *= typeof(operator[w][1])(1)
  end
  @inbounds @simd for w = 1:length(base)
    finalnum *= typeof(base[w][1])(1)
  end
  @inbounds @simd for w = 1:length(trail)
    finalnum *= typeof(trail[w][1])(1)
  end
  finalType = typeof(finalnum)

  applytrail = length(trail) > 0
  if applytrail
    fulltrail = [trail[(a-1) % length(trail) + 1] for a = 1:length(ind)]
  end

  isdens = W <: denstens || X <: denstens
  if isdens
    opString = Array{tens{finalType},1}(undef,length(base))
  else
    opString = Array{Array{finalType,2},1}(undef,length(base))
  end
  a = 0
  @inbounds for i = 1:length(base)
    opString[i] = base[i]
  end
  for i = length(ind):-1:1
    thisval = i == 1 ? val : 1.0
    opString[ind[i]] = contract(operator[i],2,opString[ind[i]],1,alpha=thisval)
    if applytrail
      for b = 1:ind[i]-1
        opString[b] = contract(fulltrail[i],2,opString[b],1)
      end
    end
  end
  return MPO(opString)
end

function mpoterm(operator::TensType,ind::Array{P,1},base::Array{G,1},trail::TensType...)::MPO where {P <: Integer, G <: TensType}
  return mpoterm(1.,operator,ind,base,trail...)
end

function mpoterm(val::Number,operator::TensType,ind::Integer,base::Array{G,1},trail::TensType...)::MPO where G <: densTensType
  return mpoterm(val,[operator],[ind],base,trail...)
end

function mpoterm(operator::TensType,ind::Integer,base::Array{G,1},trail::TensType...)::MPO where G <: densTensType
  return mpoterm(1.,[operator],[ind],base,trail...)
end

function mpoterm(val::Number,operator::Array{W,1},ind::Array{P,1},base::Array{X,1},trail::Y...)::MPO where {W <: qarray, X <: qarray, Y <: qarray, P <: Integer}
  Qlabels = [fullQnumMat(base[w])[2] for w = 1:length(base)]
  densbase = [makeArray(base[w]) for w = 1:length(base)]
  densop = [makeArray(operator[w]) for w = 1:length(operator)]
  if length(trail) > 0
    denstrail = makeArray(trail[1])
    opString = mpoterm(val,densop,ind,densbase,denstrail)
  else
    opString = mpoterm(val,densop,ind,densbase)
  end
  densmpo = MPO(opString)
  return makeqMPO(Qlabels,densmpo)
end

"""
  mpoterm(Qlabels,val,operator,ind,base,trail...)

Same as `mpoterm` but converts to quantum number MPO with `Qlabels` on the physical indices

See also: [`mpoterm`](@ref)
"""
function mpoterm(Qlabels::Array{Array{Q,1},1},val::Number,operator::Array,ind::Array{P,1},base::Array,trail::Array...)::MPO where {Q <: Qnum, P <: Integer}
  return makeqMPO(Qlabels,mpoterm(val,operator,ind,base,trail...))
end

function mpoterm(Qlabels::Array{Array{Q,1},1},operator::Array,ind::Array{P,1},base::Array,trail::Array...)::MPO where {Q <: Qnum, P <: Integer}
  return mpoterm(Qlabels,1.,operator,ind,base,trail...)
end
export mpoterm

function mpoterm(Op1::TensType,i::Integer,other...)# where U <: Union{Tuple,Array}
  return mpoterm(1.,Op1,i,other...)
end

function mpoterm(val::Number,Op1::TensType,i::Integer,other...)# where U <: Union{Tuple,Array}
  return mpoterm([(val,Op1,i,other...)])
end

import .DMRjulia.+
function +(A::MPOterm,B::MPOterm)
  return mpoterm(Any[A.T...,B.T...])
end

function +(int::Integer,tup::MPOterm)
  return tup
end

function length(R::MPOterm)
  return length(R.T)
end

function getindex(R::MPOterm,i::Integer)
  return R.T[i]
end


function prepare_autoInfo(opstring::MPOterm)
  #find the total number of sites
  Ns = 0
  @inbounds for y = 1:length(opstring)
    this_string = opstring[y]
    @inbounds for x = 1:fld(length(this_string)-1,2)
      Ns = max(this_string[2*x+1],Ns)
    end
  end

  qarrayops = false

  #find physical index sizes
  physind = zeros(intType,Ns)
  @inbounds for y = 1:length(opstring)
    this_string = opstring[y]
    @inbounds for x = 1:fld(length(this_string)-1,2)
      site = this_string[2*x+1]
      if physind[site] == 0
        physind[site] = size(this_string[2*x],2)
      elseif physind[site] != size(this_string[2*x],2)
        error("double definition of operator on site $site (had $(physind[site]) and also size $(size(this_string[2*x],2)) on two operators when they must match)")
      end
      if qarrayops && typeof(this_string[2*x]) <: densTensType
        error("must be all Qtensor all denstens or all Array types input into operator")
      end
      qarrayops = qarrayops || typeof(this_string[2*x]) <: qarray
    end
  end

  if qarrayops
    Qnumvec = Array{Array{typeof(opstring[1][2].flux),1},1}(undef,Ns)
    checksites = [true for i = 1:Ns]
    w = 0
    while sum(checksites) != 0 && w < length(opstring)
      w += 1
      this_string = opstring[w]
      for x = 1:fld(length(this_string)-1,2)
        ind = this_string[2*x+1]
        if checksites[ind]
          thisop = this_string[2*x]
          Qnumvec[ind] = recoverQNs(1,thisop)
          checksites[ind] = false
        end
      end
    end
  else
    Qnumvec = [U1[]]
  end

  #find element type for the MPO tensors
  valtype = 1.0
  @inbounds for w = 1:length(opstring)
    this_string = opstring[w]
    valtype *= typeof(this_string[1])(1)
    @inbounds @simd for g = 2:2:length(this_string)
      valtype *= eltype(this_string[g])(1)
    end
  end
  mpotype = typeof(valtype)

  base = [makeArray(makeId(mpotype,physind[w])) for w = 1:Ns]
  if qarrayops
    base = [Qtens(Qnumvec[i],base[i]) for i = 1:Ns]
  end

  return Ns,mpotype,base,Qnumvec,qarrayops,physind
end

function MPO(opstring::MPOterm,reverse::Bool=true,countreduce::intType=100,sweeps::intType=2)

  Ns,mpotype,base,Qnumvec,qarrayops,physinds = prepare_autoInfo(opstring)

  mpo = 0
  singlempo = 0
  manympo = 0

  singlesite = [1 < length(opstring[w]) < 5 for w = 1:length(opstring)]
  nosingles = findfirst(singlesite)
  if typeof(nosingles) <: Integer

    singleterms = Array{Array{mpotype,2},1}(undef,Ns)
    for i = 1:Ns
      Id = makeArray(base[i])
      O = zero(Id)
      singleterms[i] = mpotype[Id O; O Id]
    end

    singlempo = makeMPO(singleterms,physinds)

    if qarrayops
      singlempo = makeqMPO(Qnumvec,singlempo)
    end

    terms = findall(singlesite)
    for a in terms
      ind = opstring[a][3]
      value = opstring[a][1]
      operator = opstring[a][2]


      singlempo[ind][end,:,:,1] += operator * value

      trailon = length(opstring[a]) % 2 == 0

      if trailon
        trailvec = opstring[a][end]
        for g = 1:ind-1
          checkzero = true
          y = 0
          while checkzero && y < physind[ind]
            y += 1
            x = 0
            while checkzero && x < physind[ind]
              x += 1
              checkzero = searchindex(size(singleterms[g],1),x,y,1) == 0
            end
          end
          if checkzero
            singlempo[g][end,:,:,1] = trailvec[g]
          else
            singlempo[g][end,:,:,1] = contract(trailvec[g],2,singlempo[g][end,:,:,1],1)
          end
        end
      end
    end
  end

  regularterms = findall(w->length(opstring[w]) == 5 || length(opstring[w]) == 6 || length(opstring[w]) == 1,1:length(singlesite))
  mpovec = Array{Any,1}(undef,Threads.nthreads())
  value_mpo_vec = Array{Any,1}(undef,Threads.nthreads())
  for i = 1:length(mpovec)
    mpovec[i] = 0
    value_mpo_vec[i] = 0
  end

  Threads.@threads :static for a in regularterms

    numthread = Threads.threadid()

    if length(opstring) == 1
      value_mpo_vec[numthread] += opstring[1]
    else

      value = opstring[a][1]
      Opvec = [opstring[a][x] for x = 2:2:length(opstring[a])]
      posvec = [opstring[a][x] for x = 3:2:length(opstring[a])]

      trailon = length(opstring[a]) % 2 == 0
      if trailon
        mpovec[numthread] += mpoterm(value,Opvec,posvec,base,opstring[a][end])
      else
        mpovec[numthread] += mpoterm(value,Opvec,posvec,base)
      end
    end
  end
  for i = 1:length(mpovec)
    mpo += mpovec[i]
    mpo += value_mpo_vec[i]
  end

  mpo += singlempo



  manysite = [length(opstring[w]) > 6 for w = 1:length(opstring)]
  manysiteterms = findall(manysite)

  if sum(manysite) > 0

    manyvec = Array{Any,1}(undef,Threads.nthreads())
    for w = 1:length(manyvec)
      manyvec[w] = 0
    end

    counter = zeros(Int64,length(manyvec))

    Threads.@threads :static for a in manysiteterms

      numthread = Threads.threadid()

      value = opstring[a][1]
      Opvec = [opstring[a][x] for x = 2:2:length(opstring[a])]
      posvec = [opstring[a][x] for x = 3:2:length(opstring[a])]

      trailon = length(opstring[a]) % 2 == 0
      if trailon
        manyvec[numthread] *= mpoterm(value,Opvec,posvec,base,opstring[a][end])
      else
        manyvec[numthread] *= mpoterm(value,Opvec,posvec,base)
      end

      counter[numthread] += 1
#=
      if counter[numthread] % countreduce == 0
        println()
        println(numthread," ",size.(manyvec[numthread].H))
        manyvec[numthread] = compressMPO!(manyvec[numthread])
        println(numthread," ",size.(manyvec[numthread].H))
      end
=#
    end

    Threads.@threads :static for w = 1:length(manyvec)
      if manyvec[w] != 0 && !isapprox(sum(p->norm(manyvec[w][p]),1:length(manyvec[w])),0)
        manyvec[w] = compressMPO!(manyvec[w],sweeps=sweeps)
      end
    end

    for i = 1:length(manyvec)
      manympo *= manyvec[i]
    end

    if !isapprox(sum(p->norm(manympo[p]),1:length(manympo)),0)
      mpo *= compressMPO!(manympo,sweeps=sweeps)
      compressMPO!(mpo,sweeps=sweeps)
    end
  end


  if reverse # && typeof(mpo[1]) <: qarray
    for a = 1:length(mpo)
      mpo[a] = permutedims!(mpo[a],[1,3,2,4])
    end
  end

  return mpo #compressMPO!(mpo)
end


function expMPO(opstring::MPOterm)

  Ns,mpotype,base,Qnumvec,qarrayops = prepare_autoInfo(opstring)

  lambda = opstring[1][1]

  true_expterms = [mpotype[base[i] zero(base[i]) zero(base[i]); zero(base[i]) lambda*base[i] zero(base[i]); zero(base[i]) zero(base[i]) base[i]] for i = 1:Ns]


  mpo = makeMPO(true_expterms,physind)
  if qarrayops
    mpo = makeqMPO(Qnumvec,mpo)
  end

  firstop = [true for i = 1:Ns]
  lastop = [true for i = 1:Ns]

  i = 0
  while sum(firstop) != 0 && sum(secondop) != 0 && i < length(opstring)
    i += 1

    if opstring[i][3] < opstring[i][5]
      a = opstring[i][3]
      op1 = opstring[i][2]
      b = opstring[i][5]
      op2 = opstring[i][4]
    else
      a = opstring[i][5]
      op1 = opstring[i][4]
      b = opstring[i][3]
      op2 = opstring[i][2]
    end

    if firstop[b]
      mpo[b][2,:,:,1] = op2
      firstop[b] = false
    end
    if secondop[a]
      mpo[a][end,:,:,2] = ops1
      secondop[a] = false
    end
  end

  if reverse && typeof(mpo[1]) <: qarray
    for a = 1:length(mpo)
      mpo[a] = permutedims!(mpo[a],[1,3,2,4])
    end
  end

  return mpo
end

function expMPO(lambda::Number,op1::TensType,op2::TensType,Ns::Integer;start::Integer=1,stop::Integer=Ns,reverse::Bool=true)

  mpotype = typeof(typeof(lambda)(1) * eltype(op1)(1) * eltype(op2)(1))

  physind = size(op1,2)

  base = [zeros(mpotype,i == 1 ? 1 : 3,physind,physind,i == Ns ? 1 : 3) for i = 1:Ns]
  mpo = MPO(base)

  Id = makeId(mpotype,physind)
  if typeof(op1) <: qarray
    Qlabel = recoverQNs(1,op1)
    mpo = makeqMPO(Qlabel,mpo)
    Id = Qtens(Qlabel,Id)
  end

  lambda_Id = lambda*Id

  for i = 1:Ns
    if i > 1
      mpo[i][1,:,:,1] = Id
      if start <= i <= stop
        mpo[i][2,:,:,1] = op2
      end
    end
    if i < Ns
      mpo[i][end,:,:,end] = Id
      if start <= i <= stop
        mpo[i][end,:,:,2] = op1
      end
    end
    if 1 < i < Ns
      mpo[i][2,:,:,2] = lambda_Id
    end
  end

  if reverse && typeof(mpo[1]) <: qarray
    for a = 1:length(mpo)
      mpo[a] = permutedims!(mpo[a],[1,3,2,4])
    end
  end

  return mpo
end
export expMPO

function expmpoterm(lambda::Number,Op1::TensType,Op2::TensType,trailterm::U...) where U <: Union{Tuple,Array}

  mpotype = typeof(typeof(lambda)(1)*eltype(Op1)(1)*eltype(Op2)(1))
  Id = makeArray(makeId(mpotype,size(Op1,1)))
  O = zero(Id)
  if length(trailterm) > 0
    trail = trailterm[1]
    out = [Id O O;
            makeArray(Op1) lambda*makeArray(trail[i]) O;
            O makeArray(Op2*trail[i]) Id]
  else
    out = [Id O O;
            makeArray(Op1) lambda*Id O;
            O makeArray(Op2) Id]
  end
  return out
end
export expmpoterm


"""
  A + B

functionality for adding (similar to direct sum) of MPOs together; uses joinindex function to make a combined MPO

note: deparallelizes after every addition

See also: [`deparallelization`](@ref) [`add!`](@ref)
"""
function +(X::MPO...)
  finalMPO = *(X...,fct=add!)
  return finalMPO
end

#=
function +(X::MPO...;nthreads::Integer=Threads.nthreads())

  sizeparts = cld(length(X),nthreads)
  startparts = Array{intType,1}(undef,nthreads+1)
  startparts[1] = 0
  for i = 1:nthreads-1
    startparts[i+1] = sizeparts*i
  end
  startparts[end] = length(X)

  Z = Array{MPO,1}(undef,nthreads)

  #=Threads.@threads =#for w = 1:nthreads

    checktype = typeof(prod(w->eltype(X[w])(1),startparts[w]+1:startparts[w+1]))
    if checktype != eltype(X[startparts[w]+1])
      C = MPO(checktype,copy(X[startparts[w]+1]))
    else
      C = copy(X[startparts[w]+1])
    end
    for k = startparts[w]+2:startparts[w+1]
      add!(C,X[k])
    end
    Z[w] = C
  end
  R = Z[1]
  for w = 2:nthreads
    R += Z[w]
  end
  return R
end
=#

#  import .QN.add!
"""
  add!(A,B)

functionality for adding (similar to direct sum) of MPOs together and replacing `A`; uses joinindex function to make a combined MPO

note: deparallelizes after every addition

See also: [`deparallelization`](@ref) [`+`](@ref)
"""
function add!(A::MPO,B::MPO;finiteBC::Bool=true)
  mult!(A,B)
  return deparallelize!(A)
end

"""
  A + B

functionality for adding (similar to direct sum) of MPOs together; uses joinindex function to make a combined MPO

note: deparallelizes after every addition

See also: [`deparallelization`](@ref) [`add!`](@ref)
"""
function *(X::MPO...;fct::Function=mult!)
  checktype = typeof(prod(w->eltype(X[w])(1),1:length(X)))
  nthreads = Threads.nthreads()
  if length(X) > 2
    sizeparts = cld(length(X),nthreads)
    startparts = Array{intType,1}(undef,nthreads+1)
    startparts[1] = 0
    for i = 1:nthreads-1
      startparts[i+1] = sizeparts*i
    end
    startparts[end] = length(X)

    R = Array{MPO,1}(undef,nthreads)

    Threads.@threads :static for w = 1:nthreads

      start = startparts[w] + 1
      stop = startparts[w+1]

      if checktype != eltype(X[start])
        Z = MPO(checktype,copy(X[start]))
      else
        Z = copy(X[start])
      end
      for g = start+1:stop
        fct(Z,X[g])
      end
      R[w] = Z
    end
    finalMPO = R[1]
    for w = 2:length(R)
      fct(finalMPO,R[w])
    end
  else
    if checktype != eltype(X[1])
      finalMPO = MPO(checktype,copy(X[1]))
    else
      finalMPO = copy(X[1])
    end
    for w = 2:length(X)
      fct(finalMPO,X[w])
    end
  end
  return finalMPO
end

#=
function +(X::MPO...;nthreads::Integer=Threads.nthreads())

  sizeparts = cld(length(X),nthreads)
  startparts = Array{intType,1}(undef,nthreads+1)
  startparts[1] = 0
  for i = 1:nthreads-1
    startparts[i+1] = sizeparts*i
  end
  startparts[end] = length(X)

  Z = Array{MPO,1}(undef,nthreads)

  #=Threads.@threads =#for w = 1:nthreads

    checktype = typeof(prod(w->eltype(X[w])(1),startparts[w]+1:startparts[w+1]))
    if checktype != eltype(X[startparts[w]+1])
      C = MPO(checktype,copy(X[startparts[w]+1]))
    else
      C = copy(X[startparts[w]+1])
    end
    for k = startparts[w]+2:startparts[w+1]
      add!(C,X[k])
    end
    Z[w] = C
  end
  R = Z[1]
  for w = 2:nthreads
    R += Z[w]
  end
  return R
end
=#

#  import .QN.add!
"""
  mult!(A,B)

functionality for adding (similar to direct sum) of MPOs together and replacing `A`; uses joinindex function to make a combined MPO

note: Does not compress the bond dimension (recommend to use compressMPO! afterwards)

See also: [`compressMPO!`](@ref) [`*`](@ref)
"""
function mult!(A::MPO,B::MPO;finiteBC::Bool=true)
  Ns = length(A)
  if finiteBC
    A[1] = joinindex!(4,A[1],B[1])
    for a = 2:Ns-1
      A[a] = joinindex!([1,4],A[a],B[a])
    end
    A[end] = joinindex!(1,A[Ns],B[Ns])
  else
    for a = 1:Ns
      A[a] = joinindex!([1,4],A[a],B[a])
    end
  end
  return A
end

"""
  H + c

Adds a constant `c` to a Hamiltonian `H` (commutative)
"""
function +(H::MPO,c::Number;pos::Integer=1)
  const_term = MPO([i == pos ? mult!(c,makeId(H[i],[2]))  : makeId(H[i],[2]) for i = 1:length(H)])
  return copy(H) + const_term
end

function +(c::Number,H::MPO;pos::Integer=1)
  return +(H,c,pos=pos)
end

function *(c::Number,H::MPO;pos::Integer=1)
  return +(H,c,pos=pos)
end

function *(H::MPO,c::Number;pos::Integer=1)
  return +(H,c,pos=pos)
end

#import Base.-
"""
  H - c

Adds a constant `c` to a Hamiltonian `H`
"""
function -(H::MPO,c::Number;pos::Integer=1)
  return +(H,-c,pos=pos)
end


















function pullvec(M::TensType,j::Integer,left::Bool)
  return left ? M[:,j:j] : M[j:j,:]
end

"""
  deparallelize!(M[,left=])

Deparallelizes a matrix-equivalent of a rank-4 tensor `M`; toggle the decomposition into the `left` or `right`
"""
function deparallelize!(M::densTensType;left::Bool=true,zero::Float64=0.)
  sizeM = size(M)  
  group = left ? [[1,2,3],[4]] : [[1],[2,3,4]]
  rM = reshape(M,group)
  if left
    newK,finalT = deparallelize_block(rM,left,zero)

    outT = finalT[1:size(newK,2),:]
    newK = reshape!(newK,sizeM[1:3]...,size(newK,2))

    return newK,outT
  else
    finalT,newK = deparallelize_block(rM,left,zero)

    outT = finalT[:,1:size(newK,1)]
    newK = reshape!(newK,size(newK,1),sizeM[2:4]...)
    return outT,newK
  end
end

function deparallelize!(M::Qtens{W,Q};left::Bool=true,zero::Float64=0.) where {W <: Number, Q <: Qnum}
  sizeM = size(M)
  group = left ? [[1,2,3],[4]] : [[1],[2,3,4]]

  rM = changeblock(M,group[1],group[2])
  qfinalT = reshape(M,[group[1],group[2]],merge=true)

  newK = Array{Array{W,2},1}(undef,length(rM.T))
  finalT = Array{Array{W,2},1}(undef,length(newK))
  if left
    A,B = newK,finalT
  else
    A,B, = finalT,newK
  end
  for q = 1:length(newK)
    X,Y = deparallelize_block(qfinalT.T[q],left,zero)
    A[q],B[q] = X,Y
  end

  if left
    interval = [1:size(newK[q],2) for q = 1:length(newK)]
    keepvec_newK = vcat([rM.ind[q][2][interval[q]] .+ 1 for q = 1:length(rM.ind)]...)
    qnewK = rM[:,:,:,keepvec_newK]

    keepvec_finalT = vcat([qfinalT.ind[q][1][interval[q]] .+ 1 for q = 1:length(qfinalT.ind)]...)

    qfinalT = qfinalT[keepvec_finalT,:]

    qnewK.T = [newK[q][:,interval[q]] for q = 1:length(newK)]
    qfinalT.T = [finalT[q][interval[q],:] for q = 1:length(newK)]
    return qnewK,qfinalT
  else
    interval = [1:size(newK[q],1) for q = 1:length(newK)]
    keepvec_newK = vcat([rM.ind[q][1][interval[q]] .+ 1 for q = 1:length(rM.ind)]...)
    qnewK = rM[keepvec_newK,:,:,:]

    keepvec_finalT = vcat([qfinalT.ind[q][2][interval[q]] .+ 1 for q = 1:length(qfinalT.ind)]...)
    qfinalT = qfinalT[:,keepvec_finalT]

    qnewK.T = [newK[q][interval[q],:] for q = 1:length(newK)]
    qfinalT.T = [finalT[q][:,interval[q]] for q = 1:length(newK)]
    return qfinalT,qnewK
  end
end

function deparallelize_block(rM::densTensType,left::Bool,zero::Float64)

  T = zeros(rM) #maximum size for either left or right
  firstvec = pullvec(rM,1,left)

  K = [firstvec]
  normK = [norm(K[1])]

  b = left ? 2 : 1
  for j = 1:size(rM,b)
    thisvec = pullvec(rM,j,left)

    mag_thisvec = norm(thisvec) # |A|

    condition = true
    i = 0
    while condition  && i < size(K,1)
      i += 1
  #      if left
        thisdot = dot(K[i],thisvec)
  #=      else
        thisdot = contractc(K[i],thisvec)
      end
      =#

      if isapprox(thisdot,mag_thisvec * normK[i]) && !isapprox(normK[i],0) #not sure why it would be zero...
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
        if length(K) > size(T,1)
          newT = zeros(eltype(T),length(K),size(T,2))
          newT[1:end-1,:] = T
          T = newT
        end
        T[length(K),j] = 1.
      else
        if length(K) > size(T,2)
          newT = zeros(eltype(T),size(T,1),length(K))
          newT[:,1:end-1] = T
          T = newT
        end
        T[j,length(K)] = 1.
      end
    end
  end

  if left
    finalT = T
    newK = K[1]
    for a = 2:size(K,1)
      newK = joinindex!(newK,K[a],2)
    end
    return newK,finalT
  else
    finalT = T
    newK = K[1]
    for a = 2:size(K,1)
      newK = joinindex!(newK,K[a],1)
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
function deparallelize!(W::MPO;sweeps::Integer=1)
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
        @inbounds #=Threads.@threads=# for i = 1:2^j:currsize
          iL = i
          iR = iL + 2^(j-1)
          if iR < currsize
            add!(W[iL],W[iR])
            W[iL] = deparallelize!(W[iL],sweeps=sweeps)
            active[iR] = false
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
function deparallelize(W::MPO;sweeps::Integer=1)
  return deparallelize!(copy(W),sweeps=sweeps)
end

function deparallelize(W::Array{G,1};sweeps::Integer=1) where G <: TensType
  return deparallelize!(copy(W),sweeps=sweeps)
end
export deparallelize
































"""
  invDfactor(D)

Finds nearest factor of 2 to the magnitude of `D`
"""
function invDfactor(D::TensType)
  avgD = sum(D)
  avgD /= size(D,1)
  maxval = convert(intType,floor(log(2,avgD)))
  exp_factor = max(0,maxval)+1
  finaltwo = 2^(exp_factor)
  return finaltwo
end
export invDfactor

const forwardshape = Array{intType,1}[intType[1,2,3],intType[4]]
const backwardshape = Array{intType,1}[intType[1],intType[2,3,4]]

"""
  compressMPO!(W[,sweeps=,cutoff=,deltam=,minsweep=,nozeros=])

compresses MPO (`W`; or several `M`) with SVD compression for `sweeps` sweeps, `cutoff` applied to the SVD, `deltam` target for teh bond dimension compression, and `nozeros` defaulted to true to eliminate all zeros in the SVD
"""
function compressMPO!(W::MPO,M::MPO...;sweeps::Integer=100,cutoff::Float64=0.,
                    deltam::Integer=0,minsweep::Integer=1,nozeros::Bool=true)
  for a = 1:length(M)
    W = add!(W,M[a])
  end
  n = 0
  mchange = 1000
  lastmdiff = [size(W[i],4) for i = 1:length(W)-1]
  while (n < sweeps && mchange > deltam) || (n < minsweep)
    n += 1
    for i = 1:length(W)-1
      U,D,V = svd(W[i],forwardshape,cutoff=cutoff,nozeros=nozeros)
      scaleD = invDfactor(D)

      U = mult!(U,scaleD)
      W[i] = U

      scaleDV = contract(D,2,V,1,alpha=1/scaleD)
      W[i+1] = contract(scaleDV,2,W[i+1],1)
    end
    for i = length(W):-1:2
      U,D,V = svd(W[i],backwardshape,cutoff=cutoff,nozeros=nozeros)
      scaleD = invDfactor(D)
      
      V = mult!(V,scaleD)
      W[i] = V

      scaleUD = contract(U,2,D,1,alpha=1/scaleD)
      W[i-1] = contract(W[i-1],4,scaleUD,1)
    end
    thismdiff = intType[size(W[i],4) for i = 1:length(W)-1]
    mchange = sum(a->lastmdiff[a]-thismdiff[a],1:length(thismdiff))
    lastmdiff = copy(thismdiff)
  end
  return W
end

function compressMPO!(w::Number)
  return w
end

"""
  compressMPO!(W[,sweeps=,cutoff=,deltam=,minsweep=,nozeros=])

compresses an array of MPOs (`W`) in parallel with SVD compression for `sweeps` sweeps, `cutoff` applied to the SVD, `deltam` target for teh bond dimension compression, and `nozeros` defaulted to true to eliminate all zeros in the SVD
"""
function compressMPO!(W::Array{MPO,1};sweeps::Integer=1000,cutoff::Float64=1E-16,
                    deltam::Integer=0,minsweep::Integer=1,nozeros::Bool=true)
  nlevels = floor(intType,log(2,length(W)))
  active = Bool[true for i = 1:length(W)]
  if length(W) > 2
    for j = 1:nlevels
      currsize = fld(length(W),2^(j-1))
      for i = 1:2^j:currsize
        iL = i
        iR = iL + 2^(j-1)
        if iR < currsize
          W[iL] = compressMPO!(W[iL],W[iR],sweeps=sweeps,cutoff=cutoff,nozeros=nozeros)
          active[iR] = false
        end
      end
    end
    if sum(active) > 1
      compressMPO!(W[active],sweeps=sweeps,cutoff=cutoff,nozeros=nozeros)
    end
  end
  return length(W) == 2 ? compressMPO!(W[1],W[2],nozeros=nozeros) : compressMPO!(W[1],nozeros=nozeros)
end
export compressMPO!

"""
  compressMPO(W,sweeps=,cutoff=,deltam=,minsweep=,nozeros=])

Same as `compressMPO!` but a copy is made of the original vector of MPOs

See also: [`compressMPO!`](@ref)
"""
  function compressMPO(W::Array{MPO,1};sweeps::Integer=1000,cutoff::Float64=1E-16,deltam::Integer=0,minsweep::Integer=1,nozeros::Bool=true)
  M = copy(W)
  return compressMPO!(M;sweeps=sweeps,cutoff=cutoff,deltam=deltam,minsweep=minsweep,nozeros=nozeros)
end
export compressMPO

#end


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
      xpos = C[i,x]
      for w = x+1:Ncols
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





















#=

struct zeroMPO
  base::Array
end

function mpoterm(base::Array{G,1}) where G <: densTensType
  return zeroMPO(base)
end

#import Base.+
function +(X::zeroMPO,Y::MPO)
  W = elnumtype(Y)
  mpotens = Array{Array{W,4},1}(undef,length(X.base))
  O = zero(X.base[1])
  d = size(O,1)
  temp = [Y[1] X.base[1]]
  mpotens[1] = reshape(temp,1,d,d,2)
  @inbounds for i = 2:length(X.base)-1
    O = zero(X.base[i])
    d = size(O,1)
    mpotens[i] = reshape([X.base[i] O;
                  Y[i] X.base[i]],2,d,d,2)
  end
  O = zero(X.base[end])
  d = size(O,1)
  mpotens[end] = reshape([X.base[end];
                  Y[end]],2,d,d,1)
  return MPO(mpotens)
end
=#