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
#>------+  move! orthogonality center in MPS    +---------<
#       +---------------------------------------+

"""
  D,truncerr = moveR!(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center right one space, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error Modifies `Lpsi`

See also: [`moveR!`](@ref)
"""
@inline function moveR!(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                fast::Bool=false,qrfct::Function=qr!,svdfct::Function=svd!)
  if (min(size(Lpsi,1)*size(Lpsi,2),size(Lpsi,3)) <= m || m == 0) && !isapprox(cutoff,0.) && fast
    Ltens,modV = qrfct(Lpsi,[[1,2],[3]])

    DV = (condition ? getindex!(modV,:,1:size(Rpsi,1)) : modV)
    D = DV
    truncerr = 0.
  else
    Ltens,D,V,truncerr,sumD = svdfct(Lpsi,[[1,2],[3]],cutoff=cutoff,m=m,minm=minm,mag=mag)      
    modV = (condition ? getindex!(V,:,1:size(Rpsi,1)) : V)
    DV = lmul!(D,modV)
  end
  Rtens = contract(DV,2,Rpsi,1)
  return Ltens,Rtens,D,truncerr
end

"""
  D,truncerr = moveR(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center right one space, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error

See also: [`moveR!`](@ref)
"""
@inline function moveR(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                fast::Bool=false,qrfct::Function=qr!,svdfct::Function=svd!)
  return moveR!(Lpsi,Rpsi,cutoff=cutoff,m=m,minm=minm,condition=condition,mag=mag,qrfct=qr,svdfct=svd)
end
export moveR

"""
  D,truncerr = moveR!(psi[,cutoff=,m=,minm=,condition=])

acts in-place to move `psi`'s orthogonality center right one space, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`moveR`](@ref)
"""
@inline function moveR!(psi::MPS;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                  mag::Number=0.)
  iL = psi.oc
  psi[iL],psi[iL+1],D,truncerr = moveR!(psi[iL],psi[iL+1],cutoff=cutoff,m=m,minm=minm,condition=condition,mag=mag)
  psi.oc += 1
  return D,truncerr
end
export moveR!

"""
  D,truncerr = moveL(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center left one space, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error. Modifies `Rpsi`

See also: [`moveL!`](@ref)
"""
@inline function moveL!(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
                fast::Bool=false,lqfct::Function=lq!,svdfct::Function=svd!)
  if (min(size(Rpsi,1),size(Rpsi,2)*size(Rpsi,3)) <= m || m == 0) && !isapprox(cutoff,0.) && fast
    modU,Rtens = lqfct(Rpsi,[[1],[2,3]])

    UD = (condition ? getindex!(modU,1:size(Lpsi,3),:) : modU)
    D = UD
    truncerr = 0.
  else
    U,D,Rtens,truncerr,sumD = svdfct(Rpsi,[[1],[2,3]],cutoff=cutoff,m=m,minm=minm,mag=mag)
    modU = (condition ? getindex!(U,1:size(Lpsi,3),:) : U)
    UD = rmul!(modU,D) #contract(modU,(2,),D,(1,))
  end
  Ltens = contract(Lpsi,3,UD,1)
  return Ltens,Rtens,D,truncerr
end

"""
  D,truncerr = moveL(Lpsi,Rpsi[,cutoff=,m=,minm=,condition=])

moves `psi`'s orthogonality center left one space, with a maximum bond dimenion `m`, cutoff `cutoff`, minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`; returns psi[iL],psi[iR],D from the SVD, and the truncation error. Does not modify `Lpsi` and `Rpsi`

See also: [`moveL!`](@ref)
"""
@inline function moveL(Lpsi::TensType,Rpsi::TensType;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,mag::Number=0.,
  fast::Bool=false,lqfct::Function=lq,svdfct::Function=svd)
  return moveL!(Lpsi,Rpsi,cutoff=cutoff,m=m,minm=minm,condition=condition,mag=mag,lqfct=lq,svdfct=svd)
end
export moveL

"""
    moveL!(psi[,cutoff=,m=,minm=,condition=])

acts in-place to move `psi`'s orthogonality center left one space, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`moveL`](@ref)
"""
@inline function moveL!(psi::MPS;cutoff::Float64=0.,m::Integer=0,minm::Integer=0,condition::Bool=false,
                  mag::Number=0.)
  iR = psi.oc
  psi[iR-1],psi[iR],D,truncerr = moveL(psi[iR-1],psi[iR],cutoff=cutoff,m=m,minm=minm,condition=condition,mag=mag)
  psi.oc -= 1
  return D,truncerr
end
export moveL!

"""
    movecenter!(psi,newoc[,m=,cutoff=,minm=,Lfct=,Rfct=])

movement function to move `psi` to a new site, `newoc` with `Lfct` and `Rfct`, with a maximum bond dimenion `m`, cutoff `cutoff`, and minimum bond dimension `minm`; toggle truncation of tensor during movement with `condition`

See also: [`move!`](@ref) [`move`](@ref)
"""
@inline function movecenter!(psi::MPS,pos::Integer;cutoff::Float64=1E-14,m::Integer=0,minm::Integer=0,Lfct::Function=moveR,Rfct::Function=moveL)
  if m == 0
    m = maximum([maximum(size(psi[i])) for i = 1:length(psi)])
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
@inline function move!(mps::MPS,pos::Integer;m::Integer=0,cutoff::Float64=1E-14,minm::Integer=0)
  movecenter!(mps,pos,cutoff=cutoff,m=m,minm=minm)
  nothing
end
export move!

"""
    move(psi,newoc[,m=,cutoff=,minm=])

same as `move!` but makes a copy of `psi`

See also: [`move!`](@ref)
"""
@inline function move(mps::MPS,pos::Integer;m::Integer=0,cutoff::Float64=1E-14,minm::Integer=0)
  newmps = copy(mps)
  movecenter!(newmps,pos,cutoff=cutoff,m=m,minm=minm)
  return newmps
end
export move

"""
  newpsi,D,V = leftnormalize(psi)

Creates a left-normalized MPS `newpsi` from `psi` and returns the external tensors `D` and `V`
"""
function leftnormalize(psi::MPS)
  newpsi = move(psi,length(psi))
  U,D,V = svd(psi[end],[[1,2],[3]])
  newpsi[end] = U
  newpsi.oc = length(psi)+1
  return newpsi,D,V
end
export leftnormalize

"""
  psi,D,V = leftnormalize!(psi)

Creates a left-normalized MPS in-place from `psi` and returns the external tensors `D` and `V`
"""
function leftnormalize!(psi::MPS)
  move!(psi,length(psi))
  U,D,V = svd(psi[end],[[1,2],[3]])
  psi[end] = U
  psi.oc = length(psi)+1
  return psi,D,V
end
export leftnormalize!

"""
  U,D,newpsi = rightnormalize(psi)

Creates a right-normalized MPS `newpsi` from `psi` and returns the external tensors `U` and `D`
"""
function rightnormalize(psi::MPS)
  newpsi = move(psi,1)
  U,D,V = svd(psi[1],[[1],[2,3]])
  newpsi[1] = V
  newpsi.oc = 0
  return U,D,newpsi
end
export rightnormalize

"""
  U,D,psi = rightnormalize!(psi)

Creates a right-normalized MPS in-place from `psi` and returns the external tensors `U` and `D`
"""
function rightnormalize!(psi::MPS)
  move!(psi,1)
  U,D,V = svd(psi[1],[[1],[2,3]])
  psi[1] = V
  psi.oc = 0
  return U,D,psi
end
export rightnormalize!

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
    makeBoundary(qind,newArrows[,retType=])

makes a boundary tensor for an input from the quantum numbers `qind` and arrows `newArrows`; can also define type of resulting Qtensor `retType` (default `Float64`)

#Note:
+dense tensors are just ones(1,1,1,...)

See also: [`makeEnds`](@ref)
"""
function makeBoundary(dualpsi::MPS,psi::MPS,mpovec::MPO...;left::Bool=true,rightind::Integer=3)
  retType = elnumtype(dualpsi,psi,mpovec...)
  nrank = 2 + length(mpovec)
  boundary = ones(retType,ones(intType,nrank)...)
  if typeof(psi[1]) <: qarray

    Q = typeof(psi[1].flux)

    qind = Array{Q,1}(undef,nrank)
    Ns = length(psi)
    site = left ? 1 : Ns
    index = left ? 1 : rightind
    qind[1] = -(getQnum(index,1,dualpsi[site]))
    qind[end] = getQnum(index,1,psi[site])
    for i = 1:length(mpovec)
      index = left ? 1 : ndims(mpovec[i][Ns])
      qind[i+1] = getQnum(index,1,mpovec[i][site])
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

function defaultBoundary(A::TensType)
  if typeof(A) <: qarray
    out = Qtens{eltype(A),typeof(A.flux)}()
  else
    out = tens{eltype(A)}()
  end
  return out
end


function makeEdgeEnv(dualpsi::MPS,psi::MPS,mpovec::MPO...;boundary::TensType=defaultBoundary(psi[1]),left::Bool=true)
  expsize = 2+length(mpovec)
  Lnorm = norm(boundary)
  if ndims(boundary) != expsize || isapprox(Lnorm,0) || isnan(Lnorm) || isinf(Lnorm)
    Lout = makeBoundary(dualpsi,psi,mpovec...,left=left)
  else
    Lout = copy(boundary)
  end
  return Lout
end

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
function makeEnds(dualpsi::MPS,psi::MPS,mpovec::MPO...;Lbound::TensType=defaultBoundary(psi[1]),Rbound::TensType=typeof(psi[end])())
  return makeEdgeEnv(dualpsi,psi,mpovec...,boundary=Lbound),makeEdgeEnv(dualpsi,psi,mpovec...,boundary=Rbound,left=false)
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
function makeEnds(psi::MPS,mpovec::MPO...;Lbound::TensType=defaultBoundary(psi[1]),Rbound::TensType=defaultBoundary(psi[1]))
  return makeEnds(psi,psi,mpovec...,Lbound=Lbound,Rbound=Rbound)
end
export makeEnds


#=
function genEnv(dualpsi::MPS,psi::MPS,mpo::MPO...;bound::TensType=defaultBoundary(psi[1]))

end
=#



"""
    makeEnv(dualpsi,psi,mpo[,Lbound=,Rbound=])

Generates environment tensors for a MPS (`psi` and its dual `dualpsi`) with boundaries `Lbound` and `Rbound`
"""
function makeEnv(dualpsi::MPS,psi::MPS,mpo::MPO...;Lbound::TensType=defaultBoundary(psi[1]),Rbound::TensType=defaultBoundary(psi[1]))
  Ns = length(psi)
  numtype = elnumtype(dualpsi,psi,mpo...)
  C = psi[1]

  if typeof(psi) <: largeMPS || typeof(mpo) <: largeMPO
    Lenv,Renv = largeEnv(numtype,Ns)
  else
    Lenv = environment(psi)
    Renv = environment(psi)
  end
  Lenv[1],Renv[Ns] = makeEnds(dualpsi,psi,mpo...;Lbound=Lbound,Rbound=Rbound)
  for i = 1:psi.oc-1
    Lupdate!(i,Lenv,dualpsi,psi,mpo...)
  end

  for i = Ns:-1:psi.oc+1
    Rupdate!(i,Renv,dualpsi,psi,mpo...)
  end
  return Lenv,Renv
end

"""
    makeEnv(psi,mpo[,Lbound=,Rbound=])

Generates environment tensors for a MPS (`psi`) with boundaries `Lbound` and `Rbound`
"""
function makeEnv(psi::MPS,mpo::MPO;Lbound::TensType=[0],Rbound::TensType=[0])
  return makeEnv(psi,psi,mpo,Lbound=Lbound,Rbound=Rbound)
end
export makeEnv


"""
    Lupdate(Lenv,dualpsi,psi,mpo)

Updates left environment tensor `Lenv` with dual MPS `dualpsi`, MPS `psi`, and MPO `mpo`
"""
@inline function  Lupdate(Lenv::TensType,dualpsi::TensType,psi::TensType,mpo::TensType...)
  nMPOs = length(mpo)
  nLsize = nMPOs+2
  tempLenv = contractc(Lenv,1,dualpsi,1)
  for j = 1:nMPOs
    tempLenv = contract(tempLenv,(1,nLsize),mpo[j],(1,3))
  end
  return contract(tempLenv,(1,nLsize),psi,(1,2))
end
export Lupdate

"""
    Rupdate(Renv,dualpsi,psi,mpo)

Updates right environment tensor `Renv` with dual MPS `dualpsi`, MPS `psi`, and MPO `mpo`
"""
@inline function  Rupdate(Renv::TensType,dualpsi::TensType,psi::TensType,mpo::TensType...)
  nMPOs = length(mpo)
  nLsize = nMPOs+2
  nRsize = nLsize+1
  tempRenv = ccontract(dualpsi,3,Renv,nLsize)
  for j = 1:nMPOs
    tempRenv = contract(mpo[j],(3,4),tempRenv,(2,nRsize))
  end
  return contract(psi,(2,3),tempRenv,(2,nRsize))
end
export Rupdate

"""
    Lupdate!(i,Lenv,psi,dualpsi,mpo)

Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
"""
@inline function Lupdate!(i::Integer,Lenv::Env,dualpsi::MPS,psi::MPS,mpo::MPO...)
  Lenv[i+1] = Lupdate(Lenv[i],dualpsi[i],psi[i],[mpo[a][i] for a = 1:length(mpo)]...)
  nothing
end

"""
    Lupdate!(i,Lenv,psi,mpo)

Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
"""
@inline function Lupdate!(i::Integer,Lenv::Env,psi::MPS,mpo::MPO...)
  Lupdate!(i,Lenv,psi,psi,mpo...)
end
export Lupdate!

"""
    Rupdate!(i,Renv,dualpsi,psi,mpo)

Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
"""
@inline function Rupdate!(i::Integer,Renv::Env,dualpsi::MPS,psi::MPS,mpo::MPO...)
  Renv[i-1] = Rupdate(Renv[i],dualpsi[i],psi[i],[mpo[a][i] for a = 1:length(mpo)]...)
  nothing
end

"""
    Rupdate!(i,Renv,psi,mpo)

Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
"""
@inline function Rupdate!(i::Integer,Renv::Env,psi::MPS,mpo::MPO...)
  Rupdate!(i,Renv,psi,psi,mpo...)
end
export Rupdate!

"""
    boundaryMove!(psi,i,mpo,Lenv,Renv)

Move orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
@inline function boundaryMove!(psi::MPS,i::Integer,Lenv::Env,Renv::Env,mpo::MPO...;mover::Function=move!)
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

"""
    boundaryMove!(dualpsi,psi,i,mpo,Lenv,Renv)

Move orthogonality center of `psi` and `dualpsi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
@inline function boundaryMove!(dualpsi::MPS,psi::MPS,i::Integer,Lenv::Env,Renv::Env,mpo::MPO...;mover::Function=move!)
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
    boundaryMove(dualpsi,psi,i,mpo,Lenv,Renv)

Copies `psi` and moves orthogonality center of `psi` and `dualpsi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
@inline function boundaryMove(dualpsi::MPS,psi::MPS,i::Integer,Lenv::Env,Renv::Env,mpo::MPO...)
  newpsi = copy(psi)
  newdualpsi = copy(dualpsi)
  newLenv = copy(Lenv)
  newRenv = copy(Renv)
  boundaryMove!(newdualpsi,newpsi,i,newLenv,newRenv,mpo...)
  return newdualpsi,newpsi,newLenv,newRenv
end

"""
    boundaryMove(psi,i,mpo,Lenv,Renv)

Copies `psi` and moves orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
@inline function boundaryMove(psi::MPS,i::Integer,Lenv::Env,Renv::Env,mpo::MPO...)
  newpsi = copy(psi)
  newLenv = copy(Lenv)
  newRenv = copy(Renv)
  boundaryMove!(newpsi,newpsi,i,newLenv,newRenv,mpo...)
  return newpsi,newLenv,newRenv
end
export boundaryMove

#       +---------------------------------------+
#>------+       measurement operations          +---------<
#       +---------------------------------------+

"""
  psi = applyOps!(psi,sites,Op[,trail=ones(1,1)])

Applies operator `Op` (any `TensType`) in-place to the MPS `psi` at sites `sites`, a vector of integers. A trailing operator `trail` can be applied if not the default.
"""
function applyOps!(psi::MPS,sites::Array{W,1},Op::TensType;trail::TensType=ones(1,1)) where W <: Integer
  def_trail = ones(1,1)
  @inbounds for i = 1:length(sites)
    site = sites[i]
    p = site
    psi[p] = contract([2,1,3],Op,2,psi[p],2)
    if trail != def_trail
      @inbounds for j = 1:p-1
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

  thissize = length(psi)
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
  for w = 1:length(H)
    newpsi = applyMPO(newpsi,H[w],m=m,cutoff=cutoff)
  end
  return newpsi
end
export applyMPO

"""
    expect(dualpsi,psi,H[,Lbound=,Rbound=,order=])

evaluate <`dualpsi`|`H`|`psi`> for any number of MPOs `H`; can specifcy left and right boundaries (`Lbound` and `Rbound`); `dualpsi` is conjugated inside of the algorithm

See also: [`overlap`](@ref)
"""
function expect(dualpsi::MPS,psi::MPS,H::MPO...;Lbound::TensType=defaultBoundary(psi[1]),Rbound::TensType=Lbound,order::intvecType=intType[])
  Ns = length(psi)
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
    permvec = ntuple(i->ndims(Lenv)-i+1,ndims(Lenv))
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
function expect(psi::MPS,H::MPO...;Lbound::TensType=defaultBoundary(psi[1]),Rbound::TensType=Lbound,order::intvecType=intType[])
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
function correlationmatrix(dualpsi::MPS, psi::MPS, Cc::TensType, Ca::TensType; trail=intType[])
  rho = Array{eltype(psi[1]),2}(undef,length(psi),length(psi))
  if trail != []
    FCc = contract(Cc,2,trail,1)
  else
    FCc = Cc
  end
  diffTensors = !(psi == dualpsi)
  onsite = contract(Cc,2,Ca,1)
  for i = 1:length(psi)
    move!(psi,i)
    if diffTensors
      move!(dualpsi,i)
    end
    TopTerm = contract([2,1,3],onsite,2,psi[i],2)
    rho[i,i] = contractc(TopTerm,dualpsi[i])
  end
  for i = 1:length(psi)-1
    move!(psi,i)
    if diffTensors
      move!(dualpsi,i)
    end
    TopTerm = contract(FCc,2,psi[i],2)
    Lenv = contractc(TopTerm,(2,1),dualpsi[i],(1,2))
    for j = i+1:length(psi)
      Renv = contract(Ca,2,psi[j],2)
      Renv = contractc(Renv,(1,3),dualpsi[j],(2,3))
      DMElement = contract(Lenv,Renv)
      if j < length(psi)
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































function localizeOp(psi::MPS,Op::Array{G,1},insites::Array{R,1};#=centerpsi::TensType=psi[psi.oc],=#trail::Tuple=()) where {G <: TensType, R <: Integer}

  #trail operations....
  isId = [isapprox(norm(trail[r])^2,size(trail[r],1)) for r = 1:length(trail)]
  if length(trail) > 0
    for r = 1:length(isId)
      index = 0
      @inbounds while isId[r] && index < size(trail[r],1)
        index += 1
        isId[r] &= searchindex(trail[r],index,index) == 1
      end
    end
  end


  if issorted(insites)
    Oparray = Op
    sites = insites
  else
    order = sortperm(insites)
    Oparray = Op[order]
    sites = insites[order]
  end


  #needs to incorporate trail operators between operators.

  minsite = minimum(sites)
  Lstart = minsite < psi.oc ? minsite : psi.oc
  Lenv = LinearAlgebra.Diagonal(ones(eltype(psi[Lstart]),size(psi[Lstart],1)))
  currOp = [0]
  for w = minsite:psi.oc-1
    thispsi = psi[w]
    if w in sites
      p = findall(r->r==w,sites)
      for k = 1:length(p)
        thispsi = contract([2,1,3],Oparray[p[k]],2,thispsi,2)
        currOp[1] += 1
      end
    end
    for g = currOp[1]+1:length(trail)
      thispsi = contract([2,1,3],trail[g],2,thispsi,2)
    end
    Lenv = Lupdate(Lenv,psi[w],thispsi)
  end


  maxsite = maximum(sites)
  Rstart = maxsite > psi.oc ? maxsite : psi.oc
  Renv = LinearAlgebra.Diagonal(ones(eltype(psi[Rstart]),size(psi[Rstart],3)))
  currOp = [length(sites)+1]
  for w = maxsite:-1:psi.oc+1
    thispsi = psi[w]
    if w in sites
      p = findall(r->r==w,sites)
      for k = 1:length(p)
        thispsi = contract([2,1,3],Oparray[p[k]],2,thispsi,2)
        currOp[1] -= 1
      end
    end
    for g = currOp[1]+1:length(trail)
      thispsi = contract([2,1,3],trail[g],2,thispsi,2)
    end
    Renv = Rupdate(Renv,psi[w],thispsi)
  end

  if psi.oc in sites
    p = findfirst(r->psi.oc==sites[r],1:length(sites))
    outOp = Oparray[p]
  else
    outOp = makeId(eltype(psi[1]),size(psi[psi.oc],2))
  end
  return Lenv,outOp,Renv
end

function localizeOp(psi::MPS,mpo::MPO...;Hpsi::Function=singlesite_update)
  Lenv,Renv = makeEnv(psi,mpo...)
  nMPOs = length(mpo)
  nLsize = nMPOs+2

  Lpsi = contract(Lenv[psi.oc],ndims(Lenv[psi.oc]),psi[psi.oc],1)
  for w = 1:nMPOs
    Lpsi = contract(Lpsi,(2+nMPOs-w,ndims(Lpsi)-w),mpo[w][psi.oc],(1,2))
  end

  Ltup = (ntuple(w->w+1,ndims(Lpsi)-3)...,ndims(Lpsi))
  Rtup = ntuple(w->w,ndims(Renv[psi.oc])-1)

  return contract(Lpsi,Ltup,Renv[psi.oc],Rtup)
end
export localizeOp


























"""
  operator_in_order!(pos,sizes)

Increments elements of input vector `pos` by one step (last element first) with sizes of a tensor `sizes` such that `pos`.  For use in `correlation` function.
"""
@inline function operator_in_order!(pos::Array{G,1},sizes::intvecType) where G <: Integer
  w = length(pos)
  pos[w] += 1
  @inbounds while w > 1 && pos[w] > sizes[w]
    w -= 1
    pos[w] += 1
    @inbounds @simd for x = w:length(pos)-1
      pos[x+1] = pos[x]
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
                      sites::Tuple=ntuple(i->1:length(psi),length(inputoperators)),
                      trail::Union{Tuple,TensType}=()) #where S <: Union{Vector{AbstractMatrix{Float64}},TensType}

  return correlation(psi,psi,inputoperators...,sites=sites,trail=trail)
end
"""
  correlation(dualpsi,psi,inputoperators...[,sites=ntuple(i->1,length(inputoperators)),trail=()])

Computes the general correlation on the input MPS `psi` with operators defined in the tuple `inputoperators`. The `sites` command specifies which sites to allow for the operators to be applied (i.e., some operators can skip some sites) and the `trail` option allows for a trailing operator like the Fermion sign operator.

See also: [`expect`](@ref) [`correlationmatrix`](@ref)
"""
function correlation(dualpsi::MPS, psi::MPS, inputoperators...;
                      sites::Tuple=ntuple(i->1:length(psi),length(inputoperators)),
                      trail::Union{Tuple,TensType}=()#=,periodic::Bool=false,infinite::Bool=false=#) #where S <: Union{Vector{AbstractMatrix{Float64}},TensType}

  move!(psi,1)
  move!(dualpsi,1)

  istrail = length(trail) != 0
  if istrail
    if typeof(trail) <: Tuple || ndims(trail) == 1
      subtrail = [trail[(w-1) % length(inputoperators) + 1] for w = 1:length(inputoperators)]
    elseif trail != ()
      subtrail = [trail for w = 1:length(inputoperators)]
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

  temp = eltype(dualpsi[1])(1)
  temp *= eltype(psi[1])(1)
  for w = 1:numops
    @inbounds @simd for a = 1:lengthops[w]
      temp *= eltype(operators[w][a][1])(1)
    end
  end
  if istrail
    @inbounds @simd for r = 1:length(subtrail)
      temp *= eltype(subtrail[r])(1)
    end
  end
  retType = typeof(temp)

  base_sizes = Array{intType,1}(undef,length(sites))
#=  if periodic
    @inbounds @simd for i = 1:length(sites)
      base_sizes[i] = sites[i]
    end
  else=#
    @inbounds @simd for i = 1:length(sites)
      base_sizes[i] = length(sites[i]) - (ndims(inputoperators[i]) == 1 ? length(inputoperators[i])-1 : 0)
    end
#  end


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
        currsite = (pos[g] + p-1) - 1 % length(savepsi) + 1
        savepsi[currsite] = contract([2,1,3],operators[g][p],2,savepsi[currsite],2)
      end
      if istrail && length(isId) > 0 && !isId[g]
        @inbounds for w = 1:pos[g]-1
          savepsi[w] = contract([2,1,3],subtrail[g],2,savepsi[w],2)
        end
      end
    end

    @inbounds for a = maxopspos:-1:2
      Renv[a-1] = Rupdate(Renv[a],dualpsi[a],savepsi[a])
    end

    @inbounds for y = 1:base_sizes[1] #w in sites[1]
      w = sites[1][y]

      wrapcond = #=periodic &&=# w + lengthops[1]-1 > length(Lenv)
      thisLenv = wrapcond ? Lenv[1] : Lenv[w]

      @inbounds for p = 1:lengthops[1]
        currsite = (w + p-1) - 1 % length(savepsi) + 1

        newpsi = contract([2,1,3],operators[1][p],2,savepsi[currsite],2)
        thisLenv = Lupdate(thisLenv,dualpsi[currsite],newpsi)
      end
      if wrapcond
        for r = w+lengthops[1]:length(savepsi)
          thisLenv = Lupdate(thisLenv,dualpsi[r],savepsi[r])
        end
      end

      thisRenv = permutedims(Renv[w+lengthops[1]-1],(2,1))
      res = contract(thisLenv,thisRenv)
      pos[1] = w
      omega[pos...] = res

      if istrail && length(isId) > 0 && !isId[1]
        savepsi[w] = contract([2,1,3],subtrail[1],2,savepsi[w],2)
      end

      if y+1 <= length(sites[1])
        @inbounds for r = w:sites[1][y+1]
          if r < length(Lenv)
            Lenv[r+1] = Lupdate(Lenv[r],dualpsi[r],savepsi[r])
          end
        end
      end
    end
    base_pos[1] = base_sizes[1]
    position_incrementer!(base_pos,base_sizes)
  end
  return omega
end
export correlation

#       +--------------------------------------+
#>------|    (bad) Methods for excitations     |---------<
#       +--------------------------------------+

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
export transfermatrix
