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
    Lbound = makeBoundary(dualpsi,psi,mpovec...[,left=true,rightind=3])

Creates boundary tensor `Lbound` from an input MPS `psi`, dual MPS `dualpsi`, and any number of MPOs `mpovec`; `left` controls whether to make the left or right boundary tensor; `rightind` is the number index of the MPS that corresponds to the right link index

#Note:
+ dense tensors are just ones(1,1,1,...)

Current environment convention is
     LEFT              RIGHT
   +--<-- 1          3 ---<--+
   |                         |
   |                         |
   +-->-- 2          2 --->--+
   |                         |
   |                         |
   +-->-- 3          1 --->--+


See also: [`makeEnds`](@ref)
"""
function makeBoundary(dualpsi::matrixproductstate{Qtens{W,Q}},psi::matrixproductstate{Qtens{W,Q}},mpovec::matrixproductoperator{Qtens{W,Q}}...;left::Bool=true,rightind::Integer=3) where {W <: Number, Q <: Qnum}
#  retType = elnumtype(dualpsi,psi,mpovec...)
  nrank = 2 + length(mpovec)
  boundary = ones(W,ntuple(a->1,nrank)...)

  

#  Q = typeof(psi[1].flux)

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
end

function makeBoundary(dualpsi::matrixproductstate{tens{W}},psi::matrixproductstate{tens{W}},mpovec::matrixproductoperator{tens{W}}...;left::Bool=true,rightind::Integer=3) where W <: Number
#  retType = elnumtype(dualpsi,psi,mpovec...)
  nrank = 2 + length(mpovec)
  #    boundary = ones(W,ones(intType,nrank)...)
  return tens{W}(ones(intType,nrank),W[1])
end

function makeBoundary(dualpsi::matrixproductstate{Array{W,G}},psi::matrixproductstate{Array{W,G}},mpovec::matrixproductoperator{Array{W,G}}...;left::Bool=true,rightind::Integer=3) where {W <: Number, G}
#  retType = elnumtype(dualpsi,psi,mpovec...)
  nrank = 2 + length(mpovec)
  boundary = ones(W,ones(intType,nrank)...)
  return boundary
end
export makeBoundary

"""
    tensor = defaultBoundary(A)

Creates a tensor `tensor` that has no elements for any input tensor `A` (both tensors of the same type)
"""
function defaultBoundary(A::Union{tens{W},Array{W,G}}) where {W <: Number, G}
  return tens{W}()
end

function defaultBoundary(A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return Qtens{W,Q}()
end

"""
    Lbound = makeEdgeEnv(dualpsi,psi,mpovec...[,boundary=defaultBoundary,left=true])

Created leftmost boundary tensor for the MPS `psi`, dual MPS `dualpsi`, any number of MPOs `mpovec`; `boundary` creates a tensor of the appropriate type and `left` makes either a left (true) or right (false) tensor
"""
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
    Lenv,Renv = makeEnds(dualpsi,psi[,mpovec,Lbound=,Rbound=])

Generates first and last environments for a given system of variable MPOs

# Arguments:
+ `dualpsi::MPS`: dual MPS
+ `psi::MPS`: MPS
+ `mpovec::MPO`: MPOs
+ `Lbound::TensType`: left boundary
+ `Rbound::TensType`: right boundary


Current environment convention is
     LEFT              RIGHT
   +--<-- 1          3 ---<--+
   |                         |
   |                         |
   +-->-- 2          2 --->--+
   |                         |
   |                         |
   +-->-- 3          1 --->--+

"""
function makeEnds(dualpsi::MPS,psi::MPS,mpovec::MPO...;Lbound::TensType=defaultBoundary(psi[1]),Rbound::TensType=typeof(psi[end])())
  left = makeEdgeEnv(dualpsi,psi,mpovec...,boundary=Lbound)
  right = makeEdgeEnv(dualpsi,psi,mpovec...,boundary=Rbound,left=false)
  return left,right
end

"""
    Lenv,Renv = makeEnds(psi[,mpovec,Lbound=,Rbound=])

Generates first and last environment tensors for a given system of variable MPOs.  Same as other implementation but `dualpsi`=`psi`

# Arguments:
+ `psi::MPS`: MPS
+ `mpovec::MPO`: MPOs
+ `Lbound::TensType`: left boundary
+ `Rbound::TensType`: right boundary

Current environment convention is
     LEFT              RIGHT
   +--<-- 1          3 ---<--+
   |                         |
   |                         |
   +-->-- 2          2 --->--+
   |                         |
   |                         |
   +-->-- 3          1 --->--+
"""
function makeEnds(psi::MPS,mpovec::MPO...;Lbound::TensType=defaultBoundary(psi[1]),Rbound::TensType=defaultBoundary(psi[1]))
  return makeEnds(psi,psi,mpovec...,Lbound=Lbound,Rbound=Rbound)
end
export makeEnds

"""
    Lenv,Renv = makeEnv(dualpsi,psi,mpo[,Lbound=defaultBoundary,Rbound=defaultBoundary,Llabel="Lenv_",Rlabel="Renv_"])

Generates environment tensors for a MPS (`psi` and its dual `dualpsi`) with boundaries `Lbound` and `Rbound`; labels `Llabel` and `Rlabel` are for large-types
    
Current environment convention is
     LEFT              RIGHT
   +--<-- 1          3 ---<--+
   |                         |
   |                         |
   +-->-- 2          2 --->--+
   |                         |
   |                         |
   +-->-- 3          1 --->--+
"""
function makeEnv(dualpsi::MPS,psi::MPS,mpo::MPO...;Lbound::TensType=defaultBoundary(psi[1]),Rbound::TensType=defaultBoundary(psi[1]),Llabel::String="Lenv_",Rlabel::String="Renv_")

  Ns = length(psi)
  numtype = elnumtype(dualpsi,psi,mpo...)
  C = psi[1]
  if typeof(psi) <: largeMPS || typeof(mpo) <: largeMPO
    Lenv,Renv = largeEnv(numtype,Ns,Llabel=Llabel,Rlabel=Rlabel)
  else
    Lenv = environment(psi)
    Renv = environment(psi)
  end

  Lenv[1],Renv[Ns] = makeEnds(dualpsi,psi,mpo...;Lbound=Lbound,Rbound=Rbound)

  for i = 1:psi.oc-1
    Lupdate!(i,Lenv,dualpsi,psi,mpo...)
    Renv[i] = Lenv[1] #avoids any undefined elements
  end

  for i = Ns:-1:psi.oc+1
    Rupdate!(i,Renv,dualpsi,psi,mpo...)
    Lenv[i] = Renv[Ns] #avoids any undefined elements
  end

  return Lenv,Renv
end



const default_Env = environment([tens()])
const default_psiEnv = environment([[tens()]])


"""
    Lenv,Renv = makeEnv(psi,mpo[,Lbound=,Rbound=])

Generates environment tensors for a MPS (`psi`) with boundaries `Lbound` and `Rbound`

Current environment convention is
     LEFT              RIGHT
   +--<-- 1          3 ---<--+
   |                         |
   |                         |
   +-->-- 2          2 --->--+
   |                         |
   |                         |
   +-->-- 3          1 --->--+
"""
function makeEnv(psi::MPS,mpo::MPO;Lbound::TensType=default_boundary,Rbound::TensType=default_boundary)
  return makeEnv(psi,psi,mpo,Lbound=Lbound,Rbound=Rbound)
end

function makeEnv(psi::MPS;Lbound::TensType=default_boundary,Rbound::TensType=default_boundary)
  return makeEnv(psi,psi,Lbound=Lbound,Rbound=Rbound)
end
export makeEnv