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

"""
    mpo = makeqMPO(Qlabels,mpo[,arrows])

Generates quantum number MPO from a dense Hamiltonian based on `Qlabels`

# Arguments:
+ `mpo::MPO`: dense MPO
+ `Qlabels::Array{Array{Qnum,1},1}`: quantum numbers for physical indices (modulus size of vector)
+ `arrows::Array{Bool,1}`: arrow convention for quantum numbers (default: [false,false,true,true])
"""
function makeqMPO(Qlabels::Array{Array{Q,1},1},mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1)::MPO where Q <: Qnum
  Ns = infinite ? 3*unitcell*length(mpo) : length(mpo)
  W = elnumtype(mpo)
  QtensVec = Array{Qtens{W,Q},1}(undef, Ns)

  storeQnumMat = [Q()]
  theseArrows = length(arrows) == 0 ? Bool[false,false,true,true] : arrows[1]
  @inbounds for w = 1:Ns
    i = (w-1) % length(mpo) + 1
    QnumMat = Array{Q,1}[Array{Q,1}(undef,size(mpo[i],a)) for a = 1:ndims(mpo[i])]

    QnumMat[1] = inv.(storeQnumMat)
    theseQN = Qlabels[(i-1) % size(Qlabels,1) + 1]
    QnumMat[2] = inv.(theseQN)
    QnumMat[3] = theseQN
    storeVal = -ones(Float64,size(mpo[i],4))
    for a = 1:size(mpo[i],1)
      for b = 1:size(mpo[i],2)
        for c = 1:size(mpo[i],3)
          @inbounds for d = 1:size(mpo[i],4)
            absval = abs(mpo[i][a,b,c,d])
            if absval > storeVal[d]
              storeVal[d] = absval
              tempQN = QnumMat[1][a] + QnumMat[2][b] + QnumMat[3][c]
              QnumMat[4][d] = -(tempQN)
            end
          end
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
    mpo = makeqMPO(Qlabels,mpo[,arrows])

Generates quantum number MPO from a dense Hamiltonian based on `Qlabels`

# Arguments:
+ `mpo::MPO`: dense MPO
+ `Qlabels::Array{Qnum,1}`: quantum numbers for physical indices (uniform)
+ `arrows::Array{Bool,1}`: arrow convention for quantum numbers (default: [false,false,true,true])
"""
function makeqMPO(Qlabels::Array{Q,1},mpo::MPO,arrows::Array{Bool,1}...;infinite::Bool=false,unitcell::Integer=1)::MPO where Q <: Qnum
  return makeqMPO([Qlabels],mpo,arrows...,infinite=infinite,unitcell=unitcell)
end
export makeqMPO