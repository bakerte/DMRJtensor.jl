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

#const mpoval_cutoff = 1E-16

"""
    mpo = makeqMPO(Qlabels,mpo)

Generates quantum number MPO from a dense Hamiltonian based on `Qlabels`

# Arguments:
+ `mpo::MPO`: dense MPO
+ `Qlabels::Array{Array{Qnum,1},1}`: quantum numbers for physical indices (modulus size of vector)
"""
function makeqMPO(Qlabels::Array{Array{Q,1},1},mpo::MPO) where Q <: Qnum
  Ns = length(mpo)
  W = elnumtype(mpo)
  QtensVec = Array{Qtens{W,Q},1}(undef, Ns)

  storeQnumMat = [Q()]
  @inbounds for w = 1:Ns
    i = (w-1) % length(mpo) + 1
    QnumMat = Array{Q,1}[Array{Q,1}(undef,size(mpo[i],a)) for a = 1:ndims(mpo[i])]

    QnumMat[1] = inv.(storeQnumMat)
    theseQN = Qlabels[(i-1) % size(Qlabels,1) + 1]
    QnumMat[2] = inv.(theseQN)
    QnumMat[3] = theseQN

    numthreads = Threads.nthreads()

    storeQN = [Array{Q,3}(undef,size(mpo[i],1),size(mpo[i],2),size(mpo[i],3)) for w = 1:numthreads]
    storeVal = [Array{Float64,3}(undef,size(mpo[i],1),size(mpo[i],2),size(mpo[i],3)) for w = 1:numthreads]
    totsize = prod(w->size(mpo[i],w),1:3)

    #=Threads.@threads=# for d = 1:size(mpo[i],4)
      thisthread = Threads.threadid()
      for c = 1:size(mpo[i],3)
        @inbounds for b = 1:size(mpo[i],2)
          tempQN = QnumMat[3][c] + QnumMat[2][b]
          @inbounds for a = 1:size(mpo[i],1)
            storeQN[thisthread][a,b,c] = QnumMat[1][a] + tempQN

            storeVal[thisthread][a,b,c] = abs2(mpo[i][a,b,c,d])
          end
        end
      end

      sumQNs = unique(reshape(storeQN[thisthread],totsize))
      totVals = zeros(Float64,length(sumQNs))

      for w = 1:length(storeVal[thisthread])
        x = 1
        while storeQN[thisthread][w] != sumQNs[x] #&& !isapprox(storeVal[thisthread][w],0)
          x += 1
        end
        totVals[x] += abs2(storeVal[thisthread][w])
      end

      maxval = 1
      for p = 2:length(totVals)
        if totVals[p] > totVals[maxval]
          maxval = p
        end
      end
      QnumMat[4][d] = isapprox(totVals[maxval],0) ? Q() : -(sumQNs[maxval])
    end

    storeQnumMat = QnumMat[4]
    baseQtens = Qtens(QnumMat,currblock=[[1,2],[3,4]])
    QtensVec[i] = Qtens(mpo[i],baseQtens)
  end
  T = prod(a->eltype(mpo[a])(1),1:length(mpo))
  finalMPO = MPO(typeof(T),QtensVec)
  return finalMPO
end

"""
    mpo = makeqMPO(Qlabels,mpo)

Generates quantum number MPO from a dense Hamiltonian based on `Qlabels`

# Arguments:
+ `mpo::MPO`: dense MPO
+ `Qlabels::Array{Qnum,1}`: quantum numbers for physical indices (uniform)
"""
function makeqMPO(Qlabels::Array{Q,1},mpo::MPO) where Q <: Qnum
  return makeqMPO([Qlabels],mpo)
end
export makeqMPO