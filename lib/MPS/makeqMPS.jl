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
"""
function makeqMPS(Qlabels::Array{Array{Q,1},1},mps::MPS,arrows::Array{Bool,1}...;newnorm::Bool=true,setflux::Bool=false,
                  flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where Q <: Qnum
  if newnorm
    if mps.oc < length(mps)
      move!(mps,mps.oc+1)
      move!(mps,mps.oc-1)
    else
      move!(mps,mps.oc-1)
      move!(mps,mps.oc+1)
    end
    start_norm = expect(mps)
  end

  W = elnumtype(mps)
  QtensVec = Array{Qtens{W,Q},1}(undef, length(mps.A))

  Ns = length(mps)
  storeQnumMat = [Q()]
  theseArrows = length(arrows) == 0 ? Bool[false,true,true] : arrows[1]
  @inbounds for i = 1:Ns
    currSize = size(mps[i])
    QnumMat = Array{Q,1}[Array{Q,1}(undef,currSize[a]) for a = 1:ndims(mps[i])]

    QnumMat[1] = inv.(storeQnumMat)
    QnumMat[2] = Qlabels[(i-1) % size(Qlabels,1) + 1] 
    storeVal = zeros(Float64,size(mps[i],3))

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
  finalMPS = matrixproductstate(network(QtensVec),mps.oc)

  thisnorm = expect(finalMPS)

  if newnorm
    finalMPS[mps.oc] *= sqrt(start_norm)/sqrt(thisnorm)
  end
  if lastfluxzero
    @inbounds for q = 1:length(finalMPS[end].Qblocksum)
      finalMPS[end].Qblocksum[q][2] = -(finalMPS[end].flux)
    end
  else
    Qnumber = finalMPS[end].QnumMat[3][1]

    finalMPS[end].flux,newQnum = -(finalMPS[end].QnumSum[3][Qnumber]),-(finalMPS[end].flux)
    finalMPS[end].QnumSum[3][1] = newQnum
  end

  @inbounds for q = 1:length(finalMPS[end].Qblocksum)
    index = finalMPS[end].ind[q][2][:,1] .+ 1 #[x]
    pos = finalMPS[end].currblock[2]
    newQsum = Q()
    @inbounds for y = 1:length(pos)
      newQsum += getQnum(pos[y],index[y],finalMPS[end])
    end
    finalMPS[end].Qblocksum[q] = (finalMPS[end].Qblocksum[q][1],newQsum)
  end

  return finalMPS #sets initial orthogonality center at site 1 (hence the arrow definition above)
end

"""
    makeqMPS(Qlabels,mps[,arrows,newnorm=,setflux=,flux=,randomize=,override=])

creates quantum number MPS from regular MPS according to `Qlabels`

# Arguments
+ `mps::MPS`: dense MPS
+ `Qlabels::Array{Qnum,1}`: quantum number labels on each physical index (uniform physical indices)
+ `arrows::Array{Bool,1}`: first entry of this tuple is the arrow convention on MPS tensors (default: [false,true,true])
+ `newnorm::Bool`: set new norm of the MPS tensor
+ `setflux::Bool`: toggle to force this to be the total flux of the MPS tensor
+ `flux::Qnum`: quantum number to force total MPS flux to be if setflux=true
+ `randomize::Bool`: randomize last tensor if flux forces a zero tensor
"""
function makeqMPS(Qlabels::Array{Q,1},mps::MPS,arrows::Array{Bool,1}...;newnorm::Bool=true,setflux::Bool=false,
  flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where Q <: Qnum
  return makeqMPS([Qlabels],mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
end

function makeqMPS(Qlabels::W,arr::Union{network,Array},arrows::Array{Bool,1}...;oc::Integer=1,newnorm::Bool=true,setflux::Bool=false,
  flux::Q=Q(),randomize::Bool=true,override::Bool=true,lastfluxzero::Bool=false)::MPS where W <: Union{Array{Array{Q,1},1},Array{Q,1}} where Q <: Qnum
mps = MPS(arr,oc=oc)
makeqMPS(Qlabels,mps,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,lastfluxzero=lastfluxzero)
end
export makeqMPS