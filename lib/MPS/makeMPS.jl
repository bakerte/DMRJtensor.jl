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
    makeMPS(vect,physInd[,Ns=,oc=])

generates an MPS from a single vector (i.e., from exact diagonalization) for `Ns` sites and `physInd` size physical index at orthogonality center `oc`
"""
function makeMPS(vect::Array{W,1},inputphysInd::Array{P,1};Ns::Integer=length(physInd),left2right::Bool=true,oc::Integer=left2right ? Ns : 1,regtens::Bool=false) where {W <: Number, P <: Integer}


  hilbertspacesize = prod(inputphysInd)
  if hilbertspacesize != length(vect)
    prodtrack = 1
    Ns = 0
    while prodtrack != length(vect)
      Ns += 1
      prodtrack *= inputphysInd[(Ns-1) % length(inputphysInd) + 1]
    end
    physInd = [inputphysInd[(w-1) % length(inputphysInd) + 1] for w = 1:Ns]
  else
    physInd = inputphysInd
  end

  mps = Array{Array{W,3},1}(undef,Ns)
  # MPS building loop
  if left2right
    M = reshape(vect, physInd[1], div(length(vect),physInd[1]))
    Lindsize = 1 #current size of the left index
    for i=1:Ns-1
      U,DV = qr(M)
      mps[i] = reshape(U,Lindsize,physInd[i],size(DV,1))

      Lindsize = size(DV,1)
      if i == Ns-1
        mps[Ns] = unreshape(DV,Lindsize,physInd[i+1],1)
      else
        Rsize = cld(size(M,2),physInd[i+1]) #integer division, round up
        M = unreshape(DV,size(DV,1)*physInd[i+1],Rsize)
      end
    end
    finalmps = MPS(mps,oc=Ns,regtens=regtens)
  else
    M = reshape(vect, div(length(vect),physInd[end]), physInd[end])
    Rindsize = 1 #current size of the right index
    for i=Ns:-1:2
      UD,V = lq(M)
      mps[i] = reshape(V,size(UD,2),physInd[i],Rindsize)
      Rindsize = size(UD,2)
      if i == 2
        mps[1] = unreshape(UD,1,physInd[i-1],Rindsize)
      else
        Rsize = cld(size(M,1),physInd[i-1]) #integer division, round up
        M = unreshape(UD,Rsize,size(UD,2)*physInd[i-1])
      end
    end
    finalmps = MPS(mps,oc=1,regtens=regtens)
  end
  move!(finalmps,oc)
  return finalmps
end

"""
    makeMPS(vect,physInd[,Ns=,oc=])

generates an MPS from a single vector expressed as a `denstens` (i.e., from exact diagonalization) for `Ns` sites and `physInd` size physical index at orthogonality center `oc`
"""
function makeMPS(vect::denstens,physInd::Array{P,1};Ns::Integer=length(physInd),
                  left2right::Bool=true,oc::Integer=left2right ? Ns : 1,regtens::Bool=false) where P <: Integer
  newvect = copy(vect.T)
  return makeMPS(newvect,physInd;Ns=Ns,oc=oc,left2right=left2right,regtens=regtens)
end

"""
    makeMPS(vect,physInd[,Ns=,oc=])

generates an MPS from a single vector expressed as a `denstens` (i.e., from exact diagonalization) for `Ns` sites and `physInd` an integer that is equal on all sites for the size physical index at orthogonality center `oc`
"""
function makeMPS(vect::Union{Array{W,1},tens{W}},physInd::Integer;Ns::Integer=convert(Int64,log(physInd,length(vect))),
                  left2right::Bool=true,oc::Integer=left2right ? Ns : 1,regtens::Bool=false) where W <: Union{denstens,Number}
  vecPhysInd = [physInd for i = 1:Ns]
  return makeMPS(vect,vecPhysInd;Ns=Ns,oc=oc,left2right=left2right,regtens=regtens)
end
