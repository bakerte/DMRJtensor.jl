#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.1
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.0) or (v1.5)
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
  function convert2MPO(H::AbstractArray,physSize::Array{Y,1},Ns::intType;infinite::Bool=false,saveL::intType=0,saveR::intType=0,
                       lower::Bool=true,regtens::Bool=false)::MPO where {X <: Integer, Y <: Integer}
    retType = typeof(prod(a->eltype(H[a])(1),1:Ns))
    finalMPO = Array{Array{retType,4},1}(undef,Ns)
    for i = 1:Ns
      thisH = lower ? H[i] : transpose(H[i])
      QS = physSize[(i-1)%size(physSize,1) + 1]
      a1size = div(size(thisH,1),QS) #represented in LEFT link indices
      a2size = div(size(thisH,2),QS) #represented in RIGHT link indices
      P = eltype(thisH)
      W = zeros(P,a1size,QS,QS,a2size)
      if 1 < i < Ns || infinite
        finalMPO[i] = Array{P,4}(undef,a1size,QS,QS,a2size)
      elseif i == 1
        finalMPO[1] = Array{P,4}(undef,1,QS,QS,a2size)
      elseif i == Ns
        finalMPO[Ns] = Array{P,4}(undef,a1size,QS,QS,1)
      end
      let QS = QS, a1size = a1size, a2size = a2size, H = H, W = W, i = i
        Threads.@threads for m = 1:a2size
          for k = 1:QS
            for j = 1:QS
              @simd for l = 1:a1size
                W[l,j,k,m] = thisH[j + (l-1)*QS, k + (m-1)*QS]
              end
            end
          end
        end
      end
      if 1 < i < Ns || infinite
        finalMPO[i] = W #to be expanded later to include MPOs that vary on each site.
      elseif i == 1
        lind = saveL != 0 ? saveL : a1size
        finalMPO[i][1,:,:,:] = W[lind,:,:,:] #put in bottom row on first site
      elseif i == Ns
        rind = saveR != 0 ? saveR : 1
        finalMPO[i][:,:,:,1] = W[:,:,:,rind] #put in first column on last site
      end
    end
    return MPO(finalMPO,regtens=regtens)
  end

  function convert2MPO(H::AbstractArray,physSize::Y,Ns::intType;infinite::Bool=false,saveL::intType=0,saveR::intType=0,
                       lower::Bool=true,regtens::Bool=false)::MPO where X <: Integer where Y <: Integer
    return convert2MPO(H,[physSize],Ns,infinite=infinite,lower=lower,regtens=regtens,saveL=saveL,saveR=saveR)
  end

  function convert2MPO(H::Function,physSize::Array{X,1},Ns::intType;infinite::Bool=false,saveL::intType=0,saveR::intType=0,
                       lower::Bool=true,regtens::Bool=false)::MPO where X <: Integer
    thisvec = [H(i) for i = 1:Ns]
    return convert2MPO(thisvec,physSize,Ns,infinite=infinite,lower=lower,regtens=regtens,saveL=saveL,saveR=saveR)
  end

  function convert2MPO(H::Function,physSize::intType,Ns::intType;infinite::Bool=false,saveL::intType=0,saveR::intType=0,
                       lower::Bool=true,regtens::Bool=false)::MPO
    return convert2MPO(H,[physSize],Ns,infinite=infinite,lower=lower,regtens=regtens,saveL=saveL,saveR=saveR)
  end
  export convert2MPO

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
    rfullH = rpfullH[size(rpfullH,1),:,:,1]
    return rfullH #reshape(rfullH,2*ones(Int64,Ns)...,2*ones(Int64,Ns)...)
  end
  export fullH

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



#       +---------------------------------------+
#>------+           convert to qMPS             +---------<
#       +---------------------------------------+

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
    let mps = mps, i = i, storeVal = storeVal, QnumMat = QnumMat
      Threads.@threads for c = 1:size(mps[i],3)
        for b = 1:size(mps[i],2),a = 1:size(mps[i],1)
          absval = abs(mps[i][a,b,c])
          if absval > storeVal[c]
            storeVal[c] = absval
            QnumMat[3][c] = inv(QnumMat[1][a]+QnumMat[2][b])
          end
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
                    flux::Q=Q(),randomize::Bool=true,override::Bool=true,warning::Bool=true)::MPS where Q <: Qnum
    if newnorm
      if warning
        println("(makeqMPS: If converting from a non-QN MPS to a QN MPS, then beware that applying a 3S or adding noise in general to the MPS is not considered when calling makeqMPS)")
      end
      start_norm = expect(mps)
    end
    QtensVec = Array{Qtens,1}(undef, size(mps.A,1))
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
      newQt = Qtens(QnumMat)
      QtensVec[i] = Qtens(newQt,mps[i])

      if size(QtensVec[i].T,1) == 0 && randomize
        QtensVec[i] = rand(QtensVec[i])
        if size(QtensVec[i].T,1) == 0 && !override
          error("specified bad quantum number when making QN MPS...try a different quantum number")
        end
      end
    end
    finalMPS = MPS(QtensVec,mps.oc)
    finalMPS[end].flux,finalMPS[end].QnumMat[3][1] = inv(finalMPS[end].QnumMat[3][1]),(finalMPS[end].flux)
    finalMPS[end].QnumSum = unique.(finalMPS[end].QnumMat)
    thisnorm = expect(finalMPS)
    if newnorm
      finalMPS[mps.oc] *= sqrt(start_norm)/sqrt(thisnorm)
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
    flux::Q=Q(),randomize::Bool=true,override::Bool=true)::MPS where Q <: Qnum
    return makeqMPS(mps,[Qlabels],arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override)
  end

  function makeqMPS(arr::Array,Qlabels::W,arrows::Array{Bool,1}...;oc::intType=1,newnorm::Bool=true,setflux::Bool=false,
                    flux::Q=Q(),randomize::Bool=true,override::Bool=true,warning::Bool=true)::MPS where W <: Union{Array{Array{Q,1},1},Array{Q,1}} where Q <: Qnum
    mps = MPS(arr,oc)
    makeqMPS(mps,Qlabels,arrows...,newnorm=newnorm,setflux=setflux,flux=flux,randomize=randomize,override=override,warning=warning)
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
    QtensVec = Array{Qtens,1}(undef, Ns)
    storeQnumMat = Q[zeroQN]
    theseArrows = length(arrows) == 0 ? Bool[false,false,true,true] : arrows[1]
    for w = 1:Ns
      i = (w-1) % length(mpo) + 1
      QnumMat = Array{Q,1}[Array{Q,1}(undef,size(mpo[i],a)) for a = 1:ndims(mpo[i])]

      QnumMat[1] = inv.(storeQnumMat)
      theseQN = Qlabels[(i-1) % size(Qlabels,1) + 1]
      QnumMat[2] = inv.(theseQN)
      QnumMat[3] = theseQN
      storeVal = -ones(Float64,size(mpo[i],4))
      let mpo = mpo, QnumMat = QnumMat, i = i, storeVal = storeVal
        Threads.@threads for d = 1:size(mpo[i],4)
          for c = 1:size(mpo[i],3), b = 1:size(mpo[i],2), a = 1:size(mpo[i],1)
            absval = abs(mpo[i][a,b,c,d])
            if absval > storeVal[d]
              storeVal[d] = absval
              QnumMat[4][d] = inv(QnumMat[1][a] + QnumMat[2][b] + QnumMat[3][c])
            end
          end
        end
      end
      storeQnumMat = QnumMat[4]
      baseQtens = Qtens(QnumMat)
      QtensVec[i] = Qtens(baseQtens,mpo[i])
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
    #fiedler vector reordering
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
#    @assert(size(operator,1) == size(ind,1))
#    @assert(maximum(ind) <= size(base,1))
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


  function ladder(val::Number,operator::Array{W,1},ind::Array{intType,1},base::Array{X,1},trail::Y...)::MPO where {W <: AbstractArray, X <: AbstractArray, Y <: AbstractArray}
    return mpoterm(val,operator,ind,base,trail...)
  end

  function ladder(operator::Array{W,1},ind::Array{intType,1},base::Array{X,1},trail::Y...)::MPO where {W <: AbstractArray, X <: AbstractArray, Y <: AbstractArray}
    return mpoterm(1.,operator,ind,base,trail...)
  end
  export ladder

  function qladder(Qlabels::Array{Array{Q,1},1},val::Number,operator::AbstractArray,ind::Array{intType,1},base::AbstractArray,trail::AbstractArray...)::MPO where Q <: Qnum
    return mpoterm(Qlabels,operator,ind,base,trail...)
  end

  function qladder(Qlabels::Array{Array{Q,1},1},operator::AbstractArray,ind::Array{intType,1},base::AbstractArray,trail::AbstractArray...)::MPO where Q <: Qnum
    return mpoterm(Qlabels,1.,operator,ind,base,trail...)
  end
  export qladder



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

  functionality for adding (similar to direct sum) of MPOs together; uses concat function to make a combined MPO

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

  functionality for adding (similar to direct sum) of MPOs together and replacing `A`; uses concat function to make a combined MPO

  note: deparallelizes after every addition

  See also: [`deparallelization`](@ref) [`+`](@ref)
  """
  function add!(A::MPO,B::MPO;finiteBC::Bool=true)
    if finiteBC
      A[1] = concat!(4,A[1],B[1])
      for a = 2:size(A,1)-1
        A[a] = concat!([1,4],A[a],B[a])
      end
      A[size(A,1)] = concat!(1,A[size(A,1)],B[size(A,1)])
    else
      for a = 1:size(A,1)
        A[a] = concat!([1,4],A[a],B[a])
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

  import ..contractions.scalarcontractor
  """
      deparallelize!(M[,left=])

  Deparallelizes a matrix-equivalent of a rank-4 tensor `M`; toggle the decomposition into the `left` or `right`
  """
  function deparallelize!(M::TensType;left::Bool=true,zero::Float64=0.) where W <: Number
    K = Array{typeof(M),1}(undef,0)
    if left
      rsize = size(M,4)
      lsize = max(prod(a->size(M,a),1:3),rsize)
    else
      lsize = size(M,1)
      rsize = max(lsize,prod(a->size(M,a),2:4))
    end
    T = zeros(eltype(M),lsize,rsize) #maximum size for either left or right
    firstvec = pullvec(M,1,left)

    push!(K,firstvec)
    b = left ? 4 : 1
    for j = 1:size(M,b)
      thisvec = pullvec(M,j,left)

      mag_thisvec = norm(thisvec) # |A|
      condition = true
      i = 0
      while condition  && i < size(K,1)
        i += 1
        dot = scalarcontractor(K[i],thisvec,left,!left)
        normK = norm(K[i])
        if isapprox(real(dot),mag_thisvec * normK)
          if left
            T[i,j] = mag_thisvec/normK
          else
            T[j,i] = mag_thisvec/normK
          end
          condition = false
        end
      end
      if condition && !(isapprox(mag_thisvec,0.))
        push!(K,thisvec)
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
        newK = concat!(newK,K[a],4)
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
        newK = concat!(newK,K[a],1)
      end

      if QNflag
        finalT = Qtens(finalT,[M.QnumMat[1],inv.(newK.QnumMat[1])])
      end

      return finalT,newK
    end
  end

  function deparallelize!(M::tens{W};left::Bool=true) where W <: Number
    X = reshape(M.T,M.size...)
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
          Threads.@threads for i = 1:2^j:currsize
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
          Threads.@threads for i = 1:2^j:currsize
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
        mpo[i] = concat!(4,mpo[i],rho)
      elseif i == length(psi)
        mpo[i] = concat!(1,mpo[i],rho)
      else
        mpo[i] = concat!([1,4],mpo[i],rho)
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
end