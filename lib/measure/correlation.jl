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
    operator_in_order!(pos,sizes)

Increments elements of input vector `pos` by one step (last element first) with sizes of a tensor `sizes` such that `pos`.  For use in `correlation` function.
"""
 function operator_in_order!(pos::Array{G,1},sizes::intvecType) where G <: Integer
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
    T = correlation(psi,inputoperators...[,sites=ntuple(i->1,length(inputoperators)),trail=()])

Computes the general correlation on the input MPS `psi` with operators defined in the tuple `inputoperators`. The `sites` command specifies which sites to allow for the operators to be applied (i.e., some operators can skip some sites) and the `trail` option allows for a trailing operator like the Fermion sign operator; output is a tensor `T` of rank equal to the size of `inputoperators`

See also: [`expect`](@ref) [`correlationmatrix`](@ref)
"""
function correlation(psi::MPS, inputoperators...;
                      sites::Tuple=ntuple(i->1:length(psi),length(inputoperators)),
                      trail::Union{Tuple,TensType}=()) #where S <: Union{Vector{AbstractMatrix{Float64}},TensType}

  return correlation(psi,psi,inputoperators...,sites=sites,trail=trail)
end

"""
    T = correlation(dualpsi,psi,inputoperators...[,sites=ntuple(i->1,length(inputoperators)),trail=()])

Computes the general correlation on the input MPS `psi` with operators defined in the tuple `inputoperators`. The `sites` command specifies which sites to allow for the operators to be applied (i.e., some operators can skip some sites) and the `trail` option allows for a trailing operator like the Fermion sign operator; output is a tensor `T` of rank equal to the size of `inputoperators`

See also: [`expect`](@ref) [`correlationmatrix`](@ref)
"""
function correlation(dualpsi::MPS, psi::MPS, inputoperators...;
                      sites::Tuple=ntuple(i->1:length(psi),length(inputoperators)),
                      trail::Union{Tuple,TensType}=()#=,periodic::Bool=false,infinite::Bool=false=#) #where S <: Union{Vector{AbstractMatrix{Float64}},TensType}

  move!(psi,1)
  move!(dualpsi,1)

  istrail = trail != ()
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