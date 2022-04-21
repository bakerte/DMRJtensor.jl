function newcorrelation(dualpsi::MPS, psi::MPS, inputoperators...;
                      sites::intvecType=ntuple(i->1:length(psi),length(inputoperators)),
                      trail::Tuple=()) #where S <: Union{Vector{AbstractMatrix{Float64}},TensType}

  move!(psi,1)
  move!(dualpsi,1)

  Lenv,Renv = makeEnv(dualpsi,psi)
  @inbounds for b = 1:length(Renv)
    Renv[b] = permutedims(Renv[b],[2,1])
  end

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
  @inbounds @simd for r = 1:length(trail)
    temp *= eltype(trail[r])(1)
  end
  retType = typeof(temp)

  base_sizes = Array{intType,1}(undef,length(sites))
  @inbounds @simd for i = 1:length(sites)
    base_sizes[i] = length(sites[i]) - (lengthops[i] - 1)
  end




  base_pos = ones(intType,numops)
  base_pos[end] = 0

  omega = zeros(retType,base_sizes...)
  perm = permutations(numops)

  for i = 1:length(perms)

    order = perms[i]


    for w = 1:numops-1
      base_pos[w] = 1
    end
    base_pos[end] = 0

    for j = 1:totalOPs
      operator_in_order!(base_pos,base_sizes)


      for w = 1:length(pos)
        pos[w] = sites[base_pos[w]]
      end

      beginsite = 1
      finalsite = length(psi)

      for w = beginsite:finalsite
        savepsi[w] = psi[w]
      end

      @inbounds @simd for i = 1:numops
        pos[i] = sites[1][1]
        prevpos[i] = sites[1][1]
      end

      thisLenv = Lenv[beginsite]
      for w = beginsite:finalsite
        for g = 1:numops
          if w == pos[g]
            for p = 1:lengthops[g]
              currsite = w + p-1
              savepsi[currsite] = contract([2,1,3],operators[order[g]][p],2,savepsi[currsite],2)
            end
          end
        end
        thisLenv = Lupdate(thisLenv,dualpsi[w],newpsi)
#        if w < Ns
#          Lenv[w+1] = thisLenv
#        end
      end
      thisRenv = Renv[finalsite]
      res = contract(thisLenv,thisRenv)
      omega[pos...] = res
    end
  end



#
#
#
#
#
#
#
#
#
#
#
#=


  numops = length(inputoperators)
  operators = Array{Array{TensType,1},1}(undef,numops)
  lengthops = Array{intType,1}(undef,numops)
  @inbounds for k = 1:numops
    if ndims(inputoperators[k]) == 1
      operators[k] = inputoperators[k]
      lengthops[k] = length(operators[k])
    else
      operators[k] = [inputoperators[k]]
      lengthops[k] = 1
    end
  end

  Ns = length(psi)
  maxOplength = maximum(lengthops)

  temp = eltype(dualpsi[1])(1)
  temp *= eltype(psi[1])(1)
  for w = 1:numops
    @inbounds @simd for a = 1:lengthops[w]
      temp *= eltype(operators[w][a][1])(1)
    end
  end
  @inbounds @simd for r = 1:length(trail)
    temp *= eltype(trail[r])(1)
  end
  retType = typeof(temp)

  base_sizes = Array{intType,1}(undef,length(sites))
  @inbounds @simd for i = 1:length(sites)
    base_sizes[i] = length(sites[i]) - (lengthops[i] - 1)
  end

  omega = zeros(retType,base_sizes...)

  perm = permutations(numops)

  move!(psi,1)
  move!(dualpsi,1)

  Lenv,Renv = makeEnv(dualpsi,psi)
  @inbounds for b = 1:length(Renv)
    Renv[b] = permutedims(Renv[b],[2,1])
  end

  isId = Array{Bool,1}(undef,length(trail))
  @inbounds @simd for r = 1:length(trail)
    isId[r] = sum(trail[r]) == size(trail[r],1)
  end
  if length(trail) > 0
    for r = 1:length(isId)
      index = 0
      @inbounds while isId[r]
        index += 1
        isId[r] = trail[r][index,index] == 1
      end
    end
  end
  activetrail = sum(isId) > 0

  pos = Array{intType,1}(undef,numops)
  prevpos = Array{intType,1}(undef,numops)
  finalpos = Array{intType,1}(undef,numops)

  savepsi = Array{eltype(psi),1}(undef,length(psi))

  @inbounds for i = 1:length(perm)

    order = perm[i]

    base_pos = ones(intType,numops)

    @inbounds @simd for i = 1:numops
      pos[i] = sites[1][1]
      prevpos[i] = sites[1][1]
    end

    @inbounds while sum(base_sizes - pos) >= 0

      startsite = 1
      @inbounds while startsite < length(pos) && pos[startsite] == prevpos[startsite]
        startsite += 1
      end

      @inbounds while startsite > 1 && pos[startsite-1] == prevpos[startsite]
        startsite -= 1
      end

      beginsite = prevpos[startsite]
      finalsite = pos[end]+lengthops[g]

      thisLenv = Lenv[beginsite]

#      startsite = sum(isId) == length(isId) ? beginsite : 1
#=
      @inbounds for w = beginsite:finalsite+maximum(lengthops)
        savepsi[w] = psi[w]
      end
=#
      @inbounds for g = 1:numops
        for w = beginsite:finalsite
          for g = numops:-1:1
            currop = order[g]
            if pos[g] <= w <= pos[g] + lengthops[currop]
              @inbounds for p = 1:lengthops[currop]
                currsite = pos[g] + (p-1)
                savepsi[currsite] = contract([2,1,3],operators[currop][p],savepsi[currsite])
              end
            elseif length(isId) > 0 && isId[g]
              savepsi[w] = contract([2,1,3],trail[g],2,savepsi[w],2)
            end
          end

          thisLenv = Lupdate(thisLenv,dualpsi[w],newpsi)
          if w < Ns
            Lenv[w+1] = thisLenv
          end
        end
      end



      #=
      @inbounds for w = beginsite:finalsite
        newpsi = psi[w]
        @inbounds for g = 1:numops
          opdist = w - pos[g]
          if 0 <= opdist < lengthops[g]
            newpsi = contract([2,1,3],operators[order[g]][opdist + 1],2,newpsi,2)
          end
        end
        @inbounds for r = 1:numops
          if  length(isId) > 0 && w < pos[r] && !isId[r]
            newpsi = contract([2,1,3],trail[r],2,newpsi,2)
          end
        end
        thisLenv = Lupdate(thisLenv,dualpsi[w],newpsi)
        if w < Ns
          Lenv[w+1] = thisLenv
        end
      end
      =#

      thisRenv = Renv[finalsite]
      res = contract(thisLenv,thisRenv)
      @inbounds @simd for w = 1:length(pos)
        finalpos[w] = pos[order[w]]
      end
      
      omega[finalpos...] = res

      @inbounds @simd for b = 1:numops
        prevpos[b] = pos[b]
      end
      operator_in_order!(base_pos,base_sizes)
      @inbounds @simd for b = 1:numops
        pos[b] = sites[b][base_pos[b]]
      end
    end
  end
  =#
  return omega
end