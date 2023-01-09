

println("#            +-------------+")
println("#>-----------|  libalg.jl  |-----------<")
println("#            +-------------+")
global fulltest = true

import LinearAlgebra

typevec = [Float32,Float64,ComplexF32,ComplexF64]
tol = [1E-4,1E-10,1E-5,1E-11]

for g = 1:length(typevec)
  local A = rand(typevec[g],100,100)
  local A += A'
  local D,U = DMRjulia.libeigen(A)

  local tA = tens(A)
  local tD,tU = DMRjulia.libeigen(tA)

  local checkD,checkU = LinearAlgebra.eigen(A)

  local testval = isapprox(U*LinearAlgebra.Diagonal(D)*U',A)
  local newU = reshape(tU,100,100)
  local testval &= isapprox(newU*LinearAlgebra.Diagonal(D)*newU',A)
  global fulltest &= testfct(testval,"eigen $(typevec[g])")
end

println()

for g = 1:length(typevec)
  local A = rand(typevec[g],200,100)

  local U,D,V = DMRjulia.libsvd(A)

  local tA = tens(A)
  local tU,tD,tV = DMRjulia.libsvd(tA)

#  checkD,checkU = LinearAlgebra.eigen(A)

  local testval = isapprox(U*LinearAlgebra.Diagonal(D)*V,A)
  local newU = reshape(tU,200,100)
  local newV = reshape(tV,100,100)
  local testval &= isapprox(newU*LinearAlgebra.Diagonal(D)*newV,A)
  global fulltest &= testfct(testval,"svd $(typevec[g])")
end

println()

for g = 1:length(typevec)
  local A = rand(typevec[g],200,100)
  local B = rand(typevec[g],100,200)

  local alpha = typevec[g](3)
  local beta = typevec[g](3)

  local C = rand(typevec[g],200,200)

  local realC = alpha*A*B+beta*C

  local C = DMRjulia.libmult!('N','N',alpha,A,B,beta,C,200,100,100,200)
  
  local checkC = DMRjulia.libmult('N','N',alpha,A,B,200,100,100,200)

  local tA = tens(A)
  local tU,tD,tV = DMRjulia.libsvd(tA)

#  checkD,checkU = LinearAlgebra.eigen(A)

  local testval = isapprox(C,realC) && isapprox(checkC,alpha*A*B)
  global fulltest &= testfct(testval,"libmult(!) $(typevec[g])")
end

println()

for g = 1:length(typevec)
  local A = rand(typevec[g],100,100)
  local B = rand(typevec[g],100,100)
  for y = 1:100
    for x = y+1:100
      B[x,y] = 0.
    end
  end

  local C = B*A

#  display(C)

  local checkC = DMRjulia.trmm!(B,100,100,A,100,side='L')

#  display(checkC)

  local testval = isapprox(C,checkC)
  global fulltest &= testfct(testval,"trmm! $(typevec[g])")
end

println()

for g = 1:length(typevec)
  local A = rand(typevec[g],100,200)
#  tA = tens(A)

  local checkQ,checkR = LinearAlgebra.qr(A)

  local Q,R = DMRjulia.libqr!(copy(A),100,200)
  local aQ,aR = DMRjulia.libqr(copy(A),100,200)

  local out = 0
  for y = 1:size(R,2)
    @inbounds @simd for x = y+1:size(R,1)
      out += R[x,y]
    end
  end

  local testval = isapprox(checkQ*checkR,A) && isapprox(Q*R,A) && isapprox(aQ*aR,A) && isapprox(out,0)
  global fulltest &= testfct(testval,"libqr! $(typevec[g])")
end

println()

for g = 1:length(typevec)
  local A = rand(typevec[g],100,200)
#  tA = tens(A)

  local checkQ,checkR = LinearAlgebra.lq(A)

  local Q,R = DMRjulia.liblq!(copy(A),100,200)
  local aQ,aR = DMRjulia.liblq!(copy(A),100,200)

  local out = 0
  for y = 1:size(Q,1)
    @inbounds @simd for x = y+1:size(Q,2)
      out += Q[y,x]
    end
  end

  local testval = isapprox(checkQ*checkR,A) && isapprox(Q*R,A) && isapprox(aQ*aR,A) && isapprox(out,0)
  global fulltest &= testfct(testval,"liblq! $(typevec[g])")
end

println()

A = rand(100,200)
B = DMRjulia.libUpperHessenberg!(A)

global out = 0
for y = 1:size(B,2)
  @inbounds @simd for x = y+1:size(B,1)
    global out += B[x,y]
  end
end

testval = isapprox(out,0)
global fulltest &= testfct(testval,"libUpperHessenberg!")

println()

A = rand(100,200)
B = DMRjulia.libLowerHessenberg!(A)

global out = 0
for y = 1:size(B,1)
  @inbounds @simd for x = y+1:size(B,2)
    global out += B[y,x]
  end
end

testval = isapprox(out,0)
global fulltest &= testfct(testval,"libLowerHessenberg!")

