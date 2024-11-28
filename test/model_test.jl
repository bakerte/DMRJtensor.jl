
println("#            +-----------------+")
println("#>-----------|  model_test.jl  |-----------<")
println("#            +-----------------+")
fulltest = true

Ns = 10
spinmag = 0.5

hereQS = convert(Int64,2*spinmag+1)
QS = cld(hereQS,2)


Sp,Sm,Sz,Sy,Sx,O,Id = spinOps(s=spinmag)
base = [Id for i = 1:Ns]

function xone(i)
  return [Id O O O O;
          Sp/2 O O O O;
          Sm/2 O O O O;
          Sz O O O O;
          O Sm Sp Sz Id]
end

function xtwo(i)
  return [Id O O O O O O O;
          Sp/2 O O O O O O O;
          O Id O O O O O O;
          Sm/2 O O O O O O O;
          O O O Id O O O O;
          Sz O O O O O O O;
          O O O O O Id O O;
          O Sm Sm Sp Sp Sz Sz Id]
end

function xthree(i)
  return [Id O O O O O O O O O O;
          Sp/2 O O O O O O O O O O;
          O Id O O O O O O O O O;
          O O Id O O O O O O O O;
          Sm/2 O O O O O O O O O O;
          O O O O Id O O O O O O;
          O O O O O Id O O O O O;
          Sz O O O O O O O O O O;
          O O O O O O O Id O O O;
          O O O O O O O O Id O O;
          O Sm O Sm Sp O Sp Sz O Sz Id]
end

function xfour(i)
  return [Id O O O O O O O O O O O O O;
          Sp/2 O O O O O O O O O O O O O;
          O Id O O O O O O O O O O O O;
          O O Id O O O O O O O O O O O;
          O O O Id O O O O O O O O O O;
          Sm/2 O O O O O O O O O O O O O;
          O O O O O Id O O O O O O O O;
          O O O O O O Id O O O O O O O;
          O O O O O O O Id O O O O O O;
          Sz O O O O O O O O O O O O O;
          O O O O O O O O O Id O O O O;
          O O O O O O O O O O Id O O O;
          O O O O O O O O O O O Id O O;
          O Sm O O Sm Sp O O Sp Sz O O Sz Id]
end

mpos = [xone,xtwo,xthree,xfour]


testvals = Array{Bool,1}(undef,length(mpos))

#println("Regular")
for r = 1:length(mpos)
  H = makeMPO(mpos[r],2,Ns)

  checkH = 0
  rangevec = unique([1,r])
  for w in rangevec
    for i = 1:Ns-w
      checkH += mpoterm(0.5,[Sp,Sm],[i,i+w],base)
      checkH += mpoterm(0.5,[Sm,Sp],[i,i+w],base)
      checkH += mpoterm([Sz,Sz],[i,i+w],base)
    end
  end
  
  global D,U = eigen(H)
  global checkD,checkU = eigen(checkH)

  testvals[r] = isapprox(D[1],checkD[1])
  
#  println("$r-neighbor model (spin): ",D[1]," ",checkD[1])
#  println()
end

fulltest &= testfct(sum(testvals)==length(testvals),"r-neighbour models")

#println()
#println()
#println()
#println()
#println()

#println("Rainbow")

function xone(i)
  return [Id O O O O;
          Sp/2 O O O O;
          Sm/2 O O O O;
          Sz O O O O;
          O Sm Sp Sz Id]
end

function xtwo(i)
  return [Id O O O O O O O;
          Sp/2 O O O O O O O;
          O Id O O O O O O;
          Sm/2 O O O O O O O;
          O O O Id O O O O;
          Sz O O O O O O O;
          O O O O O Id O O;
          O Sm Sm Sp Sp Sz Sz Id]
end

function xthree(i)
  return [Id O O O O O O O O O O;
          Sp/2 O O O O O O O O O O;
          O Id O O O O O O O O O;
          O O Id O O O O O O O O;
          Sm/2 O O O O O O O O O O;
          O O O O Id O O O O O O;
          O O O O O Id O O O O O;
          Sz O O O O O O O O O O;
          O O O O O O O Id O O O;
          O O O O O O O O Id O O;
          O Sm Sm Sm Sp Sp Sp Sz Sz Sz Id]
end

function xfour(i)
  return [Id O O O O O O O O O O O O O;
          Sp/2 O O O O O O O O O O O O O;
          O Id O O O O O O O O O O O O;
          O O Id O O O O O O O O O O O;
          O O O Id O O O O O O O O O O;
          Sm/2 O O O O O O O O O O O O O;
          O O O O O Id O O O O O O O O;
          O O O O O O Id O O O O O O O;
          O O O O O O O Id O O O O O O;
          Sz O O O O O O O O O O O O O;
          O O O O O O O O O Id O O O O;
          O O O O O O O O O O Id O O O;
          O O O O O O O O O O O Id O O;
          O Sm Sm Sm Sm Sp Sp Sp Sp Sz Sz Sz Sz Id]
end

mpos = [xone,xtwo,xthree,xfour]

for r = 1:length(mpos)
  H = makeMPO(mpos[r],2,Ns)

  checkH = 0
  rangevec = unique([1,r])
  for w = 1:r #in rangevec
    for i = 1:Ns-w
      checkH += mpoterm(0.5,[Sp,Sm],[i,i+w],base)
      checkH += mpoterm(0.5,[Sm,Sp],[i,i+w],base)
      checkH += mpoterm([Sz,Sz],[i,i+w],base)
    end
  end
  
  global D,U = eigen(H)
  global checkD,checkU = eigen(checkH)

  testvals[r] = isapprox(D[1],checkD[1])
  
#  println("$r-neighbor model (spin): ",D[1]," ",checkD[1])
#  println()
end


fulltest &= testfct(sum(testvals)==length(testvals),"r-rainbow-neighbour models")

#println()
#println()
#println()
#println()
#println()

xlambda = exp(-1/2.)

#println("Exponentials")

function xone(i)
  return [Id O O O O O;
          Sp/2 O O O O O;
          Sm/2 O O O O O;
          Sz O O O O O;
          Sz O O O xlambda*Id O;
          O Sm Sp Sz Sz Id]
end

function xtwo(i)
  return [Id O O O O O O O O;
          Sp/2 O O O O O O O O;
          O Id O O O O O O O;
          Sm/2 O O O O O O O O;
          O O O Id O O O O O;
          Sz O O O O O O O O;
          O O O O O Id O O O;
          Sz O O O O O O xlambda*Id O;
          O Sm Sm Sp Sp Sz Sz Sz Id]
end

function xthree(i)
  return [Id O O O O O O O O O O O;
          Sp/2 O O O O O O O O O O O;
          O Id O O O O O O O O O O;
          O O Id O O O O O O O O O;
          Sm/2 O O O O O O O O O O O;
          O O O O Id O O O O O O O;
          O O O O O Id O O O O O O;
          Sz O O O O O O O O O O O;
          O O O O O O O Id O O O O;
          O O O O O O O O Id O O O;
          Sz O O O O O O O O O xlambda*Id O;
          O Sm O Sm Sp O Sp Sz O Sz Sz Id]
end

function xfour(i)
  return [Id O O O O O O O O O O O O O O;
          Sp/2 O O O O O O O O O O O O O O;
          O Id O O O O O O O O O O O O O;
          O O Id O O O O O O O O O O O O;
          O O O Id O O O O O O O O O O O;
          Sm/2 O O O O O O O O O O O O O O;
          O O O O O Id O O O O O O O O O;
          O O O O O O Id O O O O O O O O;
          O O O O O O O Id O O O O O O O;
          Sz O O O O O O O O O O O O O O;
          O O O O O O O O O Id O O O O O;
          O O O O O O O O O O Id O O O O;
          O O O O O O O O O O O Id O O O;
          Sz O O O O O O O O O O O O xlambda*Id O;
          O Sm O O Sm Sp O O Sp Sz O O Sz Sz Id]
end

mpos = [xone,xtwo,xthree,xfour]


for r = 1:length(mpos)
  H = makeMPO(mpos[r],2,Ns)



  doublecheckH = 0
  rangevec = unique([1,r])
  for w in rangevec #1:r #
    for i = 1:Ns-w
      doublecheckH += mpoterm(0.5,[Sp,Sm],[i,i+w],base)
      doublecheckH += mpoterm(0.5,[Sm,Sp],[i,i+w],base)
      doublecheckH += mpoterm([Sz,Sz],[i,i+w],base)
    end
  end

  function expSzSz(i)
    return [Id O O;
            Sz xlambda*Id O;
            O Sz Id]
  end

  expSzSz_mpo = makeMPO(expSzSz,2,Ns)

  doublecheckH += expSzSz_mpo




  checkH = 0
  rangevec = unique([1,r])
  for w in rangevec #1:r #
    for i = 1:Ns-w
      checkH += mpoterm(0.5,[Sp,Sm],[i,i+w],base)
      checkH += mpoterm(0.5,[Sm,Sp],[i,i+w],base)
      checkH += mpoterm([Sz,Sz],[i,i+w],base)
    end
  end

  checkH += expMPO(xlambda,Sz,Sz,Ns)
  
  global D,U = eigen(H)
  global checkD,checkU = eigen(checkH)
  doublecheckD,doublecheckU = eigen(doublecheckH)

  testvals[r] = isapprox(D[1],checkD[1]) && isapprox(D[1],doublecheckD[1])
  
#  println("$r-neighbor model (spin): ",D[1]," ",checkD[1]," ",doublecheckD[1])
#  println()
end


fulltest &= testfct(sum(testvals)==length(testvals),"r-neighbour models with exponential interaction")

println()


Ns = 10

@makeQNs "fermion" U1 U1
Qlabels = [[fermion(0,0),fermion(1,1),fermion(1,-1),fermion(2,0)]]

Ne = Ns
Ne_up = ceil(Int64,div(Ne,2))
Ne_dn = Ne-Ne_up
QS = 4
Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()

psi = MPS(QS,Ns)
upsites = [i for i = 1:Ne_up]
Cupdag = Matrix(Cup')
applyOps!(psi,upsites,Cupdag,trail=F)


dnsites = [i for i = 1:Ne_dn]
Cdndag = Matrix(Cdn')
applyOps!(psi,dnsites,Cdndag,trail=F)

qpsi = makeqMPS(Qlabels,psi)

mu = -2.0
HubU = 4.0
t = 1.0

function H(i::Int64)
        onsite = mu * Ndens + HubU * Nup * Ndn #- Ne*exp(-abs(i-Ns/2)/2)*Ndens
        return [Id  O O O O O;
            -t*Cup' O O O O O;
            conj(t)*Cup  O O O O O;
            -t*Cdn' O O O O O;
            conj(t)*Cdn  O O O O O;
            onsite Cup*F Cup'*F Cdn*F Cdn'*F Id]
    end

#println("Making qMPO")

mpo = makeMPO(H,QS,Ns)
qmpo = makeqMPO(Qlabels,mpo)


#println("#############")
#println("QN version")
#println("#############")

QNenergy = dmrg(qpsi,qmpo,m=45,sweeps=20,cutoff=1E-9,method="twosite",silent=true)

testval = abs(QNenergy-(-25.380567747816)) < 1E-4
fulltest &= testfct(testval,"Hubbard model")

