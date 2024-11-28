
println("#            +-----------------------+")
println("#>-----------|  classicalTN_test.jl  |-----------<")
println("#            +-----------------------+")
fulltest = true


#using QuadGK

temperatures = [2.069185314213022,
2.169185314213022,
2.269185314213022, #Tc
2.369185314213022,
2.4691853142130222]

free_energy_density =  [-2.0630846809851238,
-2.0834148107765222,
-2.109651144607396,
-2.144614269103391,
-2.185113784070822]
#from previous run of Onsager solution

out = exact_square_2D.(temperatures)

println(out)

fulltest &= testfct(isapprox(out,free_energy_density),"Checking Onsager's exact solution implementation")

#ntemps = 5
#temptest = Array{Bool,1}(undef,ntemps)

function testTNalg(Temp::Float64,exval::Float64,fct::Function;tol::Float64=1E-4,Zfct::Function=createZ)

    Z = Zfct(Temp)

    Zarray,vals = fct(Z, Temp)

#    println(vals)

    fdens = vals.f[end]

    checkval = abs(exval - fdens) < tol
    println(fct," ",exval," ",fdens," ",abs(exval - fdens))

    return testfct(checkval,"T=$(Temp)")
end

fct_tests = [

#TRG_sparse



#Matt's working algorithms imported
#LTRGxx_square


  ATRG_square,
  BTRG_square,


  dTRG,


GILT_square,

  HOSRG_square,

  HOTRG1_square,
  HOTRG2_square,
  #HOTRG3_square,
  HOTRG4_square,

  LTRG_square,
  LTRGxx_square,


  PTTRG1_square,
  PTTRG2_square,
  PTTRG3_square,


  SRG_square,

  TTRG,


  TRG_square

]

function testalgs(temperatures,free_energy_density,fct_tests::Array{W,1};x::String="square",Zfct::Function=createZ,testinnerfct::Function=testTNalg,tol::Number=1E-3) where W <: Function
  testvals = Array{Bool,2}(undef,length(temperatures),length(fct_tests))


  println()
  for w = 1:length(fct_tests)
    printstyled("Function: ",color=:blue)
    println(fct_tests[w])
    for x = 1:length(temperatures)
      testvals[x,w] = testinnerfct(temperatures[x],free_energy_density[x],fct_tests[w],Zfct=Zfct,tol=tol)
    end
    println()
  end
  boolval = testfct(sum(testvals) == length(testvals),"All classical tensor network algorithms ("*x*")")
  return boolval
end

fulltest &= testalgs(temperatures,free_energy_density,fct_tests)

println()



function testTNalg_CTM(Temp::Float64,exval::Float64,fct::Function;tol::Float64=1E-4,Zfct::Function=createZ)

  A, C1, C2, C3, C4, T1, T2, T3, T4 = Zfct(Temp)

  Z,vals = fct(A, C1, C2, C3, C4, T1, T2, T3, T4, Temp)

#    println(vals)

  fdens = vals.f[end]

  checkval = abs(exval - fdens) < tol
  println(fct," ",exval," ",fdens," ",abs(exval - fdens))

  return testfct(checkval,"T=$(Temp)")
end


#modification needed (maybe separate loop)
#CTMRG requires many more tests, hence a new test
fct_tests = [
  CTMRG1,
  CTMRG2,
  HOTRG_CTM
]

fulltest &= testalgs(temperatures,free_energy_density,fct_tests,Zfct=createZ_Corn,testinnerfct=testTNalg_CTM)



println()


fct_tests = [
  TRG_triangle
]

temperatures_tri = [4/log(3)]
free_energy_tri = [-3.2]

fulltest &= testalgs(temperatures_tri,free_energy_tri,fct_tests,Zfct=createZ_Tri,x="triangular",tol=1.)


println()




fct_tests = [
  HOTRG_Hex,
  LTRG_Hex,
  LTRGxx_Hex,
  TRG_Hex,
  TTRG_Hex
]

temperatures_Hex = [2/log(2+sqrt(3))]
free_energy_Hex = [-3.1]
fulltest &= testalgs(temperatures_Hex,free_energy_Hex,fct_tests,Zfct=createZ_Hex,x="cubic",tol=1.)


println()





fct_tests = [

#none of these work but Matt has working versions

ATRG_cubic,
HOTRG_cubic,
TTRG_cubic

]

temperatures_cube = [4.31,
4.41,
4.51, #Tc
4.61,
4.71]

free_energy_cube = [-3.5 for i = 1:length(temperatures_cube)]

fulltest &= testalgs(temperatures_cube,free_energy_cube,fct_tests,Zfct=createZ_Cube,x="cubic",tol=0.5)



#Hex done: LTRG, HOTRG, GILT, TRG,
#Hex not done: BTRG, SRG, LTRG++


#Triangular (working): TRG

#Kagome (working): TRG


#240809: Matt will check dTRG, SRG

#Not even started: gTRG, TNR +


#ignored: TERG (we did it but...)
#ignore the Monte Carlo ones

#=
println()

function testalgs(temperatures,fct_tests::Array{W,1};x::String="square",Zfct::Function=createZ_Corn,tol::Float64=1E-4,fct::Function=CTMRG1) where W <: Function
  testvals = Array{Bool,2}(undef,length(temperatures),length(fct_tests))


  step = 0.0065
  Tc = (2/log(1+sqrt(2)))#- 0.26 + slurm_val*step
  nstep = 1000
  m = 20

#  A, C1, C2, C3, C4, T1, T2, T3, T4 = createZ_Corn(Tc)
#  Zut, truncs = CTMRG1(temperatures[x],A, C1, C2, C3, C4, T1, T2, T3, T4)


  println()
  for w = 1:length(fct_tests)
    printstyled("Function: ",color=:blue)
    println(fct_tests[w])
    for x = 1:length(temperatures)

      Temp = temperatures[x]
      
      A, C1, C2, C3, C4, T1, T2, T3, T4 = Zfct(Temp)
      Zut, truncs,outval = fct(Temp, A, C1, C2, C3, C4, T1, T2, T3, T4)

      fdens = outval[end]

#      fdens = vals.f[end]


      exval = free_energy_density[x]



  
  
      checkval = abs(exval - fdens) < tol
      println(fct," ",exval," ",fdens," ",abs(exval - fdens))



      testvals[x,w] = testfct(checkval,"T=$(Temp)")
    end
    println()
  end
  boolval = testfct(sum(testvals) == length(testvals),"All classical tensor network algorithms ("*x*")")
  return boolval
end

fct_tests = [
  CTMRG1
]

#fulltest &= testalgs(temperatures,fct_tests,Zfct=createZ_Corn)


println()
=#