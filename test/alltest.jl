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

const tests = [
"MPtask_test.jl",
"MPmeas_test.jl",
"MPlarge_test.jl",
"MPOauto_test.jl",
"model_test.jl",
######
######
######
"dmrg_test.jl"
]

function checkall(fulltestrecord::Array{Bool,1},i::Integer,fulltest::Bool)
  fulltestrecord[i] = fulltest
  print("All tests passed? ")
  if fulltest
    printstyled(fulltest,color=:green)
  else
    printstyled(fulltest,color=:red)
  end
  println()
end

"""
  testlib([,tests=,path=libdir*"/test/"])

Tests all functions in the files enumerated in `tests`. Default is to check all test functionality in the library. Used in nightly builds. See format in `/tests/` folder

See also: [`libdir`](@ref)
"""
function libtest(;tests::Array{String,1}=tests,dir::String=libdir,path::String=dir*"/test/")

  fulltestrecord = Array{Bool,1}(undef,length(tests))

  for i = 1:length(tests)
    @time include(path*tests[i])
    checkall(fulltestrecord,i,fulltest)
  end

  println()

  for i = 1:length(tests)
    if fulltestrecord[i]
      printstyled(fulltestrecord[i],color=:green)
    else
      printstyled(fulltestrecord[i],color=:red)
    end
    println("    ",i,"   ",tests[i])
  end

  println()

  if sum(fulltestrecord) == length(tests)
    println("All passed. Good work. We happy :^)")
  else
    println("These passed:")
    printstyled(tests[fulltestrecord],color=:green)
    println()
    println()
    println("These did not pass:")
    notfulltestrecord = [!fulltestrecord[w] for w = 1:length(tests)]
    printstyled(tests[notfulltestrecord],color=:red)
  end
end
export libtest

#libtest()
