#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker and Jaimie Greasley (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

"""
    ox,oy,oz,Id,op,om,Rx,Ry,Rz,H,O = qubits([,d=2,angle=pi/4])

Creates a set of Pauli operators for a certain number of states `d` and angle for a rotation gate `angle`

  #Outputs:
  + `ox`: Pauli-x operator
  + `oy`: Pauli-y operator
  + `oz`: Pauli-z operator
  + `Id`: identity matrix
  + `op`: raising operator
  + `om`: lowering operator
  + `Rx`: rotation of x-axis operator
  + `Ry`: rotation of y-axis operator
  + `Rz`: rotation of z-axis operator
  + `H`: Hadamard gate
  + `O`: zero matrix
"""
function qubitOps(;d::intType=2,angle::Number=pi/4)
#  if d != 2
#    println("WARNING: Hadamard gate not defined for more than 2 states")
#  end

  s = (d-1)/2
  O = zeros(Float64,d,d) #zero matrix
  Id = Array(tens(eye(d))) #identity matrix
  oz = copy(O) # z operator
  op = copy(O) # raising operator
  for (q,m) in enumerate(s:-1:-s) #counts from m to -m (all states)
    oz[q,q] = 2*m
    if m+1 <= s
      op[q-1,q] = sqrt(s*(s+1)-m*(m+1)) #Clebsch-Gordon coefficients
    end
  end
  om = Array(op') # lowering operator
  ox = (op+om) #x matrix
  oy = (om-op)*im #y matrix

  H = [1 1;1 -1]/sqrt(2)
  Rx = exp(-im*angle/2*ox)
  Ry = exp(-im*angle/2*oy)
  Rz = exp(-im*angle/2*oz)
  return ox,oy,oz,Id,op,om,Rx,Ry,Rz,H,O
end

"""
    ox,oy,oz,Id,op,om,Rx,Ry,Rz,H,O = qubits(d[,angle=pi/4])

Creates a set of Pauli operators for a certain number of states `d` and angle for a rotation gate `angle`

  #Outputs:
  + `ox`: Pauli-x operator
  + `oy`: Pauli-y operator
  + `oz`: Pauli-z operator
  + `Id`: identity matrix
  + `op`: raising operator
  + `om`: lowering operator
  + `Rx`: rotation of x-axis operator
  + `Ry`: rotation of y-axis operator
  + `Rz`: rotation of z-axis operator
  + `H`: Hadamard gate
  + `O`: zero matrix
"""
function qubitOps(d::intType;angle::W=pi/4) where W <: Number
  return qubitOps(d=d,angle=angle)
end
export qubitOps

