#include <Eigen/SparseCore>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <math.h>


using namespace Eigen; //shorcut for eigen declaration
using SpMat=SparseMatrix<double,RowMajor>;
using namespace std;

int main(){

    int n = 99;
    SpMat A(n,n); //is sym and tridiagonal

    for(int i = 0;i<n;i++){
        A.coeffRef(i,i) = abs((n+1)/2.0 - (i+1)) + 1.0;
        if(i<n-1){A.coeffRef(i,i+1) = 0.5;}
        if(i>0){A.coeffRef(i,i-1) = 0.5;}
    }

    cout<<"matrix coeff a1 is: "<<A.coeffRef(0,0)<<endl;
    cout<<"matrix coeff an is: "<<A.coeffRef(n-1,n-1)<<endl;

    //I can use selfadjointeigen solver since A is sym!

     Eigen:: SelfAdjointEigenSolver<MatrixXd> saeigensolver(A);
     VectorXd lmbdas = saeigensolver.eigenvalues();
     cout<<"Biggest eig is: "<<lmbdas(lmbdas.size()-1)<<endl;
     cout<<"Smallest eig is: "<<lmbdas(0)<<endl;

     Eigen:: saveMarket(A,"exer2.mtx");

     /*
     # mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest1.c -o eigen1
mpirun -n 4 ./eigen1 exer2.mtx eigvec.txt hist.txt -e pi -etol 1.e-10

number of processes = 4
matrix size = 99 x 99 (295 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Power
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-10 * ||lx||_2
matrix storage format : CSR
shift                 : 0.000000e+00
eigensolver status    : normal end

Power: mode number          = 0
Power: eigenvalue           = 5.022544e+01
Power: number of iterations = 782
Power: elapsed time         = 4.851005e-03 sec.
Power:   preconditioner     = 0.000000e+00 sec.
Power:     matrix creation  = 0.000000e+00 sec.
Power:   linear solver      = 0.000000e+00 sec.
Power: relative residual    = 9.829642e-11
     
     */


//I can use Inverse PM with shift 
/*
# mpirun -n 4 ./eigen1 exer2.mtx eigvec.txt hist.txt -e ii -shift 50 -etol 1.e-10

number of processes = 4
matrix size = 99 x 99 (295 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Inverse
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-10 * ||lx||_2
matrix storage format : CSR
shift                 : 5.000000e+01
linear solver         : BiCG
preconditioner        : none
eigensolver status    : normal end

Inverse: mode number          = 0
Inverse: eigenvalue           = 5.022544e+01
Inverse: number of iterations = 17
Inverse: elapsed time         = 1.775489e-03 sec.
Inverse:   preconditioner     = 1.942300e-05 sec.
Inverse:     matrix creation  = 4.870000e-07 sec.
Inverse:   linear solver      = 1.580475e-03 sec.
Inverse: relative residual    = 5.407747e-11

*/

//Smallest with LIS

/*
# mpirun -n 4 ./eigen1 exer2.mtx eigvec.txt hist.txt -e ii -i cg -p ssor         

number of processes = 4
matrix size = 99 x 99 (295 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Inverse
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-12 * ||lx||_2
matrix storage format : CSR
shift                 : 0.000000e+00
linear solver         : CG
preconditioner        : SSOR
eigensolver status    : normal end

Inverse: mode number          = 0
Inverse: eigenvalue           = 6.099899e-01
Inverse: number of iterations = 24
Inverse: elapsed time         = 1.188502e-03 sec.
Inverse:   preconditioner     = 4.795200e-05 sec.
Inverse:     matrix creation  = 6.900000e-07 sec.
Inverse:   linear solver      = 8.423290e-04 sec.
Inverse: relative residual    = 8.275944e-13
*/
    return 0;

}