#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <string>
#include <Eigen/IterativeLinearSolvers>

/*moduli per QR o LSCG
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
*/


using namespace std;
using namespace Eigen;
using SpMat=Eigen::SparseMatrix<double,RowMajor>;


int main(int argc, char** argv){

    int n = 100;
    SpMat A(n,n);

    for(int i = 0; i<n; i++){
        A.coeffRef(i,i) = -8.;
        if(i>0){ A.coeffRef(i,i-1) = 2.; }
        if(i<n-1){ A.coeffRef(i,i+1)= 4.; }
        if(i <n-2){A.coeffRef(i,i+2) = 1.; }
    }

    cout<<"The matrix norm is: "<<A.norm()<<endl;
    
    //A not sym
    MatrixXd A_dense;
  A_dense = MatrixXd(A);
  EigenSolver<MatrixXd> eigensolver(A_dense);
  if (eigensolver.info() != Eigen::Success) abort();
  //cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << endl;
  cout << "The biggest eigenvalues of A is:\n" << eigensolver.eigenvalues().real().maxCoeff() << endl;
  cout << "The smallest eigenvalues of A is:\n" << eigensolver.eigenvalues().real().minCoeff() << endl;

  //NEGATIVES

  saveMarket(A,"Aex2.mtx");


//PER IL PIU GRANDE USO INVERSE PM
/*
# mpirun -n 4 ./eigen1 Aex2.mtx eigvec.txt hist.txt -e ii  -etol 1.0e-7 -i gmres -p ssor -emaxiter 20000

number of processes = 4
matrix size = 100 x 100 (396 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Inverse
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-07 * ||lx||_2
matrix storage format : CSR
shift                 : 0.000000e+00
linear solver         : GMRES
preconditioner        : SSOR
eigensolver status    : normal end

Inverse: mode number          = 0
Inverse: eigenvalue           = -1.913297e+00
Inverse: number of iterations = 1348
Inverse: elapsed time         = 1.903253e-01 sec.
Inverse:   preconditioner     = 6.419585e-03 sec.
Inverse:     matrix creation  = 4.164100e-05 sec.
Inverse:   linear solver      = 1.438039e-01 sec.
Inverse: relative residual    = 9.961093e-08

root@0bc56b79e068 test # 
*/

//PER CALCOLARE IL PIU PICCOLO USO PM (cerca il piu grande in modulo)
/*
 mpirun -n 4 ./eigen1 Aex2.mtx eigvec.txt hist.txt -e pi -etol 1.0e-7 -emaxiter 30000

number of processes = 4
matrix size = 100 x 100 (396 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Power
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-07 * ||lx||_2
matrix storage format : CSR
shift                 : 0.000000e+00
eigensolver status    : LIS_MAXITER(code=4)

Power: mode number          = 0
Power: eigenvalue           = -1.299901e+01
Power: number of iterations = 30000
Power: elapsed time         = 1.001006e-01 sec.
Power:   preconditioner     = 0.000000e+00 sec.
Power:     matrix creation  = 0.000000e+00 sec.
Power:   linear solver      = 0.000000e+00 sec.
Power: relative residual    = 9.486912e-07

*/


    
    return 0;
}