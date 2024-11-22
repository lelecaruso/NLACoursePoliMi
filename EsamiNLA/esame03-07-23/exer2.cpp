#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Dense>

using namespace Eigen;
using SpMat=Eigen::SparseMatrix<double,RowMajor>;

using std::endl;
using std::cout;

int main(){

    int n = 81;
    SpMat A(n,n);

    for(int i = 0; i<n; i++){
        A.coeffRef(i,i) = -3.0;
        if(i>0){
            A.coeffRef(i,i-1) = 1.0;
        }
        if(i<n-1){
            A.coeffRef(i,i+1) = 2.0;
        }
    }

    VectorXd v = VectorXd:: Ones(n);

    //vTAv A-norm of V
   std::cout << "Norm A of v: " << v.dot(A*v) << std::endl;

   // Since A has a negative norm, it's not PD 
   MatrixXd AD;
  AD = MatrixXd(A);
  EigenSolver<MatrixXd> eigensolver(AD);
  if (eigensolver.info() != Eigen::Success) abort();
  std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
  /*
    lamda_min = −5.82635
    lambda_max = −0.173648
  */

  saveMarket(A,"Aex2.mtx");

  /*
  //Largest In Module
  root@0bc56b79e068 test # mpirun -n 4 ./eigen1 Aex2.mtx eigvec.txt hist.txt -e pi -etol 1.0e-8 -emaxiter 7500

number of processes = 4
matrix size = 81 x 81 (241 nonzero entries)

eigensolver           : Power
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2

Power: eigenvalue           = -5.826360e+00
Power: number of iterations = 7500

Power: relative residual    = 3.047370e-08
  
  //Smallest in module 

  # mpirun -n 4 ./eigen1 Aex2.mtx eigvec.txt hist.txt -e ii -etol 1.0e-8 -emaxiter 5000 -i gmres

number of processes = 4
matrix size = 81 x 81 (241 nonzero entries)

eigensolver           : Inverse
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2

linear solver         : GMRES

Inverse: mode number          = 0
Inverse: eigenvalue           = -1.736483e-01
Inverse: number of iterations = 358
Inverse: elapsed time         = 1.785588e-01 sec.
Inverse:   preconditioner     = 4.510590e-04 sec.
Inverse:     matrix creation  = 1.455200e-05 sec.
Inverse:   linear solver      = 1.633726e-01 sec.
Inverse: relative residual    = 9.969008e-09
  */

 // Solve Ax = emin (min eigenvector)
 /*
 # mpirun -n 4 ./test1 Aex2.mtx eigvec_min.mtx sol.txt hist.txt -i jacobi -tol 1.0e-10

number of processes = 4
matrix size = 81 x 81 (241 nonzero entries)

initial vector x      : all components set to 0
precision             : double
linear solver         : Jacobi
preconditioner        : none
convergence condition : ||b-Ax||_2 <= 1.0e-10 * ||b-Ax_0||_2


Jacobi: number of iterations = 388

Jacobi: relative residual    = 9.519440e-11

The relation is e_min = sol * lambda.
 */
    return 0;
}