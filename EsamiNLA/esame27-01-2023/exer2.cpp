#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
using namespace Eigen;
using namespace std;
int main(int argc, char** argv)
{
// Create matrix
    int n = 200;
    SparseMatrix<double> A(n,n);

    for(int i = 0; i < n ; i++){
        A.coeffRef(i,i) = ((i+1)*2);
        if(i>0){A.coeffRef(i,i-1) = -i;}
        if(i<n-1){A.coeffRef(i,i+1) = -(i+1);}
        A.coeffRef(i,n-i-1) += 1.0;
    }
    cout<<"The norm of matrix is: "<<A.norm()<<endl;

    // Compute Eigenvalues of original matrix
  SelfAdjointEigenSolver<MatrixXd> eigensolver(A);
  if (eigensolver.info() != Eigen::Success) abort();
  std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
  VectorXd v = eigensolver.eigenvalues();
  double lmbd_min = v(0);
  double lmbd_min_2 = v(1);

  cout<<"The two smallest eigenvalues are: "<<lmbd_min<<" and "<< lmbd_min_2 <<endl;

  saveMarket(A,"Aexer2.mtx");  
  /*
  # mpirun -n 4 ./eigen1 Aexer2.mtx eigvec.txt hist.txt -e ii -i cg -p ssor -etol 1.0e-12

number of processes = 4
matrix size = 200 x 200 (796 nonzero entries)

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
Inverse: eigenvalue           = 9.536967e-02
Inverse: number of iterations = 23
Inverse: elapsed time         = 1.244853e-02 sec.
Inverse:   preconditioner     = 1.346375e-03 sec.
Inverse:     matrix creation  = 7.980000e-07 sec.
Inverse:   linear solver      = 1.072489e-02 sec.
Inverse: relative residual    = 3.351802e-13

root@0bc56b79e068 test # 

# mpirun -n 4 ./eigen1 Aexer2.mtx eigvec.txt hist.txt -e ii -i cg -p ssor -etol 1.0e-12 -shift 0.3

number of processes = 4
matrix size = 200 x 200 (796 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Inverse
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-12 * ||lx||_2
matrix storage format : CSR
shift                 : 3.000000e-01
linear solver         : CG
preconditioner        : SSOR
eigensolver status    : normal end

Inverse: mode number          = 0
Inverse: eigenvalue           = 3.600850e-01
Inverse: number of iterations = 24
Inverse: elapsed time         = 1.732054e-02 sec.
Inverse:   preconditioner     = 1.590304e-03 sec.
Inverse:     matrix creation  = 9.540000e-07 sec.
Inverse:   linear solver      = 1.528983e-02 sec.
Inverse: relative residual    = 2.370654e-13

  */

        return 0;
}