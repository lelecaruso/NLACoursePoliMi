#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/IterativeLinearSolvers>

#include"jacobi.hpp"
#include "cg.hpp"


using std::endl;
using std::cout;

// Some useful alias
using namespace Eigen;
using SpMat=Eigen::SparseMatrix<double,RowMajor>;


int main(int argc, char** argv){   

    using namespace LinearAlgebra;

    SpMat A;
    loadMarket(A,"Aex1.mtx");

    cout<<"The matrix size (row x col) is: "<< A.rows() << " X " << A.cols()<< endl;
    double tol_sym = 1.0e-14;

    SpMat AT = A.transpose();
    SpMat B = A-AT;

    if(B.norm()<= tol_sym){
        cout<<"Matrix Is sym"<<endl;
    }else{
        cout<<"Matrix IS NOT sym"<<endl;
    }

    VectorXd xe = VectorXd:: Ones(A.cols());
    VectorXd b = A*xe;
    VectorXd x(A.cols());

    cout<<"The norm of vector b is: "<<b.norm()<<endl;


  // First with Eigen Choleski direct solver
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;  // LDLT factorization
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl; // decomposition failed
      return 0;
  }

  x = solver.solve(b);                                         // solving
  std::cout << "Solution with Eigen Choleski:" << std::endl;
  std::cout << "effective error: "<<(x-xe).norm()<< std::endl;

  /*
  Jacobi iterative method implemented in the
  jacobi.hpp template. Set a tolerance sufficient to achieve an error with magnitude similar
  to the one obtained with the Cholesky method. Report on the sheet the iteration counts
  and the norm of the absolute error at the last iteration.
  */

  DiagonalPreconditioner<double>D(A); //for jacobi iterations
  int result;
  double tol = 1.0e-14;
  int maxit = 1000;
  x = 0*x; // reset of my sol

  result = Jacobi(A,x,b,D,maxit,tol);
  cout<<"Jacobi Flag: "<<result<<endl;
  cout<<"Iterations: "<<maxit<<endl;
  cout<<"Error: "<<(x-xe).norm()<<endl;

  x = 0*x;
  tol = 1.0e-14;
  maxit = int(A.cols());

  int result_cg;
  IncompleteLUT<double> P_ilu(A);
  result_cg = CG(A,x,b,P_ilu,maxit,tol);
  cout<<"CG Flag: "<<result_cg<<endl;
  cout<<"Iterations: "<<maxit<<endl;
  cout<<"Error: "<<(x-xe).norm()<<endl;

    return 0;
}