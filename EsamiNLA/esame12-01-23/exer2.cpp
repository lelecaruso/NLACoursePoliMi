#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;
using SpMat=Eigen::SparseMatrix<double,RowMajor>;
using namespace std;

int main(int argc, char** argv)
{
  SpMat A;
  loadMarket(A,"A_rect.mtx");
  cout<<"Le dimensioni di A sono: "<<A.rows()<<" X "<< A.cols()<<endl;

  SpMat AT = A.transpose();
  cout<<"The norm of ATA is : "<< SpMat(AT*A).norm()<< endl;
  
  VectorXd xe = VectorXd:: Ones(A.cols());
  VectorXd b = A*xe;
  cout<<"The norm of rhs b is: "<<b.norm()<<endl;

VectorXd xqr(A.cols()); //imporante dare a x le dimensioni delle colonne di A
  // solve with Eigen QR factorization
  Eigen::SparseQR<Eigen::SparseMatrix<double>, COLAMDOrdering<int>> solver;
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {
      std::cout << "cannot factorize the matrix" << std::endl;
      return -1;
  }
   xqr = solver.solve(b);
  std::cout << "Solution with Eigen QR:" << std::endl;
  std::cout << "effective error: "<<(xqr-xe).norm()<< std::endl;


  VectorXd xls(A.cols());
  double tol = 1.0e-10;
   // solve with Eigen LeastSquareConjugateGradient solver
  LeastSquaresConjugateGradient<SparseMatrix<double,RowMajor> > lscg;
  int maxit = 5000;
  lscg.compute(A);
  lscg.setTolerance(tol);
  lscg.setMaxIterations(maxit);
  xls = lscg.solve(b);
  std::cout << "Solution with Eigen LSCG:" << std::endl;
  std::cout << "#iterations:     " << lscg.iterations() << std::endl;
  std::cout << "#tollerance(residuo normalizato su b):     " << lscg.error() << std::endl;
  std::cout << "effective error: " << (xls-xe).norm()      << std::endl;

  //THE QR method is much more efficient. This probably means that A is ill-conditioned so K(ATA) = K(A)^2 is really high and it's hard to solve ATAx = AtB sys.

  
  return 0;
  }
