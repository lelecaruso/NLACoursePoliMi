#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/IterativeLinearSolvers>


using std::endl;
using std::cout;

// Some useful alias
using namespace Eigen;
using SpMat=Eigen::SparseMatrix<double,RowMajor>;


int main(int argc, char** argv)
{
    SpMat A;
    loadMarket(A,"Aex2.mtx");
    SpMat AT = A.transpose();

    SpMat ATA = AT*A;
    cout<<"The norm of ATA is: "<<ATA.norm()<<endl;

    VectorXd xe = VectorXd:: Ones(A.cols());
    VectorXd b = A*xe;
    cout<<"The norm of rhs b is: "<<b.norm()<<endl;

    VectorXd x_sqr(A.cols());

    Eigen::SparseQR<Eigen::SparseMatrix<double,RowMajor>, COLAMDOrdering<int>> solver;   
    solver.compute(A);
    if(solver.info()!=Eigen::Success) {                     // sanity check
      std::cout << "cannot factorize the matrix" << std::endl; 
      return 0;
    }
    x_sqr = solver.solve(b);   
    
    cout<<"The eucl. norm of the error is: "<<(xe-x_sqr).norm()<<endl;    

     // solve with Eigen LeastSquareConjugateGradient solver
  LeastSquaresConjugateGradient<SparseMatrix<double> > lscg;
  double tol = 1.0e-10;
  VectorXd x_lscg(A.cols());
  lscg.compute(A);
  lscg.setTolerance(tol);
  x_lscg = lscg.solve(b);
  std::cout << "Solution with Eigen LSCG:" << std::endl;
  std::cout << "#iterations:     " << lscg.iterations() << std::endl;
  std::cout << "relative residual (TOL): " << lscg.error()      << std::endl; 
  std::cout << "effective error (xe-xlscg).norm : " << (x_lscg-xe).norm()      << std::endl; 

  saveMarket(ATA,"AtA.mtx");
  //export rhs to solve ATA x =ATb
  VectorXd Atb = AT*b; //AT = A.transpose()
     FILE* out = fopen("Atrhs.mtx","w");
    fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out,"%d\n", Atb.size());
    for (int i=0; i<Atb.size(); i++) {
        fprintf(out,"%d %f\n", i ,Atb(i));
    }
    fclose(out);

/*
    # mpirun -n 4 ./test1 AtA.mtx Atrhs.mtx sol.txt hist.txt -i cg -tol 1.0e-10

number of processes = 4
matrix size = 400 x 400 (19590 nonzero entries)

linear solver         : CG


CG: number of iterations = 119

CG: relative residual    = 7.338696e-11

*/



    return 0;   
}