namespace LinearAlgebra
{
template <class Matrix, class Vector, class Preconditioner>
int Jacobi(const Matrix &A, Vector &x, const Vector &b, const Preconditioner &M,
   int &max_iter, typename Vector::Scalar &tol)
{
  using Real = typename Matrix::Scalar;
  Real   resid;
  Real   normb = b.norm();
  Vector r = b - A * x;

  if(normb == 0.0) normb = 1;
  if((resid = r.norm() / normb) <= tol)
    {
      tol = resid;
      max_iter = 0;
      return 0;
    }

  for(int i = 1; i <= max_iter; i++)
    {
      x = M.solve(r) + x; //M is D, then M.solve(r) it's D^-1r
      r = b - A * x;
      if((resid = r.norm() / normb) <= tol)
        {
          tol = resid;
          max_iter = i;
          return 0;
        }
    }

  tol = resid;
  return 1;
}
} // namespace LinearAlgebra