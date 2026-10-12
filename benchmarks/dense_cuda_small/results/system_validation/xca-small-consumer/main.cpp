#include <triqs_xca/dense/cuda_small/diagram_evaluator.hpp>
#include <triqs/atom_diag.hpp>
#include <triqs/operators.hpp>
#include <iostream>
int main() {
#ifdef XCA_HAVE_CUDA_SMALL
  using namespace triqs::operators;
  triqs::atom_diag::fundamental_operator_set fs;
  fs.insert("f",0); fs.insert("f",1);
  many_body_operator_complex H=.2*n("f",0)+.3*n("f",1);
  triqs::atom_diag::atom_diag<true> ad(H,fs,std::vector<many_body_operator_complex>{});
  triqs::mesh::dlr_imtime mesh(2.,triqs::mesh::Fermion,5.,1e-12,false);
  nda::vector<double> poles{-1.5,.6};
  auto coefficients=nda::zeros<nda::dcomplex>(2,2,2);
  triqs_xca::dense::cuda_small::DiagramEvaluator d(poles,coefficients,mesh,ad);
  auto data=nda::zeros<nda::dcomplex>(mesh.size(),4,4);
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> g(std::vector{triqs::gfs::gf<triqs::mesh::dlr_imtime>(mesh,data)});
  nda::array<int,2> topology{{0,1}};
  auto result=d.compute_self_energy_range(g,topology,0,1);
  if(d.dimension()!=4 || nda::max_element(nda::abs(result[0].data()))!=0.) return 1;
  std::cout << "installed CUDA-small consumer passed\n";
#else
  return 2;
#endif
}
