nda::cumatrix<Complex> resident_special;
void prepare_resident_weights() {
  if (resident_special.size() != 0) return;
  nda::matrix<Complex> host(2, hyb->values.size());
  for (long i = 0; i < hyb->values.size(); ++i) {
    host(0, i) = hyb->values.data()[i];
    host(1, i) = hyb->values_reflect.data()[i];
  }
  resident_special = nda::cumatrix<Complex>(host);
  ops->wait_for_default_uploads();
  ops->finish();
}
scalar_resident_experiment::Weight resident_pole_weight(int sign, int pole) const {
  if (sign == 0) return {};
  auto const &values = sign > 0 ? positive[pole] : negative[pole];
  return {reinterpret_cast<std::uintptr_t>(values.data()), 1, 1};
}
scalar_resident_experiment::Weight resident_special_weight(int mu, int kap, bool forward, int sign) const {
  long fields = model.num_fields();
  long offset = forward ? mu * fields + kap : hyb->values.size() + kap * fields + mu;
  return {reinterpret_cast<std::uintptr_t>(resident_special.data() + offset), fields * fields, sign};
}
