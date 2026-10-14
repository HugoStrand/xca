template <typename Input>
Tensor scalar_reference_value(std::vector<Input> const &G, Backbone &backbone, SelfEnergyPath const &path) {
  // Copy to owned tensors because the general edge API takes owning inputs.
  // This slow diagnostic route runs only for cutoff-sensitive completed lanes.
  std::vector<Tensor> input;
  input.reserve(G.size());
  for (auto const &block : G) {
    input.emplace_back(block.shape());
    ops->copy(block, input.back());
  }
  Tensor value(input[path.propagator_blocks[0]].shape());
  ops->copy(input[path.propagator_blocks[0]], value);
  for (int v = 1; v < path.special_vertex; ++v) {
    vertex(value, backbone, v, path);
    edge(value, input, backbone, v, path);
  }
  special(value, backbone, path);
  for (int v = path.special_vertex + 1; v < 2 * backbone.m; ++v) {
    edge(value, input, backbone, v - 1, path);
    vertex(value, backbone, v, path);
  }
  ops->scale(kernel_prefactor(backbone), value);
  int sign = (backbone.m % 2 == 0 ? -1 : 1) * (backbone.get_fb(0) == 0 ? -1 : 1) * backbone.prefactor_sign;
  ops->scale(sign, value);
  ops->finish();
  return value;
}
