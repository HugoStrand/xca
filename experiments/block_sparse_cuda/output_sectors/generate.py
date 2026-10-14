"""Generate an isolated complete-output sector-streaming evaluator."""
from pathlib import Path
import sys

root, destination = map(Path, sys.argv[1:])
source_dir = root / 'c++/triqs_xca/block_sparse/cuda'
header = (source_dir / 'diagram_evaluator.hpp').read_text()
source = (source_dir / 'diagram_evaluator.cpp').read_text()


def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError(f'Streaming seam changed: {old[:70]}')
    source = source.replace(old, new)


replace_once('    void sample_memory() {',
             '    std::size_t host_output_budget = 512ULL * 1024 * 1024;\n    void sample_memory() {')
replace_once('auto generic = bound();', '''auto generic = checked_add(
        bound() - function_bytes(r, model.dimensions()),
        inventory_memory(r, model.dimensions(), 0).maximum_rank_buffer);''')
replace_once('  std::size_t DiagramEvaluator::self_energy_storage_bound() const { return impl_->bound(); }', '''
  void DiagramEvaluator::set_host_output_budget(std::size_t bytes) {
    impl_->ops->require_current();
    impl_->host_output_budget = bytes;
  }
  std::size_t DiagramEvaluator::self_energy_storage_bound() const { return impl_->self_energy_bound(false); }''')
replace_once('    if (first != last) x.prepare_self_energy(topology); // Before input/output upload or path arithmetic.', '''
    if (function_bytes(x.r, x.model.dimensions()) > x.host_output_budget)
      throw std::length_error("Streamed full output exceeds host payload budget");
    if (first != last) x.prepare_self_energy(topology); // Before input/output upload or path arithmetic.''')
replace_once('    std::vector<Tensor> input(G.size()), output(G.size());', '''
    std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> result;
    for (long b = 0; b < G.size(); ++b) {
      auto d = x.model.dimensions()[b];
      result.emplace_back(x.mesh, nda::zeros<Complex>(x.r, d, d));
    }
    std::vector<Tensor> input(G.size()), output(G.size());
    long active_output = -1;
    auto flush = [&] {
      if (active_output >= 0) {
        x.ops->finish();
        result[active_output].data() = nda::to_host(output[active_output]);
        output[active_output] = Tensor();
        active_output = -1;
      }
    };''')
replace_once('        output[b] = Tensor(x.r, d, d);\n        x.ops->zero(output[b]);', '')
replace_once('        auto d = x.model.dimensions()[b];\n        if (live[b]) input[b] = Tensor(G[b].data());',
             '        if (live[b]) input[b] = Tensor(G[b].data());')
replace_once('[&](Backbone &backbone, SelfEnergyPath const &path) { x.evaluate(input, output, backbone, path); }', '''
        [&](Backbone &backbone, SelfEnergyPath const &path) {
          if (active_output != path.output_block) {
            flush();
            active_output = path.output_block;
            output[active_output] = Tensor(result[active_output].data());
            x.ops->wait_for_default_uploads();
            x.sample_memory();
          }
          x.evaluate(input, output, backbone, path);
        }''')
replace_once('''    std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> result;
    for (long b = 0; b < G.size(); ++b) {
      auto d      = x.model.dimensions()[b];
      auto values = first == last ? nda::zeros<Complex>(x.r, d, d) : nda::to_host(output[b]);
      result.emplace_back(x.mesh, values);
    }
    return {result};''', '''    flush();
    return {std::move(result)};''')
header = header.replace('    public:', '    public:\n    void set_host_output_budget(std::size_t bytes);')
header = header.replace('DiagramEvaluator', 'OutputSectorEvaluator')
source = source.replace('DiagramEvaluator', 'OutputSectorEvaluator')
source = source.replace('"diagram_evaluator.hpp"', '"output_evaluator.hpp"')
destination.mkdir(parents=True, exist_ok=True)
(destination / 'output_evaluator.hpp').write_text(header)
(destination / 'output_evaluator.cpp').write_text(source)
# Separate ownership control: original resident algorithm, explicit return move.
move_header = (source_dir / 'diagram_evaluator.hpp').read_text().replace('DiagramEvaluator', 'MoveOutputEvaluator')
move_source = (source_dir / 'diagram_evaluator.cpp').read_text()
if move_source.count('    return {result};') != 1:
    raise RuntimeError('Resident return-ownership seam changed')
move_source = move_source.replace('    return {result};', '    return {std::move(result)};')
move_source = move_source.replace('DiagramEvaluator', 'MoveOutputEvaluator').replace('"diagram_evaluator.hpp"', '"move_evaluator.hpp"')
(destination / 'move_evaluator.hpp').write_text(move_header)
(destination / 'move_evaluator.cpp').write_text(move_source)
