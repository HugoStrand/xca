#include <algorithm>
#include <stdexcept>
#include <string>

#include "triqs_xca/backbone.hpp"
#include "triqs_xca/diagram_sign.hpp"

namespace triqs_xca::backbone {

  using nda::range;

  namespace {

    // base^exp, throwing instead of overflowing a long
    long checked_pow(long base, long exp) {
      long result = 1;
      for (long i = 0; i < exp; ++i)
        if (__builtin_mul_overflow(result, base, &result))
          throw std::overflow_error("Backbone: " + std::to_string(base) + "^" + std::to_string(exp) + " does not fit in a long");
      return result;
    }

  } // namespace

  BackboneVertex::BackboneVertex() : bar(false), dag(false), hyb_ind(0), Ksign(0), orb(0) { ; }

  bool BackboneVertex::has_bar() { return bar; }
  bool BackboneVertex::has_dag() { return dag; }
  int BackboneVertex::get_hyb_ind() { return hyb_ind; }
  int BackboneVertex::get_Ksign() { return Ksign; }
  int BackboneVertex::get_orb() { return orb; }

  void BackboneVertex::set_bar(bool b) { bar = b; }
  void BackboneVertex::set_dag(bool b) { dag = b; }
  void BackboneVertex::set_hyb_ind(int i) { hyb_ind = i; }
  void BackboneVertex::set_Ksign(int i) { Ksign = i; }
  void BackboneVertex::set_orb(int i) { orb = i; }

  Backbone::Backbone(nda::array<int, 2> topology, int n, int n_int)
    : Backbone(DiagramTopology{std::move(topology)}, n, n_int) {}

  Backbone::Backbone(DiagramTopology topology, int n, int n_int)
     : m(topology.order()),
       n(n),
       n_hyb(n - n_int), // number of hybridization operators is total number of operators minus number of interaction operators
       n_int(n_int),
       fb_ix_max(checked_pow(2, m)),
       o_ix_max(checked_pow(n, m - 1)),
       prefactor_sign(1),
       vertices(std::vector<BackboneVertex>(2 * m)),
       prefactor_Ksigns(nda::vector<int>(m - 1, 0)),
       prefactor_Kexps(nda::vector<int>(m - 1, 0)),
       edges(nda::zeros<int>(2 * m - 1, m - 1)),
       topology(std::move(topology)),
       fb(nda::vector<int>(m, 0)),
       f_ix(0) {}

  void Backbone::set_directions(nda::vector_const_view<int> fb_vec) {
    auto prepared = prepare_vertices(topology, fb_vec, DiagramKind::self_energy);
    fb = fb_vec;
    for (int v = 0; v < 2 * m; ++v) {
      vertices[v].set_bar(prepared[v].barred);
      vertices[v].set_dag(prepared[v].dagger);
    }
  }

  void Backbone::set_directions(long fb_ix) {

    auto fb_vec = nda::vector<int>(m);
    for (int i = 0; i < m; i++) {
      fb_vec(i) = fb_ix % 2; // 0 for backward, 1 for forward
      fb_ix /= 2;
    }
    set_directions(fb_vec);
  }

  void Backbone::reset_directions() {
    fb = 0;
    for (int i = 0; i < 2 * m; i++) {
      vertices[i].set_bar(false);
      vertices[i].set_dag(false);
    }
  }

  void Backbone::set_pole_inds(nda::vector_const_view<int> pole_inds_vec, nda::vector_const_view<double> hyb_poles) {
    auto prepared = prepare_kernel_factors(topology, fb, pole_inds_vec, hyb_poles);
    pole_inds = pole_inds_vec;
    edges = std::move(prepared.edges);
    prefactor_Ksigns = std::move(prepared.signs);
    prefactor_Kexps = std::move(prepared.exponents);
    prefactor_sign = prepared.sign;
    for (int v = 0; v < 2 * m; ++v) {
      vertices[v].set_hyb_ind(prepared.vertices[v].pole_slot);
      vertices[v].set_Ksign(prepared.vertices[v].sign);
    }
  }

  void Backbone::set_pole_inds(long p_ix, nda::vector_const_view<double> hyb_poles) {

    long p             = hyb_poles.size();
    auto pole_inds_vec = nda::vector<int>(m - 1);
    for (int i = 0; i < m - 1; i++) {
      pole_inds_vec(i) = p_ix % p;
      p_ix /= p;
    }
    set_pole_inds(pole_inds_vec, hyb_poles);
  }

  void Backbone::reset_pole_inds() {
    pole_inds        = 0;
    edges            = 0;
    prefactor_sign   = 1;
    prefactor_Ksigns = 0;
    prefactor_Kexps  = 0;
    for (int i = 0; i < 2 * m; i++) {
      vertices[i].set_hyb_ind(0);
      vertices[i].set_Ksign(0);
      vertices[i].set_orb(0);
    }
  }

  void Backbone::set_orb_inds(nda::vector_const_view<int> orb_inds_vec) {
    // orb_inds = orbital indices (e.g. lambda, mu indices), going
    // right to left, excluding the ones associated with the special vertex
    this->orb_inds = orb_inds_vec;
    for (int i = 0; i < 2 * m; i++) {
      if (i != 0 && i != topology.arcs(0, 1)) { vertices[i].set_orb(orb_inds_vec(i)); }
    }
  }

  void Backbone::set_orb_inds(long o_ix) {
    // set orbital indices from a single integer index
    auto orb_inds_vec            = nda::vector<int>(2 * m);
    orb_inds_vec(0)              = -1;
    orb_inds_vec(topology.arcs(0, 1)) = -1; // special vertex 0 and the one connected to it have no orbital indices explicitly summed over
    for (int i = 1; i < m; i++) {      // loop over lines, skipping the one connected to vertex 0
      orb_inds_vec(topology.arcs(i, 0)) = o_ix % n;
      orb_inds_vec(topology.arcs(i, 1)) = o_ix % n;
      // orbital indices on vertices connected by a line are the same
      o_ix /= n;
    }
    set_orb_inds(orb_inds_vec);
  }

  void Backbone::reset_orb_inds() {
    for (int i = 0; i < 2 * m; i++) vertices[i].set_orb(0);
  }

  void Backbone::set_orb_inds_of_0_and_vct0(int orb_ind_0, int orb_ind_vct0) {
    // set the orbital indices of vertex 0 and the vertex connected to it, which are not included in orb_inds
    vertices[0].set_orb(orb_ind_0);
    vertices[topology.arcs(0, 1)].set_orb(orb_ind_vct0);
  }

  int Backbone::get_parity() {
    // Compatibility translation only: explicit statistics own the sign rule.
    std::vector<Statistics> statistics(2 * m);
    for (int v = 0; v < 2 * m; ++v) {
      int orbital = vertices[v].get_orb();
      if (orbital >= 0) statistics[v] = orbital < n_hyb ? Statistics::fermionic : Statistics::bosonic;
      else if (orbital == -2) statistics[v] = Statistics::fermionic;
      else if (orbital == -3) statistics[v] = Statistics::bosonic;
      else throw std::invalid_argument("Orbital index of vertex " + std::to_string(v) + " is not set");
    }
    return permutation_sign(topology, statistics);
  }

  long Backbone::num_flat_indices(long p) const { return count_assignments(m, n, p, kind); }

  DiagramAssignment Backbone::prepare_assignment(long flat_ix, nda::vector_const_view<double> hyb_poles) {
    auto prepared = triqs_xca::prepare_assignment(topology, n, hyb_poles, flat_ix, kind);
    // Commit a complete assignment only after validation/preparation succeeds.
    f_ix = flat_ix;
    fb = prepared.directions;
    pole_inds = prepared.poles;
    edges = prepared.edge_kernels;
    prefactor_Ksigns = prepared.prefactor_kernel_signs;
    prefactor_Kexps = prepared.prefactor_kernel_exponents;
    prefactor_sign = prepared.prefactor_sign;
    orb_inds.resize(2 * m);
    for (int v = 0; v < 2 * m; ++v) {
      auto const &vertex = prepared.vertices[v];
      vertices[v].set_bar(vertex.barred);
      vertices[v].set_dag(vertex.dagger);
      vertices[v].set_hyb_ind(vertex.pole_slot);
      vertices[v].set_Ksign(vertex.kernel_sign);
      // The legacy numerical API uses zero before supplying its special endpoints.
      vertices[v].set_orb(vertex.orbital < 0 ? 0 : vertex.orbital);
      orb_inds(v) = vertex.orbital;
    }
    return prepared;
  }

  void Backbone::set_flat_index(long flat_ix, nda::vector_const_view<double> hyb_poles) {
    prepare_assignment(flat_ix, hyb_poles);
  }

  void Backbone::reverse_hyb_line_zero() {
    fb(0)    = 1 - fb(0);
    int vct0 = topology.arcs(0, 1);
    vertices[0].set_dag(!vertices[0].has_dag());
    vertices[vct0].set_dag(!vertices[vct0].has_dag());
  }

  void Backbone::reset_all_inds() {
    reset_directions();
    reset_pole_inds();
    reset_orb_inds();
  }

  int Backbone::get_prefactor_Ksign(int i) { return prefactor_Ksigns(i); }
  int Backbone::get_prefactor_Kexp(int i) { return prefactor_Kexps(i); }
  bool Backbone::has_vertex_bar(int i) { return vertices[i].has_bar(); }
  bool Backbone::has_vertex_dag(int i) { return vertices[i].has_dag(); }
  int Backbone::get_vertex_hyb_ind(int i) { return vertices[i].get_hyb_ind(); }
  int Backbone::get_vertex_Ksign(int i) { return vertices[i].get_Ksign(); }
  int Backbone::get_vertex_orb(int i) { return vertices[i].get_orb(); }
  int Backbone::get_edge(int num, int pole_ind) { return edges(num, pole_ind); }
  int Backbone::get_topology(int i, int j) { return topology.arcs(i, j); }
  const nda::array<int, 2> Backbone::get_topology() const { return topology.arcs; }

  int Backbone::get_pole_ind(int i) { return pole_inds(i); }
  int Backbone::get_fb(int i) { return fb(i); }
  int Backbone::get_orb_ind(int i) { return orb_inds(i); }
  long Backbone::get_flat_index() { return f_ix; }

  CorrelatorBackbone::CorrelatorBackbone(nda::array<int, 2> topology, int n, int n_int) : Backbone(topology, n, n_int) {
    kind = DiagramKind::correlator;
    fb_ix_max = checked_pow(2, m - 1);
  }

  void CorrelatorBackbone::set_directions(long fb_ix) {

    auto fb_vec = nda::vector<int>(m - 1);
    for (int i = 0; i < m - 1; i++) {
      fb_vec(i) = fb_ix % 2; // 0 for backward, 1 for forward
      fb_ix /= 2;
    }
    set_directions(fb_vec);
  }

  void CorrelatorBackbone::set_directions(nda::vector_const_view<int> fb_vec) {
    auto prepared = prepare_vertices(topology, fb_vec, DiagramKind::correlator);
    fb(nda::range(1, m)) = fb_vec;
    fb(0) = -1;
    for (int v = 0; v < 2 * m; ++v) {
      vertices[v].set_bar(prepared[v].barred);
      vertices[v].set_dag(prepared[v].dagger);
    }
  }

  std::ostream &operator<<(std::ostream &os, Backbone &B) {

    std::string p_str = "1 / (";
    int sign          = B.prefactor_sign;
    if (sign == -1) p_str = "-" + p_str;
    for (int i = 0; i < B.m - 1; i++) {
      if (B.get_prefactor_Kexp(i) >= 1) {
        p_str += "K_{l";
        for (int j = 0; j < i; j++) p_str += "`";
        p_str += "}";
        if (B.get_prefactor_Ksign(i) == 1)
          p_str += "^+";
        else
          p_str += "^-";
      }
      if (B.get_prefactor_Kexp(i) > 1) { p_str += "^" + std::to_string(B.get_prefactor_Kexp(i)); }
      p_str += "(0)";
      if (i < B.m - 2) p_str += " ";
    }
    p_str += ")";

    int diag_str_cent    = 30;
    std::string diag_str = "";
    for (int i = 0; i < diag_str_cent; i++) diag_str += " ";
    diag_str += "0\n";
    std::string v_str_tmp = "", e_str_tmp = "";
    for (int i = 0; i < 2 * B.m; i++) {
      // K factor
      if (B.get_vertex_Ksign(i) != 0) {
        v_str_tmp += "K_{l";
        for (int j = 0; j < B.get_vertex_hyb_ind(i); j++) v_str_tmp += "`";
        v_str_tmp += "}";
        if (B.get_vertex_Ksign(i) == 1)
          v_str_tmp += "^+";
        else
          v_str_tmp += "^-";
      }

      // F operator
      v_str_tmp += "F";
      if (B.has_vertex_bar(i) || B.has_vertex_dag(i)) v_str_tmp += "^{";
      if (B.has_vertex_bar(i)) v_str_tmp += "bar";
      if (B.has_vertex_dag(i)) v_str_tmp += "dag";
      if (B.has_vertex_bar(i) || B.has_vertex_dag(i)) v_str_tmp += "}";
      if (not B.has_vertex_bar(i))
        v_str_tmp += "_" + std::to_string(i);
      else {
        for (int j = 0; j < B.m; j++)
          if (B.get_topology(j, 1) == i) v_str_tmp += "_" + std::to_string(B.get_topology(j, 0));
        v_str_tmp += "l";
        for (int j = 0; j < B.get_vertex_hyb_ind(i); j++) v_str_tmp += "`";
      }
      // hybridization
      if (i == B.get_topology(0, 1)) {
        v_str_tmp += " Delta_{";
        if (B.has_vertex_dag(i) == 1)
          v_str_tmp += "0," + std::to_string(i) + "} ";
        else
          v_str_tmp += std::to_string(i) + ",0}";
      } else
        v_str_tmp += " ";
      int vlen0 = v_str_tmp.size();
      for (int j = vlen0; j < diag_str_cent - 2; j++) v_str_tmp = " " + v_str_tmp;
      diag_str += v_str_tmp + "--| \n";
      v_str_tmp = "";

      // edges --> e_str
      if (i < 2 * B.m - 1) {
        for (int j = 0; j < B.m - 1; j++) {
          if (B.get_edge(i, j) != 0) {
            e_str_tmp += "K_{l";
            for (int k = 0; k < j; k++) e_str_tmp += "`";
            e_str_tmp += "}";
            if (B.get_edge(i, j) == 1)
              e_str_tmp += "^+ ";
            else
              e_str_tmp += "^- ";
          }
        }
        e_str_tmp += "G ";
        for (int j = 0; j < diag_str_cent; j++) diag_str += " ";
        diag_str += "| " + e_str_tmp + "\n";
        e_str_tmp = "";
      }
    }
    for (int i = 0; i < diag_str_cent - 1; i++) diag_str += " ";
    diag_str += "tau";

    os << "\nPrefactor: " << p_str << "\nDiagram: \n" << diag_str;
    return os;
  }

  std::ostream &operator<<(std::ostream &os, CorrelatorBackbone &B) {

    std::string p_str = "1 / (";
    int sign          = B.prefactor_sign;
    if (sign == -1) p_str = "-" + p_str;
    for (int i = 0; i < B.m - 1; i++) {
      if (B.get_prefactor_Kexp(i) >= 1) {
        p_str += "K_{l";
        for (int j = 0; j < i; j++) p_str += "`";
        p_str += "}";
        if (B.get_prefactor_Ksign(i) == 1)
          p_str += "^+";
        else
          p_str += "^-";
      }
      if (B.get_prefactor_Kexp(i) > 1) { p_str += "^" + std::to_string(B.get_prefactor_Kexp(i)); }
      p_str += "(0)";
      if (i < B.m - 2) p_str += " ";
    }
    p_str += ")";

    int diag_str_cent    = 30;
    std::string diag_str = "";
    for (int i = 0; i < diag_str_cent; i++) diag_str += " ";
    diag_str += "0\n";
    std::string v_str_tmp = "", e_str_tmp = "";
    for (int i = 0; i < 2 * B.m; i++) {
      // K factor
      if (B.get_vertex_Ksign(i) != 0) {
        v_str_tmp += "K_{l";
        for (int j = 0; j < B.get_vertex_hyb_ind(i); j++) v_str_tmp += "`";
        v_str_tmp += "}";
        if (B.get_vertex_Ksign(i) == 1)
          v_str_tmp += "^+";
        else
          v_str_tmp += "^-";
      }

      // F operator
      v_str_tmp += "F";
      if (B.has_vertex_bar(i) || B.has_vertex_dag(i)) v_str_tmp += "^{";
      if (B.has_vertex_bar(i)) v_str_tmp += "bar";
      if (B.has_vertex_dag(i)) v_str_tmp += "dag";
      if (B.has_vertex_bar(i) || B.has_vertex_dag(i)) v_str_tmp += "}";
      if (not B.has_vertex_bar(i))
        v_str_tmp += "_" + std::to_string(i);
      else {
        for (int j = 0; j < B.m; j++)
          if (B.get_topology(j, 1) == i) v_str_tmp += "_" + std::to_string(B.get_topology(j, 0));
        v_str_tmp += "l";
        for (int j = 0; j < B.get_vertex_hyb_ind(i); j++) v_str_tmp += "`";
      }

      v_str_tmp += " ";
      int vlen0 = v_str_tmp.size();
      for (int j = vlen0; j < diag_str_cent - 2; j++) v_str_tmp = " " + v_str_tmp;
      diag_str += v_str_tmp + "--| \n";
      v_str_tmp = "";

      // edges --> e_str
      if (i < 2 * B.m - 1) {
        for (int j = 0; j < B.m - 1; j++) {
          if (B.get_edge(i, j) != 0) {
            e_str_tmp += "K_{l";
            for (int k = 0; k < j; k++) e_str_tmp += "`";
            e_str_tmp += "}";
            if (B.get_edge(i, j) == 1)
              e_str_tmp += "^+ ";
            else
              e_str_tmp += "^- ";
          }
        }
        e_str_tmp += "G ";
        for (int j = 0; j < diag_str_cent; j++) diag_str += " ";
        diag_str += "| " + e_str_tmp + "\n";
        e_str_tmp = "";
      }
    }
    for (int i = 0; i < diag_str_cent; i++) diag_str += " ";
    diag_str += "| G\n";
    for (int i = 0; i < diag_str_cent - 1; i++) diag_str += " ";
    diag_str += "beta";

    os << "\nPrefactor: " << p_str << "\nDiagram: \n" << diag_str;
    return os;
  }

} // namespace triqs_xca::backbone