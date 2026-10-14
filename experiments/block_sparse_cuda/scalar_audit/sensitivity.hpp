// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <array>
#include <complex>
#include <vector>

namespace scalar_sensitivity {
template <unsigned Bits> using Real = boost::multiprecision::number<
  boost::multiprecision::backends::cpp_bin_float<Bits, boost::multiprecision::backends::digit_base_2>>;
template <typename R> using Z = std::complex<R>;
template <typename R> R l1_norm(Z<R> const &z) { using boost::multiprecision::abs; return abs(z.real()) + abs(z.imag()); }

template <typename R> struct Formula {
  long r;
  std::vector<Z<R>> C, H, Q, inverse;
  Formula(auto const &c, auto const &h, auto const &q, auto const &inv) : r(c.extent(0)) {
    auto append = [&](auto const &a, auto &out) {
      for (long i = 0; i < r; ++i) for (long j = 0; j < r; ++j) {
        auto z = a(i,j); out.emplace_back(R(z.real()), R(z.imag()));
      }
    };
    append(c,C); append(h,H); append(q,Q); append(inv,inverse);
  }
  std::vector<Z<R>> transform(std::vector<Z<R>> const &v) const {
    std::vector<Z<R>> out(r);
    for (long i = 0; i < r; ++i) for (long j = 0; j < r; ++j) out[i] += H[i*r+j] * v[j];
    return out;
  }
  std::vector<Z<R>> convolve(std::vector<Z<R>> const &f, std::vector<Z<R>> const &g) const {
    auto hf = transform(f), hg = transform(g);
    std::vector<Z<R>> out(r);
    for (long t = 0; t < r; ++t) for (long j = 0; j < r; ++j)
      out[t] += R(2) * (Q[t*r+j]*f[j]*g[j] + C[t*r+j]*(hf[j]*g[j] + f[j]*hg[j]));
    return out;
  }
  std::pair<R,R> jacobian_norms(std::vector<Z<R>> const &other) const {
    auto ho = transform(other);
    std::vector<Z<R>> J(r*r);
    R coefficient_norm = 0, node_norm = 0;
    for (long t = 0; t < r; ++t) {
      R row = 0;
      for (long i = 0; i < r; ++i) {
        auto &v = J[t*r+i];
        v = Q[t*r+i]*other[i] + C[t*r+i]*ho[i];
        for (long j = 0; j < r; ++j) v += C[t*r+j]*H[j*r+i]*other[j];
        v *= R(2); row += l1_norm(v);
      }
      coefficient_norm = std::max(coefficient_norm, row);
    }
    for (long t = 0; t < r; ++t) {
      R row = 0;
      for (long k = 0; k < r; ++k) {
        Z<R> sum{};
        for (long i = 0; i < r; ++i) sum += J[t*r+i]*inverse[i*r+k];
        row += l1_norm(sum);
      }
      node_norm = std::max(node_norm, row);
    }
    // Independent directional check: the first Jacobian column equals a
    // convolution with unit coefficient input, since the formula is bilinear.
    std::vector<Z<R>> unit(r); unit[0] = 1;
    auto directional = convolve(unit, other);
    R error = 0, scale = 1;
    for (long t = 0; t < r; ++t) {
      error = std::max(error, l1_norm(directional[t] - J[t*r]));
      scale = std::max(scale, l1_norm(directional[t]));
    }
    if (error / scale > R("1e-30")) throw std::runtime_error("Jacobian directional check failed");
    return {coefficient_norm, node_norm};
  }
  R absolute_scale(std::vector<Z<R>> const &f, std::vector<Z<R>> const &g) const {
    std::vector<R> hf(r),hg(r);
    for (long j=0;j<r;++j) for (long i=0;i<r;++i) {
      hf[j] += l1_norm(H[j*r+i])*l1_norm(f[i]);
      hg[j] += l1_norm(H[j*r+i])*l1_norm(g[i]);
    }
    R result=0;
    for (long t=0;t<r;++t) {
      R row=0;
      for (long j=0;j<r;++j)
        row += R(2)*(l1_norm(Q[t*r+j])*l1_norm(f[j])*l1_norm(g[j])
                      +l1_norm(C[t*r+j])*(hf[j]*l1_norm(g[j])+l1_norm(f[j])*hg[j]));
      result=std::max(result,row);
    }
    return result;
  }
};

template <unsigned Bits> auto measure(auto const &C, auto const &H, auto const &Q, auto const &inverse,
                                     auto const &nf, auto const &ng, auto const &wf, auto const &wg) {
  using R = Real<Bits>;
  Formula<R> formula(C,H,Q,inverse);
  auto vector = [&](auto const &v) {
    std::vector<Z<R>> out;
    for (long t = 0; t < formula.r; ++t) { auto z=v(t,0,0); out.emplace_back(R(z.real()),R(z.imag())); }
    return out;
  };
  auto f=vector(nf), g=vector(ng), a=vector(wf), b=vector(wg);
  auto [jf,tf]=formula.jacobian_norms(g); auto [jg,tg]=formula.jacobian_norms(f);
  auto narrow=formula.convolve(f,g), wide=formula.convolve(a,b);
  std::vector<Z<R>> df(formula.r),dg(formula.r);
  for (long i=0;i<formula.r;++i) {df[i]=a[i]-f[i];dg[i]=b[i]-g[i];}
  auto firstf=formula.convolve(df,g),firstg=formula.convolve(f,dg),second=formula.convolve(df,dg);
  R nn=0,wn=0,effect=0,quadratic=0,identity_error=0;
  for (long t=0;t<formula.r;++t) {
    nn=std::max(nn,l1_norm(narrow[t])); wn=std::max(wn,l1_norm(wide[t]));
    effect=std::max(effect,l1_norm(wide[t]-narrow[t])); quadratic=std::max(quadratic,l1_norm(second[t]));
    identity_error=std::max(identity_error,l1_norm(wide[t]-narrow[t]-firstf[t]-firstg[t]-second[t]));
  }
  if (identity_error/std::max(R(1),std::max(nn,wn)) > R("1e-30")) throw std::runtime_error("Bilinear identity check failed");
  return std::array<R,10>{jf,tf,jg,tg,nn,wn,effect,quadratic,formula.absolute_scale(f,g),formula.absolute_scale(a,b)};
}
}
