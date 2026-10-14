# ThinkStation P620 weighted reference schedule

CPU TRIQS, one OMP/BLAS thread, source50bec42. Operation counts model the GPU
reference schedule; optimization target remains GH200-class hardware. These are
not device timings or traffic measurements. Eight cases use actual atomic G0 block
masks and full DLR rank; two Lambda4000 cases report rank64. No rank truncation or
large atomN14 construction is used. Work remains conditional on structural paths;
output-cutoff accumulation is separate.

| Case | Modeled events | Model MFLOP | Dense-equivalent MFLOP |
| --- | ---: | ---: | ---: |
| n10-number-nca | 3210 | 5096.174 | 684927.860 |
| n10-number-nca-r64 | 3210 | 14180.659 | 1905886.218 |
| n4-number-nca | 228 | 0.325 | 1.153 |
| n4-number-oca | 249 | 3.314 | 3.830 |
| n4-number-oca-r64 | 249 | 23.035 | 24.470 |
| n4-occupation-oca | 384 | 0.646 | 0.646 |
| n6-mixed-retarded | 1500 | 65.602 | 74.204 |
| n8-number-nca | 1672 | 228.256 | 8539.164 |

Raw traces expose field/nonzero counts and source/output sectors. Weighted-final
reports work and operand-footprint shares by kind; solve cost is leading-order.
Dense-equivalent operator work can be much larger than actual sparse arithmetic;
it must not be reported as executed sparse FLOPs. Footprints are addressable operands,
not actual HBM bytes, cache behavior or peak admission. Model exclusions are in the
summary manifest and generic README.

Independent checks verify NCA q² right/q left counts, sparse8*r*nnz*RHS, structural
path counts and mask/rank reporting. Guards reject analytic work/masks, negative
work summaries and a trace beyond200000 rows. The incomplete row-cap file is hashed
and counted but not committed as a valid trace. No runtime evaluator code changed.
Commands and source/binary/input hashes are retained. No Arrhenius allocation was
used to collect this inventory.
