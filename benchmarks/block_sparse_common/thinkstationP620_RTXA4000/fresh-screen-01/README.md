# Local fresh-process recorder screen

RTX A4000 is a development proxy; optimization target is GH200-class hardware.
Three-process pilot: three selected cases × three controls × three fresh processes
=27 distinct PIDs,54 samples. Each case/control group has exactly three process
records. Separate final summary control runs three CPU processes and verifies the
aggregated fresh-process count. Production cohort screen: six declared cases ×
seven controls=42 successful processes/samples, including the actual production
mapped-special frontend. All use single-thread OMP/BLAS, sequential backend calls.
Raw signal/error/count/setup/call rows retained; no evaluator arithmetic changed.

The pilot verifies process identity, independent launches and recording, not a
GH200 crossover or statistically established local winner. It predates the final
summary renderer; the later CPU control checks that renderer. Executed runner and
binary hashes remain in each manifest. The supplied revisions file describes the
retained dependency snapshot; it is not substituted for actual local binary/runner
provenance. Source code under c++/triqs_xca matches85ecb2d. The target campaign
uses its own pinned source/archive and fresh-process records.
