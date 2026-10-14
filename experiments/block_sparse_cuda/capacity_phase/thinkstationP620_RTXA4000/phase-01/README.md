# RTX A4000 full-shape phase screens

Code edb8b73, target recipe290d65f. Rank64 (lambda4000,eps1e-10,nonsymmetric), square
sizes65/129/252/512, two calls each. Three spatial entries at every rank node are
checked against independent long-double full inner/rank sums; sampled replay exact.
Largest explicit phase payload2181366020 bytes, first/warm convolution906.982/784.097 ms,
sampled oracle error5.275e-23 at norm4.810e-08. Setup/upload and oracle time separate.
Not complete Solver timings or full output/physical model/cutoff qualification.

Final native two CTests pass14.52 s, including fourteen stream controls; library-only
two pass11.15 s, stream thirteen active plus one native-policy skip. Native flags OFF
in library tree. All four library/ordered and reference/mapped selections pass exact
SE/SPGF controls with dynamic interactions. Incompatible RealBatch arena rejects.
Two new native policy controls pass memcheck34.377 s, zero errors/leaks. Phase probe
13x9x11/rank64 passes memcheck zero errors/leaks. Earlier stream lifetime memcheck
remains separate. CPU configuration omits both experiments. Declared-budget rejection
(status2), large-execution workstation guard(status1) and malformed suffix guard
(status1) checked explicitly; no automatic retry/relaxed cap.

Host-only3432x3003x3432 inventory at lambda4000 gives rank64/90493714692 bytes
(84.28 GiB), including one retained opposite-square tensor and32 MiB reserve. This
uses N14 sector dimensions, not a full atom/solver. No large workstation allocation.
The source permits large execution only with allow-large, GH200 and Slurm; actual free
memory/headroom checked before large arrays. Reduced screens preceded a logging and
BLAS-limit hardening patch; final small tests/guards use final code, arithmetic unchanged.
Pre-hardening screen binary hash was not retained; do not attribute those runs to the
final binary hash. Qualification.json distinguishes provenance. GPU runs sequential;
no local work overlaps the subsequent GH200 qualification. Target GH200, local proxy.
