# Cutoff-heavy cohort and ownership qualification

Seventeen replay GoogleTests passed, CTest 46.29 seconds. The new order-two scalar
case calibrates one isolated path, then exercises a six-mode eight-index range at
seven near-threshold amplitudes with zero/nonzero pole configurations. Structural
paths span a full 512-lane cohort and a tail. The replay observer requires actual
replay across that boundary for the zero-pole case and verifies reset/repeatability.
Outputs and contributing counts agree with ordered general CUDA.

Memcheck of the new cohort/pole case passed in 482.049 seconds with zero errors and
zero leaked bytes. This expensive local instrumentation was run once after the
numerical suite; no concurrent GPU work or cluster time was used. Preserved constants
suite was rebuilt and passed afterward; timing is in baseline.log.

Replay input owners are now allocated/copied once per cohort and shared by its
sensitive lanes. The admission bound is unchanged because the maximum live payload
is unchanged. No new performance claim is made. A fixed replay band remains an
experimental policy requiring broader conditioning/cancellation and GH200 evidence
before production promotion. All older experiments and production remain retained.
