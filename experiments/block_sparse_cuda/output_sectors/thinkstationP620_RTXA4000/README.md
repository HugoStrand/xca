# ThinkStation P620 / RTX A4000

Target hardware remains GH200-class. `outputs-01` is the preliminary two-route
comparison (98 rows), superseded for decisions by the ownership-controlled
`outputs-02` campaign (147 rows). Both retain raw output, bounds and captured hashes.
First-campaign launcher/benchmark code evolved before commit; use the final
three-route source and `outputs-02` for reproduction and comparative conclusions.

Seven physical static self-energy cases span number/mixed/occupation partitions,
orders 1–3, nonzero index offsets, and 4/6/8/10 fermionic modes. All final complete
outputs, accepted counts and contributing counts match exactly. Both baseline
and candidate use Ordered/Mapped policies. No build/profiler overlapped timings.
First-call rows share context/JIT state; conclusions below use three-sample medians.

Ten-mode number NCA changed-G medians: original resident 204.181 ms, explicit-move
resident 162.227 ms, streamed 171.698 ms. Thus the explicit move improves the local
resident baseline, while streaming costs 5.8% against that control. Eight-mode
changed-G medians are 21.355/18.835/18.891 ms, respectively; streaming costs 0.3%
against the move control. Smaller changed-G cases cost 0.2–1.2% against the control.
These are complete API costs, not PCIe bandwidth or whole-Solver speedups.

Ten-mode changed-G median sampled device deltas are 234,881,024 bytes for resident
move versus 186,646,528 for streaming; warm medians are 232,783,872/186,646,528.
The exact explicit bound reduction is 44,620,736 bytes (42.554 MiB), while samples
show allocator-granularity differences. Eight-mode deltas are 12,582,912/8,388,608
bytes, with a 2,932,960-byte explicit bound reduction. Smaller samples are often
zero because the diagnostic observes global allocation changes at selected points.
No continuous/process-private peak or per-route host RSS measurement is claimed.

The three-route benchmark also matches on a native-disabled Library/Reference
control. Focused three-route memcheck reports zero errors and leaks. Conclusions:
investigate explicit return ownership separately, prefer resident execution when
it fits, and retain sector streaming as a candidate capacity route. Large N=14
systems, target hardware, input/slab streaming and independent peak traces remain
unqualified.

`outputs-03` rechecks the promoted production return move against an explicitly
generated CopyOutputEvaluator and streaming. All 147 rows match outputs/counts
exactly. Changed-G eight-mode medians are 21.375454/18.823287/18.766029 ms copy/
production/streamed; ten-mode medians are 207.280464/162.649520/173.526853 ms.
Production reduces local call time 11.94%/21.53% against copying in these selections.
The device bound and CUDA arithmetic are unchanged by the return move. Source hashes
record the promotion and copying-control generation. These are local API medians,
not GH200 predictions or a whole-Solver improvement claim.
