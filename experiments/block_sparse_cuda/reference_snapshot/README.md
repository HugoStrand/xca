# Resident evaluator snapshot

Evaluator source frozen before production storage-policy integration; original hashes
and revisions in origin.json. Existing experiment generators read this source so their
algorithms and replacement seams survive production changes. DeviceOps/DlrOps, model
and traversal helpers still link from production; changing those requires a separate
baseline audit. The optional reference library renames the class for direct comparisons.
Preserve all experiments, including failed numerical candidates. GH200-class hardware
is the target; workstation timings remain development evidence.
