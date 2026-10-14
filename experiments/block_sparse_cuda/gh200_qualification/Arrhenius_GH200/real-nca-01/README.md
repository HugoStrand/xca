# GH200 NCA job records

Job 3843116 was cancelled during the initial preparation/build attempt (392 s).
Job 3843405 reused its isolated build and completed all three CTests / 20 controls
and the 48-row campaign (182 s). See accounting.txt, both job directories and
scheduler logs. The submitted retry recipe and source/runner hashes are preserved.

Timestamp normalization was removed from GPU allocation time and staged before
transfer. Both jobs used one GPU with fifteen-minute limits; no install/shared source
update occurred. Combined benchmark interpretation and memory limits are in
[the common record](../../../../../benchmarks/block_sparse_common/Arrhenius_GH200/real-nca-01/README.md).
