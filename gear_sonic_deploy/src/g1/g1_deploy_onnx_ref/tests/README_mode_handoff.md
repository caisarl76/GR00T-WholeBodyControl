# Managed ZMQ mode handoff regression

With the normal deployment dependencies configured and `BUILD_DEPLOY_TESTS=ON`:

```sh
cmake --build build --target zmq_mode_handoff_test g1_deploy_onnx_ref
ctest --test-dir build -R '^zmq_mode_handoff_test$' --output-on-failure
```

The executable invokes the real ZMQ manager/endpoint callbacks and decoder with
in-memory v1 and v3 packets. Subscribers are stopped before callbacks are invoked.
A synthetic inference backend exercises the real planner initialization,
resampling, and generation stamping. No robot, running publisher, model inference,
or hardware control is needed. `-fno-access-control` is confined to this test target
so callbacks can be driven without adding test hooks to production classes.

Coverage includes fresh first-pose retention on both subscriber orderings,
stale and malformed first poses, reference/playback preservation while preparing,
planner readiness and initialization deadlines, stop precedence, rapid reversals,
control-source ownership at commit, SMPL encoder preservation while waiting,
world-heading rebasing (including wraparound and chained commits), and obsolete
inference generations.

These checks establish software handoff behavior, not physical transition quality.
A pending planner uses the remaining old streamed window and then holds its last
reference; it does not receive continuing pose tracking after the sender switches.
The first destination reference and encoder can still differ from the source;
there is no cross-encoder action blending. Pose freshness is a local receive-age
check (100 ms), not a wire-level session identifier. Simulation and supervised
hardware validation remain necessary to assess actual robot motion.
