# Live A+X handoff metrics

9 switches, 4603 control samples; monotonic range 267244.280767–267336.342340 s.

Intervals use shared monotonic clocks; pre [-0.5,0), early [0,1), late [1,3), truncated before the next switch. Maximums are over all joints. Action is raw decoder output in IsaacLab order (dimensionless), logged one control tick after it was computed; first new-policy action is row switch+1. q is measured position minus default angle in IsaacLab order (radians); dq is measured velocity in the same order (radians per second); motor torque is hardware order. Logging intervals measure control-loop start cadence, not full inference latency. The final uncontrolled shutdown tail is not used for switch-local metrics.

|#|switch time (relative s)|direction|window|tilt max °|z min m|action Δ max|q Δ max rad|dq max rad/s|tick max ms|
|---|---|---|---|---|---|---|---|---|---|
|1|5.800|planner_motion→streamed|pre|3.30|0.7858|0.027|0.002|0.116|20.050|
|1|5.800|planner_motion→streamed|early|4.44|0.7770|2.455|0.057|3.013|22.319|
|1|5.800|planner_motion→streamed|late|6.35|0.7835|1.381|0.052|4.770|20.299|
|2|19.540|streamed→planner_motion|pre|4.17|0.7825|0.238|0.023|1.188|20.022|
|2|19.540|streamed→planner_motion|early|5.39|0.7750|2.719|0.079|4.513|20.027|
|2|19.540|streamed→planner_motion|late*|2.97|0.7873|0.330|0.025|1.284|20.202|
|3|22.280|planner_motion→streamed|pre|1.58|0.7873|0.048|0.002|0.108|20.035|
|3|22.280|planner_motion→streamed|early|7.25|0.7703|4.404|0.089|5.219|20.394|
|3|22.280|planner_motion→streamed|late|4.92|0.7841|1.758|0.069|3.513|21.207|
|4|25.940|streamed→planner_motion|pre|4.00|0.7852|0.145|0.015|0.932|20.020|
|4|25.940|streamed→planner_motion|early|10.48|0.7683|3.906|0.103|7.360|20.025|
|4|25.940|streamed→planner_motion|late*|3.33|0.7863|1.128|0.045|2.246|20.021|
|5|28.700|planner_motion→streamed|pre|2.75|0.7863|0.039|0.003|0.128|20.011|
|5|28.700|planner_motion→streamed|early|5.50|0.7735|2.872|0.072|7.587|20.018|
|5|28.700|planner_motion→streamed|late|4.45|0.7838|0.521|0.080|6.081|20.119|
|6|31.740|streamed→planner_motion|pre|3.36|0.7845|0.196|0.026|1.360|20.119|
|6|31.740|streamed→planner_motion|early|9.09|0.7703|3.758|0.089|4.124|20.018|
|6|31.740|streamed→planner_motion|late|3.35|0.7865|0.717|0.072|4.605|20.018|
|7|67.640|planner_motion→streamed|pre|1.73|0.7869|0.000|0.000|0.002|21.299|
|7|67.640|planner_motion→streamed|early|2.39|0.7833|1.314|0.045|3.055|25.472|
|7|67.640|planner_motion→streamed|late|3.31|0.7793|1.119|0.089|6.902|25.799|
|8|71.240|streamed→planner_motion|pre|2.27|0.7827|0.170|0.019|1.376|22.571|
|8|71.240|streamed→planner_motion|early|4.81|0.7787|1.045|0.064|3.356|20.301|
|8|71.240|streamed→planner_motion|late*|2.32|0.7874|0.176|0.053|2.816|22.194|
|9|74.060|planner_motion→streamed|pre|1.33|0.7878|0.010|0.001|0.044|21.897|
|9|74.060|planner_motion→streamed|early|3.82|0.7823|1.820|0.099|2.613|23.856|
|9|74.060|planner_motion→streamed|late|12.25|0.7650|3.473|0.136|21.819|25.459|

*Late window truncated before next switch. See JSON for all metrics and file digests.

## First newly selected policy action

|#|direction|action step max|action step L2|action index|
|---|---|---|---|---|
|1|planner_motion→streamed|0.970|2.107|23|
|2|streamed→planner_motion|0.204|0.437|25|
|3|planner_motion→streamed|1.219|2.434|28|
|4|streamed→planner_motion|0.243|0.586|19|
|5|planner_motion→streamed|1.359|2.375|28|
|6|streamed→planner_motion|0.201|0.490|28|
|7|planner_motion→streamed|1.314|2.390|23|
|8|streamed→planner_motion|0.161|0.350|27|
|9|planner_motion→streamed|1.518|2.450|28|

## Peak timing

|#|raw action step peak delay s (logged)|action index|tilt peak delay s|
|---|---|---|---|
|1|0.642|14|0.141|
|2|0.560|13|0.146|
|3|0.540|14|0.547|
|4|0.580|14|0.564|
|5|0.620|13|0.480|
|6|0.600|14|0.577|
|7|0.020|23|0.160|
|8|0.520|10|0.489|
|9|0.583|14|0.959|

Indices 13/14 are left/right ankle pitch; 10 is right knee; 23 is left wrist roll. Subtract one control tick (~20 ms) from action logged delay for the output computation time. The q/dq values use IsaacLab order as verified in GatherRobotStateToLogger.

## Findings

All four POSE→PLANNER events have first-second tilt above both their pre-window and subsequent 1–3-second maxima: 5.39°, 10.48°, 9.09°, 4.81°. The strongest is switch 4 (monotonic 267270.220747): 4.00° before, 10.48° early, 3.33° late; minimum height 0.7852→0.7683→0.7863 m; maximum joint speed 0.932→7.360→2.246 rad/s. This substantiates a transient balance disturbance rather than a no-fall pass.

Eight of nine switches have their first-second maximum raw action step at logged +0.52 to +0.64 seconds, predominantly ankle pitch. The source has a 1-second smoothstep blend of a captured initial joint position with live policy targets; the peak is around mid-ramp, not its endpoint. The temporal pattern is consistent with that path contributing, but this capture alone cannot prove causality. The logged action is before the ramp, so it is not the applied target.

No switch-local control tick gap exceeds 25.8 ms. The whole capture has one gap above 30 ms (48.31 ms), at monotonic 267332.709077, +14.368 seconds after switch 9; it does not explain the repeatable 1-second disturbances.

The final POSE interval also contains separate fall/reset-like episodes BEFORE the final shutdown: first z<0.4 at monotonic 267323.766166 (+5.425 s after switch 9), first low episode reaches z=0.202 m and tilt≈68° and is followed by upright height the next second; another at +16 seconds reaches z=0.204 m and tilt≈89°. Simulator.log explicitly reports fallen-height warnings. These later events are not included as evidence that the mode switch itself caused a fall; input/body motion and reset behavior require separate attribution.

## Scope and interpretation

This is an observational live capture: operator/body motion, mode switch, and reference changes are not independently controlled. Temporal alignment can establish a repeatable transition-associated disturbance but cannot by itself identify the causal source. No-fall is not a success criterion.

{
  "n_rows": 4603,
  "time_start": 267244.280767,
  "time_end": 267336.34234000003,
  "n_switches": 9,
  "intertick_global_max_ms": 48.309999983757734,
  "intertick_global_p99_ms": 24.019780011731196,
  "intertick_over30": 1,
  "first_z_under_04_during_logging": 267323.766165637
}
