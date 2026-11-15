# Physical AI Studio changes

[Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) is Intel's app for recording
demonstrations, training policies and running them on the arm. We used it for all teaching and training, and changed
it where it blocked us or where our task needed something it lacked. Everything below ran on the robot PC during the
hackathon. `patches/physical-ai-studio.patch` is exactly the code that ran there: Studio backend, UI and tests, plus
one fix in its `library/`.

| Patch | Applies to |
|---|---|
| `patches/physical-ai-studio.patch` | a checkout of the upstream `junction2026` branch at `285924d9` (`git apply`) |
| `patches/physicalai/so101-telemetry.patch` | the installed `physicalai_so101_plugin` 0.3.0 (backend venv) |
| `patches/physicalai/capture-depth.patch`, `capture-selfheal.patch` | the installed `physicalai.capture` (backend venv) |
| `patches/physicalai/openvino-gpu-f32-fallback.patch` | the installed `physicalai.inference` (backend venv) |

```bash
git clone -b junction2026 https://github.com/open-edge-platform/physical-ai-studio && cd physical-ai-studio
git checkout 285924d9 && (cd application/backend && uv sync --extra xpu)
../NeatPick/studio/apply.sh .        # run it again after every `uv sync`, which reinstalls the venv packages
```

After a backend change, regenerate the UI's API types from the running backend (`openapi-typescript`, as upstream
does).

Checked on a clean upstream checkout with only this patch applied: the patch's own 149 backend tests pass (telemetry,
benchmark trials, corrections, snapshots, loss stop, episode tags and markers, camera crop and calibration, Modal
backend).

The sections below give, for each change, what it does, why we needed it and what it gave us.

---

## Recording

### SO-101 motor telemetry
**What.** The SO-101 driver reads the STS3215 registers 56-70 of all six servos in one sync read: speed, load, voltage,
temperature, current. Studio records them as a separate `observation.telemetry` vector (36 values) next to
`observation.state`. The episode viewer plots them. *Include motor telemetry* in Train model decides whether the
policy sees them.
**Why.** Joint positions alone cannot tell a gripper closed on a brick from a gripper closed on nothing: both stop at
almost the same position. The load says it plainly: 17-20 holding a Duplo, 2-5 empty. We also wanted to see the
temperature and current of servos that were working hard. Telemetry is recorded separately, so datasets stay
compatible with policies that do not use it. Two follow-up fixes: the recorder no longer looks up each telemetry name
in the observation dict, and a dataset with telemetry still trains ACT when the box is unchecked.
**Gave us.** The grasp detector SortOps uses to decide that a pick succeeded, and thermal monitoring of the arm.
*(`robots/features.py`, `internal_datasets/lerobot/lerobot_dataset.py`, `training/job.py`, episode viewer;
`physicalai/so101-telemetry.patch`)*

### RealSense depth as a recorded camera
**What.** A RealSense set to output depth also records its depth image, aligned to colour, as a camera `<name>_depth`.
It is encoded as a 3-channel image, so it records, previews and trains like any other camera. A UVC camera replugged
under a new `/dev/video` number is found again by its serial.
**Why.** Depth is the cheapest cue for where the brick sits on the table, and the RealSense provides it at no extra
cost. Camera indices changed every time a cable was replugged during the event.
**Gave us.** Datasets with and without depth (the `-nodepth` copies) to compare. Cameras that survive a replug.
*(`workers/camera_worker.py`, `runtime_session/environment.py`; `physicalai/capture-depth.patch`)*

### Lens undistortion
**What.** If `<storage>/camera_calibration/<camera>.xml` exists (an OpenCV calibration file), the camera is undistorted
right after capture, at the same frame size. Calibrations computed from fewer than 10 board views are refused.
**Why.** The wide-angle overhead and wrist cameras bend straight edges. A policy trained on distorted images learns
the distortion of that particular lens.
**Gave us.** Images that do not depend on one lens, at the same shapes, with no change to datasets or policies.
*(`utils/camera_calibration.py`)*

### Camera crop and zoom, settings import/export
**What.** In *Cameras → Edit*, a zoom slider (1× to 4×) and a draggable frame over the preview. The crop is applied
right after the frame is read, so the preview, the recording and inference see the same 640×480 image, and depth gets
the same crop. *Export / Import JSON* saves all the cameras of a project and restores them by name on another machine.
**Why.** The brick takes a few dozen pixels of a full frame. Zooming before recording gives the policy that many more
pixels on the object. Rebuilding the camera setup by hand after a reinstall was slow and error-prone.
**Gave us.** Sharper task-relevant images at the same model input size, and a camera setup we could reproduce.
*(`utils/camera_crop.py`, `services/camera_settings_io.py`, `features/robots/camera-form/*`, `features/cameras/*`)*

### Camera self-healing
**What.** The capture publisher no longer gives up after 5 failed reads. It reopens the device for up to 45 s, with a
RealSense hardware reset on the third try. A watchdog ends the process if a driver call blocks for 20 s, which frees
the device. A subscriber treats a channel as alive only if its control endpoint answers, and starts a new publisher on
a silent one.
**Why.** A RealSense dropping off USB (`-71` errors in the kernel log) left Studio failing with "no publisher
responded" in a loop: 57 failures in 15 minutes, until someone restarted it by hand.
**Gave us.** A simulated 6 s USB cut now recovers by itself (8 s gap), a frozen driver within about 20 s, and a killed
publisher in 0.5 s. Each case was tested with a fake camera. *(`physicalai/capture-selfheal.patch`)*

### Episode tags, markers and dataset copies
**What.** Tag an episode while recording or afterwards ("do nothing", "failed to pick"). Press a marker button during
recording to note a moment ("turn" at frame 95). *Copy dataset* builds a new dataset that keeps only some cameras, the
telemetry or not, some tags, or the segment between two markers. Tags and markers live in the dataset's `meta/` folder
and follow its copies.
**Why.** Failed demonstrations should not be learned, but they are worth keeping. Markers let one long demonstration
(pick, turn, place) feed separate policies. Comparing "with depth" against "without front camera" needed copies of
the same episodes, not new recordings.
**Gave us.** Dataset variants (fewer cameras, no telemetry, failed demos left out, one segment) without re-recording.
*(`internal_datasets/lerobot/episode_tags.py`, `api/dataset.py`, `features/datasets/*`, record mode)*

## Training

### Modal: parallel runs and bigger machines
**What.** Several Modal jobs can run at once. L40S is added to the GPU choices, each job gets 16 vCPU and 64 GB, and the
telemetry field is sent only when it is used.
**Why.** One training at a time wasted the 48 hours. The official Modal image rejected the new telemetry field even when
it was unused.
**Gave us.** Several ACT and π0.5 runs side by side, compared on the same dataset.
*(`services/training_backends/modal.py`, `trainer/modal_app.py`, Train model dialog)*

### Snapshots during training, and stop on a loss target
**What.** *Snapshot every N steps* keeps a checkpoint every N steps. Each checkpoint is exported and imported into
Studio as `<name>-stepN` **while the run continues**, so it can be tried on the arm right away. *Stop when train loss
≤ x / val loss ≤ y* ends a run when it reaches the target. Snapshots download to disk with at most 3 tries, never
during a stop. A job started from another Modal workspace can still be stopped locally.
**Why.** An ACT run takes hours, and the final checkpoint is not always the best one on the robot. Without snapshots,
we only learned at the end of a run that it had overfitted. Downloading 6.6 GB π0.5 checkpoints into `/tmp` (RAM)
filled it and blocked Studio. A run launched from a teammate's Modal account stayed in "Stop pending" forever.
**Gave us.** Real-arm tests of intermediate checkpoints (step 300, 600, …) while training ran. Runs that stop when good
enough. No more `/tmp` saturation.
*(`services/training_snapshots.py`, `training/job.py`, `trainer/modal_app.py`, `workers/training_worker.py`)*

### π0.5 LoRA exports load again
**What.** If a checkpoint has no LoRA weights (they were merged into the base weights at export), loading drops the
empty LoRA adapters instead of failing.
**Why.** π0.5 LoRA exports kept `lora_enabled=True`, so their torch export failed to load in *Run model* (missing
`lora_A` keys).
**Gave us.** Every π0.5 export trained during the event usable, without retraining and without rewriting a checkpoint.
*(`library/src/physicalai/policies/mixins/peft/policy.py`)*

### OpenVINO GPU fallback to f32
**What.** If compiling a model for the GPU fails in fp16 with "ProgramBuilder build failed", compilation is retried with
`INFERENCE_PRECISION_HINT=f32`.
**Why.** Some ACT graphs hit an Intel GPU fp16 kernel error, and the model did not run on the iGPU at all.
**Gave us.** Those models run on the iGPU instead of failing. *(`physicalai/openvino-gpu-f32-fallback.patch`)*

## Evaluation on the arm

### Benchmark trials and Return home
**What.** In the inference view: *Set home here* and *Return home*, a smooth move with bounded time that then holds the
pose. *Benchmark* runs N trials of D seconds. During a trial the model drives. Space starts a trial, R marks Success
and E marks Fail. At the end, the arm freezes, waits for the verdict and returns home on its own. The timer runs on the
server. Statistics: success rate with a Wilson 95 % interval, mean and median time to success, aborted trials counted
apart, CSV export. Leaving a move or trial control keeps the arm holding instead of dropping it.
**Why.** "It works sometimes" is not a measurement. Comparing two models takes many identical trials, and resetting the
arm by hand between trials made them neither identical nor fast.
**Gave us.** Comparable success rates between checkpoints, recorded the same way every time.
*(`runtime_session/controls/trial.py`, `controls/move.py`, `services/benchmarks.py`, `api/benchmarks.py`,
`benchmark-panel.tsx`)*

### Human corrections during inference (HG-DAgger)
**What.** *Pause* holds the follower in place with torque on. Bars show the alignment between leader and follower.
*Take over* is refused until every joint of the leader is within 5 % of the follower's, held for 0.5 s; a take-over
lasts at most 120 s. Only the corrected segment is recorded, tagged `correction`, into `<dataset>-corrections`.
*Fine-tune now* starts a Modal job from the model's own weights (`<model>-ft`, version n+1, parent recorded).
**Why.** Policies fail in states the demonstrations never visited. Recording the human correction from exactly those
states is the most data-efficient fix: HG-DAgger, i.e. imitation learning on the corrections. The alignment gate
prevents a jump when control passes from model to human.
**Gave us.** Corrections recorded where the model failed. A fine-tune that starts at loss 0.136 instead of ≈ 4.5 from
scratch.
*(`runtime_session/controls/hold.py`, `controls/teleop.py`, `session.py`, `corrections-panel.tsx`, `training/job.py`)*

### Arm page: saved poses and teleop outside a session
**What.** A page that moves the follower to named poses (for example `inspect`, the brick held up to the camera) and
teleoperates it from the leader, without opening a session. Every action takes the robots' Studio claim. It refuses
while another program holds the serial port, and it compares pose joint names with or without the `.pos` suffix.
**Why.** Teaching the inspection pose and the home pose needed the arm to stay put at a precise position. Two programs
on one servo bus garble each other's packets.
**Gave us.** The home and inspection poses SortOps and the benchmark use, taught in one place.
*(`services/hackathon_arm.py`, `services/arm_poses.py`, `api/hackathon.py`, `features/hackathon/*`)*

### Torque off when idle
**What.** When nothing has commanded the follower for `idle_torque_off_s` (no teleop, recording or model), its torque is
cut. When a command comes back, the goal is first set to the current position, so the arm cannot jump to an old goal.
**Why.** An arm left holding a pose between sessions heats its servos for nothing.
**Gave us.** Cooler servos between sessions, and no jump when a session starts. *(`workers/robot_worker.py`)*

### Models page by dataset
**What.** One section per dataset, with snapshots and fine-tunes listed under their model, favorites first, and models
renamed in place.
**Why.** With snapshots and fine-tunes, the flat list grew to hundreds of entries.
**Gave us.** The model to test found in seconds. *(`features/models/models-table/*`, `api/models.py`)*
