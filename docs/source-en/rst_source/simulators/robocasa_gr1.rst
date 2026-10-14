RoboCasa GR1
===========

Run a native GR1 tabletop episode with an Agent selecting Joy-Brain actions and
joint-target corrections. Select ``--robot robocasa_gr1``; this backend is
independent of RoboCasa365/PandaOmron and RoboTwin.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      Original Joy-Brain VLA with ``robocasa_gr1`` statistics.

   .. grid-item-card:: Planners

      RPent tool-calling planners, including the API planner.

   .. grid-item-card:: Tasks

      Native GR1 tabletop ID and OOD splits.

   .. grid-item-card:: Hardware

      CUDA model inference and MuJoCo EGL rendering; measure memory before concurrency.

Tasks
~~~~~

.. list-table:: Original Joy-Brain GR1 task lists
   :header-rows: 1

   * - ``--eval-type``
     - Scope
   * - ``id``
     - 24 in-distribution tasks.
   * - ``ood_object_appearance``
     - 18 object-appearance tasks.
   * - ``ood_container_combination``
     - 14 container-combination tasks.
   * - ``ood_object_type``
     - 32 object-type tasks.

Task membership comes from the configured Joy-Brain checkout's
``benchmark/Robocasa_GR1/task_lists.py``. These are not RoboCasa365 Target50 tasks.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Native contracts
   :header-rows: 1

   * - Interface
     - Behavior
   * - Observation
     - Native ego RGB, task language, joints and episode status. VLA receives raw44 state and one image; the original server owns crop and normalization.
   * - Action
     - 29 effective joint targets: left arm7, left hand6, right arm7, right hand6, waist3. The raw44 reserved slots follow training's mapping, not arbitrary padding.
   * - Reward / success
     - Latch native ``info.success`` or native termination, matching the original GR1 client. Budget truncation and planner prose do not imply success.
   * - Task prompt
     - The native observation's task language is authoritative.

Installation and Resources
--------------------------

Use three independently configured interpreters: RPent's Agent environment,
the original Joy-Brain model environment, and the native GR1 simulator environment.
Install only the lightweight Agent-side extra in the RPent environment:

.. code-block:: bash

   JD_SOURCE=https://mirrors.jd.com/pypi/simple
   /path/to/rpent/python -m pip install -e '.[robocasa-gr1]' --index-url "$JD_SOURCE"

Reuse the existing ``robocasa-gr1-tabletop-tasks`` fork and assets. Its native
runtime requires MuJoCo 3.2.6 and robosuite 1.5.0/1.5.1; use its supported NumPy
version. Do not install that fork into the PandaOmron environment.
Reuse a complete, explicit GR1 checkpoint containing ``config.json``,
``stats.json`` and ``model.safetensors``, plus the VLM/processor files referenced
by its config. The backend does not choose a checkpoint automatically.

.. code-block:: bash

   export JOYBRAIN_REPO_ROOT=/path/to/Joy-Brain-vla
   export JOYBRAIN_MODEL_PATH=/path/to/explicit/gr1/checkpoint/pretrained_model
   export JOYBRAIN_PYTHON=/path/to/model-env/bin/python
   export ROBOCASA_GR1_REPO="$JOYBRAIN_REPO_ROOT/third_party/robocasa-gr1-tabletop-tasks"
   export ROBOCASA_GR1_PYTHON=/path/to/gr1-env/bin/python
   export ROBOCASA_GR1_LD_LIBRARY_PATH=/path/to/gr1-env/lib:/path/to/verified/gl-libraries

Verify the GL library paths on the target node. GPU ordinals refer to the
allocated node. EGL defaults to the single numeric Env CUDA device;
set ``--egl-device-id`` explicitly for UUID or multiple-device masks.
Use an allocated compute node, not a login node.

Run a Task
----------

First run the bounded component check with a new output directory:

.. code-block:: bash

   /path/to/rpent/python -m tests.e2e_tests.robocasa_gr1.smoke \
     --task-name PnPCupToDrawerClose --seed 7 --cuda-device 0 \
     --max-episode-steps 17 --output-dir logs/gr1-components

This checks native reset/RGB, inference without actions, a native joint hold and
one learned action window. It makes **zero Agent API calls**. It is an explicit
entry point, not a target of ``run_gpu_suite.sh``.

After confirming API credentials and permission to transmit simulation images,
use the normal Agent entry point:

.. code-block:: bash

   /path/to/rpent/python -m rpent.cli.main --robot robocasa_gr1 \
     --planner api --model openai:YOUR_MODEL --base-url "$AGENT_BASE_URL" \
     --memory-profile local --task-name PnPCupToDrawerClose --seed 7 \
     --cuda-device 0 --max-episode-steps 32 --replan-steps 16 \
     --max-turns 6 --max-tokens 2048 --output-dir logs/gr1-agent

Supply the provider credential through the environment, not a command-line
argument or tracked file. See :doc:`../guides/configure_planner` for planner configuration.
``gr1_act(chunks=1..4)`` uses the checkpoint's full predictions and executes
``--replan-steps`` actions per window, capped by the remaining native-step budget.
Its default is 16, matching the original launcher. The model chunk is unchanged.
``set_joint_targets`` exposes only native joint groups and holds unmentioned groups.

View Results
------------

``result.json`` reports native success separately from the planner's finish.
Uncertain execution or failed recording sets ``measurement_valid=false`` and
leaves ``official_success`` unknown rather than inventing a task verdict.
``states.json``, ego images and ``episode.mp4`` describe the observed trajectory;
``trace/step_*/transition.json`` records actual actions and native verdicts.
``proposal_*.npy`` contains full predictions, not necessarily executed actions.
Server logs and ``policy-metadata.json`` identify the native runtimes and model.
The component check also records phase time and device-wide sampled peak memory.
Component success does not establish task success, Agent speed or benchmark score.

Task Memory
-----------

Only ``--memory-profile local`` is supported. No PandaOmron memory is downloaded
or reused. Exploration and multi-episode worker reuse are not implemented for this
backend. Each environment owns one recorded episode and cannot reset it again.

Diagnostics
-----------

For a borrowed server, pass ``--vla-endpoint ws://HOST:PORT`` and the same explicit
checkpoint path. The client validates GR1 metadata and never stops a borrowed
server. Use ``--env-cuda-device`` and ``--vla-cuda-device`` instead of
``--cuda-device`` for split deployment. This does not implement a parallel layout
launcher or establish same-card capacity.

Check ``env-server.log`` for missing assets or EGL libraries and
``vla-server.log`` for missing VLM files or CUDA errors. Wrong state/action
dimensions fail rather than being cropped. Uncertain action execution or failed
recording blocks further mutations; start a new run rather than replaying actions.
