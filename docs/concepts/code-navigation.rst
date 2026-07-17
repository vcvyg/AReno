Code Navigation
===============

Use this map to choose the first files to read for a small change or review.
It maps behavior to its owner; :file:`AGENTS.md` remains the source for setup,
working rules, and repository conventions.

Main execution paths
--------------------

* CLI dispatch starts at ``ArenoCli`` and ``main`` in
  :file:`areno/cli/main.py`.
* Training enters ``train_command`` and then ``run`` in
  :file:`areno/cli/train.py`. ``run`` resolves the algorithm, dataset, reward,
  loss, and concrete trainer before calling ``fit``.
* Serving enters ``serve_command`` and ``create_app`` in
  :file:`areno/cli/serve.py`. ``create_app`` constructs ``ArenoEngine`` and the
  OpenAI-compatible FastAPI routes.
* SDK callers start with ``Trainer`` in :file:`areno/api/trainer.py`. Its
  ``init``, ``rollout_batch``, ``train``, and ``close`` methods own the public
  lifecycle.

The SDK training runtime crosses these ownership boundaries:

.. code-block:: text

   Trainer
     -> Backend
       -> ArenoBackend
         -> ArenoEngine
           -> ArenoWorker

``Backend`` in :file:`areno/api/backend/base.py` defines the execution
contract. ``ArenoBackend`` in :file:`areno/api/backend/areno/backend.py` packs
public API data for AReno. ``ArenoEngine`` in :file:`areno/engine/api.py`
dispatches rollout and train operations to ``ArenoWorker`` in
:file:`areno/engine/worker.py`. Rank-side rollout is owned by
``InferenceManager`` in :file:`areno/engine/inference.py`; rank-side training
is owned by ``TrainingManager`` in :file:`areno/engine/training.py`, with
shared helpers under :file:`areno/engine/runtime/`. See
:doc:`backend-topology` for the shorter backend-only view.

Registries and extension points
-------------------------------

* Algorithms: ``AlgorithmSpec`` and ``register_algorithm`` in
  :file:`areno/api/algorithms.py`; concrete loops in
  :file:`areno/api/trainers/`; construction in
  :file:`areno/api/trainer_factory.py`.
* Losses: ``sft_loss_fn``, ``dpo_loss_fn``, ``gspo_loss_fn``,
  ``grpo_loss_fn``, and ``ppo_loss_fn`` under
  :file:`areno/api/loss_fns/`.
* Models: the ``ModelAdapter`` contract in :file:`areno/models/base.py`,
  ``register_adapter`` and checkpoint dispatch in
  :file:`areno/models/registry.py`, and one implementation directory per model
  family under :file:`areno/models/`.

Where to start
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 45 29

   * - Task
     - Start here
     - Nearby verification
   * - Change train or serve CLI behavior
     - :file:`areno/cli/train.py` or :file:`areno/cli/serve.py`
     - :file:`tests/test_train_cli_config_cpu.py` or
       :file:`tests/test_serve_cli_cpu.py`
   * - Change SDK rollout or training behavior
     - :file:`areno/api/trainer.py`, then follow the ``Backend`` call
     - :file:`tests/test_trainer_api_cpu.py` and
       :file:`tests/test_protocol_cpu.py`
   * - Add or change an algorithm or loss
     - :file:`areno/api/algorithms.py`, :file:`areno/api/trainers/`, and
       :file:`areno/api/loss_fns/`
     - :file:`tests/test_algorithms_cpu.py` and
       :file:`tests/test_losses_rewards_cpu.py`
   * - Add or change a model family
     - :file:`areno/models/base.py`, :file:`areno/models/registry.py`, then the
       closest family under :file:`areno/models/`
     - :file:`tests/test_registry_cpu.py` and
       :file:`tests/test_registry_discovery_cpu.py`
   * - Change agentic rollout behavior
     - :file:`areno/api/agentic.py`, then a matching task under
       :file:`examples/agentic/`
     - :file:`tests/test_agentic_cpu.py` and the matching example test

Runnable context lives in :file:`examples/`: math RLVR under
:file:`examples/math/`, SFT under :file:`examples/sft/`, and multi-turn agents
under :file:`examples/agentic/`. CPU-safe tests use the ``*_cpu.py`` suffix in
:file:`tests/`. Repeatable model-adaptation guidance lives in
:file:`skills/areno-model-adaptation/SKILL.md`.
