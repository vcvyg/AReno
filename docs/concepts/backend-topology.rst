Backend Topology
================

The current SDK runtime follows one AReno backend path:

.. code-block:: text

   Trainer
     -> Backend
       -> ArenoBackend
         -> ArenoEngine

``Trainer`` is the public coordinator. In ``areno/api/trainer.py``,
``Trainer.init`` resolves a registered backend implementation, while
``Trainer.rollout_token_batch`` and ``Trainer.train`` delegate rollout and
training to that backend.

``Backend`` is the execution contract in ``areno/api/backend/base.py``. Its
``rollout_batch`` and ``train`` methods define the operations required by the
training loop. ``ArenoBackend`` is the registered AReno implementation in
``areno/api/backend/areno/backend.py``.

One colocated engine
--------------------

``ArenoBackend.initialize`` creates one ``ArenoEngine`` and stores it in
``self._engine``. The same engine handles both sides of the loop:

* ``ArenoBackend.rollout_batch`` calls ``ArenoEngine.generate_rollout``.
* ``ArenoBackend.train`` calls ``ArenoEngine.step``.

``ArenoEngine`` is implemented in ``areno/engine/api.py``. It coordinates the
worker cluster used by both rollout and training, so the current backend does
not split those calls across separate engines or external runtimes.

Current runtime scope
---------------------

This path uses AReno-owned engine code under ``areno/engine`` and AReno's CUDA
extensions under ``areno/accel``. The supported runtime described here is the
PyTorch and NVIDIA CUDA environment documented by the project.
