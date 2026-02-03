# genui/src/genui/generators/extensions/genuireinvent/genuimodels/algorithms.py

from __future__ import annotations

from abc import ABC
import pickle

from genui.models.genuimodels import bases
from genui.models.models import ModelFileFormat


class ReinventAlgorithm(bases.Algorithm, ABC):
    """
    Thin adapter around the external REINVENT CLI.

    Training is delegated to the model facade returned by ReinventNet.getModel().
    The builder/GenUI training pipeline will call Algorithm.fit(); internally that
    triggers the CLI-based transfer learning in ReinventNet.run_transfer_learning().
    """

    def __init__(self, builder, callback=None):
        super().__init__(builder, callback)
        self.train_params = {}
        self._model = None

    @classmethod
    def getFileFormats(cls, attach_to=None):
        pkg, _ = ModelFileFormat.objects.get_or_create(
            fileExtension=".pkg",
            defaults={
                "description": "Serialized metadata for REINVENT (e.g., produced checkpoint path).",
            },
        )
        if attach_to:
            cls.attachToInstance(attach_to, [pkg], attach_to.fileFormats)

    @classmethod
    def getModes(cls):
        return [cls.GENERATOR]

    @property
    def model(self):
        return self._model

    def predict(self, X):
        # REINVENT net here is used as a generator; prediction isn't applicable.
        return [], None

    def sample(self, n_samples, from_inputs=None):
        raise NotImplementedError("Sampling is not implemented for the REINVENT CLI adapter.")

    def getSerializer(self):
        """
        Persist minimal metadata needed to re-associate a trained artifact with this Algorithm.

        NOTE:
        - The real checkpoint is managed by ReinventNet.checkpointFile (AUX ModelFile).
        - This .pkg payload is primarily for GenUI's generic model-serialization contract.
        """
        def _save(path: str):
            payload = {}
            try:
                # facade.getModel() returns {"checkpoint": "..."} in your setup
                payload = self.model.getModel() if self.model else {}
            except Exception:
                payload = {}
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return _save

    def getDeserializer(self):
        """
        Restore the facade. Nothing is loaded into RAM.
        """
        def _load(path: str):
            try:
                with open(path, "rb") as f:
                    _ = pickle.load(f)  # kept for compatibility; optional
            except Exception:
                pass

            # facade may be lazily created by subclasses; if present, let it no-op load.
            if self.model:
                try:
                    self.model.loadStatesFromFile(path)
                except Exception:
                    pass
            return self.model
        return _load


class ReinventNetwork(ReinventAlgorithm):
    name = "ReinventNet"

    def __init__(self, builder, callback=None):
        super().__init__(builder, callback)
        # builder.instance is a ReinventNet (Django model) which returns the CLI facade
        self._model = self.builder.instance.getModel()

    def fit(self, X=None, y=None):
        # Refresh facade (safe in case builder.instance mutated)
        self._model = self.builder.instance.getModel()

        # Triggers: prepareData() + run_transfer_learning() (subprocess)
        self._model.fit(X=X, y=y)

        # Inform pipeline that "an epoch-like thing happened"
        if self.callback:
            self.callback(None)

        return self