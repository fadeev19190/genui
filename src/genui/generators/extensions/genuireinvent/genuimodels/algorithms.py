# genui/src/genui/generators/extensions/genuireinvent/genuimodels/algorithms.py
from abc import ABC

import torch  # only needed for the serializer; safe to keep
from genui.models.genuimodels import bases
from genui.models.models import ModelFileFormat


class ReinventAlgorithm(bases.Algorithm, ABC):
    """
    Thin adapter around the external REINVENT CLI.
    - No dataloaders, no in-RAM training loops.
    - Training is delegated to the model facade returned by ReinventNet.getModel().
    """

    def __init__(self, builder, callback=None):
        super().__init__(builder, callback)
        # minimal progress – DrugEx-style stage names are set in the builder
        self.train_params = {}

    @classmethod
    def getFileFormats(cls, attach_to=None):
        # keep a single pkg format; this artifact typically stores a tiny dict
        # with the produced checkpoint path (see getSerializer()).
        pkg = ModelFileFormat.objects.get_or_create(
            fileExtension=".pkg",
            description="State of a neural network built with PyTorch (or path to external checkpoint)."
        )[0]
        if attach_to:
            cls.attachToInstance(attach_to, [pkg], attach_to.fileFormats)

    @classmethod
    def getModes(cls):
        return [cls.GENERATOR]

    @property
    def model(self):
        return self._model

    def predict(self, X):
        return [], None

    # REINVENT doesn't support in-RAM sampling through this adapter; you can
    # wire a sampling endpoint later if you add a runtime that loads checkpoints.
    def sample(self, n_samples, from_inputs=None):
        raise NotImplementedError("Sampling is not implemented for the REINVENT CLI adapter.")

    def getSerializer(self):
        # Persist the minimal payload that our facade's getModel() returns
        # (e.g., {"checkpoint": "/path/to/reinvent_<pk>_tl.model"})
        return lambda path: torch.save(self.model.getModel(), path)

    def getDeserializer(self):
        # There is nothing to restore into RAM. Keep as a no-op that just returns the facade.
        def _noop(path):
            self.model.loadStatesFromFile(path)
            return self.model
        return _noop


class ReinventNetwork(ReinventAlgorithm):
    name = "ReinventNet"
    # No algorithm-level hyperparameters here; your TrainingStrategy (ReinventNetTraining)
    # already holds epochs/batch sizes compatible with REINVENT 4.x.

    def __init__(self, builder, callback=None):
        super().__init__(builder, callback)
        # get the facade that knows how to run REINVENT and where artifacts live
        self._model = self.builder.instance.getModel()

    def fit(self, X=None, y=None):
        """
        Delegate to the facade. It will:
          - self.builder.instance.prepareData()
          - self.builder.instance.run_transfer_learning()
          - store a training log to AUX ModelFile
        """
        # ensure we hold a fresh wrapper and just call its fit()
        self._model = self.builder.instance.getModel()
        self._model.fit()
        # record progress stage end (builder’s monitor)
        if self.callback:
            self.callback(None)