# genui/generators/extensions/genuireinvent/genuimodels/builders.py

from abc import ABC, abstractmethod
from genui.models.genuimodels import bases
from genui.models.models import Model
from ..models import ReinventNet

ALGO_PACKAGE_PATH = "genui.generators.extensions.genuireinvent.genuimodels"

class ReinventBuilder(bases.ProgressMixIn, bases.ModelBuilder, ABC):

    @property
    def corePackage(self):
        from .. import genuimodels
        return genuimodels

    def getY(self):
        return None

    @abstractmethod
    def sample(self, n_samples, from_inputs=None):
        pass


class ReinventNetBuilder(ReinventBuilder):

    def __init__(self, instance: ReinventNet, initial: ReinventNet = None, progress=None, noMonitor=False):
        # ↓↓↓ ключевая правка: передаём свой пакет алгоритмов
        super().__init__(instance, progress, ALGO_PACKAGE_PATH)

        self.initial = initial
        self.progressStages.append("Creating Corpus...")
        self.progressStages.append("Corpus Done.")

    def getX(self, update=True):
        self.recordProgress()
        if update:
            corpus_path = self.instance.prepareData()
            with open(corpus_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            X_train, X_valid = lines, lines
        else:
            X_train, X_valid = self.instance.corpusTrain, self.instance.corpusTrain
        self.recordProgress()
        return X_train, X_valid

    def build(self) -> Model:
        if self.instance.molset:
            return super().build()
        raise NotImplementedError("Building Reinvent network requires a MolSet with an input file.")

    def sample(self, n_samples, from_inputs=None):
        return self.model.sample(n_samples, from_inputs)