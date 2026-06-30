import os
import joblib
from backend.ml.win_model import WinProbModel
from backend.ml.goals_model import GoalsModel
from backend.config import settings


class ModelLoader:
    def __init__(self):
        self._win_model = None
        self._goals_model = None

    def get_win_model(self) -> WinProbModel:
        if self._win_model is None:
            path = os.path.join(settings.MODEL_DIR, "win_model.joblib")
            if os.path.exists(path):
                clf = joblib.load(path)
                self._win_model = WinProbModel(clf=clf)
            else:
                self._win_model = WinProbModel()
        return self._win_model

    def get_goals_model(self) -> GoalsModel:
        if self._goals_model is None:
            path = os.path.join(settings.MODEL_DIR, "goals_model.joblib")
            if os.path.exists(path):
                params = joblib.load(path)
                self._goals_model = GoalsModel(params=params)
            else:
                self._goals_model = GoalsModel()
        return self._goals_model

    def reload(self):
        self._win_model = None
        self._goals_model = None
