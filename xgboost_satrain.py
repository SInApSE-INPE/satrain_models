import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import DataLoader
from satrain.input import GMI
from satrain.target import TargetConfig
from satrain.pytorch.datasets import SatRainTabular
from satrain.pytorch import PytorchRetrieval
from satrain.evaluation import Evaluator

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

inputs = [GMI(normalize="minmax", nan=-1.5, include_angles=False)]
target_config = TargetConfig(min_rqi=0.5)
satrain_path = "/prj/cptec"
geometry = "gridded"
batch_size = 1024
subset = "xs"

training_data = SatRainTabular(
    base_sensor="gmi",
    geometry=geometry,
    split="training",
    subset=subset,
    retrieval_input=inputs,
    batch_size=batch_size,
    target_config=target_config,
    stack=True,
    data_path=satrain_path,
    download=False,
)
validation_data = SatRainTabular(
    base_sensor="gmi",
    geometry=geometry,
    split="validation",
    subset=subset,
    retrieval_input=inputs,
    batch_size=batch_size,
    target_config=target_config,
    stack=True,
    data_path=satrain_path,
    download=False
)

training_loader = DataLoader(training_data, shuffle=True, batch_size=None, num_workers=1)
validation_loader = DataLoader(validation_data, shuffle=False, batch_size=None)

# Converte para numpy
def load_limited(loader):
    X_list, y_surface_list, y_precip_mask_list, y_heavy_mask_list = [], [], [], []

    for x, y in loader:
        x = x.numpy()
        surface_precip = y["surface_precip"].numpy()
        precip_mask = y["precip_mask"].numpy().astype(np.int32)
        heavy_precip_mask = y["heavy_precip_mask"].numpy().astype(np.int32)

        X_list.append(x)
        y_surface_list.append(surface_precip)
        y_precip_mask_list.append(precip_mask)
        y_heavy_mask_list.append(heavy_precip_mask)

    X = np.concatenate(X_list, axis=0)
    y_surface = np.concatenate(y_surface_list, axis=0)
    y_precip_mask = np.concatenate(y_precip_mask_list, axis=0)
    y_heavy_mask = np.concatenate(y_heavy_mask_list, axis=0)
    return X, y_surface, y_precip_mask, y_heavy_mask

X_train, y_train_surface, y_train_precip_mask, y_train_heavy_mask = load_limited(training_loader)
X_val, y_val_surface, y_val_precip_mask, y_val_heavy_mask = load_limited(validation_loader)

def train_model(model_class, params, X_train, y_train, X_val, y_val):
    print(f"Iniciando treinamento do modelo {model_class.__name__} ...")
    model = model_class(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    print(f"Treinamento finalizado para {model_class.__name__}.")
    return model

params_reg = {
    'n_estimators': 2428, 'learning_rate': 0.008747, 'max_depth': 11, 'gamma': 0.1303,
    'subsample': 0.7792, 'colsample_bytree': 0.8892, 'reg_alpha': 0.0006835,
    'reg_lambda': 9.5364, "tree_method": "gpu_hist", "objective": "reg:squarederror", "random_state": 42
}

params_clf = {
    'n_estimators': 2428, 'learning_rate': 0.008747, 'max_depth': 11, 'gamma': 0.1303,
    'subsample': 0.7792, 'colsample_bytree': 0.8892, 'reg_alpha': 0.0006835,
    'reg_lambda': 9.5364, "tree_method": "gpu_hist", "objective": "binary:logistic", "random_state": 42
}

model_surface = train_model(xgb.XGBRegressor, params_reg, X_train, y_train_surface, X_val, y_val_surface)
model_prob_precip = train_model(xgb.XGBClassifier, params_clf, X_train, y_train_precip_mask, X_val, y_val_precip_mask)
model_prob_heavy = train_model(xgb.XGBClassifier, params_clf, X_train, y_train_heavy_mask, X_val, y_val_heavy_mask)

class XGBoostOutput(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        return {
            "surface_precip": torch.from_numpy(self.models["surface_precip"].predict(x_np)).unsqueeze(-1).float(),
            "probability_of_precip": torch.from_numpy(self.models["probability_of_precip"].predict(x_np)).unsqueeze(-1).float(),
            "probability_of_heavy_precip": torch.from_numpy(self.models["probability_of_heavy_precip"].predict(x_np)).unsqueeze(-1).float(),
        }

model = XGBoostOutput({
    "surface_precip": model_surface,
    "probability_of_precip": model_prob_precip,
    "probability_of_heavy_precip": model_prob_heavy
})

xgboost_retrieval = PytorchRetrieval(
    model=model,
    retrieval_input=inputs,
    stack=True,
    device=torch.device("cuda"),
    logits=False
)

evaluator = Evaluator(
    base_sensor="gmi",
    domain="korea",
    geometry="gridded",
    retrieval_input=inputs,
    data_path=satrain_path,
    download=False
)

evaluator.evaluate(
    retrieval_fn=xgboost_retrieval,
    input_data_format="tabular",
    batch_size=4048,
    n_processes=1
)

print("\nPrecipitation quantification")
print(evaluator.get_precip_quantification_results(name="XGBOOST (GMI)").T.to_string())
print("\nPrecipitation detection")
print(evaluator.get_precip_detection_results(name="XGBOOST (GMI)").T.to_string())
print("\nProbabilistic precipitation detection")
print(evaluator.get_prob_precip_detection_results(name="XGBOOST (GMI)").T.to_string())
print("\nHeavy precipitation detection")
print(evaluator.get_heavy_precip_detection_results(name="XGBOOST (GMI)").T.to_string())
print("\nHeavy probabilistic precipitation detection")
print(evaluator.get_prob_heavy_precip_detection_results(name="XGBOOST (GMI)").T.to_string())
