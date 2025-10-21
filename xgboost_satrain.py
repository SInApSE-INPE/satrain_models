import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from ipwgml.input import GMI,Ancillary, Geo, GeoIR
from ipwgml.target import TargetConfig
from ipwgml.pytorch.datasets import SatRainTabular
from ipwgml.pytorch import PytorchRetrieval
from ipwgml.evaluation import Evaluator

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

inputs = [GMI(normalize="minmax", nan=-1.5, include_angles=False)]
target_config = TargetConfig(min_rqi=0.5)
ipwgml_path = "/storage/satrain"
batch_size = 1024

training_data = SatRainTabular(
    base_sensor="gmi",
    geometry="gridded",
    split="training",
    subset="xs",
    retrieval_input=inputs,
    batch_size=batch_size,
    target_config=target_config,
    stack=True,
    ipwgml_path=ipwgml_path,
    download=False,
)
validation_data = SatRainTabular(
    base_sensor="gmi",
    geometry="gridded",
    split="validation",
    subset="xs",
    retrieval_input=inputs,
    batch_size=batch_size,
    target_config=target_config,
    stack=True,
    ipwgml_path=ipwgml_path,
    download=False,
    shuffle=False
)

training_loader = DataLoader(training_data, shuffle=True, batch_size=None, num_workers=4)
validation_loader = DataLoader(validation_data, shuffle=False, batch_size=None)

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

def train_model(model_class, params, gpu_id, X_train, y_train, X_val, y_val):
    print(f"Iniciando treinamento do modelo {model_class.__name__} na GPU {gpu_id}...")
    model_params = params.copy()
    model_params["gpu_id"] = gpu_id
    model = model_class(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    print(f"Treinamento na GPU {gpu_id} finalizado.")
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

training_jobs = [
    {"model_class": xgb.XGBRegressor, "params": params_reg, "y_train": y_train_surface, "y_val": y_val_surface, "name": "surface_precip"},
    {"model_class": xgb.XGBClassifier, "params": params_clf, "y_train": y_train_precip_mask, "y_val": y_val_precip_mask, "name": "probability_of_precip"},
    {"model_class": xgb.XGBClassifier, "params": params_clf, "y_train": y_train_heavy_mask, "y_val": y_val_heavy_mask, "name": "probability_of_heavy_precip"},
]

trained_models = Parallel(n_jobs=3)( 
    delayed(train_model)(
        model_class=job["model_class"], params=job["params"], gpu_id=gpu_id,
        X_train=X_train, y_train=job["y_train"], X_val=X_val, y_val=job["y_val"]
    ) for gpu_id, job in enumerate(training_jobs)
)

model_surface, model_prob_precip, model_prob_heavy = trained_models

class XGBMultiOutput(torch.nn.Module):
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

xgb_multi = XGBMultiOutput({
    "surface_precip": model_surface,
    "probability_of_precip": model_prob_precip,
    "probability_of_heavy_precip": model_prob_heavy
})

wrapped = PytorchRetrieval(
    model=xgb_multi,
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
    download=True
)

evaluator.evaluate(
    retrieval_fn=wrapped,
    input_data_format="tabular",
    batch_size=4048,
    n_processes=1
)

print("\nPrecipitation quantification")
print(evaluator.get_precip_quantification_results(name="XGBOOST (GMI)").T.to_string())
print("\nPrecipitation detection")
print(evaluator.get_precip_detection_results(name="XGBOOST (GMI)").T.to_string())
print("\nHeavy precipitation detection")
print(evaluator.get_heavy_precip_detection_results(name="XGBOOST (GMI)").T.to_string())
