import os
import glob
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from ultralytics import YOLO

class RailGuardModelLoader:
    def __init__(self, config_path="configs/serve.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Prioritize environment variable MLFLOW_TRACKING_URI over configs/serve.yaml
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.config["model_registry"]["tracking_uri"])
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self.model_name = os.getenv("MLFLOW_MODEL_NAME", self.config["model_registry"]["name"])
        self.alias = os.getenv("MLFLOW_MODEL_ALIAS", self.config["model_registry"]["alias"])
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.model = None
        self.version = None

    def load_model(self):
        print(f"Connecting to MLflow tracking server at: {self.tracking_uri}")
        
        # 1. Resolve model version details via alias or fallback
        source = None
        try:
            model_version_details = self.client.get_model_version_by_alias(self.model_name, self.alias)
            self.version = model_version_details.version
            source = model_version_details.source
            print(f"Resolved model '{self.model_name}' alias '{self.alias}' to version {self.version}")
        except Exception as e:
            print(f"Notice: Could not resolve alias '{self.alias}' directly: {e}. Will attempt artifact download.")
            self.version = "1"

        # 2. Path resolution / Artifact download
        local_path = None
        
        # Windows/Linux local sqlite workspace path resolution (if running against local sqlite DB)
        if source and source.startswith("runs:/") and self.tracking_uri.startswith("sqlite"):
            try:
                parts = source.split("/")
                run_id = parts[1]
                artifact_path = "/".join(parts[2:])
                run = self.client.get_run(run_id)
                artifact_uri = run.info.artifact_uri
                
                if artifact_uri.startswith("file:"):
                    local_dir = artifact_uri.replace("file:///", "").replace("file://", "").replace("file:", "")
                    if os.name == 'nt' and local_dir.startswith("/"):
                        local_dir = local_dir[1:]
                    if "mlruns" in local_dir:
                        mlruns_idx = local_dir.find("mlruns")
                        local_dir = local_dir[mlruns_idx:]
                        local_dir = os.path.join(os.getcwd(), local_dir)
                    local_path = os.path.normpath(os.path.join(local_dir, artifact_path))
            except Exception as ex:
                print(f"Local sqlite path resolution notice: {ex}")
                local_path = None

        if not local_path or not os.path.exists(local_path):
            print(f"Downloading model artifact from MLflow registry: models:/{self.model_name}@{self.alias}")
            try:
                local_path = download_artifacts(artifact_uri=f"models:/{self.model_name}@{self.alias}")
            except Exception as e:
                print(f"Download by model alias failed: {e}. Trying by model version...")
                if self.version and str(self.version) != "unknown":
                    local_path = download_artifacts(artifact_uri=f"models:/{self.model_name}/{self.version}")
                else:
                    raise e

        # 3. Locate the actual .pt weights file if local_path is a directory
        if os.path.isdir(local_path):
            pt_files = glob.glob(os.path.join(local_path, "**", "*.pt"), recursive=True)
            if pt_files:
                local_path = pt_files[0]
            else:
                raise FileNotFoundError(f"No .pt model file found inside downloaded artifact directory: {local_path}")

        print(f"Loading YOLOv8 model into memory from {local_path}...")
        self.model = YOLO(local_path)
        return self.model

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "version": str(self.version) if self.version else "unknown",
            "alias": self.alias,
            "task": "railway_defect_detection"
        }