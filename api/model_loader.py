import yaml
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from ultralytics import YOLO

class RailGuardModelLoader:
    def __init__(self, config_path="configs/serve.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        mlflow.set_tracking_uri(self.config["model_registry"]["tracking_uri"])
        self.model_name = self.config["model_registry"]["name"]
        self.alias = self.config["model_registry"]["alias"]
        self.client = MlflowClient()
        self.model = None
        self.version = None

    def load_model(self):
        # Resolve the alias to a specific version
        model_version_details = self.client.get_model_version_by_alias(self.model_name, self.alias)
        self.version = model_version_details.version
        
        import os
        import shutil
        print(f"Downloading model from registry alias '{self.alias}'...")
        
        # Windows MLflow artifact download bug workaround
        source = model_version_details.source
        if source.startswith("runs:/"):
            parts = source.split("/")
            run_id = parts[1]
            artifact_path = "/".join(parts[2:])
            run = self.client.get_run(run_id)
            artifact_uri = run.info.artifact_uri
            
            # Convert file URI to local path
            if artifact_uri.startswith("file:"):
                local_dir = artifact_uri.replace("file:///", "").replace("file://", "").replace("file:", "")
                if os.name == 'nt' and local_dir.startswith("/"):
                    local_dir = local_dir[1:] # strip leading slash on windows
                
                # Cross-platform Docker Fix: MLflow SQLite stores absolute Windows paths.
                # If running in a Linux container, this will fail. We rewrite it to a relative path.
                if "mlruns" in local_dir:
                    mlruns_idx = local_dir.find("mlruns")
                    local_dir = local_dir[mlruns_idx:]
                    local_dir = os.path.join(os.getcwd(), local_dir)
                    
                local_path = os.path.normpath(os.path.join(local_dir, artifact_path))
            else:
                local_path = download_artifacts(f"models:/{self.model_name}@{self.alias}")
        else:
            local_path = download_artifacts(f"models:/{self.model_name}@{self.alias}")
            
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
