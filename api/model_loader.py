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
            
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.config["model_registry"]["tracking_uri"])
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self.model_name = os.getenv("MLFLOW_MODEL_NAME", self.config["model_registry"]["name"])
        self.alias = os.getenv("MLFLOW_MODEL_ALIAS", self.config["model_registry"]["alias"])
        self.version_env = os.getenv("MLFLOW_MODEL_VERSION", "1")
        self.run_id_env = os.getenv("MLFLOW_MODEL_RUN_ID", "80842181f7e345bb85b961dc2fdcfd3c")
        
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.model = None
        self.version = "unknown"

    def load_model(self):
        print("Connecting to DagsHub MLflow...")
        print(f"Tracking URI: {self.tracking_uri}")
        
        local_path = None

        # Strategy 1: Registered model + candidate alias
        print(f"Trying registered model alias '{self.alias}' for '{self.model_name}'...")
        try:
            mv = self.client.get_model_version_by_alias(self.model_name, self.alias)
            self.version = str(mv.version)
            source = mv.source
            print(f"Alias lookup succeeded: Version {self.version}")
            
            # Check local sqlite uri workaround if running locally
            if source and source.startswith("runs:/") and self.tracking_uri.startswith("sqlite"):
                local_path = self._resolve_local_sqlite_path(source)
                
            if not local_path or not os.path.exists(local_path):
                print(f"Downloading from source: {source}")
                local_path = download_artifacts(artifact_uri=source)
                
        except Exception as e:
            print(f"Alias lookup failed ({e}). Trying model version...")

        # Strategy 2: Model version fallback
        if not local_path or not os.path.exists(local_path):
            target_version = self.version_env if self.version == "unknown" else self.version
            print(f"Trying model version fallback '{target_version}' for '{self.model_name}'...")
            try:
                mv = self.client.get_model_version(self.model_name, target_version)
                self.version = str(mv.version)
                source = mv.source
                
                if source and source.startswith("runs:/") and self.tracking_uri.startswith("sqlite"):
                    local_path = self._resolve_local_sqlite_path(source)
                    
                if not local_path or not os.path.exists(local_path):
                    print(f"Downloading from source: {source}")
                    local_path = download_artifacts(artifact_uri=source)
            except Exception as e:
                print(f"Model version lookup failed ({e}). Trying run artifact fallback...")

        # Strategy 3: Run artifact fallback
        if not local_path or not os.path.exists(local_path):
            print(f"Trying run artifact fallback for run ID '{self.run_id_env}'...")
            try:
                local_path = download_artifacts(artifact_uri=f"runs:/{self.run_id_env}/model")
                if self.version == "unknown":
                    self.version = self.version_env
            except Exception as e:
                print(f"Run artifact fallback failed: {e}")

        # Strategy 4: Local weight fallback (e.g. yolov8n.pt if present)
        if not local_path or not os.path.exists(local_path):
            local_fallback = os.path.join(os.getcwd(), "yolov8n.pt")
            if os.path.exists(local_fallback):
                print(f"Fallback to local weight file: {local_fallback}")
                local_path = local_fallback
                self.version = "1"

        if not local_path or not os.path.exists(local_path):
            raise RuntimeError("Failed to resolve or download RailGuard model weights from any MLflow strategy or local path.")

        # Resolve .pt file if local_path is a directory
        if os.path.isdir(local_path):
            pt_files = glob.glob(os.path.join(local_path, "**", "*.pt"), recursive=True)
            if pt_files:
                local_path = pt_files[0]
            else:
                raise FileNotFoundError(f"No .pt model weight file found in downloaded directory: {local_path}")

        print(f"Model downloaded successfully to: {local_path}")
        print("Loading YOLO model into memory...")
        self.model = YOLO(local_path)
        print("Model ready.")
        return self.model

    def _resolve_local_sqlite_path(self, source):
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
                return os.path.normpath(os.path.join(local_dir, artifact_path))
        except Exception as ex:
            print(f"Local path resolution notice: {ex}")
        return None

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "version": str(self.version) if self.version else "unknown",
            "alias": self.alias,
            "task": "railway_defect_detection"
        }