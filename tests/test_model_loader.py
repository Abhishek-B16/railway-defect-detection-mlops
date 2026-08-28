import os
import pytest
from unittest.mock import MagicMock, patch
from api.model_loader import RailGuardModelLoader

def test_model_loader_init_defaults():
    loader = RailGuardModelLoader()
    assert loader.model_name == "RailGuard-Detector"
    assert loader.alias == "candidate"

@patch("api.model_loader.download_artifacts")
@patch("api.model_loader.YOLO")
def test_model_loader_alias_success(mock_yolo, mock_download, tmp_path):
    loader = RailGuardModelLoader()
    mock_mv = MagicMock()
    mock_mv.version = 1
    mock_mv.source = "models:/RailGuard-Detector@candidate"
    loader.client.get_model_version_by_alias = MagicMock(return_value=mock_mv)
    
    dummy_pt = tmp_path / "best.pt"
    dummy_pt.write_bytes(b"dummy_pt_weights")
    mock_download.return_value = str(dummy_pt)
    
    model = loader.load_model()
    assert loader.version == "1"
    mock_yolo.assert_called_once_with(str(dummy_pt))

@patch("api.model_loader.download_artifacts")
@patch("api.model_loader.YOLO")
def test_model_loader_version_fallback(mock_yolo, mock_download, tmp_path):
    loader = RailGuardModelLoader()
    # Alias fails with RESOURCE_DOES_NOT_EXIST
    loader.client.get_model_version_by_alias = MagicMock(side_effect=Exception("RESOURCE_DOES_NOT_EXIST"))
    
    # Version succeeds
    mock_mv = MagicMock()
    mock_mv.version = 1
    mock_mv.source = "models:/RailGuard-Detector/1"
    loader.client.get_model_version = MagicMock(return_value=mock_mv)
    
    dummy_pt = tmp_path / "best.pt"
    dummy_pt.write_bytes(b"dummy_pt_weights")
    mock_download.return_value = str(dummy_pt)
    
    model = loader.load_model()
    assert loader.version == "1"
    mock_yolo.assert_called_once_with(str(dummy_pt))

@patch("api.model_loader.download_artifacts")
@patch("api.model_loader.YOLO")
def test_model_loader_run_id_fallback(mock_yolo, mock_download, tmp_path):
    loader = RailGuardModelLoader()
    # Both Alias and Version fail
    loader.client.get_model_version_by_alias = MagicMock(side_effect=Exception("RESOURCE_DOES_NOT_EXIST"))
    loader.client.get_model_version = MagicMock(side_effect=Exception("RESOURCE_DOES_NOT_EXIST"))
    
    dummy_pt = tmp_path / "best.pt"
    dummy_pt.write_bytes(b"dummy_pt_weights")
    mock_download.return_value = str(dummy_pt)
    
    model = loader.load_model()
    assert loader.version == "1"
    mock_yolo.assert_called_once_with(str(dummy_pt))

def test_model_loader_missing_credentials_or_all_failed():
    loader = RailGuardModelLoader()
    loader.client.get_model_version_by_alias = MagicMock(side_effect=Exception("Auth error"))
    loader.client.get_model_version = MagicMock(side_effect=Exception("Auth error"))
    
    with patch("api.model_loader.download_artifacts", side_effect=Exception("Download failed")):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                loader.load_model()
            assert "Failed to resolve or download RailGuard model weights" in str(exc_info.value)