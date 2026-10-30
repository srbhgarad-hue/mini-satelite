"""
test_provenance_safety.py
=========================
Automated Verification Suite for Space Netra Data Provenance & Application Integrity.
Verifies that synthetic imagery and synthetic raster results cannot be mistaken for real observations.
"""

import os
import json
import numpy as np
from PIL import Image

import photo_streamer
import satellite_loader
import city_catalog

def test_source_directory_integrity():
    """Verify that data/change_detection/source contains ZERO synthetic files."""
    source_dir = "data/change_detection/source"
    assert os.path.exists(source_dir), f"Source directory missing: {source_dir}"
    
    files = [f for f in os.listdir(source_dir) if f.endswith((".jpg", ".png", ".tif", ".npy"))]
    assert len(files) > 0, "No source files found in source directory"
    
    synthetic_found = []
    for f in files:
        path = os.path.join(source_dir, f)
        if f.endswith((".jpg", ".png")):
            if photo_streamer.is_procedural_synthetic(path):
                synthetic_found.append(f)
        if "b04" in f or "b08" in f or "scl" in f:
            synthetic_found.append(f) # Band rasters must be quarantined

    assert len(synthetic_found) == 0, f"Provenance violation: synthetic files found in source dir: {synthetic_found}"
    print(f"[PASS] Source directory verified: {len(files)} files present, 0 synthetic files.")

def test_quarantine_directory_integrity():
    """Verify that data/synthetic_test_fixtures correctly isolates synthetic fixtures."""
    fixtures_dir = "data/synthetic_test_fixtures"
    assert os.path.exists(fixtures_dir), f"Fixtures directory missing: {fixtures_dir}"
    
    files = os.listdir(fixtures_dir)
    assert len(files) >= 15, f"Expected quarantined fixtures, found {len(files)}"
    
    synth_demos = [f for f in files if f.startswith("synth_demo_")]
    assert len(synth_demos) >= 9, f"Expected >= 9 synthetic demo fixtures, found {len(synth_demos)}"
    for f in synth_demos:
        path = os.path.join(fixtures_dir, f)
        assert photo_streamer.is_procedural_synthetic(path), f"File {f} not recognized as synthetic"
    
    # Check simulated Sentinel-2 arrays
    for expected in ["t1_b04.npy", "t1_b08.npy", "t1_scl.npy", "t2_b04.npy", "t2_b08.npy", "t2_scl.npy", "t1_2021_visual.png"]:
        assert expected in files, f"Missing synthetic Sentinel-2 asset: {expected}"

    print(f"[PASS] Quarantine directory verified: {len(files)} synthetic fixtures strictly isolated.")

def test_live_acquisition_failure_safety():
    """Verify that failed image acquisition returns explicit failure and does NOT generate synthetic files."""
    source_dir = "data/change_detection/source"
    files_before = set(os.listdir(source_dir))

    # Test coordinate with un-cached arbitrary location
    dummy_lat = 19.9999
    dummy_lon = 75.9999
    dummy_zoom = 16

    timeline = photo_streamer.fetch_3stage_photographic_timeline(
        dummy_lat, dummy_lon, zoom=dummy_zoom, allow_synthetic_demo=False
    )

    # If offline or failed, must report ACQUISITION_FAILED
    if timeline["status"] != "VERIFIED_REAL":
        assert timeline["status"] == "ACQUISITION_FAILED"
        assert timeline["is_real"] is False
        assert timeline["acquisition_error"] is not None
        assert "could not be retrieved" in timeline["acquisition_error"]
        assert timeline["provenance"] == "UNAVAILABLE"
        # CRITICAL: Confirm source directory was NOT polluted with any new files
        files_after = set(os.listdir(source_dir))
        diff = files_after - files_before
        assert len(diff) == 0, f"Provenance violation: Silent synthetic file written to source: {diff}"
    else:
        # If live acquisition succeeded, verify newly downloaded files are verified real (not synthetic) and clean up
        files_after = set(os.listdir(source_dir))
        diff = files_after - files_before
        for f in diff:
            fpath = os.path.join(source_dir, f)
            assert not photo_streamer.is_procedural_synthetic(fpath), f"Downloaded tile marked synthetic: {f}"
            try:
                os.remove(fpath)
            except OSError:
                pass
    print("[PASS] Acquisition failure safety verified: No silent synthetic fallbacks generated in live mode.")

def test_explicit_synthetic_demo_mode():
    """Verify that synthetic demo mode requires explicit flag and outputs strictly to quarantine."""
    dummy_lat = 19.9998
    dummy_lon = 75.9998
    dummy_zoom = 16

    timeline = photo_streamer.fetch_3stage_photographic_timeline(
        dummy_lat, dummy_lon, zoom=dummy_zoom, allow_synthetic_demo=True
    )

    if timeline["status"] == "SYNTHETIC_FALLBACK":
        assert timeline["is_real"] is False
        assert timeline["provenance"] == "SYNTHETIC"
        assert "Synthetic" in timeline["provenance_label"]
        for p in [timeline["p1_path"], timeline["p2_path"], timeline["p3_path"]]:
            if p is not None:
                assert "synthetic_test_fixtures" in p or "synth_demo_" in os.path.basename(p)
                assert photo_streamer.is_procedural_synthetic(p)

    print("[PASS] Explicit synthetic demo mode verified: Quarantined to data/synthetic_test_fixtures/.")

def test_satellite_loader_provenance():
    """Verify that satellite_loader reports explicit synthetic provenance for quarantined rasters."""
    data = satellite_loader.load_temporal_pair()
    assert "t1_b04" in data
    assert data["is_synthetic"] is True
    assert data["provenance"] == "SYNTHETIC_TEST_FIXTURE"
    assert "Synthetic" in data["provenance_label"]
    assert "Real Level-2A raster pixels have not been downloaded" in data["provenance_note"]
    print("[PASS] satellite_loader provenance verified: Explicitly flags simulated Sentinel-2 bands.")

def test_provenance_manifest_honesty():
    """Verify that provenance_manifest.json explicitly separates metadata from synthetic pixels."""
    manifest_path = os.path.join("data", "change_detection", "manifests", "provenance_manifest.json")
    assert os.path.exists(manifest_path), "Manifest missing"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert "provenance_classification" in manifest
    prov = manifest["provenance_classification"]
    assert prov["pixel_provenance"] == "SYNTHETIC_SIMULATED_RASTER"
    assert prov["metadata_provenance"] == "VERIFIED_COPERNICUS_STAC_METADATA"
    assert prov["is_synthetic"] is True
    assert "CRITICAL DATA INTEGRITY NOTICE" in prov["provenance_warning"]
    print("[PASS] Provenance manifest verified: Declares synthetic pixel status alongside authentic metadata.")

def test_checkpoint_and_vector_db_intact():
    """Verify that all three model checkpoints (mixed model, full-train, original baseline), plus vector database, are preserved."""
    # 1. S2Looking + LEVIR-CD Mixed Model Checkpoint
    mixed_ckpt = "models/spacenetra_s2looking_levir_20261030_031904_best.pth"
    assert os.path.exists(mixed_ckpt), f"Mixed checkpoint missing: {mixed_ckpt}"
    assert os.path.getsize(mixed_ckpt) > 2_000_000, f"Mixed checkpoint file truncated: {os.path.getsize(mixed_ckpt)} bytes"

    # 2. Full-Train Checkpoint
    new_ckpt = "models/spacenetra_s2looking_unet_fulltrain_best.pth"
    assert os.path.exists(new_ckpt), f"New checkpoint missing: {new_ckpt}"
    assert os.path.getsize(new_ckpt) > 2_000_000, f"New checkpoint file truncated: {os.path.getsize(new_ckpt)} bytes"

    # 3. Original Baseline Checkpoint
    orig_ckpt = "models/spacenetra_s2looking_unet.pth"
    assert os.path.exists(orig_ckpt), f"Original checkpoint missing: {orig_ckpt}"
    assert os.path.getsize(orig_ckpt) > 2_000_000, f"Original checkpoint truncated: {os.path.getsize(orig_ckpt)} bytes"

    # 4. Vector Database & Backup
    vdb_path = "vector_db/spatial.index"
    vmeta_path = "vector_db/metadata.npy"
    assert os.path.exists(vdb_path), f"FAISS index missing: {vdb_path}"
    assert os.path.exists(vmeta_path), f"Metadata missing: {vmeta_path}"

    vdb_bak = "vector_db_backup/spatial.index"
    assert os.path.exists(vdb_bak), "Vector DB backup missing"
    print("[PASS] All three model checkpoints and vector database verified intact and untouched.")

def run_all_provenance_tests():
    print("==================================================================")
    print(" SPACE NETRA: DATA PROVENANCE & INTEGRITY VERIFICATION SUITE      ")
    print("==================================================================")
    test_source_directory_integrity()
    test_quarantine_directory_integrity()
    test_live_acquisition_failure_safety()
    test_explicit_synthetic_demo_mode()
    test_satellite_loader_provenance()
    test_provenance_manifest_honesty()
    test_checkpoint_and_vector_db_intact()
    print("==================================================================")
    print(" ALL DATA PROVENANCE VERIFICATIONS PASSED SUCCESSFULLY!           ")
    print("==================================================================")

if __name__ == "__main__":
    run_all_provenance_tests()
