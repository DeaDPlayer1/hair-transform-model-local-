# HairTryOn Robustness Campaign — Issue Log

| Issue ID | Failing Pair / Module | Pipeline Stage | Evidence | Root Cause Hypothesis | Status | Fix Applied | Retest Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ISSUE-001** | `tests/test_transfer.py` | Unit Testing / Smoke | `FileNotFoundError: assets/download.jpg` | Hardcoded legacy asset filenames (`download.jpg`, `image.png`) in test setup | OPEN | Update test to discover available assets or use synthetic portraits | Pending |
| **ISSUE-002** | `scripts/test_local_generative_transfer.py` vs `tests/test_local_generative_transfer.py` | Test Discovery | `pytest: import file mismatch` | Pytest discovers both files because they share the same basename `test_local_generative_transfer.py` | OPEN | Configure `pytest.ini` testpaths or rename script | Pending |
| **ISSUE-003** | `scripts/try_hair.py`, `scripts/try_hair_two_stage.py` | CLI & Imports | 30+ Streamlit warnings on CLI execution | Direct import `from app import segment_hair` loads `app.py` top-level Streamlit code (`st.set_page_config`) | OPEN | Modularize `segment_hair` into `src/hair_segmentation.py` so scripts do not import Streamlit | Pending |
| **ISSUE-004** | `src/two_stage_transfer.py` | Stage 2 Compositing | Outside-envelope pixel restoration uses `broad_mask_np == 0` without unioning `stage1_edit_mask` | If Stage 1 cleared hair outside Stage 2's broad envelope, the untouched original target hair is restored over the result | OPEN | Union Stage 1 and Stage 2 envelopes before outside pixel restoration | Pending |
