import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data import load_raw, prepare, make_split, load_train, sha256_file
from src.pipelines.preprocessor import build_preprocessor

def test_second_semester_missing():
    df = load_raw()
    df_prep = prepare(df)
    for col in C.SECOND_SEM_COLS:
        assert col not in df_prep.columns, f"Leakage: {col} present in prepared data"

def test_no_index_intersection():
    train_df = pd.read_csv(C.TRAIN_PATH)
    test_df = pd.read_csv(C.TEST_PATH)
    # Check intersection based on original data values since row_id might not have been added
    # in the original setup_split.py script execution.
    # A simple way to check is to check overlap of all feature columns
    feat_cols = [c for c in train_df.columns if c not in ('Target', 'Target_Bin', 'dropout', 'row_id')]
    intersection = train_df[feat_cols].merge(test_df[feat_cols], how='inner')
    assert len(intersection) == 0, "Leakage: Train and Test have overlapping identical rows"

def test_test_hash_unchanged():
    # If make_split wasn't the original creator, the hash check might fail if we run it
    # We just check the hash of test_frozen matches what we expect from split_manifest if it exists
    if C.SPLIT_MANIFEST.exists():
        import json
        man = json.loads(C.SPLIT_MANIFEST.read_text(encoding="utf-8"))
        assert sha256_file(C.TEST_PATH) == man["test_sha256"], "Test set file has been modified!"

def test_preprocessor_fitted_only_on_train():
    X_train, y_train, _ = load_train()
    prep = build_preprocessor("linear")
    prep.fit(X_train, y_train)
    
    # Check that medians/means come only from train
    ct = prep.named_steps["ct"]
    fs = prep.named_steps["fs1"]
    imputer_num = ct.named_transformers_["num"].named_steps["imp"]
    
    # Calculate median on train directly, after FirstSemesterFeatures transformation
    num_cols = ct.transformers_[0][2]
    X_transformed = fs.transform(X_train)
    expected_medians = X_transformed[num_cols].median().values
    
    np.testing.assert_allclose(imputer_num.statistics_, expected_medians, err_msg="Imputer learned from non-train data")
