"""
backend/tests/test_phase2_phase3.py
Comprehensive test suite for Phase 2 (Baselines, Anomaly Fix, Metrics)
and Phase 3 (M0/M1/M2/M3 Deep Learning Encoders, Fusion, Training, Calibration, ONNX Export).
"""
import json
import math
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "backend" / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "backend" / "src"))

# ── T2.3: Metrics Library Tests ───────────────────────────────────────────────

def test_metrics_calculations():
    from ml.eval.metrics import Metrics

    # Synthetic binary classification data
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.15, 0.05, 0.8, 0.7, 0.9, 0.85, 0.3, 0.65])

    m = Metrics(y_true, y_prob)
    summary = m.summary()

    assert "pr_auc" in summary
    assert "roc_auc" in summary
    assert "f1" in summary
    assert "ece" in summary
    assert "precision_at_50" in summary
    assert "recall_at_fpr5" in summary

    assert 0.0 <= summary["pr_auc"] <= 1.0
    assert 0.0 <= summary["roc_auc"] <= 1.0
    assert 0.0 <= summary["f1"] <= 1.0
    assert 0.0 <= summary["ece"] <= 1.0

    # Bootstrap CI test
    ci_low, ci_high = m.bootstrap_ci(metric="pr_auc", n=50, alpha=0.10)
    assert ci_low <= ci_high
    assert 0.0 <= ci_low <= 1.0
    assert 0.0 <= ci_high <= 1.0


def test_metrics_edge_cases():
    from ml.eval.metrics import Metrics

    # All zeros
    y_true_zero = np.zeros(20)
    y_prob = np.linspace(0.1, 0.9, 20)
    m_zero = Metrics(y_true_zero, y_prob)
    sum_zero = m_zero.summary()
    assert sum_zero["pr_auc"] == 0.0

    # All ones
    y_true_one = np.ones(20)
    m_one = Metrics(y_true_one, y_prob)
    sum_one = m_one.summary()
    assert sum_one["roc_auc"] == 0.5  # degenerate ROC is 0.5


# ── T2.1 & T2.2: Baselines & Percentile-Rank Anomaly Fix Tests ────────────────

def test_isolation_forest_percentile_rank():
    """
    T2.2: Fixes I5 (saturating anomaly scores).
    Verify percentile rank gives uniform [0, 1] spread without clipping at 1.0.
    """
    from ml.baselines import PercentileRankIsolationForest

    np.random.seed(42)
    # Train data: standard normal
    X_train = np.random.randn(200, 5)
    # Anomaly test point
    X_test = np.vstack([np.random.randn(20, 5), np.ones((5, 5)) * 10.0])

    pif = PercentileRankIsolationForest(n_estimators=50, random_state=42)
    pif.fit(X_train)

    scores = pif.predict_anomaly_score(X_test)
    assert len(scores) == 25
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    # The extreme anomalies should have high anomaly scores (> 0.8)
    assert (scores[-5:] > 0.8).all()
    # The normal points should NOT all saturate at 1.0
    normal_scores = scores[:20]
    assert np.mean(normal_scores) < 0.8
    assert len(np.unique(normal_scores)) > 5  # good distribution spread


def test_tabular_baselines_fit_predict():
    from ml.baselines import TabularBaselines

    np.random.seed(42)
    n_train, n_test, n_feat = 100, 30, 8
    X_train = np.random.randn(n_train, n_feat)
    y_train = (np.random.rand(n_train) > 0.8).astype(int)
    # Ensure at least some positives
    y_train[:10] = 1
    y_train[10:20] = 0

    X_test = np.random.randn(n_test, n_feat)
    y_test = (np.random.rand(n_test) > 0.8).astype(int)

    tb = TabularBaselines()
    tb.fit(X_train, y_train)

    results = tb.evaluate(X_test, y_test)
    for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        assert model_name in results
        assert "pr_auc" in results[model_name]
        assert "roc_auc" in results[model_name]


# ── T3.2 / T3.3 / T3.4: Encoders Tests ─────────────────────────────────────────

def test_m0_lstm_autoencoder():
    """
    T3.3: LSTM Autoencoder reconstruction and anomaly score.
    """
    from ml.models.encoders import LSTMAutoencoder

    B, T, F_dim, hidden = 4, 30, 12, 32
    ae = LSTMAutoencoder(input_dim=F_dim, hidden_dim=hidden, n_layers=2)

    x = torch.randn(B, T, F_dim)
    x_hat, z = ae(x)

    assert x_hat.shape == (B, T, F_dim)
    assert z.shape == (B, hidden)

    recon_error = torch.mean((x - x_hat) ** 2, dim=(1, 2))
    assert recon_error.shape == (B,)
    assert (recon_error >= 0.0).all()


@pytest.mark.parametrize("enc_type", ["gru", "tcn", "transformer"])
def test_m1_price_encoder_architectures(enc_type):
    """
    T3.2: Price encoder architectures: GRU, TCN, and Transformer.
    """
    from ml.models.encoders import PriceEncoder

    B, T, F_dim, hidden = 4, 30, 12, 32
    enc = PriceEncoder(input_dim=F_dim, hidden_dim=hidden, encoder_type=enc_type)

    x = torch.randn(B, T, F_dim)
    emb, logits = enc(x, return_embedding=True)

    assert emb.shape == (B, hidden)
    assert logits.shape == (B, 1)

    # Standard forward returns logits
    out_logits = enc(x)
    assert out_logits.shape == (B, 1)


def test_m2_text_encoder_attention_pooling():
    """
    T3.4: Text encoder with projection and masked attention pooling over days.
    """
    from ml.models.encoders import TextEncoder

    B, T, in_dim, proj_dim = 4, 30, 768, 64
    text_enc = TextEncoder(input_dim=in_dim, proj_dim=proj_dim, p_modality_dropout=0.0)

    x_text = torch.randn(B, T, in_dim)
    # Mask where sample 0 has 5 days of news, sample 1 has 0 days of news
    mask = torch.ones(B, T)
    mask[0, 5:] = 0.0  # only 5 days active
    mask[1, :] = 0.0   # no news days at all

    emb = text_enc(x_text, mask)
    assert emb.shape == (B, proj_dim)

    # When mask is all 0, output should be zero vector
    assert torch.allclose(emb[1], torch.zeros(proj_dim), atol=1e-6)


def test_focal_loss():
    from ml.models.encoders import FocalLoss

    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.tensor([2.0, -2.0, 0.5, -0.5])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0.0


# ── T3.5: Fusion Model Tests ──────────────────────────────────────────────────

def test_fusion_model_forward_and_loss():
    """
    T3.5: M3 Fusion Model cross-attention, modality dropout, gating, and loss.
    """
    from ml.models.fusion import FusionModel, FusionOutput

    B, T, F_dim, text_dim, hidden = 4, 30, 12, 768, 32
    fusion = FusionModel(
        price_input_dim=F_dim,
        text_input_dim=text_dim,
        hidden_dim=hidden,
        encoder_type="gru",
        n_heads=2,
        use_ae_aux=True,
        T=T,
    )

    x_price = torch.randn(B, T, F_dim)
    x_text = torch.randn(B, T, text_dim)
    text_mask = torch.ones(B, T)

    out = fusion(x_price, x_text, text_mask, return_attn=True)
    assert isinstance(out, FusionOutput)
    assert out.hype_prob.shape == (B,)
    assert (out.hype_prob >= 0.0).all() and (out.hype_prob <= 1.0).all()
    assert out.embedding.shape == (B, hidden)
    assert out.attn_weights.shape == (B, T)
    assert out.ae_recon.shape == (B, T, F_dim)
    assert out.model_mode == "fusion"

    # Test loss function
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_dict = fusion.loss(out, y, x_price, ae_weight=0.1)
    assert "total" in loss_dict
    assert "focal" in loss_dict
    assert "ae" in loss_dict
    assert loss_dict["total"].item() > 0.0


def test_fusion_graceful_degradation_price_only():
    """
    Verifies that when text_mask is zeroed (no news), model degrades gracefully
    to price_only mode without crashing or producing NaNs.
    """
    from ml.models.fusion import FusionModel

    B, T, F_dim, text_dim, hidden = 2, 30, 12, 768, 32
    fusion = FusionModel(
        price_input_dim=F_dim,
        text_input_dim=text_dim,
        hidden_dim=hidden,
        encoder_type="gru",
        n_heads=2,
    )
    fusion.eval()

    x_price = torch.randn(B, T, F_dim)
    x_text = torch.zeros(B, T, text_dim)
    text_mask = torch.zeros(B, T)  # no text available

    with torch.no_grad():
        out = fusion(x_price, x_text, text_mask)

    assert out.model_mode == "price_only"
    assert not torch.isnan(out.hype_prob).any()
    assert 0.0 <= out.hype_prob[0].item() <= 1.0


# ── T3.7: Calibration & Operating Thresholds Tests ────────────────────────────

def test_calibration_and_threshold_derivation():
    from ml.evaluate import calibrate_temperature, derive_thresholds

    np.random.seed(42)
    y_val = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1])
    # Overconfident uncalibrated probabilities
    p_val = np.array([0.05, 0.1, 0.2, 0.05, 0.95, 0.85, 0.92, 0.88, 0.3, 0.75])

    dummy_model = nn.Identity()
    device = torch.device("cpu")
    t_star = calibrate_temperature(y_val, p_val, dummy_model, device)
    assert 0.1 <= t_star <= 10.0

    thresholds = derive_thresholds(y_val, p_val, alert_precision_target=0.6, hype_precision_target=0.4)
    assert "alert" in thresholds
    assert "hype" in thresholds
    assert "neutral" in thresholds
    assert thresholds["alert"] >= thresholds["hype"]


# ── T3.8: ONNX Export and Parity Check ─────────────────────────────────────────

def test_onnx_export_and_parity(tmp_path):
    from ml.models.fusion import FusionModel
    from ml.export import export_onnx, check_onnx_parity, build_and_save_model_card

    B, T, F_dim, text_dim, hidden = 2, 30, 12, 768, 32
    fusion = FusionModel(
        price_input_dim=F_dim,
        text_input_dim=text_dim,
        hidden_dim=hidden,
        encoder_type="gru",
        n_heads=2,
        use_ae_aux=False,
    )
    fusion.eval()

    cfg = {"model_type": "fusion", "encoder_type": "gru", "seed": 42}
    device = torch.device("cpu")
    onnx_file = tmp_path / "test_fusion.onnx"

    exported_path = export_onnx(fusion, cfg, onnx_file, device, T=T, price_features=F_dim, text_dim=text_dim)
    assert exported_path.exists()
    assert exported_path.stat().st_size > 1000

    parity_ok = check_onnx_parity(
        fusion,
        onnx_file,
        cfg,
        device,
        atol=1e-4,
        T=T,
        price_features=F_dim,
        text_dim=text_dim,
        batch_size=2,
    )
    assert parity_ok is True

    card_path = build_and_save_model_card(tmp_path, cfg, onnx_file, parity_ok)
    assert card_path.exists()
    card_data = json.loads(card_path.read_text())
    assert card_data["parity_check_passed"] is True
    assert card_data["model_type"] == "fusion"
