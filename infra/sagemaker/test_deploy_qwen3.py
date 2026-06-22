"""
Tests for deploy_qwen3.py

Run: pytest infra/sagemaker/test_deploy_qwen3.py -v
"""
import json
import os
import sys
import argparse
from unittest.mock import MagicMock, patch, mock_open


# conftest.py adds infra/sagemaker/ to sys.path before collection.
# deploy_qwen3 removes it again at module level to prevent shadowing
# the installed sagemaker package — so import order matters here.
import deploy_qwen3 as dq

_this_dir = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_iam(role_exists=True):
    """Return a mock IAM client. NoSuchEntityException mirrors botocore."""
    iam = MagicMock()
    exc_cls = type("NoSuchEntityException", (Exception,), {})
    iam.exceptions.NoSuchEntityException = exc_cls
    if not role_exists:
        iam.get_role.side_effect = exc_cls("no such role")
    return iam


def _make_args(**kwargs):
    defaults = {"profile": "experimental", "hf_token": None, "delete": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# get_hf_token
# ─────────────────────────────────────────────────────────────────────────────

class TestGetHfToken:
    def test_explicit_arg_takes_priority(self):
        assert dq.get_hf_token("hf_explicit") == "hf_explicit"

    def test_explicit_arg_skips_cache_file(self, tmp_path):
        token_file = tmp_path / "token"
        token_file.write_text("hf_from_cache")
        with patch("deploy_qwen3.os.path.expanduser", return_value=str(token_file)):
            assert dq.get_hf_token("hf_explicit") == "hf_explicit"

    def test_reads_cache_file_when_no_explicit(self, tmp_path):
        token_file = tmp_path / "token"
        token_file.write_text("hf_cached\n")
        with patch("deploy_qwen3.os.path.expanduser", return_value=str(token_file)):
            assert dq.get_hf_token(None) == "hf_cached"

    def test_strips_whitespace_from_cache_file(self, tmp_path):
        token_file = tmp_path / "token"
        token_file.write_text("  hf_cached  \n")
        with patch("deploy_qwen3.os.path.expanduser", return_value=str(token_file)):
            assert dq.get_hf_token(None) == "hf_cached"

    def test_falls_back_to_env_var_when_no_cache(self, tmp_path):
        missing = str(tmp_path / "nonexistent")
        with patch("deploy_qwen3.os.path.expanduser", return_value=missing):
            with patch.dict(os.environ, {"HF_TOKEN": "hf_env"}):
                assert dq.get_hf_token(None) == "hf_env"

    def test_returns_none_when_nothing_available(self, tmp_path):
        missing = str(tmp_path / "nonexistent")
        with patch("deploy_qwen3.os.path.expanduser", return_value=missing):
            with patch.dict(os.environ, {}, clear=True):
                # Remove HF_TOKEN if present in environment
                env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
                with patch.dict(os.environ, env, clear=True):
                    assert dq.get_hf_token(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# get_or_create_role
# ─────────────────────────────────────────────────────────────────────────────

class TestGetOrCreateRole:
    ACCOUNT = "123456789012"
    EXPECTED_ARN = f"arn:aws:iam::{ACCOUNT}:role/SageMakerExecutionRole"

    def test_returns_arn_when_role_exists(self):
        iam = _make_iam(role_exists=True)
        result = dq.get_or_create_role(iam, self.ACCOUNT)
        assert result == self.EXPECTED_ARN
        iam.create_role.assert_not_called()

    def test_does_not_attach_policies_when_role_exists(self):
        iam = _make_iam(role_exists=True)
        dq.get_or_create_role(iam, self.ACCOUNT)
        iam.attach_role_policy.assert_not_called()

    def test_creates_role_when_missing(self):
        iam = _make_iam(role_exists=False)
        with patch("deploy_qwen3.time.sleep"):
            dq.get_or_create_role(iam, self.ACCOUNT)
        iam.create_role.assert_called_once()
        call_kwargs = iam.create_role.call_args.kwargs
        assert call_kwargs["RoleName"] == "SageMakerExecutionRole"

    def test_trust_policy_allows_sagemaker(self):
        iam = _make_iam(role_exists=False)
        with patch("deploy_qwen3.time.sleep"):
            dq.get_or_create_role(iam, self.ACCOUNT)
        doc = json.loads(iam.create_role.call_args.kwargs["AssumeRolePolicyDocument"])
        principals = [s["Principal"]["Service"] for s in doc["Statement"]]
        assert "sagemaker.amazonaws.com" in principals

    def test_attaches_both_policies(self):
        iam = _make_iam(role_exists=False)
        with patch("deploy_qwen3.time.sleep"):
            dq.get_or_create_role(iam, self.ACCOUNT)
        attached = [c.kwargs["PolicyArn"] for c in iam.attach_role_policy.call_args_list]
        assert "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess" in attached
        assert "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly" in attached

    def test_sleeps_for_iam_propagation(self):
        iam = _make_iam(role_exists=False)
        with patch("deploy_qwen3.time.sleep") as mock_sleep:
            dq.get_or_create_role(iam, self.ACCOUNT)
        mock_sleep.assert_called_once_with(15)

    def test_returns_arn_after_creation(self):
        iam = _make_iam(role_exists=False)
        with patch("deploy_qwen3.time.sleep"):
            result = dq.get_or_create_role(iam, self.ACCOUNT)
        assert result == self.EXPECTED_ARN


# ─────────────────────────────────────────────────────────────────────────────
# deploy
# ─────────────────────────────────────────────────────────────────────────────

class TestDeploy:
    def _run_deploy(self, hf_token="hf_test", role_exists=True):
        """Run deploy() with all AWS calls mocked. Returns (mock_model, predictor)."""
        args = _make_args(hf_token=hf_token)

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {"generated_text": "I am Qwen."}

        mock_model_instance = MagicMock()
        mock_model_instance.deploy.return_value = mock_predictor

        mock_iam = _make_iam(role_exists=role_exists)
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "491794274773"}

        mock_boto_session = MagicMock()
        mock_boto_session.client.side_effect = lambda svc: (
            mock_iam if svc == "iam" else mock_sts
        )

        mock_sm_session = MagicMock()

        with patch("deploy_qwen3.boto3.session.Session", return_value=mock_boto_session), \
             patch("deploy_qwen3.sagemaker.session.Session", return_value=mock_sm_session), \
             patch("deploy_qwen3.sagemaker.utils.name_from_base", return_value="qwen3-8b-test-123"), \
             patch("deploy_qwen3.Model", return_value=mock_model_instance), \
             patch("deploy_qwen3.time.sleep"), \
             patch("builtins.open", mock_open()):
            dq.deploy(args)

        return mock_model_instance, mock_predictor

    def test_model_created_with_correct_container(self):
        with patch("deploy_qwen3.boto3.session.Session") as mock_bs, \
             patch("deploy_qwen3.sagemaker.session.Session"), \
             patch("deploy_qwen3.sagemaker.utils.name_from_base", return_value="ep-123"), \
             patch("deploy_qwen3.Model") as MockModel, \
             patch("deploy_qwen3.time.sleep"), \
             patch("builtins.open", mock_open()):
            mock_iam = _make_iam()
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123"}
            mock_bs.return_value.client.side_effect = lambda s: mock_iam if s == "iam" else mock_sts
            MockModel.return_value.deploy.return_value.predict.return_value = {}

            dq.deploy(_make_args(hf_token="hf_x"))

        _, kwargs = MockModel.call_args
        assert dq.CONTAINER_URI in (MockModel.call_args.kwargs.get("image_uri") or MockModel.call_args.args[0] if MockModel.call_args.args else "")  or \
               MockModel.call_args.kwargs.get("image_uri") == dq.CONTAINER_URI

    def test_hf_token_injected_into_env(self):
        args = _make_args(hf_token="hf_mytoken")
        captured_env = {}

        def capture_model(**kwargs):
            captured_env.update(kwargs.get("env", {}))
            m = MagicMock()
            m.deploy.return_value.predict.return_value = {}
            return m

        mock_iam = _make_iam()
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123"}
        mock_bs = MagicMock()
        mock_bs.client.side_effect = lambda s: mock_iam if s == "iam" else mock_sts

        with patch("deploy_qwen3.boto3.session.Session", return_value=mock_bs), \
             patch("deploy_qwen3.sagemaker.session.Session"), \
             patch("deploy_qwen3.sagemaker.utils.name_from_base", return_value="ep"), \
             patch("deploy_qwen3.Model", side_effect=capture_model), \
             patch("deploy_qwen3.time.sleep"), \
             patch("builtins.open", mock_open()):
            dq.deploy(args)

        assert captured_env.get("HF_TOKEN") == "hf_mytoken"

    def test_hf_token_absent_when_none(self):
        args = _make_args(hf_token=None)
        captured_env = {}

        def capture_model(**kwargs):
            captured_env.update(kwargs.get("env", {}))
            m = MagicMock()
            m.deploy.return_value.predict.return_value = {}
            return m

        mock_iam = _make_iam()
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123"}
        mock_bs = MagicMock()
        mock_bs.client.side_effect = lambda s: mock_iam if s == "iam" else mock_sts

        missing_cache = "/tmp/nonexistent_hf_token_xyz"
        with patch("deploy_qwen3.boto3.session.Session", return_value=mock_bs), \
             patch("deploy_qwen3.sagemaker.session.Session"), \
             patch("deploy_qwen3.sagemaker.utils.name_from_base", return_value="ep"), \
             patch("deploy_qwen3.Model", side_effect=capture_model), \
             patch("deploy_qwen3.time.sleep"), \
             patch("deploy_qwen3.os.path.expanduser", return_value=missing_cache), \
             patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}, clear=True), \
             patch("builtins.open", mock_open()):
            dq.deploy(args)

        assert "HF_TOKEN" not in captured_env

    def test_required_env_vars_present(self):
        args = _make_args(hf_token="hf_x")
        captured_env = {}

        def capture_model(**kwargs):
            captured_env.update(kwargs.get("env", {}))
            m = MagicMock()
            m.deploy.return_value.predict.return_value = {}
            return m

        mock_iam = _make_iam()
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123"}
        mock_bs = MagicMock()
        mock_bs.client.side_effect = lambda s: mock_iam if s == "iam" else mock_sts

        with patch("deploy_qwen3.boto3.session.Session", return_value=mock_bs), \
             patch("deploy_qwen3.sagemaker.session.Session"), \
             patch("deploy_qwen3.sagemaker.utils.name_from_base", return_value="ep"), \
             patch("deploy_qwen3.Model", side_effect=capture_model), \
             patch("deploy_qwen3.time.sleep"), \
             patch("builtins.open", mock_open()):
            dq.deploy(args)

        assert captured_env["HF_MODEL_ID"] == dq.MODEL_ID
        assert captured_env["OPTION_DTYPE"] == "bf16"
        assert captured_env["TENSOR_PARALLEL_DEGREE"] == "max"
        assert captured_env["OPTION_MAX_MODEL_LEN"] == "8192"

    def test_deploy_called_with_correct_instance(self):
        args = _make_args(hf_token="hf_x")
        mock_model_instance = MagicMock()
        mock_model_instance.deploy.return_value.predict.return_value = {}

        mock_iam = _make_iam()
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123"}
        mock_bs = MagicMock()
        mock_bs.client.side_effect = lambda s: mock_iam if s == "iam" else mock_sts

        with patch("deploy_qwen3.boto3.session.Session", return_value=mock_bs), \
             patch("deploy_qwen3.sagemaker.session.Session"), \
             patch("deploy_qwen3.sagemaker.utils.name_from_base", return_value="ep"), \
             patch("deploy_qwen3.Model", return_value=mock_model_instance), \
             patch("deploy_qwen3.time.sleep"), \
             patch("builtins.open", mock_open()):
            dq.deploy(args)

        deploy_kwargs = mock_model_instance.deploy.call_args.kwargs
        assert deploy_kwargs["instance_type"] == dq.INSTANCE_TYPE
        assert deploy_kwargs["initial_instance_count"] == 1
        assert deploy_kwargs["container_startup_health_check_timeout"] == 900

    def test_endpoint_json_written(self, tmp_path):
        args = _make_args(hf_token="hf_x")
        mock_model_instance = MagicMock()
        mock_model_instance.deploy.return_value.predict.return_value = {}

        mock_iam = _make_iam()
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123"}
        mock_bs = MagicMock()
        mock_bs.client.side_effect = lambda s: mock_iam if s == "iam" else mock_sts

        endpoint_file = tmp_path / "endpoint.json"
        real_open = open

        def patched_open(path, *a, **kw):
            if path == "endpoint.json":
                return real_open(str(endpoint_file), *a, **kw)
            return real_open(path, *a, **kw)

        with patch("deploy_qwen3.boto3.session.Session", return_value=mock_bs), \
             patch("deploy_qwen3.sagemaker.session.Session"), \
             patch("deploy_qwen3.sagemaker.utils.name_from_base", return_value="qwen3-8b-test"), \
             patch("deploy_qwen3.Model", return_value=mock_model_instance), \
             patch("deploy_qwen3.time.sleep"), \
             patch("builtins.open", side_effect=patched_open):
            dq.deploy(args)

        data = json.loads(endpoint_file.read_text())
        assert data["endpoint_name"] == "qwen3-8b-test"
        assert data["model_id"] == dq.MODEL_ID
        assert data["instance"] == dq.INSTANCE_TYPE


# ─────────────────────────────────────────────────────────────────────────────
# delete
# ─────────────────────────────────────────────────────────────────────────────

class TestDelete:
    def test_calls_delete_endpoint_with_correct_name(self):
        args = _make_args(delete="qwen3-8b-some-endpoint")
        mock_sm = MagicMock()
        mock_bs = MagicMock()
        mock_bs.client.return_value = mock_sm

        with patch("deploy_qwen3.boto3.session.Session", return_value=mock_bs):
            dq.delete(args)

        mock_sm.delete_endpoint.assert_called_once_with(EndpointName="qwen3-8b-some-endpoint")

    def test_uses_correct_profile_and_region(self):
        args = _make_args(delete="ep-name", profile="experimental")

        with patch("deploy_qwen3.boto3.session.Session") as MockSession:
            MockSession.return_value.client.return_value = MagicMock()
            dq.delete(args)

        MockSession.assert_called_once_with(profile_name="experimental", region_name=dq.REGION)


# ─────────────────────────────────────────────────────────────────────────────
# sys.path guard (regression: infra/sagemaker/ must not shadow sagemaker pkg)
# ─────────────────────────────────────────────────────────────────────────────

class TestSysPathGuard:
    def test_script_dir_not_in_sys_path(self):
        # The module-level fix should have removed the script's own directory.
        # Verify sagemaker.model is importable (would fail if shadowed).
        from sagemaker.model import Model
        assert Model is not None

    def test_infra_sagemaker_dir_absent_from_path(self):
        assert _this_dir not in sys.path
