"""Device-flow state transitions against real Redis, including Lua JSON encoding."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from extensions.ext_redis import redis_client
from models.account import Account
from models.oauth import OpenAPIOAuthAccessToken
from repositories.oauth_device_token_repository import SQLAlchemyOAuthDeviceTokenRepository
from services.entities.account_entities import AccountSnapshot
from services.oauth_device_adapters import EnvironmentOAuthDeviceTokenTTLPolicy, OAuthDeviceTokenIssuanceGateway
from services.oauth_device_contracts import DeviceFlowStatus, InvalidTransitionError, PollPayload
from services.oauth_device_flow import DEVICE_CODE_KEY_FMT, USER_CODE_KEY_FMT, DeviceFlowRedis

pytestmark = pytest.mark.usefixtures("flask_app_with_containers")


@pytest.mark.parametrize("subject_type", ["external_sso", "account"])
@pytest.mark.parametrize(
    "workspaces",
    [[], [{"id": "workspace-1", "name": 'Workspace "quoted"\\path\n测试', "role": "owner"}]],
    ids=["empty-workspaces", "nonempty-workspaces"],
)
def test_approval_preserves_payload_and_is_consumed_once(
    subject_type: str, workspaces: list[dict[str, object]]
) -> None:
    store = DeviceFlowRedis(redis_client)
    device_code, user_code, _ = store.start("difyctl", "CLI", "127.0.0.1")
    payload: PollPayload = {
        "token": "test-device-token",
        "expires_at": "2030-01-01T00:00:00+00:00",
        "subject_type": subject_type,
        "account": (
            {"id": "account-1", "email": "user@example.com", "name": "Example"} if subject_type == "account" else None
        ),
        "workspaces": workspaces,
        "default_workspace_id": "workspace-1" if workspaces else None,
        "token_id": "token-1",
    }
    if subject_type == "external_sso":
        payload.update(subject_email="user@example.com", subject_issuer="https://idp.example")

    try:
        store.approve(device_code, "approve-1", "token-1", payload)
        # Retrying the same transition must not rewrite or corrupt the approved payload.
        store.approve(device_code, "approve-1", "token-1", payload)
        with pytest.raises(InvalidTransitionError):
            store.deny(device_code, "deny-1")

        stored = store.load_by_device_code(device_code)
        assert stored is not None
        assert stored.status is DeviceFlowStatus.APPROVED
        assert stored.poll_payload == payload

        consumed = store.consume_on_poll(device_code)
        assert consumed is not None
        assert consumed.poll_payload == payload
        assert store.consume_on_poll(device_code) is None
        assert store.load_by_user_code(user_code) is None
    finally:
        redis_client.delete(DEVICE_CODE_KEY_FMT.format(code=device_code), USER_CODE_KEY_FMT.format(code=user_code))


def test_denied_state_has_no_payload_and_is_consumed_once() -> None:
    store = DeviceFlowRedis(redis_client)
    device_code, user_code, _ = store.start("difyctl", "CLI", "127.0.0.1")
    try:
        store.deny(device_code, "deny-1")
        store.deny(device_code, "deny-1")

        consumed = store.consume_on_poll(device_code)
        assert consumed is not None
        assert consumed.status is DeviceFlowStatus.DENIED
        assert consumed.poll_payload is None
        assert store.consume_on_poll(device_code) is None
        assert store.load_by_user_code(user_code) is None
    finally:
        redis_client.delete(DEVICE_CODE_KEY_FMT.format(code=device_code), USER_CODE_KEY_FMT.format(code=user_code))


def test_concurrent_logins_respect_cap(db_session_with_containers: Session) -> None:
    max_sessions = 3
    account = Account(name="Cap owner", email=f"cap-{uuid4().hex}@example.com")
    db_session_with_containers.add(account)
    db_session_with_containers.commit()
    snapshot = AccountSnapshot(
        id=account.id,
        name=account.name,
        email=account.email,
        avatar=None,
        is_password_set=False,
        interface_language=None,
        interface_theme=None,
        timezone=None,
        last_login_at=None,
        last_login_ip=None,
        status=account.status.value,
        initialized_at=None,
        created_at=account.created_at,
    )
    sessions = sessionmaker(bind=db_session_with_containers.get_bind(), expire_on_commit=False)
    issuer = OAuthDeviceTokenIssuanceGateway(
        tokens=SQLAlchemyOAuthDeviceTokenRepository(
            session_factory=sessions,
            redis=redis_client,
            max_sessions_per_account=max_sessions,
        ),
        ttl_policy=EnvironmentOAuthDeviceTokenTTLPolicy(),
    )

    def login(device_label: str) -> None:
        issuer.issue_account_token(
            account=snapshot,
            workspace_id="workspace-1",
            client_id="difyctl",
            device_label=device_label,
        )

    for index in range(max_sessions - 1):
        login(f"existing-{index}")
    start = Barrier(2)

    def concurrent_login(device_label: str) -> None:
        start.wait(timeout=5)
        login(device_label)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(concurrent_login, f"concurrent-{index}") for index in range(2)]
        for future in futures:
            future.result(timeout=10)

    with sessions() as check:
        live = check.scalar(
            select(func.count())
            .select_from(OpenAPIOAuthAccessToken)
            .where(OpenAPIOAuthAccessToken.account_id == account.id, OpenAPIOAuthAccessToken.revoked_at.is_(None))
        )
    assert live == max_sessions
