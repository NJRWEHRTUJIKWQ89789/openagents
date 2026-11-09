"""Real convenience pipelines must not expand the audience of private data."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from app import cache
from tests.test_network_event_publish import emit_setup  # noqa: F401
from tests.test_visibility import people  # noqa: F401


@pytest.fixture
def published(monkeypatch):
    out = []
    monkeypatch.setattr(cache, "publish_event", lambda channel, data: out.append(data))
    return out


def stream(client, workspace, monkeypatch, published, viewer):
    async def subscribe(channel):
        assert channel == f"ws:{workspace['id']}:events"
        for data in published:
            yield data

    monkeypatch.setattr(cache, "subscribe_events", subscribe)
    if viewer == "machine":
        headers = {"X-Workspace-Token": workspace["token"]}
    elif viewer.startswith("agent:"):
        headers = {"X-Workspace-Token": workspace["token"], "X-Agent-Name": viewer[6:]}
    else:
        headers = {"Authorization": f"Bearer {viewer}"}
    response = client.get(
        f"/v1/events/stream?network={workspace['id']}",
        headers=headers,
    )
    assert response.status_code == 200
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ")]


def test_join_stream_does_not_disclose_rotated_session(client, workspace, people, monkeypatch, published):
    response = client.post("/v1/join", json={
        "network": workspace["id"], "token": workspace["token"], "agent_name": "joined-agent",
    })
    assert response.status_code == 200
    assert response.json()["data"]["session_id"]
    events = stream(client, workspace, monkeypatch, published, "vic")
    assert len(events) == 1
    assert events[0]["type"] == "network.agent.join"
    assert events[0]["metadata"]["role"] == "member"
    assert "session_id" not in events[0]["metadata"]


@pytest.mark.parametrize("event_type,source,is_credential", [
    ("network.agent.join", "openagents:agent", True),
    ("workspace.message.posted", "openagents:agent", True),
    ("workspace.message.posted", "human:person@example.test", False),
    ("review.business-event", "system", False),
])
def test_session_projection_preserves_other_metadata(emit_setup, event_type, source, is_credential):
    from app.routers.network import _emit_event
    s = emit_setup
    result = s.event.model_copy(update={"type": event_type, "source": source,
        "metadata": {"session_id": "synthetic-session", "trace": "keep"},
        "payload": {"session_id": "business-session"}})
    s.process.return_value = result
    assert asyncio.run(_emit_event(s.event, s.workspace, s.db)) is result
    snapshot = json.loads(s.publish.call_args.args[1])
    assert ("session_id" in snapshot["metadata"]) is not is_credential
    assert snapshot["metadata"]["trace"] == "keep"
    assert snapshot["payload"]["session_id"] == "business-session"
    assert result.metadata["session_id"] == "synthetic-session"


@pytest.mark.parametrize("kind", ["file", "knowledge"])
@pytest.mark.parametrize("visibility", ["private", "public"])
def test_resource_stream_matches_read_access(
    client, db, workspace, people, monkeypatch, published, tmp_path, kind, visibility,
):
    headers = {"Authorization": "Bearer mia", "X-Workspace-Token": workspace["token"]}
    if kind == "file":
        from app.storage import LocalFileStore
        monkeypatch.setattr("app.routers.files.get_file_store", lambda: LocalFileStore(str(tmp_path)))
        response = client.post("/v1/files", headers=headers,
            data={"network": workspace["id"], "visibility": visibility},
            files={"file": ("synthetic-report.txt", b"synthetic content", "text/plain")})
    else:
        response = client.post("/v1/knowledge", headers=headers, json={
            "network": workspace["id"], "title": "Synthetic report",
            "content": "synthetic content", "visibility": visibility,
        })
    assert response.status_code == 200, response.text
    resource_id = response.json()["data"]["id"]
    read_path = f"/v1/files/{resource_id}/info" if kind == "file" else f"/v1/knowledge/{resource_id}"
    outsider_read = client.get(read_path, headers={"Authorization": "Bearer vic"})
    assert outsider_read.status_code == (200 if visibility == "public" else 403)
    assert len(published) == 1
    assert len(stream(client, workspace, monkeypatch, published, "mia")) == 1
    outsider_events = stream(client, workspace, monkeypatch, published, "vic")
    assert len(outsider_events) == (1 if visibility == "public" else 0)
    assert len(stream(client, workspace, monkeypatch, published, "machine")) == 1
    from app.models import WorkspaceMember
    member = db.get(WorkspaceMember, (workspace["id"], "agent-alpha"))
    member.owner_email = "vic@acme.test"
    db.commit()
    assert len(stream(client, workspace, monkeypatch, published, "agent:agent-alpha")) == (1 if visibility == "public" else 0)
    member.owner_email = "mia@acme.test"
    db.commit()
    assert len(stream(client, workspace, monkeypatch, published, "agent:agent-alpha")) == 1
    if visibility == "private":
        from app.services.access_model import create_grant, revoke_grant
        grant, _ = create_grant(db, workspace["id"], resource_kind=kind, resource_id=resource_id,
            grantee_kind="human", grantee_id="vic@acme.test", rights=["read"], granted_by="mia@acme.test")
        db.commit()
        assert len(stream(client, workspace, monkeypatch, published, "vic")) == 1
        revoke_grant(db, grant, "mia@acme.test")
        db.commit()
        assert stream(client, workspace, monkeypatch, published, "vic") == []

    # Actual DELETE routes retain the resource row for permission evaluation.
    published.clear()
    delete_path = f"/v1/files/{resource_id}" if kind == "file" else f"/v1/knowledge/{resource_id}?network={workspace['id']}"
    deleted = client.delete(delete_path, headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert len(published) == 1
    assert json.loads(published[0])["type"] == f"workspace.{kind}.deleted"
    assert len(stream(client, workspace, monkeypatch, published, "mia")) == 1
    assert len(stream(client, workspace, monkeypatch, published, "vic")) == (1 if visibility == "public" else 0)
    assert len(stream(client, workspace, monkeypatch, published, "agent:agent-alpha")) == 1
    assert len(stream(client, workspace, monkeypatch, published, "machine")) == 1


@pytest.mark.parametrize("outcome", ["commit", "commit_failure", "rejected"])
def test_real_pipeline_persistence_precedes_publication(db, workspace, monkeypatch, outcome):
    from app.models import EventRecord, Workspace
    from app.routers.network import _emit_event
    from openagents.core.onm_events import Event

    ws = db.get(Workspace, workspace["id"])
    event = Event(id="synthetic-transaction", type="review.notice", source="system", target="core")
    published = []
    from sqlalchemy import event as sa_event
    commits = []
    sa_event.listen(db, "after_commit", lambda session: commits.append(True))

    def publish(channel, data):
        assert commits == [True]
        # Read persistence independently of the writer's identity map.
        from sqlalchemy.orm import Session
        with Session(db.get_bind()) as reader:
            record = reader.get(EventRecord, event.id)
            assert record is not None
            assert json.loads(data)["id"] == record.id
        published.append(data)

    # Exercise the real cache API through a synthetic Redis transport.
    monkeypatch.setattr(cache, "_lazy_client", lambda: SimpleNamespace(publish=publish))
    if outcome == "commit_failure":
        def fail_commit():
            raise RuntimeError("synthetic commit failure")
        monkeypatch.setattr(db, "commit", fail_commit)
        with pytest.raises(RuntimeError, match="synthetic commit failure"):
            asyncio.run(_emit_event(event, ws, db, token=workspace["token"]))
        db.rollback()
    else:
        result = asyncio.run(_emit_event(event, ws, db,
            token="synthetic-invalid" if outcome == "rejected" else workspace["token"]))
        assert (result is not None) == (outcome == "commit")
    assert len(published) == (1 if outcome == "commit" else 0)
    assert (db.get(EventRecord, event.id) is not None) == (outcome == "commit")


def test_resource_stream_does_not_cache_revoked_group_membership(client, db, workspace, people, monkeypatch, published):
    from app.models import SecurityGroup, SecurityGroupMember
    from app.services.access_model import create_grant

    response = client.post("/v1/knowledge", headers={"Authorization": "Bearer mia", "X-Workspace-Token": workspace["token"]},
        json={"network": workspace["id"], "title": "Synthetic private knowledge", "content": "synthetic", "visibility": "private"})
    assert response.status_code == 200
    group = SecurityGroup(workspace_id=workspace["id"], name="Readers", slug="synthetic-readers")
    db.add(group)
    db.flush()
    membership = SecurityGroupMember(group_id=group.id, principal_kind="human", principal_id="vic@acme.test")
    db.add(membership)
    create_grant(db, workspace["id"], resource_kind="knowledge", resource_id=response.json()["data"]["id"],
        grantee_kind="group", grantee_id=group.id, rights=["read"])
    db.commit()
    wire = published[0]

    async def subscribe(channel):
        yield wire
        db.delete(membership)
        db.commit()
        yield wire

    monkeypatch.setattr(cache, "subscribe_events", subscribe)
    result = client.get(f"/v1/events/stream?network={workspace['id']}", headers={"Authorization": "Bearer vic"})
    assert result.status_code == 200
    assert sum(line.startswith("data: ") for line in result.text.splitlines()) == 1


@pytest.mark.parametrize('change', ['transfer', 'removed', 'missing', 'ownerless'])
def test_open_agent_stream_refreshes_current_membership(
    client, db, workspace, people, monkeypatch, change,
):
    from app.models import KnowledgeEntry, WorkspaceMember
    member = db.get(WorkspaceMember, (workspace['id'], 'agent-alpha'))
    member.owner_email = 'mia@acme.test'
    # Losing ownership must not upgrade an established agent stream to machine.
    from app.models import Workspace
    db.get(Workspace, workspace['id']).legacy_machine_access = True
    old = KnowledgeEntry(workspace_id=workspace['id'], slug='old-owner', title='Old owner private',
                         created_by='human:mia@acme.test', owner='human:mia@acme.test', visibility='private')
    new = KnowledgeEntry(workspace_id=workspace['id'], slug='new-owner', title='New owner private',
                         created_by='human:vic@acme.test', owner='human:vic@acme.test', visibility='private')
    public = KnowledgeEntry(workspace_id=workspace['id'], slug='public-resource', title='Public',
                            created_by='human:mia@acme.test', visibility='public')
    db.add_all([old, new, public])
    db.commit()

    def wire(event_id, resource):
        return json.dumps({'id': event_id, 'type': 'workspace.knowledge.updated', 'target': 'core',
                           'visibility': 'channel', 'payload': {'entry_id': resource.id}}).encode()

    async def subscribe(channel):
        yield wire('before', old)
        if change == 'transfer':
            member.owner_email = 'vic@acme.test'
        elif change == 'removed':
            member.status = 'removed'
        elif change == 'missing':
            db.delete(member)
        else:
            member.owner_email = None
        db.commit()
        yield wire('old-after', old)
        yield wire('new-after', new)
        yield wire('public-after', public)

    monkeypatch.setattr(cache, 'subscribe_events', subscribe)
    response = client.get(f"/v1/events/stream?network={workspace['id']}", headers={
        'X-Workspace-Token': workspace['token'], 'X-Agent-Name': 'agent-alpha',
    })
    assert response.status_code == 200
    ids = [json.loads(line[6:])['id'] for line in response.text.splitlines() if line.startswith('data: ')]
    expected = ['before']
    if change == 'transfer':
        expected += ['new-after', 'public-after']
    elif change == 'ownerless':
        expected += ['public-after']
    assert ids == expected


def test_open_human_stream_refreshes_role_derived_groups(client, db, workspace, people, monkeypatch):
    from sqlalchemy import select
    from app.models import KnowledgeEntry, User, WorkspaceMembership
    from app.services.access_model import create_grant, get_or_create_builtin_groups
    user = db.scalar(select(User).where(User.email == 'vic@acme.test'))
    membership = db.get(WorkspaceMembership, (workspace['id'], user.id))
    membership.role = 'guest'
    entry = KnowledgeEntry(workspace_id=workspace['id'], slug='guest-only', title='Guest grant',
                           created_by='human:mia@acme.test', visibility='private')
    db.add(entry)
    db.flush()
    guest = get_or_create_builtin_groups(db, workspace['id'])['guest']
    create_grant(db, workspace['id'], resource_kind='knowledge', resource_id=entry.id,
                 grantee_kind='group', grantee_id=guest.id, rights=['read'])
    db.commit()
    wire = json.dumps({'type': 'workspace.knowledge.updated', 'target': 'core', 'visibility': 'channel',
                       'payload': {'entry_id': entry.id}}).encode()

    async def subscribe(channel):
        yield wire
        membership.role = 'member'
        db.commit()
        yield wire

    monkeypatch.setattr(cache, 'subscribe_events', subscribe)
    response = client.get(f"/v1/events/stream?network={workspace['id']}", headers={'Authorization': 'Bearer vic'})
    assert response.status_code == 200
    assert sum(line.startswith('data: ') for line in response.text.splitlines()) == 1


def test_explicit_legacy_machine_stream_remains_unfiltered(client, db, workspace, monkeypatch):
    from app.models import Workspace, WorkspaceMember
    db.get(Workspace, workspace['id']).legacy_machine_access = True
    db.get(WorkspaceMember, (workspace['id'], 'agent-alpha')).owner_email = None
    db.commit()
    wire = json.dumps({'id': 'legacy', 'type': 'workspace.knowledge.deleted', 'target': 'core',
                       'payload': {'entry_id': 'synthetic-missing-resource'}}).encode()

    async def subscribe(channel):
        yield wire

    monkeypatch.setattr(cache, 'subscribe_events', subscribe)
    response = client.get(f"/v1/events/stream?network={workspace['id']}", headers={
        'X-Workspace-Token': workspace['token'], 'X-Agent-Name': 'agent-alpha',
    })
    assert response.status_code == 200
    assert sum(line.startswith('data: ') for line in response.text.splitlines()) == 1
