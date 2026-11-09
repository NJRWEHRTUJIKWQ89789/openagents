"""Independent PR 982 authorization, lifecycle and creation-race regressions."""
import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models import Channel, ChannelMember, CloudAgentConfig, EventRecord, Workspace, WorkspaceMember
from app.mods.workspace_mod import _handle_message_posted
from openagents.core.onm_events import Event
from openagents.core.onm_mods import EventRejected, PipelineContext
from tests.test_visibility import people, no_push  # noqa: F401 - isolated auth/background fixtures


def post(client, workspace, name, *, payload=None, source="human:user", headers=None):
    return client.post('/v1/events', headers=headers or {'Authorization': 'Bearer mia'}, json={
        'network': workspace['id'], 'type': 'workspace.message.posted',
        'source': source, 'target': f'channel/{name}',
        'payload': {'content': 'synthetic message', **(payload or {})},
    })


def channel(db, workspace, name):
    db.expire_all()
    return db.scalar(select(Channel).where(Channel.workspace_id == workspace['id'], Channel.name == name))


@pytest.mark.parametrize('status', ['online', 'removed', 'missing'])
def test_implicit_create_matches_explicit_agent_admission(client, workspace, db, people, status):
    if status != 'missing':
        db.add(WorkspaceMember(workspace_id=workspace['id'], agent_name='private-bot',
                               owner_email='adam@acme.test', visibility='personal', status=status))
        db.commit()
    payload = {'participants': ['private-bot']}
    explicit = client.post('/v1/events', headers={'Authorization': 'Bearer mia'}, json={
        'network': workspace['id'], 'type': 'network.channel.create', 'source': 'human:user',
        'target': 'core', 'payload': {'name': 'explicit', **payload},
    })
    assert explicit.status_code == 403
    implicit = post(client, workspace, 'implicit', payload=payload)
    assert implicit.status_code == 403, implicit.text
    assert channel(db, workspace, 'implicit') is None
    assert db.scalar(select(EventRecord).where(EventRecord.target == 'channel/implicit')) is None


def test_pipeline_token_human_cannot_invite_another_owners_agent(db, workspace):
    db.add(WorkspaceMember(workspace_id=workspace['id'], agent_name='private-bot',
                           owner_email='adam@acme.test', visibility='personal', status='online'))
    db.commit()
    ws = db.get(Workspace, workspace['id'])
    event = Event(type='workspace.message.posted', source='human:user', target='channel/pipeline',
                  payload={'content': 'hello', 'sender_email': 'mia@acme.test', 'participants': ['private-bot']})
    ctx = PipelineContext(network_id=workspace['id'], db=db, workspace=ws, token=workspace['token'])
    with pytest.raises(EventRejected, match='forbidden'):
        asyncio.run(_handle_message_posted(event, ctx))
    assert channel(db, workspace, 'pipeline') is None


def test_implicit_create_respects_personal_assistant_boundary(client, workspace, db):
    db.add(WorkspaceMember(workspace_id=workspace['id'], agent_name='yumi',
                           agent_type='cloud:openagents', status='online'))
    db.add(CloudAgentConfig(workspace_id=workspace['id'], agent_name='yumi', provider='openagents', model='synthetic', api_key=''))
    db.commit()
    headers = {'X-Workspace-Token': workspace['token']}
    explicit = client.post('/v1/events', headers=headers, json={
        'network': workspace['id'], 'type': 'network.channel.create', 'source': 'openagents:agent-alpha',
        'target': 'core', 'payload': {'name': 'explicit-yumi', 'participants': ['yumi']},
    })
    assert explicit.status_code != 200
    implicit = post(client, workspace, 'implicit-yumi', source='system:synthetic', headers=headers,
                    payload={'participants': ['yumi', 'agent-alpha']})
    assert implicit.status_code == explicit.status_code, implicit.text
    assert channel(db, workspace, 'implicit-yumi') is None


def test_authenticated_identity_overrides_spoofed_owner(client, workspace, db, people):
    response = post(client, workspace, 'identity', source='human:adam@acme.test', payload={
        'sender_email': 'adam@acme.test', 'owner_email': 'adam@acme.test', 'participants': ['agent-alpha'],
    })
    assert response.status_code == 200, response.text
    ch = channel(db, workspace, 'identity')
    assert (ch.owner_email, ch.director_email, ch.visibility, ch.created_by) == (
        'mia@acme.test', 'mia@acme.test', 'private', 'human:mia@acme.test')
    denied = post(client, workspace, 'identity', headers={'Authorization': 'Bearer adam'})
    assert denied.status_code == 403


def test_deleted_task_rejects_late_agent_message(client, workspace, db):
    """Task DELETE must keep its working thread retired after a late reply."""
    from tests.test_tasks import _create, _headers
    task = _create(client, workspace).json()['data']
    assigned = client.post(f"/v1/tasks/{task['id']}/assign",
                           json={'network': workspace['id'], 'agent': 'agent-alpha'}, headers=_headers(workspace))
    assert assigned.status_code == 200, assigned.text
    task['channel_name'] = f"task:{task['id']}"
    assert client.delete(f"/v1/tasks/{task['id']}?network={workspace['id']}", headers=_headers(workspace)).status_code == 200
    assert channel(db, workspace, task['channel_name']).status == 'deleted'
    response = post(client, workspace, task['channel_name'], source='openagents:agent-alpha', headers=_headers(workspace))
    assert response.status_code == 403
    assert response.json()['message'] == 'channel_deleted_forbidden'
    assert channel(db, workspace, task['channel_name']).status == 'deleted'


@pytest.mark.parametrize('message_type', ['chat', 'status'])
@pytest.mark.parametrize('visibility,status,source,expected', [
    ('public', 'active', 'openagents:agent-alpha', 200),
    ('private', 'active', 'human:user', 401),
    ('private', 'active', 'openagents:agent-alpha', 401),
    ('public', 'deleted', 'openagents:agent-alpha', 403),
])
def test_creation_race_reuses_winner_without_invitations(
    client, workspace, db, monkeypatch, people, visibility, status, source, expected, message_type,
):
    """Real uniqueness conflict at a deterministic resolver/insert interleaving.

    This is a two-session SQLite test, not a parallel PostgreSQL test.
    """
    from app.mods import workspace_mod
    from app.models import ChannelHumanMember
    from tests.conftest import TestingSessionLocal
    materialize = workspace_mod._materialize_channel
    winner_id = None

    def competing_insert(*args, **kwargs):
        nonlocal winner_id
        with TestingSessionLocal() as other:
            winner = Channel(workspace_id=workspace['id'], name='race', status=status,
                             visibility=visibility, owner_email='adam@acme.test', title='Winner')
            other.add(winner)
            other.flush()
            winner_id = winner.id
            other.add(ChannelHumanMember(channel_id=winner.id, user_email='adam@acme.test'))
            other.add(ChannelMember(channel_id=winner.id, agent_name='winner-agent'))
            other.commit()
        return materialize(*args, **kwargs)

    monkeypatch.setattr(workspace_mod, '_materialize_channel', competing_insert)
    headers = {'Authorization': 'Bearer mia'} if source.startswith('human:') else {'X-Workspace-Token': workspace['token']}
    response = post(client, workspace, 'race', source=source, headers=headers,
                    payload={'participants': ['agent-alpha'], 'visibility': 'public', 'channel_title': 'Loser', 'message_type': message_type})
    assert response.status_code == expected, response.text
    winner = channel(db, workspace, 'race')
    assert (winner.id, winner.status, winner.visibility, winner.owner_email, winner.title) == (
        winner_id, status, visibility, 'adam@acme.test', 'Winner')
    assert set(db.scalars(select(ChannelMember.agent_name).where(ChannelMember.channel_id == winner_id))) == {'winner-agent'}
    assert set(db.scalars(select(ChannelHumanMember.user_email).where(ChannelHumanMember.channel_id == winner_id))) == {'adam@acme.test'}
    records = db.scalars(select(EventRecord).where(EventRecord.target == 'channel/race')).all()
    assert len(records) == (1 if expected == 200 else 0)
    if records:
        assert records[0].metadata_['channel_id'] == winner_id
        assert not records[0].metadata_.get('channel_materialized')


def test_explicit_agent_grant_allows_implicit_invitation(client, workspace, db, people):
    from app.services.access_model import create_grant
    db.add(WorkspaceMember(workspace_id=workspace['id'], agent_name='granted-bot',
                           owner_email='adam@acme.test', visibility='personal', status='online'))
    db.flush()
    create_grant(db, workspace['id'], resource_kind='agent', resource_id='granted-bot',
                 grantee_kind='human', grantee_id='mia@acme.test', rights=['act'], granted_by='adam@acme.test')
    db.commit()
    response = post(client, workspace, 'granted', payload={'participants': ['granted-bot']})
    assert response.status_code == 200, response.text
    ch = channel(db, workspace, 'granted')
    assert db.get(ChannelMember, (ch.id, 'granted-bot')) is not None


def test_existing_channel_does_not_accept_payload_invitations(client, workspace, db, people):
    assert post(client, workspace, 'existing').status_code == 200
    assert post(client, workspace, 'existing', payload={'participants': ['unknown-bot']}).status_code == 200
    ch = channel(db, workspace, 'existing')
    assert db.get(ChannelMember, (ch.id, 'unknown-bot')) is None


def test_unauthorized_person_cannot_revive_private_tombstone(client, workspace, db, people):
    db.add(Channel(workspace_id=workspace['id'], name='private-deleted', status='deleted',
                   visibility='private', owner_email='adam@acme.test'))
    db.commit()
    assert post(client, workspace, 'private-deleted').status_code == 403
    assert channel(db, workspace, 'private-deleted').status == 'deleted'


@pytest.mark.parametrize('message_type', ['thinking', 'status', 'todos', 'approval', 'artifact', 'error'])
def test_non_chat_message_materializes_without_routing(client, workspace, db, message_type, monkeypatch):
    """Persistence has a channel even when routing is deliberately skipped."""
    from unittest.mock import AsyncMock
    monkeypatch.setattr('app.mods.workspace_mod._conversation_targets', AsyncMock(side_effect=AssertionError('must not route')))
    response = post(client, workspace, 'early-return', source='openagents:agent-alpha',
                    headers={'X-Workspace-Token': workspace['token']}, payload={'message_type': message_type})
    assert response.status_code == 200
    assert channel(db, workspace, 'early-return') is not None
    assert db.scalar(select(EventRecord).where(EventRecord.target == 'channel/early-return')) is not None


@pytest.mark.parametrize('message_type', ['chat', 'thinking', 'status', 'todos', 'approval', 'artifact', 'error'])
def test_deleted_thread_rejects_all_message_types(client, workspace, db, message_type):
    db.add(Channel(workspace_id=workspace['id'], name='retired', status='deleted'))
    db.commit()
    response = post(client, workspace, 'retired', source='openagents:agent-alpha',
                    headers={'X-Workspace-Token': workspace['token']}, payload={'message_type': message_type})
    assert response.status_code == 403, response.text
    assert channel(db, workspace, 'retired').status == 'deleted'
    assert db.scalar(select(EventRecord).where(EventRecord.target == 'channel/retired')) is None


def test_savepoint_does_not_commit_channel_on_later_failure(client, workspace, db):
    # Participant PK failure happens after the channel savepoint is released.
    # The enclosing request must still roll back that newly inserted channel.
    with pytest.raises(IntegrityError, match='channel_members.channel_id, channel_members.agent_name'):
        post(client, workspace, 'rollback', source='system:synthetic',
             headers={'X-Workspace-Token': workspace['token']},
             payload={'participants': ['agent-alpha', 'agent-alpha']})
    assert channel(db, workspace, 'rollback') is None
    assert db.scalar(select(EventRecord).where(EventRecord.target == 'channel/rollback')) is None


def test_unrelated_channel_constraint_is_not_swallowed(client, workspace, db):
    from sqlalchemy import event as sql_event
    existing_id = channel(db, workspace, workspace['channel']['name']).id

    def collide_primary_key(mapper, connection, target):
        if target.name == 'wrong-constraint':
            target.id = existing_id

    sql_event.listen(Channel, 'before_insert', collide_primary_key)
    try:
        with pytest.raises(IntegrityError, match='UNIQUE constraint failed: channels.id'):
            post(client, workspace, 'wrong-constraint', source='system:synthetic',
                 headers={'X-Workspace-Token': workspace['token']})
    finally:
        sql_event.remove(Channel, 'before_insert', collide_primary_key)
    assert channel(db, workspace, 'wrong-constraint') is None


def test_explicit_legacy_machine_keeps_private_thread_access(client, workspace, db):
    ws = db.get(Workspace, workspace['id'])
    ws.legacy_machine_access = True
    db.get(WorkspaceMember, (workspace['id'], 'agent-alpha')).owner_email = None
    db.add(Channel(workspace_id=workspace['id'], name='legacy-private', visibility='private',
                   owner_email='someone@example.test', status='active'))
    db.commit()
    response = post(client, workspace, 'legacy-private', source='openagents:agent-alpha',
                    headers={'X-Workspace-Token': workspace['token']}, payload={'message_type': 'status'})
    assert response.status_code == 200, response.text
    assert db.scalar(select(EventRecord).where(EventRecord.target == 'channel/legacy-private')) is not None
