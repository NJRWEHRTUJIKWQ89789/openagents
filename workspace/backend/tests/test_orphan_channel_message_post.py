# -*- coding: utf-8 -*-
"""A message posted to a thread name that has no Channel row must not vanish
into an invisible thread.

`POST /v1/events` with ``target="channel/<name>"`` accepts any name. The
message pipeline resolved the Channel by name, got ``None`` for a name that
was never created, and carried on: the event was stored and read back through
``GET /v1/events?channel=<name>``, but because no ``channels`` row existed the
thread was absent from ``/v1/discover`` and every thread endpoint
(``/v1/channels/<name>/participants``, ``PATCH /v1/channels/<name>``) answered
404 "Thread not found". The thread could never be opened, so the messages were
unreachable from the dashboard.

Two live cases produced this: the dashboard verification harness (it uses a
fresh ``dashboard-verification-<hex>`` name as a scratch scope and posts into
it) and the workflow engine's task threads (``task:<uuid>``).

Deleted rows are deliberate tombstones. Posting must reject rather than undo
an explicit task/issue deletion, preserving the row and its identity.
"""

from sqlalchemy import select

from app.models import Channel, EventRecord


MARK = "orphan-repro-thread"


def _headers(workspace):
    return {"X-Workspace-Token": workspace["token"]}


def _post(client, workspace, target, content="hello"):
    return client.post(
        "/v1/events",
        json={
            "type": "workspace.message.posted",
            "source": "openagents:agent-alpha",
            "target": f"channel/{target}",
            "network": workspace["id"],
            "payload": {"content": content, "message_type": "chat"},
        },
        headers=_headers(workspace),
    )


def test_message_to_unknown_thread_creates_the_thread(client, workspace, db):
    """The first message to an unknown name materializes the thread."""
    assert db.execute(
        select(Channel).where(
            Channel.workspace_id == workspace["id"], Channel.name == MARK,
        )
    ).scalar_one_or_none() is None

    response = _post(client, workspace, MARK)
    assert response.status_code == 200, response.text
    assert response.json()["code"] == 0

    channel = db.execute(
        select(Channel).where(
            Channel.workspace_id == workspace["id"], Channel.name == MARK,
        )
    ).scalar_one_or_none()
    assert channel is not None, "posting created no Channel row — thread is invisible"
    assert channel.status == "active"

    event = db.execute(
        select(EventRecord).where(EventRecord.target == f"channel/{MARK}")
    ).scalar_one_or_none()
    assert event is not None, "the message was not stored"
    assert event.metadata_.get("channel_id") == str(channel.id)


def test_created_thread_is_discoverable_and_answerable(client, workspace):
    """The materialized thread shows up in discovery and its endpoints answer."""
    assert _post(client, workspace, MARK).status_code == 200

    discovery = client.get(
        f"/v1/discover?network={workspace['id']}", headers=_headers(workspace),
    ).json()["data"]
    assert any(c["address"] == f"channel/{MARK}" for c in discovery["channels"])

    participants = client.get(
        f"/v1/channels/{MARK}/participants?network={workspace['id']}",
        headers=_headers(workspace),
    )
    assert participants.status_code == 200, participants.text
    assert participants.json()["code"] == 0

    # A thread whose only message came from an agent keeps working as a target.
    assert _post(client, workspace, MARK, "second").status_code == 200


def test_existing_thread_is_reused_not_duplicated(client, workspace, db):
    """A channel that already exists is not recreated on the next message."""
    existing = workspace["channel"]["name"]
    before = len(db.execute(
        select(Channel).where(Channel.workspace_id == workspace["id"])
    ).scalars().all())

    assert _post(client, workspace, existing).status_code == 200
    assert _post(client, workspace, existing, "again").status_code == 200

    after = db.execute(
        select(Channel).where(Channel.workspace_id == workspace["id"])
    ).scalars().all()
    assert len(after) == before
    assert [c for c in after if c.name == existing]


def test_materialized_thread_keeps_visibility_invariant(client, workspace, db):
    """An agent's post cannot widen a thread: a public thread stays public and
    a private one stays private and keeps its owner."""
    assert _post(client, workspace, MARK).status_code == 200
    channel = db.execute(
        select(Channel).where(
            Channel.workspace_id == workspace["id"], Channel.name == MARK,
        )
    ).scalar_one()
    # No identified human creator on an agent post → public, like channel.create.
    assert channel.visibility == "public"
    assert channel.name == MARK
    assert channel.title is None
    assert channel.created_by == "openagents:agent-alpha"


def test_message_to_a_deleted_thread_is_rejected(client, workspace, db):
    """A late reply must not undo explicit deletion or persist new content."""
    assert _post(client, workspace, MARK, "before delete").status_code == 200
    channel = db.execute(
        select(Channel).where(
            Channel.workspace_id == workspace["id"], Channel.name == MARK,
        )
    ).scalar_one()
    channel.status = "deleted"
    db.commit()

    response = _post(client, workspace, MARK, "after delete")
    assert response.status_code == 403, response.text
    assert response.json()["message"] == "channel_deleted_forbidden"
    db.refresh(channel)
    assert channel.status == "deleted"
    assert len(db.execute(select(EventRecord).where(EventRecord.target == f"channel/{MARK}")).scalars().all()) == 1

    # Id is preserved (the messages still belong to the same thread) ...
    assert db.execute(
        select(Channel).where(
            Channel.workspace_id == workspace["id"], Channel.name == MARK,
        )
    ).scalar_one().id == channel.id
    # ... and the tombstone remains hidden and unopenable.
    discovery = client.get(
        f"/v1/discover?network={workspace['id']}", headers=_headers(workspace),
    ).json()["data"]
    assert not any(c["address"] == f"channel/{MARK}" for c in discovery["channels"])
    participants = client.get(
        f"/v1/channels/{MARK}/participants?network={workspace['id']}",
        headers=_headers(workspace),
    )
    assert participants.status_code == 404, participants.text


def test_rejection_keeps_deleted_thread_identity(client, workspace, db):
    """Rejection preserves the tombstone's title, owner, visibility and members."""
    from app.models import ChannelHumanMember, ChannelMember

    assert _post(client, workspace, MARK).status_code == 200
    channel = db.execute(
        select(Channel).where(
            Channel.workspace_id == workspace["id"], Channel.name == MARK,
        )
    ).scalar_one()
    channel.title = "Parked work"
    channel.visibility = "private"
    channel.owner_email = "owner@example.test"
    db.add(ChannelHumanMember(channel_id=channel.id, user_email="owner@example.test"))
    db.add(ChannelMember(channel_id=channel.id, agent_name="agent-beta"))
    channel.status = "deleted"
    db.commit()
    channel_id = channel.id

    assert _post(client, workspace, MARK, "resumed").status_code == 403
    db.expire_all()  # the request used its own session; drop our cached copy
    tombstone = db.execute(
        select(Channel).where(Channel.id == channel_id)
    ).scalar_one()
    assert tombstone.status == "deleted"
    assert tombstone.title == "Parked work"
    assert tombstone.visibility == "private"
    assert tombstone.owner_email == "owner@example.test"
    assert {m.user_email for m in db.execute(
        select(ChannelHumanMember).where(ChannelHumanMember.channel_id == channel_id)
    ).scalars()} == {"owner@example.test"}
    assert {m.agent_name for m in db.execute(
        select(ChannelMember).where(ChannelMember.channel_id == channel_id)
    ).scalars()} == {"agent-alpha", "agent-beta"}
