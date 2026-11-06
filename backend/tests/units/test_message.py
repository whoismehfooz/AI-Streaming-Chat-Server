import pytest

from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_create_message_rejects_invalid_role():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        response = await client.post(
            "/conversations",
            json={
                "title": "Role Validation Test",
            },
        )

        assert response.status_code == 201

        conversation_id = response.json()["id"]

        response = await client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "role": "banana",
                "content": "This role is invalid.",
            },
        )

        assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [
        "user",
        "assistant",
        "system",
    ],
)
async def test_create_message_accepts_valid_roles(role):
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        response = await client.post(
            "/conversations",
            json={
                "title": f"{role} Role Test",
            },
        )

        assert response.status_code == 201

        conversation_id = response.json()["id"]

        response = await client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "role": role,
                "content": f"Valid {role} message.",
            },
        )

        assert response.status_code == 201

        data = response.json()

        assert data["role"] == role
        assert data["content"] == f"Valid {role} message."


@pytest.mark.asyncio
async def test_create_message_rejects_empty_content():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        response = await client.post(
            "/conversations",
            json={
                "title": "Empty Content Test",
            },
        )

        assert response.status_code == 201

        conversation_id = response.json()["id"]

        response = await client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "role": "user",
                "content": "",
            },
        )

        assert response.status_code == 422
