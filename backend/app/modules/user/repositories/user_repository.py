import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        email: str,
        password_hash: str,
    ) -> User:
        normalized_email = email.strip().lower()

        user = User(
            email=normalized_email,
            password_hash=password_hash,
        )

        self.session.add(user)
        await self.session.flush()

        return user

    async def get_by_id(
        self,
        user_id: uuid.UUID,
    ) -> User | None:
        result = await self.session.execute(
            select(User).where(
                User.id == user_id
            )
        )

        return result.scalar_one_or_none()

    async def get_by_email(
        self,
        email: str,
    ) -> User | None:
        normalized_email = email.strip().lower()

        result = await self.session.execute(
            select(User).where(
                User.email == normalized_email
            )
        )

        return result.scalar_one_or_none()

    async def list(
        self,
        limit: int,
        offset: int,
    ) -> list[User]:
        result = await self.session.execute(
            select(User)
            .order_by(
                User.created_at.desc(),
                User.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )

        return list(result.scalars().all())

    async def update(
        self,
        user: User,
        *,
        email: str | None = None,
        password_hash: str | None = None,
        is_active: bool | None = None,
        is_verified: bool | None = None,
    ) -> User:
        if email is not None:
            user.email = email.strip().lower()

        if password_hash is not None:
            user.password_hash = password_hash

        if is_active is not None:
            user.is_active = is_active

        if is_verified is not None:
            user.is_verified = is_verified

        await self.session.flush()

        return user

    async def delete(
        self,
        user: User,
    ) -> None:
        await self.session.delete(user)
        await self.session.flush()
