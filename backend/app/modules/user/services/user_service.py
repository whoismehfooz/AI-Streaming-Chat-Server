import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.custom_exceptions import (
    EmailAlreadyRegisteredException,
    UserNotFoundException,
)
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserUpdate


class UserService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = UserRepository(session)

    @staticmethod
    def _is_email_unique_violation(exc: IntegrityError) -> bool:
        original_error = exc.orig

        diagnostic = getattr(
            original_error,
            "diag",
            None,
        )

        constraint_name = getattr(
            diagnostic,
            "constraint_name",
            None,
        )

        if constraint_name is None:
            constraint_name = getattr(
                original_error,
                "constraint_name",
                None,
            )

        return constraint_name == "uq_users_email"

    async def create_user(
        self,
        email: str,
        password_hash: str,
    ) -> User:
        normalized_email = email.strip().lower()

        if not normalized_email:
            raise ValueError("Email cannot be empty.")

        if not password_hash or not password_hash.strip():
            raise ValueError("Password hash cannot be empty.")

        try:
            existing_user = await self.repository.get_by_email(
                normalized_email,
            )

            if existing_user is not None:
                raise EmailAlreadyRegisteredException()

            user = await self.repository.create(
                email=normalized_email,
                password_hash=password_hash,
            )

            await self.session.commit()
            await self.session.refresh(user)

            return user

        except IntegrityError as exc:
            await self.session.rollback()

            if self._is_email_unique_violation(exc):
                raise EmailAlreadyRegisteredException() from exc

            raise

        except Exception:
            await self.session.rollback()
            raise

    async def get_user(
        self,
        user_id: uuid.UUID,
    ) -> User:
        user = await self.repository.get_by_id(user_id)

        if user is None:
            raise UserNotFoundException()

        return user

    async def get_user_by_email(
        self,
        email: str,
    ) -> User:
        normalized_email = email.strip().lower()

        if not normalized_email:
            raise ValueError("Email cannot be empty.")

        user = await self.repository.get_by_email(
            normalized_email,
        )

        if user is None:
            raise UserNotFoundException()

        return user

    async def list_users(
        self,
        limit: int,
        offset: int,
    ) -> list[User]:
        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

        return await self.repository.list(
            limit=limit,
            offset=offset,
        )

    async def update_user(
        self,
        user_id: uuid.UUID,
        user_data: UserUpdate,
    ) -> User:
        if "email" not in user_data.model_fields_set:
            raise ValueError("At least one field must be provided.")

        if user_data.email is None:
            raise ValueError("Email cannot be null.")

        normalized_email = str(user_data.email).strip().lower()

        if not normalized_email:
            raise ValueError("Email cannot be empty.")

        try:
            user = await self.get_user(user_id)

            existing_user = await self.repository.get_by_email(
                normalized_email,
            )

            if (
                existing_user is not None
                and existing_user.id != user.id
            ):
                raise EmailAlreadyRegisteredException()

            updated_user = await self.repository.update(
                user,
                email=normalized_email,
            )

            await self.session.commit()
            await self.session.refresh(updated_user)

            return updated_user

        except IntegrityError as exc:
            await self.session.rollback()

            if self._is_email_unique_violation(exc):
                raise EmailAlreadyRegisteredException() from exc

            raise

        except Exception:
            await self.session.rollback()
            raise

    async def set_user_active_status(
        self,
        user_id: uuid.UUID,
        is_active: bool,
    ) -> User:
        try:
            user = await self.get_user(user_id)

            updated_user = await self.repository.update(
                user,
                is_active=is_active,
            )

            await self.session.commit()
            await self.session.refresh(updated_user)

            return updated_user

        except Exception:
            await self.session.rollback()
            raise

    async def set_user_verified_status(
        self,
        user_id: uuid.UUID,
        is_verified: bool,
    ) -> User:
        try:
            user = await self.get_user(user_id)

            updated_user = await self.repository.update(
                user,
                is_verified=is_verified,
            )

            await self.session.commit()
            await self.session.refresh(updated_user)

            return updated_user

        except Exception:
            await self.session.rollback()
            raise

    async def delete_user(
        self,
        user_id: uuid.UUID,
    ) -> None:
        try:
            user = await self.get_user(user_id)

            await self.repository.delete(user)
            await self.session.commit()

        except Exception:
            await self.session.rollback()
            raise
