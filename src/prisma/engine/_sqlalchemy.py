from __future__ import annotations

import importlib
import logging
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from typing_extensions import override, Literal

from ._abstract import SyncAbstractEngine, AsyncAbstractEngine
from .._types import TransactionId

if TYPE_CHECKING:
    from ..types import MetricsFormat, DatasourceOverride  # noqa: TID251
    from .._types import Datasource


__all__ = (
    'SyncSQLAlchemyEngine',
    'AsyncSQLAlchemyEngine',
)

log: logging.Logger = logging.getLogger(__name__)

_EMPTY_METRICS: dict[str, Any] = {
    'counters': [],
    'gauges': [],
    'histograms': [],
    'version': '0',
}


def _prisma_url_to_sqlalchemy(url: str, *, async_mode: bool = False) -> str:
    """Convert a Prisma datasource URL to a SQLAlchemy connection URL."""
    if url.startswith('file:') or url.startswith('sqlite:'):
        path = url.removeprefix('file:').removeprefix('sqlite:')
        driver = 'sqlite+aiosqlite' if async_mode else 'sqlite'
        return f'{driver}:///{path}'

    if url.startswith('postgresql://') or url.startswith('postgres://'):
        if async_mode:
            return url.replace('postgresql://', 'postgresql+asyncpg://', 1).replace(
                'postgres://', 'postgresql+asyncpg://', 1
            )
        return url.replace('postgres://', 'postgresql://', 1)

    if url.startswith('mysql://'):
        if async_mode:
            return url.replace('mysql://', 'mysql+aiomysql://', 1)
        return url

    return url


def _resolve_datasource_url(
    datasource: 'Datasource',
    datasources_override: 'list[DatasourceOverride] | None',
) -> str:
    if datasources_override:
        for ds in datasources_override:
            if ds.get('name') == datasource.get('name') or not ds.get('name'):
                return ds['url']
    return datasource['url']


class BaseSQLAlchemyEngine:
    _datasource: 'Datasource'
    _active_provider: str
    _sa_url: str | None
    _sessions: dict[TransactionId, Any]

    def __init__(self, *, datasource: 'Datasource', active_provider: str) -> None:
        self._datasource = datasource
        self._active_provider = active_provider
        self._sa_url = None
        self._sessions = {}
        self.dml = ''

    def _get_tables_module(self) -> Any:
        try:
            return importlib.import_module('prisma._tables')
        except ModuleNotFoundError:
            # generated client not in prisma package — try relative import
            from ..client import SCHEMA_PATH  # type: ignore[attr-defined]  # noqa: TID251

            parent = str(SCHEMA_PATH.parent)
            import sys

            if parent not in sys.path:
                sys.path.insert(0, parent)
            import _tables  # type: ignore[import]

            return _tables


class SyncSQLAlchemyEngine(BaseSQLAlchemyEngine, SyncAbstractEngine):
    _engine: Any  # sqlalchemy.engine.Engine
    _session_factory: Any  # sqlalchemy.orm.sessionmaker

    @override
    def connect(
        self,
        timeout: timedelta = timedelta(seconds=10),
        datasources: 'list[DatasourceOverride] | None' = None,
    ) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        url = _resolve_datasource_url(self._datasource, datasources)
        self._sa_url = _prisma_url_to_sqlalchemy(url, async_mode=False)
        log.debug('Connecting sync SQLAlchemy engine to %s', self._sa_url)
        self._engine = create_engine(self._sa_url)
        self._session_factory = sessionmaker(bind=self._engine)

    @override
    def close(self, *, timeout: timedelta | None = None) -> None:
        if hasattr(self, '_engine') and self._engine is not None:
            self._engine.dispose()

    @override
    async def aclose(self, *, timeout: timedelta | None = None) -> None:
        self.close(timeout=timeout)

    @override
    def query(self, content: Any, *, tx_id: TransactionId | None) -> Any:
        from ._interpreter import SyncQueryInterpreter

        session = self._sessions.get(tx_id) if tx_id else None
        owns_session = session is None

        if owns_session:
            session = self._session_factory()

        try:
            tables = self._get_tables_module()
            interpreter = SyncQueryInterpreter(session=session, tables_module=tables)
            result = interpreter.execute(content)
            if owns_session:
                session.commit()
            return result
        except Exception:
            if owns_session:
                session.rollback()
            raise
        finally:
            if owns_session:
                session.close()

    @override
    def start_transaction(self, *, content: str) -> TransactionId:
        tx_id = TransactionId(str(uuid.uuid4()))
        session = self._session_factory()
        session.begin()
        self._sessions[tx_id] = session
        log.debug('Started transaction %s', tx_id)
        return tx_id

    @override
    def commit_transaction(self, tx_id: TransactionId) -> None:
        session = self._sessions.pop(tx_id, None)
        if session is not None:
            session.commit()
            session.close()
            log.debug('Committed transaction %s', tx_id)

    @override
    def rollback_transaction(self, tx_id: TransactionId) -> None:
        session = self._sessions.pop(tx_id, None)
        if session is not None:
            session.rollback()
            session.close()
            log.debug('Rolled back transaction %s', tx_id)

    @override
    def metrics(
        self,
        *,
        format: 'MetricsFormat',
        global_labels: dict[str, str] | None,
    ) -> str | dict[str, Any]:
        if format == 'prometheus':
            return ''
        return _EMPTY_METRICS


class AsyncSQLAlchemyEngine(BaseSQLAlchemyEngine, AsyncAbstractEngine):
    _engine: Any  # sqlalchemy.ext.asyncio.AsyncEngine
    _session_factory: Any  # sqlalchemy.ext.asyncio.async_sessionmaker

    @override
    async def connect(
        self,
        timeout: timedelta = timedelta(seconds=10),
        datasources: 'list[DatasourceOverride] | None' = None,
    ) -> None:
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

        url = _resolve_datasource_url(self._datasource, datasources)
        self._sa_url = _prisma_url_to_sqlalchemy(url, async_mode=True)
        log.debug('Connecting async SQLAlchemy engine to %s', self._sa_url)
        self._engine = create_async_engine(self._sa_url)
        self._session_factory = async_sessionmaker(bind=self._engine, expire_on_commit=False)

    @override
    def close(self, *, timeout: timedelta | None = None) -> None:
        pass  # async cleanup handled by aclose

    @override
    async def aclose(self, *, timeout: timedelta | None = None) -> None:
        if hasattr(self, '_engine') and self._engine is not None:
            await self._engine.dispose()

    @override
    async def query(self, content: Any, *, tx_id: TransactionId | None) -> Any:
        from ._interpreter import AsyncQueryInterpreter

        session = self._sessions.get(tx_id) if tx_id else None
        owns_session = session is None

        if owns_session:
            session = self._session_factory()

        try:
            tables = self._get_tables_module()
            interpreter = AsyncQueryInterpreter(session=session, tables_module=tables)
            result = await interpreter.execute(content)
            if owns_session:
                await session.commit()
            return result
        except Exception:
            if owns_session:
                await session.rollback()
            raise
        finally:
            if owns_session:
                await session.close()

    @override
    async def start_transaction(self, *, content: str) -> TransactionId:
        tx_id = TransactionId(str(uuid.uuid4()))
        session = self._session_factory()
        await session.begin()
        self._sessions[tx_id] = session
        log.debug('Started async transaction %s', tx_id)
        return tx_id

    @override
    async def commit_transaction(self, tx_id: TransactionId) -> None:
        session = self._sessions.pop(tx_id, None)
        if session is not None:
            await session.commit()
            await session.close()
            log.debug('Committed async transaction %s', tx_id)

    @override
    async def rollback_transaction(self, tx_id: TransactionId) -> None:
        session = self._sessions.pop(tx_id, None)
        if session is not None:
            await session.rollback()
            await session.close()
            log.debug('Rolled back async transaction %s', tx_id)

    @override
    async def metrics(
        self,
        *,
        format: 'MetricsFormat',
        global_labels: dict[str, str] | None,
    ) -> str | dict[str, Any]:
        if format == 'prometheus':
            return ''
        return _EMPTY_METRICS
