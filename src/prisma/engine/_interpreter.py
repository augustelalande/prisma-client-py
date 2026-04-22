"""Query interpreter: translates QueryRequest → SQLAlchemy Core statements."""
from __future__ import annotations

import base64
import datetime
import decimal
import logging
from types import ModuleType
from typing import Any

from .._query_repr import QueryRequest
from ..errors import RecordNotFoundError, UniqueViolationError, ForeignKeyViolationError, RawQueryError

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(value: Any) -> dict[str, Any]:
    return {'data': {'result': value}}


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a SQLModel/SQLAlchemy row to a plain dict."""
    if row is None:
        return {}
    # SQLModel instances have __dict__ but also __sqlmodel_fields__
    if hasattr(row, 'model_dump'):
        raw = row.model_dump()
    elif hasattr(row, '__dict__'):
        raw = {k: v for k, v in row.__dict__.items() if not k.startswith('_')}
    else:
        raw = dict(row._mapping)  # SQLAlchemy Row

    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, bytes):
            out[k] = base64.b64encode(v).decode()
        elif isinstance(v, decimal.Decimal):
            out[k] = str(v)
        elif isinstance(v, datetime.datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def _get_table(tables_module: ModuleType, model_name: str) -> Any:
    table_class_name = f'{model_name}Table'
    table_class = getattr(tables_module, table_class_name, None)
    if table_class is None:
        raise RuntimeError(f'Table class {table_class_name!r} not found in generated _tables module')
    return table_class


def _build_where_clause(table: Any, where: dict[str, Any] | None) -> Any:
    """Translate a Prisma where dict into a SQLAlchemy where clause."""
    from sqlalchemy import and_, or_, not_

    if not where:
        return None

    clauses = []
    for key, value in where.items():
        if key == 'AND':
            sub = [_build_where_clause(table, w) for w in value if w]
            if sub:
                clauses.append(and_(*[c for c in sub if c is not None]))
            continue
        if key == 'OR':
            sub = [_build_where_clause(table, w) for w in value if w]
            if sub:
                clauses.append(or_(*[c for c in sub if c is not None]))
            continue
        if key == 'NOT':
            if isinstance(value, list):
                sub = [_build_where_clause(table, w) for w in value if w]
                if sub:
                    clauses.append(not_(and_(*[c for c in sub if c is not None])))
            else:
                inner = _build_where_clause(table, value)
                if inner is not None:
                    clauses.append(not_(inner))
            continue

        col = getattr(table, key, None)
        if col is None:
            log.warning('Unknown field %r in where clause for %s', key, table.__name__)
            continue

        if value is None:
            clauses.append(col.is_(None))
        elif isinstance(value, dict):
            clauses.extend(_build_field_filters(col, value))
        else:
            clauses.append(col == value)

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    from sqlalchemy import and_
    return and_(*clauses)


def _build_field_filters(col: Any, filters: dict[str, Any]) -> list[Any]:
    """Build per-field filter clauses from a Prisma filter object."""
    clauses = []
    for op, val in filters.items():
        if op == 'equals':
            clauses.append(col == val if val is not None else col.is_(None))
        elif op == 'not':
            if val is None:
                clauses.append(col.isnot(None))
            elif isinstance(val, dict):
                for inner in _build_field_filters(col, val):
                    from sqlalchemy import not_
                    clauses.append(not_(inner))
            else:
                clauses.append(col != val)
        elif op == 'in':
            clauses.append(col.in_(val))
        elif op == 'notIn':
            clauses.append(col.notin_(val))
        elif op == 'lt':
            clauses.append(col < val)
        elif op == 'lte':
            clauses.append(col <= val)
        elif op == 'gt':
            clauses.append(col > val)
        elif op == 'gte':
            clauses.append(col >= val)
        elif op == 'contains':
            mode = filters.get('mode', 'default')
            if mode == 'insensitive':
                clauses.append(col.ilike(f'%{val}%'))
            else:
                clauses.append(col.like(f'%{val}%'))
        elif op == 'startsWith':
            clauses.append(col.like(f'{val}%'))
        elif op == 'endsWith':
            clauses.append(col.like(f'%{val}'))
        elif op == 'mode':
            pass  # handled inside 'contains'
        elif op == 'search':
            clauses.append(col.like(f'%{val}%'))
        else:
            log.warning('Unknown filter operator %r', op)
    return clauses


def _apply_order_by(stmt: Any, table: Any, order_by: Any) -> Any:
    """Apply orderBy to a select statement."""
    if not order_by:
        return stmt

    if isinstance(order_by, dict):
        order_by = [order_by]

    for item in order_by:
        for field_name, direction in item.items():
            col = getattr(table, field_name, None)
            if col is None:
                continue
            if isinstance(direction, str) and direction.lower() == 'desc':
                stmt = stmt.order_by(col.desc())
            else:
                stmt = stmt.order_by(col.asc())
    return stmt


def _split_scalar_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract only scalar values from a create/update data dict,
    skipping nested relation operations (create, connect, connectOrCreate, etc.)
    """
    relation_ops = {'create', 'connect', 'disconnect', 'connectOrCreate', 'set', 'upsert', 'update', 'delete'}
    result = {}
    for key, val in data.items():
        if isinstance(val, dict) and set(val.keys()) & relation_ops:
            continue
        if isinstance(val, dict) and any(k.startswith('set') or k == 'increment' or k == 'decrement' or k == 'multiply' or k == 'divide' for k in val):
            # Atomic update operations
            atomic_val = val.get('set', val.get('increment', val.get('decrement', val.get('multiply', val.get('divide')))))
            if atomic_val is not None:
                result[key] = atomic_val
        else:
            result[key] = val
    return result


def _wrap_error(exc: Exception) -> Exception:
    """Map SQLAlchemy exceptions to Prisma error types."""
    exc_str = str(exc).lower()
    orig = getattr(exc, 'orig', None)
    orig_str = str(orig).lower() if orig else ''

    if 'unique' in exc_str or 'unique' in orig_str:
        return UniqueViolationError(str(exc))
    if 'foreign key' in exc_str or 'foreign key' in orig_str or 'violates foreign' in orig_str:
        return ForeignKeyViolationError(str(exc))
    return exc


# ---------------------------------------------------------------------------
# Base interpreter
# ---------------------------------------------------------------------------

class BaseInterpreter:
    def __init__(self, *, tables_module: ModuleType) -> None:
        self._tables = tables_module

    def _table(self, model_name: str) -> Any:
        return _get_table(self._tables, model_name)

    def _load_includes(
        self,
        session: Any,
        rows: list[dict[str, Any]],
        model_name: str,
        include: dict[str, Any] | None,
        *,
        is_async: bool = False,
    ) -> list[dict[str, Any]]:
        """Sync eager-load included relations. Not called for async — see AsyncQueryInterpreter."""
        if not include or not rows:
            return rows
        return rows  # placeholder — overridden below

    # ------------------------------------------------------------------
    # Shared query building logic (no I/O)
    # ------------------------------------------------------------------

    def _build_select_stmt(self, table: Any, req: QueryRequest) -> Any:
        from sqlalchemy import select

        args = req.arguments
        stmt = select(table)
        where = args.get('where')
        if where:
            clause = _build_where_clause(table, where)
            if clause is not None:
                stmt = stmt.where(clause)

        stmt = _apply_order_by(stmt, table, args.get('order_by') or args.get('orderBy'))

        take = args.get('take')
        if take is not None:
            stmt = stmt.limit(take)

        skip = args.get('skip')
        if skip is not None:
            stmt = stmt.offset(skip)

        return stmt

    def _build_count_stmt(self, table: Any, req: QueryRequest) -> Any:
        from sqlalchemy import select, func

        args = req.arguments
        stmt = select(func.count()).select_from(table)
        where = args.get('where')
        if where:
            clause = _build_where_clause(table, where)
            if clause is not None:
                stmt = stmt.where(clause)
        take = args.get('take')
        if take is not None:
            stmt = stmt.limit(take)
        skip = args.get('skip')
        if skip is not None:
            stmt = stmt.offset(skip)
        return stmt


# ---------------------------------------------------------------------------
# Sync interpreter
# ---------------------------------------------------------------------------

class SyncQueryInterpreter(BaseInterpreter):
    def __init__(self, *, session: Any, tables_module: ModuleType) -> None:
        super().__init__(tables_module=tables_module)
        self._session = session

    def execute(self, req: QueryRequest) -> dict[str, Any]:
        try:
            return self._dispatch(req)
        except (UniqueViolationError, ForeignKeyViolationError, RecordNotFoundError):
            raise
        except Exception as exc:
            try:
                from sqlalchemy.exc import IntegrityError
                if isinstance(exc, IntegrityError):
                    raise _wrap_error(exc) from exc
            except ImportError:
                pass
            raise

    def _dispatch(self, req: QueryRequest) -> dict[str, Any]:
        m = req.method
        if m == 'find_unique':
            return self._find_single(req, raise_if_missing=False)
        if m == 'find_unique_or_raise':
            return self._find_single(req, raise_if_missing=True)
        if m == 'find_first':
            return self._find_first(req, raise_if_missing=False)
        if m == 'find_first_or_raise':
            return self._find_first(req, raise_if_missing=True)
        if m == 'find_many':
            return self._find_many(req)
        if m == 'create':
            return self._create(req)
        if m == 'create_many':
            return self._create_many(req)
        if m == 'update':
            return self._update(req)
        if m == 'update_many':
            return self._update_many(req)
        if m == 'upsert':
            return self._upsert(req)
        if m == 'delete':
            return self._delete(req)
        if m == 'delete_many':
            return self._delete_many(req)
        if m == 'count':
            return self._count(req)
        if m == 'group_by':
            return self._group_by(req)
        if m == 'execute_raw':
            return self._execute_raw(req)
        if m == 'query_raw':
            return self._query_raw(req)
        raise NotImplementedError(f'Unsupported method: {m!r}')

    def _find_single(self, req: QueryRequest, *, raise_if_missing: bool) -> dict[str, Any]:
        table = self._table(req.model_name)
        stmt = self._build_select_stmt(table, req)
        stmt = stmt.limit(1)
        result = self._session.execute(stmt).scalars().first()
        if result is None:
            if raise_if_missing:
                raise RecordNotFoundError('Record not found')
            return _result(None)
        row_dict = _row_to_dict(result)
        row_dict = self._eager_load(row_dict, req.model_name, req.arguments.get('include'))
        return _result(row_dict)

    def _find_first(self, req: QueryRequest, *, raise_if_missing: bool) -> dict[str, Any]:
        return self._find_single(req, raise_if_missing=raise_if_missing)

    def _find_many(self, req: QueryRequest) -> dict[str, Any]:
        table = self._table(req.model_name)
        stmt = self._build_select_stmt(table, req)
        rows = self._session.execute(stmt).scalars().all()
        row_dicts = [_row_to_dict(r) for r in rows]
        row_dicts = self._eager_load_many(row_dicts, req.model_name, req.arguments.get('include'))
        return _result(row_dicts)

    def _create(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import insert

        table = self._table(req.model_name)
        data = _split_scalar_data(req.arguments.get('data') or {})
        stmt = insert(table.__table__).values(**data)
        self._session.execute(stmt)
        # Fetch the created row
        where_clause = _build_where_clause(table, {k: v for k, v in data.items() if k})
        from sqlalchemy import select
        fetch_stmt = select(table)
        if where_clause is not None:
            fetch_stmt = fetch_stmt.where(where_clause)
        fetch_stmt = fetch_stmt.order_by(*[
            col.desc() for col in table.__table__.columns if col.primary_key
        ]).limit(1)
        result = self._session.execute(fetch_stmt).scalars().first()
        row_dict = _row_to_dict(result)
        row_dict = self._eager_load(row_dict, req.model_name, req.arguments.get('include'))
        return _result(row_dict)

    def _create_many(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import insert

        table = self._table(req.model_name)
        data_list = req.arguments.get('data') or []
        scalar_rows = [_split_scalar_data(d) for d in data_list]
        if not scalar_rows:
            return _result({'count': 0})
        result = self._session.execute(insert(table.__table__), scalar_rows)
        return _result({'count': result.rowcount})

    def _update(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import update, select

        table = self._table(req.model_name)
        where_clause = _build_where_clause(table, req.arguments.get('where'))
        data = _split_scalar_data(req.arguments.get('data') or {})

        stmt = update(table.__table__).values(**data)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        result = self._session.execute(stmt)
        if result.rowcount == 0:
            raise RecordNotFoundError('Record not found')

        fetch_stmt = select(table)
        if where_clause is not None:
            fetch_stmt = fetch_stmt.where(where_clause)
        row = self._session.execute(fetch_stmt).scalars().first()
        row_dict = _row_to_dict(row)
        row_dict = self._eager_load(row_dict, req.model_name, req.arguments.get('include'))
        return _result(row_dict)

    def _update_many(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import update

        table = self._table(req.model_name)
        where_clause = _build_where_clause(table, req.arguments.get('where'))
        data = _split_scalar_data(req.arguments.get('data') or {})

        stmt = update(table.__table__).values(**data)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        result = self._session.execute(stmt)
        return _result({'count': result.rowcount})

    def _upsert(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import select

        table = self._table(req.model_name)
        where_clause = _build_where_clause(table, req.arguments.get('where'))

        fetch_stmt = select(table)
        if where_clause is not None:
            fetch_stmt = fetch_stmt.where(where_clause)
        existing = self._session.execute(fetch_stmt).scalars().first()

        if existing is None:
            create_data = _split_scalar_data(req.arguments.get('create') or {})
            from sqlalchemy import insert
            self._session.execute(insert(table.__table__).values(**create_data))
            row = self._session.execute(fetch_stmt).scalars().first()
        else:
            update_data = _split_scalar_data(req.arguments.get('update') or {})
            if update_data:
                from sqlalchemy import update
                stmt = update(table.__table__).values(**update_data)
                if where_clause is not None:
                    stmt = stmt.where(where_clause)
                self._session.execute(stmt)
            row = self._session.execute(fetch_stmt).scalars().first()

        row_dict = _row_to_dict(row)
        row_dict = self._eager_load(row_dict, req.model_name, req.arguments.get('include'))
        return _result(row_dict)

    def _delete(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import delete, select

        table = self._table(req.model_name)
        where_clause = _build_where_clause(table, req.arguments.get('where'))

        fetch_stmt = select(table)
        if where_clause is not None:
            fetch_stmt = fetch_stmt.where(where_clause)
        row = self._session.execute(fetch_stmt).scalars().first()
        if row is None:
            raise RecordNotFoundError('Record not found')

        row_dict = _row_to_dict(row)
        stmt = delete(table.__table__)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        self._session.execute(stmt)
        return _result(row_dict)

    def _delete_many(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import delete

        table = self._table(req.model_name)
        where_clause = _build_where_clause(table, req.arguments.get('where'))
        stmt = delete(table.__table__)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        result = self._session.execute(stmt)
        return _result({'count': result.rowcount})

    def _count(self, req: QueryRequest) -> dict[str, Any]:
        table = self._table(req.model_name)
        stmt = self._build_count_stmt(table, req)
        total = self._session.execute(stmt).scalar_one()
        root_selection = req.root_selection or []
        # Determine if select was used (client passes root_selection with field names)
        if root_selection and root_selection != ['_count { _all }'] and not any('_all' in s for s in root_selection):
            # Per-field counts not easily done without separate queries — approximate
            fields = [s.replace('_count { ', '').replace(' }', '').strip() for s in root_selection]
            field_counts = {f: total for f in fields if f and not f.startswith('_')}
            return _result({'_count': field_counts})
        return _result({'_count': {'_all': total}})

    def _group_by(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import select, func

        table = self._table(req.model_name)
        args = req.arguments
        by_fields = args.get('by') or []

        group_cols = [getattr(table, f) for f in by_fields if hasattr(table, f)]
        stmt = select(*group_cols, func.count().label('_count_all')).group_by(*group_cols)

        where_clause = _build_where_clause(table, args.get('where'))
        if where_clause is not None:
            stmt = stmt.where(where_clause)

        stmt = _apply_order_by(stmt, table, args.get('orderBy'))
        take = args.get('take')
        if take is not None:
            stmt = stmt.limit(take)
        skip = args.get('skip')
        if skip is not None:
            stmt = stmt.offset(skip)

        rows = self._session.execute(stmt).all()
        results = []
        for row in rows:
            mapping = dict(row._mapping)
            count_all = mapping.pop('_count_all', 0)
            mapping['_count'] = {'_all': count_all}
            results.append(mapping)
        return _result(results)

    def _execute_raw(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import text

        query = req.arguments.get('query', '')
        params = req.arguments.get('parameters') or []
        try:
            result = self._session.execute(text(query), list(params))
            return _result(result.rowcount)
        except Exception as exc:
            raise RawQueryError({'message': str(exc), 'code': 'raw_query_error'}) from exc

    def _query_raw(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import text

        query = req.arguments.get('query', '')
        params = req.arguments.get('parameters') or []
        try:
            result = self._session.execute(text(query), list(params))
            rows = [dict(r._mapping) for r in result.all()]
            return _result(rows)
        except Exception as exc:
            raise RawQueryError({'message': str(exc), 'code': 'raw_query_error'}) from exc

    # ------------------------------------------------------------------
    # Relation loading
    # ------------------------------------------------------------------

    def _eager_load(
        self,
        row_dict: dict[str, Any],
        model_name: str,
        include: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self._eager_load_many([row_dict], model_name, include)[0]

    def _eager_load_many(
        self,
        row_dicts: list[dict[str, Any]],
        model_name: str,
        include: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not include or not row_dicts:
            return row_dicts

        table = self._table(model_name)
        for relation_name, include_spec in include.items():
            if not include_spec:
                continue
            rel_attr = getattr(table, relation_name, None)
            if rel_attr is None:
                continue
            self._load_relation(row_dicts, table, model_name, relation_name, include_spec)
        return row_dicts

    def _load_relation(
        self,
        parent_rows: list[dict[str, Any]],
        parent_table: Any,
        parent_model_name: str,
        relation_name: str,
        include_spec: Any,
    ) -> None:
        from sqlalchemy import select, inspect as sa_inspect

        rel_prop = getattr(parent_table, relation_name).property
        related_table = rel_prop.mapper.class_

        related_model_name = type(related_table).__name__.replace('Table', '') if hasattr(related_table, '__tablename__') else relation_name

        # Find FK column mapping
        fk_pairs = []
        for fk in rel_prop.synchronize_pairs:
            local_col, remote_col = fk
            fk_pairs.append((local_col.key, remote_col.key))

        nested_include = include_spec.get('include') if isinstance(include_spec, dict) else None
        nested_where = include_spec.get('where') if isinstance(include_spec, dict) else None
        nested_order = include_spec.get('orderBy') if isinstance(include_spec, dict) else None
        nested_take = include_spec.get('take') if isinstance(include_spec, dict) else None
        nested_skip = include_spec.get('skip') if isinstance(include_spec, dict) else None

        is_list = rel_prop.uselist

        if not fk_pairs:
            return

        if is_list:
            # parent holds NO FK; child holds FK pointing to parent
            # Find the parent's PK column values
            parent_pk_col, child_fk_col = fk_pairs[0]
            parent_ids = list({row.get(parent_pk_col) for row in parent_rows if row.get(parent_pk_col) is not None})
            if not parent_ids:
                for row in parent_rows:
                    row[relation_name] = []
                return

            fk_col = getattr(related_table, child_fk_col)
            stmt = select(related_table).where(fk_col.in_(parent_ids))
            if nested_where:
                clause = _build_where_clause(related_table, nested_where)
                if clause is not None:
                    stmt = stmt.where(clause)
            stmt = _apply_order_by(stmt, related_table, nested_order)
            if nested_take is not None:
                stmt = stmt.limit(nested_take)
            if nested_skip is not None:
                stmt = stmt.offset(nested_skip)

            related_rows = self._session.execute(stmt).scalars().all()
            related_dicts = [_row_to_dict(r) for r in related_rows]

            if nested_include:
                related_dicts = self._eager_load_many(related_dicts, related_model_name, nested_include)

            # Group by FK
            from collections import defaultdict
            grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for r in related_dicts:
                grouped[r.get(child_fk_col)].append(r)

            for row in parent_rows:
                row[relation_name] = grouped.get(row.get(parent_pk_col), [])
        else:
            # parent holds FK; load single related object
            child_pk_col, parent_fk_col = fk_pairs[0]
            fk_values = list({row.get(parent_fk_col) for row in parent_rows if row.get(parent_fk_col) is not None})
            if not fk_values:
                for row in parent_rows:
                    row[relation_name] = None
                return

            pk_col = getattr(related_table, child_pk_col)
            stmt = select(related_table).where(pk_col.in_(fk_values))
            related_rows = self._session.execute(stmt).scalars().all()
            related_by_pk = {_row_to_dict(r).get(child_pk_col): _row_to_dict(r) for r in related_rows}

            if nested_include:
                related_list = list(related_by_pk.values())
                related_list = self._eager_load_many(related_list, related_model_name, nested_include)
                related_by_pk = {r.get(child_pk_col): r for r in related_list}

            for row in parent_rows:
                fk_val = row.get(parent_fk_col)
                row[relation_name] = related_by_pk.get(fk_val)


# ---------------------------------------------------------------------------
# Async interpreter
# ---------------------------------------------------------------------------

class AsyncQueryInterpreter(BaseInterpreter):
    def __init__(self, *, session: Any, tables_module: ModuleType) -> None:
        super().__init__(tables_module=tables_module)
        self._session = session

    async def execute(self, req: QueryRequest) -> dict[str, Any]:
        try:
            return await self._dispatch(req)
        except (UniqueViolationError, ForeignKeyViolationError, RecordNotFoundError):
            raise
        except Exception as exc:
            try:
                from sqlalchemy.exc import IntegrityError
                if isinstance(exc, IntegrityError):
                    raise _wrap_error(exc) from exc
            except ImportError:
                pass
            raise

    async def _dispatch(self, req: QueryRequest) -> dict[str, Any]:
        m = req.method
        if m == 'find_unique':
            return await self._find_single(req, raise_if_missing=False)
        if m == 'find_unique_or_raise':
            return await self._find_single(req, raise_if_missing=True)
        if m == 'find_first':
            return await self._find_first(req, raise_if_missing=False)
        if m == 'find_first_or_raise':
            return await self._find_first(req, raise_if_missing=True)
        if m == 'find_many':
            return await self._find_many(req)
        if m == 'create':
            return await self._create(req)
        if m == 'create_many':
            return await self._create_many(req)
        if m == 'update':
            return await self._update(req)
        if m == 'update_many':
            return await self._update_many(req)
        if m == 'upsert':
            return await self._upsert(req)
        if m == 'delete':
            return await self._delete(req)
        if m == 'delete_many':
            return await self._delete_many(req)
        if m == 'count':
            return await self._count(req)
        if m == 'group_by':
            return await self._group_by(req)
        if m == 'execute_raw':
            return await self._execute_raw(req)
        if m == 'query_raw':
            return await self._query_raw(req)
        raise NotImplementedError(f'Unsupported method: {m!r}')

    async def _find_single(self, req: QueryRequest, *, raise_if_missing: bool) -> dict[str, Any]:
        table = self._table(req.model_name)
        stmt = self._build_select_stmt(table, req)
        stmt = stmt.limit(1)
        result = (await self._session.execute(stmt)).scalars().first()
        if result is None:
            if raise_if_missing:
                raise RecordNotFoundError('Record not found')
            return _result(None)
        row_dict = _row_to_dict(result)
        row_dict = await self._eager_load(row_dict, req.model_name, req.arguments.get('include'))
        return _result(row_dict)

    async def _find_first(self, req: QueryRequest, *, raise_if_missing: bool) -> dict[str, Any]:
        return await self._find_single(req, raise_if_missing=raise_if_missing)

    async def _find_many(self, req: QueryRequest) -> dict[str, Any]:
        table = self._table(req.model_name)
        stmt = self._build_select_stmt(table, req)
        rows = (await self._session.execute(stmt)).scalars().all()
        row_dicts = [_row_to_dict(r) for r in rows]
        row_dicts = await self._eager_load_many(row_dicts, req.model_name, req.arguments.get('include'))
        return _result(row_dicts)

    async def _create(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import insert, select

        table = self._table(req.model_name)
        data = _split_scalar_data(req.arguments.get('data') or {})
        await self._session.execute(insert(table.__table__).values(**data))
        where_clause = _build_where_clause(table, {k: v for k, v in data.items() if k})
        fetch_stmt = select(table)
        if where_clause is not None:
            fetch_stmt = fetch_stmt.where(where_clause)
        fetch_stmt = fetch_stmt.order_by(*[
            col.desc() for col in table.__table__.columns if col.primary_key
        ]).limit(1)
        result = (await self._session.execute(fetch_stmt)).scalars().first()
        row_dict = _row_to_dict(result)
        row_dict = await self._eager_load(row_dict, req.model_name, req.arguments.get('include'))
        return _result(row_dict)

    async def _create_many(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import insert

        table = self._table(req.model_name)
        data_list = req.arguments.get('data') or []
        scalar_rows = [_split_scalar_data(d) for d in data_list]
        if not scalar_rows:
            return _result({'count': 0})
        result = await self._session.execute(insert(table.__table__), scalar_rows)
        return _result({'count': result.rowcount})

    async def _update(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import update, select

        table = self._table(req.model_name)
        where_clause = _build_where_clause(table, req.arguments.get('where'))
        data = _split_scalar_data(req.arguments.get('data') or {})
        stmt = update(table.__table__).values(**data)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        result = await self._session.execute(stmt)
        if result.rowcount == 0:
            raise RecordNotFoundError('Record not found')
        fetch_stmt = select(table)
        if where_clause is not None:
            fetch_stmt = fetch_stmt.where(where_clause)
        row = (await self._session.execute(fetch_stmt)).scalars().first()
        row_dict = _row_to_dict(row)
        row_dict = await self._eager_load(row_dict, req.model_name, req.arguments.get('include'))
        return _result(row_dict)

    async def _update_many(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import update

        table = self._table(req.model_name)
        where_clause = _build_where_clause(table, req.arguments.get('where'))
        data = _split_scalar_data(req.arguments.get('data') or {})
        stmt = update(table.__table__).values(**data)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        result = await self._session.execute(stmt)
        return _result({'count': result.rowcount})

    async def _upsert(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import select

        table = self._table(req.model_name)
        where_clause = _build_where_clause(table, req.arguments.get('where'))
        fetch_stmt = select(table)
        if where_clause is not None:
            fetch_stmt = fetch_stmt.where(where_clause)
        existing = (await self._session.execute(fetch_stmt)).scalars().first()

        if existing is None:
            create_data = _split_scalar_data(req.arguments.get('create') or {})
            from sqlalchemy import insert
            await self._session.execute(insert(table.__table__).values(**create_data))
            row = (await self._session.execute(fetch_stmt)).scalars().first()
        else:
            update_data = _split_scalar_data(req.arguments.get('update') or {})
            if update_data:
                from sqlalchemy import update
                stmt = update(table.__table__).values(**update_data)
                if where_clause is not None:
                    stmt = stmt.where(where_clause)
                await self._session.execute(stmt)
            row = (await self._session.execute(fetch_stmt)).scalars().first()

        row_dict = _row_to_dict(row)
        row_dict = await self._eager_load(row_dict, req.model_name, req.arguments.get('include'))
        return _result(row_dict)

    async def _delete(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import delete, select

        table = self._table(req.model_name)
        where_clause = _build_where_clause(table, req.arguments.get('where'))
        fetch_stmt = select(table)
        if where_clause is not None:
            fetch_stmt = fetch_stmt.where(where_clause)
        row = (await self._session.execute(fetch_stmt)).scalars().first()
        if row is None:
            raise RecordNotFoundError('Record not found')
        row_dict = _row_to_dict(row)
        stmt = delete(table.__table__)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        await self._session.execute(stmt)
        return _result(row_dict)

    async def _delete_many(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import delete

        table = self._table(req.model_name)
        where_clause = _build_where_clause(table, req.arguments.get('where'))
        stmt = delete(table.__table__)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        result = await self._session.execute(stmt)
        return _result({'count': result.rowcount})

    async def _count(self, req: QueryRequest) -> dict[str, Any]:
        table = self._table(req.model_name)
        stmt = self._build_count_stmt(table, req)
        total = (await self._session.execute(stmt)).scalar_one()
        root_selection = req.root_selection or []
        if root_selection and not any('_all' in s for s in root_selection):
            fields = [s.replace('_count { ', '').replace(' }', '').strip() for s in root_selection]
            field_counts = {f: total for f in fields if f and not f.startswith('_')}
            return _result({'_count': field_counts})
        return _result({'_count': {'_all': total}})

    async def _group_by(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import select, func

        table = self._table(req.model_name)
        args = req.arguments
        by_fields = args.get('by') or []
        group_cols = [getattr(table, f) for f in by_fields if hasattr(table, f)]
        stmt = select(*group_cols, func.count().label('_count_all')).group_by(*group_cols)
        where_clause = _build_where_clause(table, args.get('where'))
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        stmt = _apply_order_by(stmt, table, args.get('orderBy'))
        take = args.get('take')
        if take is not None:
            stmt = stmt.limit(take)
        skip = args.get('skip')
        if skip is not None:
            stmt = stmt.offset(skip)
        rows = (await self._session.execute(stmt)).all()
        results = []
        for row in rows:
            mapping = dict(row._mapping)
            count_all = mapping.pop('_count_all', 0)
            mapping['_count'] = {'_all': count_all}
            results.append(mapping)
        return _result(results)

    async def _execute_raw(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import text

        query = req.arguments.get('query', '')
        params = req.arguments.get('parameters') or []
        try:
            result = await self._session.execute(text(query), list(params))
            return _result(result.rowcount)
        except Exception as exc:
            raise RawQueryError({'message': str(exc), 'code': 'raw_query_error'}) from exc

    async def _query_raw(self, req: QueryRequest) -> dict[str, Any]:
        from sqlalchemy import text

        query = req.arguments.get('query', '')
        params = req.arguments.get('parameters') or []
        try:
            result = await self._session.execute(text(query), list(params))
            rows = [dict(r._mapping) for r in result.all()]
            return _result(rows)
        except Exception as exc:
            raise RawQueryError({'message': str(exc), 'code': 'raw_query_error'}) from exc

    async def _eager_load(
        self,
        row_dict: dict[str, Any],
        model_name: str,
        include: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result = await self._eager_load_many([row_dict], model_name, include)
        return result[0]

    async def _eager_load_many(
        self,
        row_dicts: list[dict[str, Any]],
        model_name: str,
        include: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not include or not row_dicts:
            return row_dicts
        table = self._table(model_name)
        for relation_name, include_spec in include.items():
            if not include_spec:
                continue
            rel_attr = getattr(table, relation_name, None)
            if rel_attr is None:
                continue
            await self._load_relation(row_dicts, table, model_name, relation_name, include_spec)
        return row_dicts

    async def _load_relation(
        self,
        parent_rows: list[dict[str, Any]],
        parent_table: Any,
        parent_model_name: str,
        relation_name: str,
        include_spec: Any,
    ) -> None:
        from sqlalchemy import select
        from collections import defaultdict

        rel_prop = getattr(parent_table, relation_name).property
        related_table = rel_prop.mapper.class_
        related_model_name = related_table.__name__.replace('Table', '')

        fk_pairs = []
        for fk in rel_prop.synchronize_pairs:
            local_col, remote_col = fk
            fk_pairs.append((local_col.key, remote_col.key))

        if not fk_pairs:
            return

        nested_include = include_spec.get('include') if isinstance(include_spec, dict) else None
        nested_where = include_spec.get('where') if isinstance(include_spec, dict) else None
        nested_order = include_spec.get('orderBy') if isinstance(include_spec, dict) else None
        nested_take = include_spec.get('take') if isinstance(include_spec, dict) else None
        nested_skip = include_spec.get('skip') if isinstance(include_spec, dict) else None

        is_list = rel_prop.uselist

        if is_list:
            parent_pk_col, child_fk_col = fk_pairs[0]
            parent_ids = list({row.get(parent_pk_col) for row in parent_rows if row.get(parent_pk_col) is not None})
            if not parent_ids:
                for row in parent_rows:
                    row[relation_name] = []
                return

            fk_col = getattr(related_table, child_fk_col)
            stmt = select(related_table).where(fk_col.in_(parent_ids))
            if nested_where:
                clause = _build_where_clause(related_table, nested_where)
                if clause is not None:
                    stmt = stmt.where(clause)
            stmt = _apply_order_by(stmt, related_table, nested_order)
            if nested_take is not None:
                stmt = stmt.limit(nested_take)
            if nested_skip is not None:
                stmt = stmt.offset(nested_skip)

            related_rows = (await self._session.execute(stmt)).scalars().all()
            related_dicts = [_row_to_dict(r) for r in related_rows]

            if nested_include:
                related_dicts = await self._eager_load_many(related_dicts, related_model_name, nested_include)

            grouped: dict[Any, list] = defaultdict(list)
            for r in related_dicts:
                grouped[r.get(child_fk_col)].append(r)

            for row in parent_rows:
                row[relation_name] = grouped.get(row.get(parent_pk_col), [])
        else:
            child_pk_col, parent_fk_col = fk_pairs[0]
            fk_values = list({row.get(parent_fk_col) for row in parent_rows if row.get(parent_fk_col) is not None})
            if not fk_values:
                for row in parent_rows:
                    row[relation_name] = None
                return

            pk_col = getattr(related_table, child_pk_col)
            stmt = select(related_table).where(pk_col.in_(fk_values))
            related_rows = (await self._session.execute(stmt)).scalars().all()
            related_by_pk = {_row_to_dict(r).get(child_pk_col): _row_to_dict(r) for r in related_rows}

            if nested_include:
                related_list = list(related_by_pk.values())
                related_list = await self._eager_load_many(related_list, related_model_name, nested_include)
                related_by_pk = {r.get(child_pk_col): r for r in related_list}

            for row in parent_rows:
                fk_val = row.get(parent_fk_col)
                row[relation_name] = related_by_pk.get(fk_val)
