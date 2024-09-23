from __future__ import annotations


class SqliteBackendFactory:
    @staticmethod
    def pickle(
        actor: Actor,
        compression: Optional[Compression] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        filename: Optional[str] = None,
        log_serde_time: bool | str = False,
    ):
        backend = SqliteBackend(
            ser=pickle.dumps,
            deser=pickle.loads,
            dbdir=actor.actor_dir,
            filename=filename,
            compression=compression,
        )
        return wrap_backend(backend, mem_persist, log_serde_time)
