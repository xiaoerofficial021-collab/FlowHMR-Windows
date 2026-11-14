"""Portable character files and preview preferences, independent of the desktop port."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from shutil import rmtree
import struct
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Lock
from time import sleep
from uuid import uuid4

from .character_files import MAX_MODEL_BYTES, MAX_TEXTURE_BYTES, MODEL_EXTENSIONS, texture_parts

_ID = re.compile(r"[0-9a-f]{64}")
_RETIRED = re.compile(r"\.removed-[0-9a-f]{32}")
_MAX_BUNDLE_BYTES = 1 << 30
THUMBNAIL_NAME = "thumbnail.png"
MAX_THUMBNAIL_BYTES = 512 << 10
MAX_THUMBNAIL_SIDE = 512
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def thumbnail_is_valid(data):
    """Accept only a small, real PNG: the file is served back to the web UI."""
    if not isinstance(data, (bytes, bytearray)) or not 33 <= len(data) <= MAX_THUMBNAIL_BYTES:
        return False
    if bytes(data[:8]) != _PNG_SIGNATURE or bytes(data[12:16]) != b"IHDR":
        return False
    width, height = struct.unpack(">II", bytes(data[16:24]))
    return 0 < width <= MAX_THUMBNAIL_SIDE and 0 < height <= MAX_THUMBNAIL_SIDE



def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def _pose(value):
    def triple(vector, limit):
        if not isinstance(vector, list) or len(vector) != 3:
            return [0, 0, 0]
        return [max(-limit, min(limit, x)) if type(x) in (int, float)
                and (type(x) is int or isfinite(x)) else 0 for x in vector]

    value = value if isinstance(value, dict) else {}
    rotations = value.get("rotations")
    rotations = rotations if isinstance(rotations, dict) else {}
    normalized = {}
    for key in map(str, range(37)):
        angles = triple(rotations.get(key), 180)
        if any(angles):
            normalized[key] = angles
    pose = {"rotations": normalized, "mirror": value.get("mirror") is True,
            "ikMode": "ik" if value.get("ikMode") == "ik" else "fk",
            "rootRotation": triple(value.get("rootRotation"), 180),
            "rootTranslation": triple(value.get("rootTranslation"), 5)}
    baseline = value.get("pmxBaselineVersion")
    if type(baseline) is int and baseline == 1:
        pose["pmxBaselineVersion"] = baseline
    return pose


class CharacterLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self._lock = Lock()
        self._checked_retired = False

    def _entry(self, identifier):
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
            return None
        directory = self.root / identifier
        canonical = self.root.resolve() / identifier
        # A manually moved or replaced entry must never grant access outside the library.
        if not directory.is_dir() or directory.resolve() != canonical:
            return None
        manifest_path = directory / "character.json"
        if manifest_path.resolve() != canonical / "character.json":
            return None
        record = _read_json(manifest_path)
        filename = record.get("filename")
        if (not isinstance(filename, str) or texture_parts(filename) != (filename,)
                or Path(filename).suffix.lower() not in MODEL_EXTENSIONS):
            return None
        file = directory / "files" / filename
        if not file.is_file() or file.resolve() != canonical / "files" / filename:
            return None
        name = record.get("name")
        thumbnail = directory / THUMBNAIL_NAME
        try:
            size = file.stat().st_size
        except OSError:
            return None
        return {"id": identifier, "filename": filename, "name": name if isinstance(name, str) and name else filename,
                "kind": file.suffix[1:].lower(), "bytes": size,
                "hasThumb": thumbnail.is_file() and thumbnail.resolve() == canonical / THUMBNAIL_NAME}, file

    def open(self, identifier, selections):
        with self._lock:
            entry = self._entry(identifier)
            if entry is None:
                raise FileNotFoundError("Character is not in the library")
            return selections.register(str(entry[1].resolve()))

    def save(self, selections, ticket, name, thumbnail=None):
        model, textures = selections.bundle(ticket)
        if texture_parts(model.name) != (model.name,):
            raise ValueError("Invalid character filename")
        label = " ".join(name.split())[:200] if isinstance(name, str) else ""
        record = {"filename": model.name, "name": label or model.stem}
        if thumbnail is not None and not thumbnail_is_valid(thumbnail):
            raise ValueError("Invalid character thumbnail")
        self.root.mkdir(parents=True, exist_ok=True)
        digest = sha256()
        total = 0
        with TemporaryDirectory(prefix=".import-", dir=self.root) as scratch:
            entry = Path(scratch) / "entry"
            for relative, source, limit in [
                (model.name, model, MAX_MODEL_BYTES),
                *((path, file, MAX_TEXTURE_BYTES) for path, file in sorted(textures)),
            ]:
                parts = texture_parts(relative)
                candidate = model.parent.joinpath(*parts)
                if (not parts or source.resolve(strict=True) != source
                        or candidate.resolve(strict=True) != source or not source.is_relative_to(model.parent)
                        or not source.is_file() or not 0 < source.stat().st_size <= limit):
                    raise ValueError("Character file changed while importing")
                dest = entry.joinpath("files", *parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                file_digest, size = sha256(), 0
                with source.open("rb") as src, dest.open("xb") as out:
                    while block := src.read(1 << 20):
                        size += len(block)
                        total += len(block)
                        if size > limit or total > _MAX_BUNDLE_BYTES:
                            raise ValueError("Character bundle is too large")
                        out.write(block)
                        file_digest.update(block)
                if size == 0:
                    raise ValueError("Character file changed while importing")
                digest.update(relative.encode("utf-8") + b"\0" + file_digest.digest())
            identifier = digest.hexdigest()
            if thumbnail is not None:
                (entry / THUMBNAIL_NAME).write_bytes(bytes(thumbnail))
            (entry / "character.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            with self._lock:
                existing = self._entry(identifier)
                if existing is not None:
                    return existing[0]
                entry.rename(self.root / identifier)
        return {"id": identifier, **record, "kind": model.suffix[1:].lower()}

    @property
    def _preferences(self):
        return self.root / ".preview.json"

    @staticmethod
    def _purge_retired(root, directory):
        # Only unlink an app-owned staging directory inside this library.
        try:
            if (directory.parent != root or not _RETIRED.fullmatch(directory.name)
                    or not directory.is_dir() or directory.resolve(strict=True) != directory):
                return False
            rmtree(directory)
            return True
        except OSError:
            return False

    def remove(self, identifier, selections):
        with self._lock:
            entry = self._entry(identifier)
            if entry is None:
                raise FileNotFoundError("Character is not in the library")
            root = self.root.resolve(strict=True)
            directory = root / identifier
            if directory.resolve(strict=True) != directory:
                raise ValueError("Character directory is outside the library")
            state = _read_json(self._preferences)
            for record in state.get("actors", {}).values():
                if record.get("id") == identifier:
                    record["id"] = None
                record.get("poses", {}).pop(identifier, None)
            defaults = state.get("defaults", {})
            for slot in [slot for slot, value in defaults.items() if value.get("id") == identifier]:
                del defaults[slot]
            state.get("presets", {}).pop(identifier, None)

            # Move this bundle out of the catalog first. Roll back the move if
            # saving the updated assignments fails, before removing any files.
            retired = root / f".removed-{uuid4().hex}"
            if retired.resolve() != retired or retired.exists():
                raise ValueError("Invalid character removal directory")
            directory.rename(retired)
            try:
                self._write(state)
            except BaseException:
                retired.rename(directory)
                raise
            selections.revoke_directory(directory)
            return self._purge_retired(root, retired)

    def snapshot(self, actor_ids):
        models = []
        with self._lock:
            if self.root.is_dir():
                root = self.root.resolve(strict=True)
                for directory in root.iterdir():
                    if _RETIRED.fullmatch(directory.name):
                        if not self._checked_retired:
                            self._purge_retired(root, directory)
                        continue
                    entry = self._entry(directory.name)
                    if entry is not None:
                        models.append(entry[0])
                self._checked_retired = True
            state = _read_json(self._preferences)
        models.sort(key=lambda item: (item["name"].casefold(), item["filename"].casefold(), item["id"]))
        actors = state.get("actors", {})
        return {"models": models, "defaults": state.get("defaults", {}), "presets": state.get("presets", {}),
                "actors": {identifier: actors[identifier] for identifier in actor_ids if identifier in actors}}

    def write_thumbnail(self, identifier, data):
        """Store a rendered thumbnail for an entry that does not have one yet."""
        if not thumbnail_is_valid(data):
            raise ValueError("Invalid character thumbnail")
        with self._lock:
            if self._entry(identifier) is None:
                raise FileNotFoundError("Character is not in the library")
            directory = self.root / identifier
            target = directory / THUMBNAIL_NAME
            stream = directory / f".thumb-{uuid4().hex}.tmp"
            try:
                stream.write_bytes(bytes(data))
                stream.replace(target)
            finally:
                stream.unlink(missing_ok=True)
        return True

    def thumbnail_path(self, identifier):
        with self._lock:
            if self._entry(identifier) is None:
                return None
        path = self.root / identifier / THUMBNAIL_NAME
        try:
            return path if path.is_file() and 0 < path.stat().st_size <= MAX_THUMBNAIL_BYTES else None
        except OSError:
            return None

    def set_actor(self, actor, slot, identifier, offsets, *, select):
        with self._lock:
            if identifier is not None and self._entry(identifier) is None:
                raise FileNotFoundError("Character is not in the library")
            state = _read_json(self._preferences)
            defaults = state.setdefault("defaults", {})
            actors = state.setdefault("actors", {})
            record = actors.setdefault(actor, {"id": identifier, "poses": {}})
            if not select and record["id"] != identifier:
                return
            record["id"] = identifier
            offsets = _pose(offsets)
            if identifier is not None:
                record["poses"][identifier] = offsets
                if not select:
                    state.setdefault("presets", {})[identifier] = offsets
            if select or defaults.get(slot, {}).get("actor") == actor:
                defaults[slot] = {"actor": actor, "id": identifier, "offsets": offsets}
            self._write(state)

    def forget_actors(self, ids):
        if not ids:
            return
        with self._lock:
            state = _read_json(self._preferences)
            actors = state.get("actors", {})
            removed = any(identifier in actors for identifier in ids)
            if removed:
                for identifier in ids:
                    actors.pop(identifier, None)
                self._write(state)

    def _write(self, state):
        self.root.mkdir(parents=True, exist_ok=True)
        stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root, prefix=".preview-",
                                    suffix=".tmp", delete=False)
        path = Path(stream.name)
        try:
            with stream:
                json.dump(state, stream, ensure_ascii=False, separators=(",", ":"))
            for delay in (0.01, 0.025, 0.05, 0.1, 0.2, None):
                try:
                    path.replace(self._preferences)
                    break
                except PermissionError as error:
                    # Windows readers can momentarily block an atomic replace.
                    if delay is None or getattr(error, "winerror", None) not in (5, 32, 33):
                        raise
                    sleep(delay)
        finally:
            path.unlink(missing_ok=True)
