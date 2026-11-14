"""Short-lived access to a character selected in the desktop file picker."""
from dataclasses import dataclass, field
from os.path import normcase
from pathlib import Path
import re
from secrets import token_urlsafe
from threading import Lock
from time import monotonic


KEY_ENV = "FLOWHMR_DESKTOP_CHARACTER_KEY"
MODEL_EXTENSIONS = {".pmx", ".fbx"}
TEXTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp", ".sph", ".spa", ".dds"}
MAX_MODEL_BYTES = 256 << 20
MAX_TEXTURE_BYTES = 64 << 20
LIFETIME_SECONDS = 600
MAX_TICKETS = 24
MAX_TEXTURES = 2048
_INVALID_PATH = re.compile(r'[<>:"|?*\x00-\x1f]')
_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.I)


def texture_parts(value):
    if (not isinstance(value, str) or not value or len(value) > 2048
            or value[0] in "/\\" or _INVALID_PATH.search(value)):
        return ()
    parts = tuple(part for part in value.replace("\\", "/").split("/") if part and part != ".")
    if any(part.endswith((".", " ")) or _DEVICE.match(part) for part in parts):
        return ()
    return parts


@dataclass
class Selection:
    expires: float
    model: Path
    textures: dict = field(default_factory=dict)
    too_many_textures: bool = False


class CharacterFiles:
    def __init__(self):
        self._tickets = {}
        self._lock = Lock()

    def register(self, value):
        if not isinstance(value, str) or not value or len(value) > 4096:
            raise ValueError("Invalid character file path")
        path = Path(value)
        if not path.is_absolute() or path.suffix.lower() not in MODEL_EXTENSIONS:
            raise ValueError("Select a PMX or FBX")
        path = path.resolve(strict=True)
        if path.suffix.lower() not in MODEL_EXTENSIONS or not path.is_file():
            raise ValueError("Not a character file")
        size = path.stat().st_size
        if not 0 < size <= MAX_MODEL_BYTES:
            raise ValueError("Character file size out of range")
        now = monotonic()
        wanted = normcase(str(path))
        with self._lock:
            self._tickets = {key: entry for key, entry in self._tickets.items() if entry.expires > now}
            # Reuse the ticket for a path that is already registered. The resource
            # URL then stays stable across character switches, which is what lets
            # the browser serve the model and its textures from cache instead of
            # re-downloading them every time.
            for ticket, entry in self._tickets.items():
                if normcase(str(entry.model)) == wanted:
                    entry.expires = now + LIFETIME_SECONDS
                    return {"ticket": ticket, "name": path.name}
            if len(self._tickets) >= MAX_TICKETS:
                del self._tickets[next(iter(self._tickets))]
            ticket = token_urlsafe(32)
            self._tickets[ticket] = Selection(now + LIFETIME_SECONDS, path)
        return {"ticket": ticket, "name": path.name}

    def bundle(self, ticket):
        """Snapshot the relative paths actually requested by the model loader."""
        with self._lock:
            entry = self._tickets.get(ticket)
            if entry is None or entry.expires <= monotonic():
                raise ValueError("Character selection expired")
            if entry.too_many_textures:
                raise ValueError("Too many external character textures")
            return entry.model, tuple(entry.textures.values())

    def revoke_directory(self, directory):
        # Tickets store the resolved model path, so a caller handing over a
        # junction or symlinked directory must be matched on its resolved form
        # too, or the entries would linger until they expire.
        candidates = [Path(directory)]
        try:
            candidates.append(Path(directory).resolve())
        except (OSError, RuntimeError):
            pass
        with self._lock:
            self._tickets = {key: entry for key, entry in self._tickets.items()
                             if not any(entry.model.is_relative_to(candidate) for candidate in candidates)}

    def resource(self, ticket, texture=None):
        with self._lock:
            entry = self._tickets.get(ticket)
            if entry is None or entry.expires <= monotonic():
                return None
        model = entry.model
        try:
            if texture is None:
                # Do not follow a replacement link after the file was approved.
                return model if model.resolve(strict=True) == model and model.is_file() \
                    and 0 < model.stat().st_size <= MAX_MODEL_BYTES else None
            parts = texture_parts(texture)
            if not parts:
                return None
            candidate = model.parent.joinpath(*parts)
            if candidate.suffix.lower() not in TEXTURE_EXTENSIONS:
                return None
            path = candidate.resolve(strict=True)
            if (path.is_relative_to(model.parent) and path.suffix.lower() in TEXTURE_EXTENSIONS
                    and path.is_file() and 0 < path.stat().st_size <= MAX_TEXTURE_BYTES):
                relative = "/".join(parts)
                with self._lock:
                    key = normcase(relative)
                    if key not in entry.textures and len(entry.textures) >= MAX_TEXTURES:
                        entry.too_many_textures = True
                    else:
                        entry.textures.setdefault(key, (relative, path))
                return path
        except (OSError, ValueError, RuntimeError):
            return None
        return None
