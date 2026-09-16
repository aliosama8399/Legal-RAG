from ..config import Settings, settings as default_settings


class BaseController:
    """Shared base for controllers: settings access and small helpers."""

    def __init__(self, settings: Settings = default_settings) -> None:
        self.app_settings = settings
