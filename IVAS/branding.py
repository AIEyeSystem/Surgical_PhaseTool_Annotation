"""Single source of truth for IVAS product-facing identity.

The Python package, plugin entry points, model configuration targets, UI, files,
and build artifacts use the IVAS identity defined here.
"""

APP_NAME = "IVAS"
APP_DESCRIPTION = "Visual annotation toolkit powered by Segment Anything"
AUTHOR = "zack zhao"
COPYRIGHT = "Copyright (C) 2026 zack zhao"

DISTRIBUTION_NAME = "ivas-sam"
PLUGIN_ENTRYPOINT_GROUP = "ivas.plugins"
LEGACY_PLUGIN_ENTRYPOINT_GROUPS = ("ivat.plugins",)

ANNOTATION_FORMAT = "IVAS"
LEGACY_ANNOTATION_FORMATS = frozenset({"IVAT"})
SUPPORTED_ANNOTATION_FORMATS = frozenset(
    {ANNOTATION_FORMAT, *LEGACY_ANNOTATION_FORMATS}
)

CATEGORY_CONFIG_NAME = "ivas.yaml"
LEGACY_CATEGORY_CONFIG_NAMES = ("ivat.yaml", "ivas_legacy_categories.yaml")
VIDEO_PROJECT_NAME = "project.ivas.json"
LEGACY_VIDEO_PROJECT_NAMES = ("project.ivat.json",)

LOGO_RESOURCE = ":/icon/logo/ivas.png"
