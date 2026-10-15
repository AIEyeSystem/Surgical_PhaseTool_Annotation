---
version: "1.0"
name: "IVAS Visual Annotation Toolkit"
description: "A focused desktop workspace for temporal video annotation, frame tracking, and precise image segmentation."
colors:
  canvas: "#111827"
  canvas-raised: "#1F2937"
  surface: "#F6F8FB"
  surface-raised: "#FFFFFF"
  surface-subtle: "#E8EEF4"
  text: "#172033"
  text-muted: "#526174"
  border: "#718096"
  border-subtle: "#CBD5E1"
  primary: "#087F8C"
  primary-hover: "#066B76"
  primary-active: "#075B66"
  on-primary: "#FFFFFF"
  danger: "#B42318"
  danger-hover: "#8F1C14"
  danger-active: "#731610"
  on-danger: "#FFFFFF"
  focus: "#087F8C"
  selection: "#C9E9ED"
  on-selection: "#0B3E45"
  disabled-bg: "#E2E8EF"
  disabled-text: "#526174"
  timeline-phase: "#21A6B3"
  timeline-task: "#F4A340"
  success: "#207A4B"
  warning: "#9A6700"
typography:
  body:
    fontFamily: "Segoe UI, Microsoft YaHei UI, sans-serif"
    fontSize: 10pt
    fontWeight: 400
    lineHeight: 1.35
  label:
    fontFamily: "Segoe UI, Microsoft YaHei UI, sans-serif"
    fontSize: 10pt
    fontWeight: 600
    lineHeight: 1.3
  numeric:
    fontFamily: "Segoe UI, sans-serif"
    fontSize: 10pt
    fontWeight: 400
    lineHeight: 1.3
rounded:
  sm: "4px"
  md: "6px"
spacing:
  xs: "3px"
  sm: "6px"
  md: "10px"
  lg: "16px"
---

## Overview

IVAS is a canvas-first annotation interface for annotators who move rapidly
between video phases, an image canvas, category controls, model state, and saved
geometry. The main window stays intentionally sparse: frequent actions live in
compact toolbars, while complete and advanced workflows remain discoverable in
menus, dropdowns, docks, and focused dialogs.
The visual thesis is a calm clinical instrument: a dark inspection field keeps
the image and coloured masks dominant, while cool light panels and a restrained
teal action color keep controls legible. The signature is the contrast between
the imaging field and the surrounding measurement console.

## Colors

The canvas uses near-black slate to avoid competing with image data and mask
colors. Panels use tonal light surfaces and quiet blue-grey boundaries. Teal is
reserved for focus, selection, and active tools; category colors remain owned by
annotation data. Error, warning, and success colors communicate state and must
not be used as decoration.

## Typography

Segoe UI is the primary Windows interface face, with Microsoft YaHei UI as the
CJK fallback. Labels use weight rather than multiple sizes for hierarchy.
Changing GPU, coordinate, and image values should use aligned numerals where a
widget supports them.

## Layout

The active media is always the focal region. Mode is the first toolbar dropdown,
before Project, and displays Video or Frame from the actual central surface.
File owns open/save/import/export; Project owns directory settings and media
relinking. The central surface shows either the image canvas or
video preview for the active case, while shared temporal controls, case panels,
transport, and the Phase/Task timeline stay in the main-window shell. Opening an
indexed video frame must retain that case context rather than creating a second
workspace. Selecting Frame within Video extracts the playhead frame before
changing Mode; failed or cancelled transitions retain the current editor.
Standalone Frame work disables the retained video's commands. Within a video
frame, canvas shortcuts remain frame operations; video shortcuts apply when the
video panels or transport have focus. Video undo restores only annotations and
templates, preserving media, frame links and tracking state.
Toolbars provide the frequent path and labelled dropdown menus
provide the complete path. At the verified compact 640×480 size, secondary
controls remain reachable through toolbar overflow and dock scrolling.
Right-hand inspector docks default to 360 logical pixels. Their tabs use normal
body text, a padded target and a selected underline. Labels never elide or shrink
to fit; overflow uses visible scroll arrows, including after language/layout changes.

Tool dialogs use `ResponsiveDialog` to bound their decorated window to the
current screen and expose overflow through a scroll area. Simple action footers
stay outside that area; shortcut editors reflow without recreating their values.
Only the active central editor contributes a minimum size. The temporal inspector
scrolls, the video surface has no large fixed height, and transport controls adapt
to their own host width, independently of the hidden video page. Timeline lane
captions follow lane geometry at compact heights. The image canvas follows resize
only while fit-to-window is active; manual zoom remains unchanged.

Semantic and instance previews disable annotation mutations at the action,
canvas-entry and asynchronous-result boundaries. Selecting another drawing tool
must preserve an unfinished shape. Changing language or synchronizing video state
must preserve label text selection, custom shortcuts and pending intervals.

During extraction and frame editing, seek controls are disabled and backend
position updates cannot move the annotation playhead. In Frame, the transport's
play button becomes Return to video, keeping the exit visible even on small
windows. Returning preserves pending intervals and seeks to the edited frame;
playback resumes only on the next Play command. Point/box inference runs outside
the UI thread, retaining at most one pending prompt set. Finish waits for the
latest result; Cancel invalidates it immediately. Same-frame reuse requires
unchanged image and label file signatures.

## Elevation & Depth

Depth is expressed with tonal surface steps and one-pixel boundaries. Shadows
are intentionally avoided because this is a focused desktop tool, not a layered
marketing surface.

## Shapes

Controls use 4–6 px radii, consistent icon sizes, and visible one-pixel
boundaries. Data swatches and mask colors remain square enough to read as
samples rather than decorative badges.

## Components

Buttons expose hover, pressed, focused, checked, disabled, and busy states.
Lists expose hover, keyboard focus, and selected rows. The active annotation
tool uses the same teal selection language as focused inputs. Long-running SAM
operations must preserve a visible progress or status state and restore all
controls on success or failure.
Temporal intervals use text and lane position in addition to colour. Extracting
a video frame must produce a visible status message before the image canvas is
shown.
Template management, import/export, validation, project relinking, and tracking
setup belong in the labelled overflow menu or focused secondary dialogs. A
tracking run must visibly distinguish running, staged, committed, cancelled,
failed, and rolled-back states; generated masks must never look accepted before
commit. Cancellable media and tracking jobs restore enabled controls on success,
failure, cancellation, and stale-project replacement.

## Do's and Don'ts

- Keep category colors for segmentation semantics; do not reuse them as chrome.
- Keep the dark canvas visually quiet so masks remain the strongest color.
- Keep only frequent actions visible; place advanced operations in labelled
  menus, dropdowns, docks, or focused dialogs.
- Preserve keyboard-visible focus and the existing shortcut-heavy workflow.
- Route shortcuts by the active case and focused editor; Space controls video
  playback only when a video surface is active and never changes hidden state.
- Do not add large cards, gradients, ornamental shadows, or animation that
  consumes attention during repetitive annotation.
