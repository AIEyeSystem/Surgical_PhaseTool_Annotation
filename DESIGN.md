---
version: "1.0"
name: "IVAS Visual Annotation Toolkit"
description: "A focused desktop workspace for temporal video annotation, frame tracking, and precise image segmentation."
colors:
  canvas: "#0B1220"
  canvas-raised: "#16233A"
  canvas-text: "#C9D6EA"
  chrome: "#FFFFFF"
  surface: "#F3F7FC"
  surface-raised: "#FFFFFF"
  surface-subtle: "#E7EEF8"
  input: "#FFFFFF"
  hover: "#EAF2FE"
  pressed: "#D6E6FD"
  text: "#0F1E36"
  text-muted: "#50607A"
  icon: "#2D4C7C"
  border: "#7D8DA6"
  border-subtle: "#D5DEEA"
  primary: "#1463D8"
  primary-hover: "#0F55BE"
  primary-active: "#0B469F"
  on-primary: "#FFFFFF"
  danger: "#C42B22"
  danger-hover: "#A3231C"
  danger-active: "#831C16"
  on-danger: "#FFFFFF"
  focus: "#1463D8"
  selection: "#DCEAFE"
  on-selection: "#0A3D8F"
  disabled-bg: "#EEF2F7"
  disabled-text: "#97A3B5"
  timeline-phase: "#3B8BEB"
  timeline-task: "#F2A93B"
  success: "#1C7C54"
  warning: "#A15F00"
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
the image and coloured masks dominant, while the surrounding console stays quiet
and legible. Five bundled themes share one set of semantic roles and one
stylesheet template (`IVAS/theme.py`); the frontmatter lists the default,
Medical Blue. Users switch themes in View → Theme or Application → Theme; the
choice is stored as `software.theme` and applied before the first window paints.

| Key | Name | Character |
| --- | --- | --- |
| `medical` | 医学蓝白 Medical Blue | White console, clinical blue actions (default) |
| `geek` | 极客终端 Geek Terminal | Near-black, neon-green accent, monospace UI |
| `graphite` | 午夜石墨 Midnight Graphite | Neutral dark grey with a blue accent |
| `surgical` | 手术室青 Surgical Teal | The original light teal console, refined |
| `aurora` | 北极光 Nordic Aurora | Cool blue-grey dark with frost accents |

## Colors

The canvas is always dark to avoid competing with image data and mask colors;
`canvas-text` is the only role for marks drawn on it (timeline captions,
playhead). `chrome` colours the menu bar, toolbars, tab strips and status bar;
`hover`/`pressed` are the only tool-button fills, so disabled controls fade to
`disabled-text` instead of becoming grey blocks. The primary colour is reserved
for focus, the checked Mode segment, primary buttons and Export; category colors
remain owned by annotation data. Error, warning, and success colors communicate
state and must not be used as decoration. Every theme must meet the contrast
pairs in `theme-matrix.json` (text 4.5:1, icons and boundaries 3:1); the
regression test enforces them.

Icons come from one stroke glyph set (`IVAS/icon_set.py`, 24×24, 1.75 px round
strokes) rendered per theme: `icon` normally, `disabled-text` when disabled,
`on-selection` when checked, `on-primary` on filled controls.

## Typography

Segoe UI is the primary Windows interface face, with Microsoft YaHei UI as the
CJK fallback. Labels use weight rather than multiple sizes for hierarchy.
Changing GPU, coordinate, and image values should use aligned numerals where a
widget supports them.

## Layout

The active media is always the focal region. Mode is the first toolbar control:
a two-segment Video | Frame switch that mirrors the actual central surface (also
in View → Mode). There is no Project toolbar menu; label/output directories and
relinking are File commands.
File groups Image, Video and General commands under visible headings separated
by dashed rules. Image contains image and frame-label directories; Video contains
media/project opening, output directory, relinking and video annotation exchange.
General contains Save, Export dataset…, Import dataset…, Settings…, Keyboard
shortcuts… and Exit. Export writes the
open label folder as COCO, YOLO segmentation, YOLO detection, LabelMe, VOC PNG,
VOC XML or an IVAS copy; Import converts COCO, YOLO, LabelMe or VOC XML into an
IVAS label folder, warning before overwriting and merging (never replacing) an
existing category file. The central surface shows either the image canvas or
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
provide the complete path, and each command has exactly one menu-bar home:
File holds data, import/export and preferences (Settings, Keyboard shortcuts);
View holds layout and appearance (Mode, Panels, Theme, Language); SAM holds
models; Tools holds utilities and Plugins; Help holds docs and About. The
toolbar never repeats preference or help commands.
Row one is frame editing: Mode, frame handoff, Prev/Next, Polygon, BBox, SAM
point, SAM box, Repaint, Finish/Cancel/Save/Export, Undo/Redo, Fit and Frame
tools (rotated box, layer, boolean, zoom and display commands). Drawing tools
keep their labels longest, then Export; narrow windows fall back to icons with
tooltips. Row two holds only the Phase/Task controls and the Video tools and
Tracking menus; it appears once a video case exists or Video mode is shown, so
image-only work has a single row. The loaded SAM model is state, not a command:
a status-bar indicator shows it (none, loading, ready, failed) and opens the
model manager on click. Frame/video handoff buttons appear beside Mode so
drawing tools do not displace them.
At the verified compact 640×480 size, secondary
controls remain reachable through toolbar overflow and dock scrolling.
Right-hand inspector widths follow window space, the minimum size of their
styled, translated forms (reserving a vertical scrollbar) and the full width of
the inspector tab strip: the canvas shrinks before any tab is hidden, up to the
48% inspector budget; only beyond that do tab scroll arrows appear. Larger
windows give inspectors more room; small windows retain scrolling and at least
half the window for the media area in the default layout. Splitter adjustments
persist until window space or content requirements change; floating/left-side
panels retain independent sizing. Text buttons use their natural size hints.
Tabs use normal body text, a padded target and a selected underline. Labels
never elide or shrink to fit; overflow uses visible scroll arrows, including
after language/layout changes.

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
latest result and saves completed annotations immediately, independently of the
global auto-save setting. A failed save retains the shapes and unsaved state.
Cancel invalidates the pending result immediately. Same-frame reuse requires
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

Frame annotation supports polygons, axis-aligned boxes (BBox: two clicks or
one drag, Shift for a square; stored as `shape_type: rectangle`) and rotated
boxes (OBB). Frame undo/redo keeps up to 100 snapshots of the current image;
each finished shape, delete or edit is one step and a drag is one step recorded
on mouse release. Opening another image starts a new history.

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
