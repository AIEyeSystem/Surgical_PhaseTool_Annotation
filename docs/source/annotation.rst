Annotate
=================================

1. Start Annotation
---------------------------------

The ``File`` menu separates ``Image``, ``Video`` and ``General`` commands with
dashed dividers. Image contains the image and frame-label directories. Video
contains opening media/projects, the output directory, relinking and video
annotation import/export. Save current work, Settings and Exit are shared.

In Frame mode, ``Polygon``, ``SAM point``, ``SAM box`` and ``Repaint`` are direct
toolbar buttons. Wide windows show their icons and names; narrow windows keep
the icons and hover descriptions. SAM tools become available when the model and
current image encoding are ready.

The right-hand inspectors adapt their width to the window, display scaling and
translated controls. Very small windows or large fonts retain scrollbars so all
controls remain reachable. Drag the divider to adjust the width manually; it
adapts again when window space or content requirements change. Floating panels
keep their own size. Use ``View → Reset layout`` to restore the default layout.

1.1 Manual Annotation
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

Click the ``Polygon`` button ``[C]`` in the toolbar to start manually annotating a polygon.

.. image:: ../../display/draw_polygon_button.png
    :alt: draw_polygon_button.png


**Manual annotation offers two methods: clicking and dragging.**

1.1.1 Clicking Method
.................................

Clicking is the most common annotation method, with each click adding a polygon vertex.

.. image:: ../../display/clicking.gif
    :alt: clicking.gif

1.1.2 Dragging Method
.................................

By pressing and holding the left mouse button and dragging the mouse, (0.15s per point)

.. image:: ../../display/dragging.gif
    :alt: dragging.gif

.. tip:: With the Dragging method, you can easily draw curves.

1.1.3 Shift Key Constraint
.................................

Hold the ``Shift`` key while drawing to constrain lines to horizontal, vertical, or 45-degree angles.

.. image:: ../../display/shift_constraint.gif
    :alt: shift_constraint.gif

.. tip:: This feature is useful for annotating objects with regular shapes like buildings, documents, or signs. The line will automatically snap to the nearest 0°, 45°, 90°, 135°, or 180° direction.

1.2 Semi-automatic Annotation
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

Utilizes SAM with point and bounding box prompts.

.. image:: ../../display/point_box_prompt_button.png
    :alt: point_box_prompt_button.png

1.2.1 Point Prompt
.................................

Click the ``SAM point`` button ``[Q]`` in the toolbar to start SAM point prompt.

.. image:: ../../display/point_prompts.gif
    :alt: point_prompts.gif

1.2.2 Box Prompt
.................................

Click the ``SAM box`` button ``[W]`` in the toolbar to start SAM box prompt.

.. image:: ../../display/box_prompts.gif
    :alt: box_prompts.gif

1.2.3 Text Prompt
.................................

Since v1.5.0, IVAS supports text prompts. SAM 3.1 can reuse the current image
embedding for one or more prompts.

1. Select a foreground annotation category.

2. Keep the category name as the prompt, or edit it to an English alias that is
   easier for the model to ground.

3. Click ``Segment`` to explicitly confirm prediction. Changing a
   category does not automatically run the model.

4. Review the created polygons. An empty model result creates no annotation.

.. image:: ../../display/text_prompt1.gif

The multiple-category action uses the configured category names as prompts and
shares the SAM 3.1 image embedding.

.. image:: ../../display/text_prompt2.gif

.. note:: The selected annotation category and the editable model prompt are
          intentionally separate. The category is saved in the annotation;
          the prompt only guides segmentation.

Folder segmentation
~~~~~~~~~~~~~~~~~~~

In Frame mode with SAM3 or SAM3.1 loaded, select a foreground category in
``Categories``. The row below ``Segment / Change`` offers:

* ``SegAll``: segment all images in the currently opened folder using the
  selected category name as the prompt.
* ``SegMiss``: segment only images whose annotation JSON contains no object of
  that category. Missing JSON files, empty annotations and annotations of other
  categories are eligible. Display aliases do not affect this check.

Unsaved edits on the current image are saved before the batch. New polygons are
appended to the existing JSON files in the current label directory as SAM drafts;
existing objects are preserved. Repeating ``SegAll`` appends another set of
predictions. Use ``SegMiss`` to fill missing categories without repeating images
that already have them. Empty predictions remain eligible for a later run.

The progress window supports cancellation and reports errors per image. Completed
images stay saved; cancelled in-flight predictions are discarded. The batch uses
the current file list without recursing into subfolders. For video projects this
means already extracted frames, whose saved annotation links are also updated.

1.2.4 Visual Prompt
.................................

Since v1.5.2, IVAS supports the visual prompt function based on SAM3.

Use the positive box and negative box for prediction.

.. image:: ../../display/visual_prompt.gif

2. Finish Annotation
---------------------------------

Click the ``Finish annotation`` checkmark ``[E]`` in the toolbar to finish the
annotation and immediately save the current image's JSON, even when the global
auto-save option is disabled. This also updates the saved frame link when
annotating a video frame.

When using sam ``point prompt`` or ``box prompt``, ``finish annotation`` will convert the mask into a polygon on the canvas.

If SAM inference is still running, Finish waits for the latest result before
saving. Cancelled or invalid shapes do not trigger a save. If saving fails, the
completed shape remains on the canvas and the image stays marked as unsaved;
resolve the error and retry with ``Ctrl+S``.

.. tip:: IVAS offers multiple conversion methods, which you can find in the settings interface.

         Contour mode:

             - External contours
             - Only max contour
             - All contours

         For detailed usage, please refer to `Contour mode <other.html#contour-mode>`_.

3. Cancel Annotation
---------------------------------

Click the ``Cancel annotation`` button ``[Esc]`` in the toolbar to end the annotation.

4. Backspace
---------------------------------

Click the ``Backspace`` button ``[Z]`` in the toolbar to return to the previous annotation state.

.. image:: ../../display/backspace.gif
    :alt: backspace.gif

5. Video Annotation
---------------------------------

5.1 Unified Phase/Task and frame annotation
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

Use ``File → Open video`` to open the integrated video workspace. Select
the Phase or Task lane, enter a label, press ``I`` at the interval start, and
press ``O`` at its end. Intervals can also be selected from the timeline or
table, undone, redone, and deleted.

``Annotate current frame`` pauses playback, maps the current time through the
decoded presentation-timestamp index, extracts that frame in a cancellable
background job, links a label file in ``labels``, and opens it in the normal IVAS
polygon/SAM canvas. ``Mode → Frame`` uses the same extraction workflow; Mode
changes only after the frame loads. Use ``Mode → Video`` or ``Return to video
preview`` to return to the same case, retaining unfinished intervals.
While extracting or editing a frame, the playhead stays pinned and seeking is
disabled. The bottom transport provides ``Return to video`` in Frame mode;
returning checks unfinished shapes and unsaved changes, restores that frame's
position and leaves playback paused until Play is pressed. VLC pause is
idempotent, so repeated frame transitions cannot accidentally resume playback.
Reopening an unchanged frame in the same session reuses its extracted image,
canvas, zoom and model features. Changed label files are reloaded.

Point and box prompts run in a background worker. Rapid changes replace pending
requests; only a result matching the current frame, model and prompts can appear.
Finishing during inference waits for the latest mask without blocking the UI;
Escape cancels the pending annotation. Right-hand inspector tabs retain full
labels and expose scroll arrows when the available width is too small.

Opening an image directory activates independent Frame mode and disables commands
for any retained video. Video undo/redo affects annotations and templates only;
it does not undo frame links, saved segmentation, media metadata, or tracking.
While annotating a video frame, video shortcuts operate only when video panels
or transport have focus; canvas focus retains the frame shortcuts.
The player supports 0.25×--2× rates and exact frame stepping. Scroll the timeline
to zoom, Shift-scroll to pan, and double-click it to reset the view. When Qt
cannot decode a codec, the application attempts its VLC fallback.

The recoverable ``project.ivas.json`` manifest is stored below ``outputs`` in a
video-fingerprint case directory. Existing ``project.ivat.json`` and ``project.ivas.json`` manifests are
still discovered for compatibility. Existing manifests can be opened directly and
moved media can be relinked. A fingerprint mismatch is never accepted silently:
the user must confirm it and the old identity is preserved in media history. If
the same case contains a legacy ``project.surgui.json``, its temporal records are
migrated.

The schema-v3 editor creates and edits template-driven intervals, points/key
frames and ratings with notes and review status. ``Video annotation`` contains
template management, validation, coverage and frame-index controls. ``File``
owns opening, saving and interchange; ``Project`` configures directories and
media relinking; ``Tracking`` contains staged propagation and review.

5.2 Legacy image-sequence propagation (retired)
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

The former ``Video segment once/five/all`` actions wrote generated objects
directly into later image-sequence labels. Those actions are now disabled and
hidden because they did not provide review, cancellation-safe persistence, or
rollback. Open the source video in the unified workspace and use the transaction
workflow below.

5.3 Transactional unified-video tracking
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

With a compatible SAM2/SAM3/SAM3.1 video model loaded, choose ``Tracking → Start
transactional tracking``. Select a forward, backward or bidirectional range and
one or more reviewed correction frames. Generated dense masks are stored in a
run-specific staging directory and do not change accepted IVAS labels until the
user opens ``Review / commit staged tracking`` and confirms the commit.

Manual objects win identity conflicts by default. A failed save restores every
label and leaves the run staged for retry; a committed run can be rolled back
from the same menu. Stable ``object_id`` and ``track_id`` fields survive normal
polygon editing and project reopen.

.. attention:: A bare SAM 3.1 detector checkpoint has no video-predictor weights
               and cannot be used for tracking. The legacy image-sequence
               tracker is retired because it wrote accepted labels directly;
               use the staged unified-video workflow instead.

