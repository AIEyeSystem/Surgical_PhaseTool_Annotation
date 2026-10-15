# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
from collections.abc import Mapping
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr
from .model.decoder import (
    DecoupledTransformerDecoderLayerv2,
    SimpleRoPEAttention,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
    TransformerEncoderDecoupledCrossAttention,
)
from .model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .model.geometry_encoders import SequenceGeometryEncoder
from .model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from .model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from .model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from .model.multiplex_utils import MultiplexController
from .model.necks import Sam3DualViTDetNeck, Sam3TriViTDetNeck
from .model.position_encoding import PositionEmbeddingSine
from .model.sam1_task_predictor import SAM3InteractiveImagePredictor
from .model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU
from .model.sam3_multiplex_detector import Sam3MultiplexDetector
from .model.sam3_tracking_predictor import Sam3TrackerPredictor
from .model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from .model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from .model.text_encoder_ve import VETextEncoder
from .model.tokenizer_ve import SimpleTokenizer
from .model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo
from .model.vitdet import ViT
from .model.vl_combiner import SAM3VLBackbone, SAM3VLBackboneTri, TriHeadVisionOnly
from .sam.transformer import RoPEAttention
from ..precision import apply_precision


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding()
    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    matcher = None
    if not eval_mode:
        from .train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    return model


def _create_tracker_maskmem_backbone():
    """Create the SAM3 Tracker memory encoder."""
    # Position encoding for mask memory backbone
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    # Mask processing components
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone


def _create_tracker_transformer():
    """Create the SAM3 Tracker transformer components."""
    # Self attention
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )

    # Cross attention
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )

    # Encoder layer
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    # Encoder
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )

    # Transformer wrapper
    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer


def build_tracker(
    apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
) -> Sam3TrackerPredictor:
    """
    Build the SAM3 Tracker module for video tracking.

    Returns:
        Sam3TrackerPredictor: Wrapped SAM3 Tracker module
    """

    # Create model components
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    # Create the Tracker module
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        # SAM parameters
        multimask_output_in_sam=True,
        # Evaluation
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        # Multimask
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        # Additional settings
        always_start_from_first_ann_frame=False,
        # Mask overlap
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        # SAM decoder settings
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )

    return model


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer(has_presence_token: bool = True) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_checkpoint(model, checkpoint_path):
    """Load model checkpoint from file."""
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing and/or unexpected keys:\n{missing_keys=}"
        )


def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    model = model.to(device=device)
    if eval_mode:
        model.eval()
    return model


def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path to model checkpoint
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile_mode: To enable compilation, set to "default"

    Returns:
        A SAM3 image model
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
    # Create visual components
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
    )

    # Create text components
    text_encoder = _create_text_encoder(bpe_path)

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model


def download_ckpt_from_hf():
    SAM3_MODEL_ID = "facebook/sam3"
    SAM3_CKPT_NAME = "sam3.pt"
    SAM3_CFG_NAME = "config.json"
    _ = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CFG_NAME)
    checkpoint_path = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CKPT_NAME)
    return checkpoint_path


def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """
    Build SAM3 dense tracking model.

    Args:
        checkpoint_path: Optional path to checkpoint file
        bpe_path: Path to the BPE tokenizer file

    Returns:
        Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )

    # Build Tracker module
    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

    # Build Detector components
    visual_neck = _create_vision_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    # Create main dot product scoring
    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )

    # Build Detector module
    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )

    # Build the main SAM3 video model
    if apply_temporal_disambiguation:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=16,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )
    else:
        # a version without any heuristics for ablation studies
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=0,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )

    # Load checkpoint if provided
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt, strict=strict_state_dict_loading
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    model.to(device=device)
    return model


def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )
# ===================== SAM 3.1 (multiplex) support =====================

def _create_multiplex_maskmem_backbone(multiplex_count=16):
    """Create the multiplex memory encoder with per-object mask channels."""
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
        interpol_size=[1152, 1152],
        multiplex_count=multiplex_count,
        starting_out_chan=4,
        input_channel_multiplier=2,
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=256,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone


def _create_multiplex_transformer(use_fa3=False, use_rope_real=False):
    """Create the decoupled transformer for multiplex memory attention."""
    self_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )

    cross_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )

    encoder_layer = DecoupledTransformerDecoderLayerv2(
        activation="gelu",
        d_model=256,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pos_enc_at_attn=False,
        pre_norm=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention_rope=self_attention_rope,
        cross_attention_rope=cross_attention_rope,
    )

    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        use_image_in_output=False,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
        batch_first=True,
    )

    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer


def _create_multiplex_tri_backbone(compile_mode=None):
    """Create the TriHead vision backbone for the SAM 3.1 multiplex model."""
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(compile_mode=compile_mode)
    tri_neck = Sam3TriViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )
    return tri_neck


def _find_base_checkpoint(checkpoint_path, names=("sam3.1_multiplex.pt", "sam3.pt")):
    """Locate a base checkpoint (containing tracker weights) for tracker merging.

    Search order:
      1. The directory of the given checkpoint
      2. IVAS's checkpoint directory (IVAS/checkpoints/)
      3. The HuggingFace hub cache
      4. The SAM3_BASE_CKPT environment variable

    Returns the first match, or None.
    """
    candidates = []
    if checkpoint_path:
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        for name in names:
            candidates.append(os.path.join(ckpt_dir, name))
    try:
        from IVAS.configs import CHECKPOINT_PATH
    except Exception:
        CHECKPOINT_PATH = None
    if CHECKPOINT_PATH:
        for name in names:
            candidates.append(os.path.join(CHECKPOINT_PATH, name))
    hf_cache = os.path.expanduser(
        os.path.join("~", ".cache", "huggingface", "hub")
    )
    if os.path.isdir(hf_cache):
        for root, dirs, files in os.walk(hf_cache):
            for name in names:
                if name in files:
                    candidates.append(os.path.join(root, name))
    env_ckpt = os.getenv("SAM3_BASE_CKPT")
    if env_ckpt:
        candidates.append(env_ckpt)
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _remap_official_prefixes(ckpt):
    """Remap official checkpoint naming (sam3_model.* -> detector.*, sam2_predictor.* -> tracker.*)."""
    needs_remap = any(
        k.startswith("sam3_model.") or k.startswith("sam2_predictor.") for k in ckpt
    )
    if not needs_remap:
        return ckpt
    remapped = {}
    for k, v in ckpt.items():
        if k.startswith("sam3_model."):
            k = "detector." + k[len("sam3_model.") :]
        elif k.startswith("sam2_predictor."):
            k = "tracker." + k[len("sam2_predictor.") :]
        remapped[k] = v
    return remapped


def _normalize_ckpt_keys_31(ckpt, model_has_detector_module, model_has_tracker_module):
    """Map checkpoint keys to the given model's module layout.

    - Wrapper models (Sam3MultiplexTrackingWithInteractivity, which has
      `detector` / `tracker` submodules) expect `detector.*` / `tracker.*`
      prefixed keys; bare checkpoints are wrapped accordingly.
    - Bare models (Sam3Image family, Sam3VideoTrackingMultiplexDemo) register
      submodules directly; `detector.*` and `tracker.model.*` prefixes are
      stripped.
    Returns the normalized dict.
    """
    has_detector = any(k.startswith("detector.") for k in ckpt)
    has_tracker = any(k.startswith("tracker.") for k in ckpt)
    normalized = dict(ckpt)
    if model_has_detector_module:
        if not has_detector:
            normalized = {"detector." + k: v for k, v in normalized.items()}
        if model_has_tracker_module and not has_tracker:
            normalized = {"tracker." + k: v for k, v in normalized.items()}
    else:
        if has_detector:
            normalized = {
                k[len("detector.") :] if k.startswith("detector.") else k: v
                for k, v in normalized.items()
            }
        if has_tracker:
            normalized = {
                k[len("tracker.model.") :]
                if k.startswith("tracker.model.")
                else k: v
                for k, v in normalized.items()
            }
    return normalized


def _unwrap_checkpoint_state(checkpoint, checkpoint_path):
    """Return a validated tensor state dict from common checkpoint wrappers."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "SAM 3.1 checkpoint {!r} must contain a mapping.".format(
                checkpoint_path
            )
        )
    for wrapper_key in ("model", "state_dict"):
        wrapped = checkpoint.get(wrapper_key)
        if isinstance(wrapped, Mapping):
            checkpoint = wrapped
            break
    invalid = [
        key
        for key, value in checkpoint.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid:
        raise ValueError(
            "SAM 3.1 state dict contains non-tensor entries: {}".format(
                invalid[:10]
            )
        )
    return dict(checkpoint)


def _strip_uniform_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _dominant_checkpoint_dtype(state_dict):
    counts = {}
    for tensor in state_dict.values():
        if tensor.is_floating_point():
            counts[tensor.dtype] = counts.get(tensor.dtype, 0) + tensor.numel()
    return max(counts.items(), key=lambda item: item[1])[0] if counts else torch.float32


def _extract_sam31_detector_state_dict(checkpoint, model_state):
    """Select exactly the detector branch and validate every learned tensor."""

    checkpoint = _strip_uniform_module_prefix(checkpoint)
    has_official = any(key.startswith("sam3_model.") for key in checkpoint)
    has_merged = any(key.startswith("detector.") for key in checkpoint)
    if has_official and has_merged:
        raise ValueError(
            "Ambiguous SAM 3.1 checkpoint: both sam3_model.* and detector.* exist."
        )

    if has_official:
        invalid = [
            key
            for key in checkpoint
            if not key.startswith(("sam3_model.", "sam2_predictor."))
        ]
        if invalid:
            raise ValueError(
                "Unexpected keys beside official SAM 3.1 branches: {}".format(
                    invalid[:10]
                )
            )
        mapped = {
            key[len("sam3_model.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("sam3_model.")
        }
    elif has_merged:
        invalid = [
            key
            for key in checkpoint
            if not key.startswith(("detector.", "tracker."))
        ]
        if invalid:
            raise ValueError(
                "Unexpected keys beside merged SAM 3.1 branches: {}".format(
                    invalid[:10]
                )
            )
        mapped = {
            key[len("detector.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("detector.")
        }
    else:
        mapped = dict(checkpoint)

    model_keys = set(model_state)
    mapped_keys = set(mapped)
    missing = sorted(model_keys - mapped_keys)
    unexpected = sorted(mapped_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(
            "SAM 3.1 detector checkpoint does not match the image model; "
            "missing={} unexpected={}".format(missing[:10], unexpected[:10])
        )
    shape_mismatches = [
        (key, tuple(mapped[key].shape), tuple(model_state[key].shape))
        for key in sorted(model_keys)
        if mapped[key].shape != model_state[key].shape
    ]
    if shape_mismatches:
        raise RuntimeError(
            "SAM 3.1 detector checkpoint shape mismatch: {}".format(
                shape_mismatches[:10]
            )
        )
    return mapped


def _load_sam31_detector_checkpoint(model, checkpoint_path):
    """Strictly load a bare, merged, or official SAM 3.1 detector branch."""

    with g_pathmgr.open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = torch.load(
            checkpoint_file, map_location="cpu", weights_only=True
        )
    checkpoint = _unwrap_checkpoint_state(checkpoint, checkpoint_path)
    checkpoint = _extract_sam31_detector_state_dict(
        checkpoint, model.state_dict()
    )
    model.load_state_dict(checkpoint, strict=True, assign=True)
    model._checkpoint_dtype = _dominant_checkpoint_dtype(checkpoint)
    return model._checkpoint_dtype


def _load_checkpoint_31(model, checkpoint_path, base_checkpoint_path=None):
    """Load a SAM 3.1 checkpoint into a multiplex (or bare detector) model.

    Supports three checkpoint formats:
      1. Official merged format:  sam3_model.* -> detector.*, sam2_predictor.* -> tracker.*
      2. Merged format:           detector.* / tracker.* prefixed keys
      3. Bare detector format:    no prefixes (e.g. fine-tuned detector checkpoints
                                 such as checkpoint-bf16.pt)

    The checkpoint tensor dtypes (bf16/fp16/fp32) are preserved as-is.
    When tracker weights are missing, they are merged from a base checkpoint
    if one can be found (see _find_base_checkpoint) or supplied explicitly.
    """
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    ckpt = _remap_official_prefixes(ckpt)

    model_has_detector_module = hasattr(model, "detector")
    model_has_tracker_module = hasattr(model, "tracker")
    ckpt = _normalize_ckpt_keys_31(
        ckpt, model_has_detector_module, model_has_tracker_module
    )

    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    missing = model_keys - set(ckpt.keys())
    if model_has_tracker_module:
        missing_tracker = {k for k in missing if k.startswith("tracker.")}
    else:
        # bare demo model: every missing key is tracker-side (the detector
        # grounding stack is not part of the demo model at all, and the shared
        # backbone / tri-head keys come from the detector checkpoint)
        missing_tracker = set(missing)

    if missing_tracker and base_checkpoint_path is None:
        base_checkpoint_path = _find_base_checkpoint(checkpoint_path)
    if missing_tracker and base_checkpoint_path is not None:
        print(
            f"  - {len(missing_tracker)} tracker weights missing; "
            f"merging from base checkpoint: {base_checkpoint_path}"
        )
        with g_pathmgr.open(base_checkpoint_path, "rb") as f:
            base_ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in base_ckpt and isinstance(base_ckpt["model"], dict):
            base_ckpt = base_ckpt["model"]
        base_ckpt = _remap_official_prefixes(base_ckpt)
        base_ckpt = _normalize_ckpt_keys_31(
            base_ckpt, model_has_detector_module, model_has_tracker_module
        )
        merged = {
            k: v
            for k, v in base_ckpt.items()
            if k in missing_tracker
            and v.shape == model_state_dict[k].shape
        }
        if not merged:
            print(
                "  - warning: base checkpoint contains no compatible tracker weights "
                "(tracker.* keys not found). Video tracking will not work; "
                "detection (image) inference is unaffected."
            )
        ckpt.update(merged)
        if merged:
            non_fp32_dtypes = {
                v.dtype
                for k, v in ckpt.items()
                if v.is_floating_point()
                and not torch.is_complex(v)
                and v.dtype != torch.float32
                and not k.startswith("tracker.")
            }
            if len(non_fp32_dtypes) == 1:
                target_dtype = non_fp32_dtypes.pop()
                print(f"  - casting merged tracker weights to {target_dtype}")
                ckpt.update(
                    {
                        k: v.to(target_dtype) if v.is_floating_point() else v
                        for k, v in merged.items()
                    }
                )
    elif missing_tracker:
        print(
            "  - warning: no tracker weights available for this checkpoint and no "
            "base checkpoint (sam3.1_multiplex.pt / sam3.pt) was found. "
            "Video tracking will not work; detection (image) inference is unaffected."
        )

    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False, assign=True)
    if missing_keys:
        print(f"  - missing keys ({len(missing_keys)}): {missing_keys[:10]} ...")
    if unexpected_keys:
        print(f"  - unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]} ...")


def build_sam3_1_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=False,
    precision="original",
):
    """
    Build a SAM 3.1 image detection model (Sam3MultiplexDetector with the
    tri-head vision backbone). Works with bare detector checkpoints such as
    checkpoint-bf16.pt, and supports bf16/fp16 dtypes.
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz"
        )
    tri_neck = _create_multiplex_tri_backbone(compile_mode=None)
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=text_encoder)
    transformer = _create_sam3_transformer()
    segmentation_head = _create_segmentation_head()
    geometry_encoder = _create_geometry_encoder()
    dot_prod_scoring = _create_dot_product_scoring()

    detector = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=dot_prod_scoring,
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    if checkpoint_path is not None:
        _load_sam31_detector_checkpoint(detector, checkpoint_path)
    detector._effective_dtype = apply_precision(detector, precision)
    model = _setup_device_and_mode(detector, device, eval_mode)
    return model


def build_sam3_1_video_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=False,
    multiplex_count=16,
    use_fa3=False,
    use_rope_real=False,
    compile=False,
    base_checkpoint_path=None,
):
    """
    Build a SAM 3.1 multiplex video tracking model (Sam3VideoTrackingMultiplexDemo).

    If the checkpoint lacks tracker weights (e.g. a bare fine-tuned detector
    checkpoint), they are merged from a base checkpoint that is either passed
    explicitly or located automatically (see _find_base_checkpoint).
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz"
        )
    maskmem_backbone = _create_multiplex_maskmem_backbone(
        multiplex_count=multiplex_count
    )
    transformer = _create_multiplex_transformer(
        use_fa3=use_fa3, use_rope_real=use_rope_real
    )
    tri_neck = _create_multiplex_tri_backbone(
        compile_mode="max-autotune" if compile else None
    )
    backbone = TriHeadVisionOnly(
        visual=tri_neck,
        n_features=256,
        scalp=0,
    )
    multiplex_controller = MultiplexController(
        multiplex_count=multiplex_count,
        eval_multiplex_count=multiplex_count,
    )

    model = Sam3VideoTrackingMultiplexDemo(
        backbone=backbone,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multiplex_controller=multiplex_controller,
        image_size=1008,
        backbone_stride=14,
        num_maskmem=7,
        # Multiplex-specific settings
        use_high_res_features_in_sam=True,
        use_obj_ptrs_in_encoder=True,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_mlp_for_obj_ptr_proj=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        use_no_obj_ptr=True,
        use_linear_no_obj_ptr=True,
        no_obj_embed_spatial=True,
        sincos_tpos_enc=True,
        # Multimask settings
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_multimask_token_for_obj_ptr=True,
        num_multimask_outputs=3,
        # Memory encoder settings
        apply_sigmoid_to_mask_logits_for_mem_enc=True,
        sigmoid_scale_for_mem_enc=2.0,
        sigmoid_bias_for_mem_enc=-1.0,
        non_overlap_masks_for_mem_enc=False,
        # Suppression/conditional embeddings
        add_output_suppression_embeddings=True,
        add_object_conditional_embeddings=False,
        condition_as_mask_input=True,
        condition_as_mask_input_fg=1.0,
        condition_as_mask_input_bg=0.0,
        # Memory settings
        use_maskmem_tpos_v2=True,
        save_image_features=True,
        randomness_fix=True,
        # Interaction settings
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        forward_backbone_per_frame_for_eval=True,
        offload_output_to_cpu_for_eval=False,
        trim_past_non_cond_mem_for_eval=False,
        max_cond_frames_in_attn=4,
        # Dynamic multiplex settings
        is_dynamic_model=True,
        # SAM mask decoder extra args
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        compile_all_components=compile,
        use_memory_selection=False,
    )

    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        _load_checkpoint_31(
            model, checkpoint_path, base_checkpoint_path=base_checkpoint_path
        )

    model = _setup_device_and_mode(model, device, eval_mode)
    return model
