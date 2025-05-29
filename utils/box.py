"""General methods for working with boxes

For more details see https://d2l.ai/chapter_computer-vision/anchor.html.
"""

import torch


def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU across two lists of anchor or bounding boxes

    :param boxes1: anchors [num_anchors, 4] - (ulw, ulh, drw, drh).
    :type boxes1: torch.Tensor
    :param boxes2: ground truth [num_gt_box, 4] - (ulw, ulh, drw, drh).
    :type boxes2: torch.Tensor
    :return: IoU. Element x_ij in the i-th row and j-th column is
        the IoU of the anchor box i and the ground-truth bounding box j.

        Shape [num_anchors, num_gt_box].
    :rtype: torch.Tensor
    """
    assert boxes1.shape == (boxes1.shape[0], 4), "Wrong box shape"
    assert boxes2.shape == (boxes2.shape[0], 4), "Wrong box shape"

    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = torch.prod(boxes1[:, 2:] - boxes1[:, :2], dim=1)
    areas2 = torch.prod(boxes2[:, 2:] - boxes2[:, :2], dim=1)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_up_lefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_low_rights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = torch.clamp(inter_low_rights - inter_up_lefts, min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = torch.prod(inters, dim=2)
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.concat([offset_xy, offset_wh], axis=1)
    return offset


def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets"""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, class_id, num_classes, iou_threshold):
    """Sort confidence scores of predicted bounding boxes"""
    keep = []  # Indices of predicted bounding boxes that will be kept
    for class_idx in range(num_classes - 1):
        scores_cls = torch.nonzero(class_id == class_idx).squeeze(dim=1)
        boxes_cls = boxes[scores_cls]
        B = torch.argsort(scores[scores_cls], descending=True)
        while B.numel() > 0:
            i = B[0]
            keep.append(scores_cls[i])
            if B.numel() == 1:
                break
            iou = box_iou(
                boxes_cls[i, :].reshape(-1, 4), boxes_cls[B[1:], :].reshape(-1, 4)
            ).reshape(-1)
            inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
            B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def multibox_detection(
    cls_probs: torch.Tensor,
    offset_preds: torch.Tensor,
    anchors: torch.Tensor,
    nms_threshold: float = 0.1,
    pos_threshold: float = 0.009999999,
) -> torch.Tensor:
    """Predict bounding boxes using non-maximum suppression

    :param cls_probs: Shape [batch, anchor, num_classes + 1]
    :type cls_probs: torch.Tensor
    :param offset_preds: Shape [batch, anchor, 4]
    :type offset_preds: torch.Tensor
    :param anchors: Shape [anchor, 4]
    :type anchors: torch.Tensor
    :param nms_threshold: Defaults to 0.1
    :type nms_threshold: float, optional
    :param pos_threshold: Defaults to 0.009999999
    :type pos_threshold: float, optional
    :return: Shape [batch, anchor, 6]

        One label contains (class, iou, luw, luh, rdw, rdh)
    :rtype: torch.Tensor
    """
    device = cls_probs.device
    batch_size, num_anchors, num_classes = cls_probs.shape
    out = []
    for batch_idx in range(batch_size):
        cls_prob, offset_pred = cls_probs[batch_idx], offset_preds[batch_idx]
        conf, class_id = torch.max(cls_prob, 1)
        predicted_bb = offset_inverse(anchors, offset_pred)
        class_id -= 1
        keep = nms(predicted_bb, conf, class_id, num_classes, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = conf < pos_threshold
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)
