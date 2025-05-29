"""Label anchor boxes using ground-truth bounding boxes"""

import torch
import utils.box as box


class RoI:
    """Label anchor boxes using ground-truth bounding boxes"""

    def __init__(self, iou_threshold=0.5) -> None:
        """
        :param iou_threshold: Minimum acceptable iou. Defaults to 0.5.
        :type iou_threshold: float, optional
        """
        self.iou_threshold = iou_threshold

    def __call__(
        self, anchors: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Label anchor boxes using ground-truth bounding boxes

        :param anchors: Shape [anchor, 4]
        :type anchors: torch.Tensor
        :param labels: Shape [batch, gt_boxes, 5]

            One label contains (class, luw, luh, rdw, rdh)
        :type labels: torch.Tensor
        :return: List of 3 tensors:

            1. Ground truth offsets for each box. Shape [batch, anchor, 4].
            2. Box mask. Shape [batch, anchor, 4]. (0)*4 for background, (1)*4 for object.
            3. Class of each box (0 - background). Shape [batch, anchor].
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        batch_size = labels.shape[0]
        device, num_anchors = anchors.device, anchors.shape[0]
        batch_offset, batch_mask, batch_class_labels = [], [], []
        for i in range(batch_size):
            label = labels[i][labels[i, :, 0] >= 0]
            anchors_bbox_map = (
                self._assign_anchor_to_box(label[:, 1:], anchors)
                if label.shape[0]
                else torch.full((num_anchors,), -1, device=anchors.device)
            )
            bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
            # Initialize class labels and assigned bounding box coordinates with zeros
            class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
            assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
            # Label classes of anchor boxes using their assigned ground-truth
            # bounding boxes. If an anchor box is not assigned any, we label its
            # class as background (the value remains zero)
            indices_true = torch.nonzero(anchors_bbox_map >= 0)
            bb_idx = anchors_bbox_map[indices_true]
            class_labels[indices_true] = label[bb_idx, 0].long() + 1
            assigned_bb[indices_true] = label[bb_idx, 1:]
            # Offset transformation
            offset = box.offset_boxes(anchors, assigned_bb) * bbox_mask
            batch_offset.append(offset)
            batch_mask.append(bbox_mask)
            batch_class_labels.append(class_labels)
        bbox_offset = torch.stack(batch_offset)
        bbox_mask = torch.stack(batch_mask)
        class_labels = torch.stack(batch_class_labels)
        return bbox_offset, bbox_mask, class_labels

    def _assign_anchor_to_box(
        self, ground_truth: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        """Assign closest ground-truth bounding boxes to anchor boxes

        See https://d2l.ai/chapter_computer-vision/anchor.html#assigning-ground-truth-bounding-boxes-to-anchor-boxes

        :param ground_truth: The ground-truth bounding boxes [gt_box, 4] - ulw, ulh, drw, drh
        :type ground_truth: torch.Tensor
        :param anchors: Anchors boxes [anchor, 4] - ulw, ulh, drw, drh
        :type anchors: torch.Tensor
        :return: Tensor with ground truth box indices [anchor]
        :rtype: torch.Tensor
        """
        num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
        # The Jaccard index measures the similarity between two sets [num_anchors, num_gt_box]
        jaccard = box.box_iou(anchors, ground_truth)

        # Initialize the tensor to hold the assigned ground-truth bounding box for each anchor
        anchors_box_map = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)

        # Assign ground-truth bounding boxes according to the threshold
        max_ious, indices = torch.max(jaccard, dim=1)
        # Indexes of non-empty boxes
        mask = max_ious >= self.iou_threshold
        anc_i = torch.nonzero(mask).reshape(-1)
        if len(anc_i) == 0:
            w = ground_truth[:, 2] - ground_truth[:, 0]
            h = ground_truth[:, 3] - ground_truth[:, 1]
            print(
                f"[WARN]: There is no suitable anchor \
                \n\tSizes: \
                \n\t\tW: {w} \
                \n\t\tH: {h} \
                \n\tRatios: {h / w} \
                \n\tCoordinates: \
                \n\t\tX: {(ground_truth[:, 2] + ground_truth[:, 0]) / 2} \
                \n\t\tY: {(ground_truth[:, 3] + ground_truth[:, 1]) / 2}",
            )
        box_j = indices[mask]
        # Each anchor is assigned a gt_box with the highest iou if it is greater than the threshold
        anchors_box_map[anc_i] = box_j

        # For each gt_box we assign an anchor with maximum iou
        col_discard = torch.full((num_anchors,), -1)
        row_discard = torch.full((num_gt_boxes,), -1)
        for _ in range(num_gt_boxes):
            max_idx = torch.argmax(jaccard)  # Find the largest IoU
            box_idx = (max_idx % num_gt_boxes).long()
            anc_idx = (max_idx / num_gt_boxes).long()
            anchors_box_map[anc_idx] = box_idx
            jaccard[:, box_idx] = col_discard
            jaccard[anc_idx, :] = row_discard
        return anchors_box_map
