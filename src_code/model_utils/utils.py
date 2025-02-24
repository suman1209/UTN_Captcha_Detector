import torch


def cxcy_to_xy(bboxes):
    '''
        Convert bboxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    '''
    return torch.cat([bboxes[:, :2] - (bboxes[:, 2:] / 2),
                      bboxes[:, :2] + (bboxes[:, 2:] / 2)], 1)


def cxcy_to_torch_xy(bboxes):
    '''
        Convert bboxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
        changing this to get bounding boxes in torch coordinate system
    '''
    bboxes_xy = cxcy_to_xy(bboxes)
    bboxes_inverted = torch.stack([
        bboxes_xy[:, 1],  # ymin becomes xmin
        bboxes_xy[:, 0],  # xmin becomes ymin
        bboxes_xy[:, 3],  # ymax becomes xmax
        bboxes_xy[:, 2]   # xmax becomes ymax
    ], dim=1)
    return bboxes_inverted


def xy_to_cxcy(bboxes):
    '''
        Convert bboxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h)
        bboxes: Bounding boxes, a tensor of dimensions (n_object, 4)

        Out: bboxes in center coordinate
    '''
    return torch.cat([(bboxes[:, 2:] + bboxes[:, :2]) / 2,
                      bboxes[:, 2:] - bboxes[:, :2]], 1)


def torch_xy_to_cxcy(bboxes_torch_xy):
    '''
    Convert bboxes from (xmin, ymin, xmax, ymax) with inverted axes
    back to (cx, cy, w, h) format.
    '''
    # Swap x and y axes back to their original positions
    bboxes_xy = torch.stack([
        bboxes_torch_xy[:, 1],  # ymin becomes xmin
        bboxes_torch_xy[:, 0],  # xmin becomes ymin
        bboxes_torch_xy[:, 3],  # ymax becomes xmax
        bboxes_torch_xy[:, 2]   # xmax becomes ymax
    ], dim=1)
    # Convert (xmin, ymin, xmax, ymax) to (cx, cy, w, h)
    widths = bboxes_xy[:, 2] - bboxes_xy[:, 0]  # xmax - xmin
    heights = bboxes_xy[:, 3] - bboxes_xy[:, 1]  # ymax - ymin
    cx = bboxes_xy[:, 0] + (widths / 2)  # cx = xmin + (width / 2)
    cy = bboxes_xy[:, 1] + (heights / 2)  # cy = ymin + (height / 2)
    # Stack (cx, cy, w, h) into a single tensor
    bboxes_cxcy = torch.stack([cx, cy, widths, heights], dim=1)
    return bboxes_cxcy


def encode_bboxes(bboxes,  default_boxes):
    '''
        Encode bboxes corresponding default boxes (center form)
        Out: Encodeed bboxes to 4 offset,
             a tensor of dims (n_defaultboxes, 4)
    '''
    db = default_boxes
    return torch.cat([(bboxes[:, :2] - db[:, :2]) / (db[:, 2:] / 10),
                      torch.log(bboxes[:, 2:] / db[:, 2:]) * 5], 1)


def decode_bboxes(offsets, default_boxes):
    '''
        Decode offsets
    '''
    db = default_boxes
    return torch.cat([offsets[:, :2] * db[:, 2:] / 10 + db[:, :2],
                      torch.exp(offsets[:, 2:] / 5) * db[:, 2:]], 1)


def intersect(boxes1, boxes2):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)
        Out: Intersection each of boxes1 with respect to each of boxes2,
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))
    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  # (n1, n2)


def find_IoU(boxes1, boxes2):
    '''
        Find IoU between every boxes set of boxes
        boxes1: a tensor of dimensions (n1, 4) (left, top, right , bottom)
        boxes2: a tensor of dimensions (n2, 4)
        Out: IoU each of boxes1 with respect to each of boxes2, a tensor of
             dimensions (n1, n2)
        Formula:
        i = (box1 ∩ box2)
        u = (box1 u box2)
        IoU = i / (area(box1) + area(box2) - i)
    '''
    inter = intersect(boxes1, boxes2)
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(inter)  # (n1, n2)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(inter)  # (n1, n2)
    union = (area_boxes1 + area_boxes2 - inter)
    return inter / union
