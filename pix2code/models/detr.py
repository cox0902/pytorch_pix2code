import torch
import torch.nn as nn
import torchvision


class Detr(nn.Module):

    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)  # pretrained ImageNet ResNet-50
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        
        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        self.criterion_class = nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch):
        inputs = batch["image"]
        caps = batch["code"].long()
        caplens = batch["code_len"]
        bboxes = batch["rect"]
        equals = batch["equal"].long()
        ignores = batch["ignore"].long()

        # propagate inputs through ResNet-50 up to avg-pool layer
        # x = self.backbone.conv1(inputs)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)

        # x = self.backbone.layer1(x)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
        x = self.backbone(inputs)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        logits = self.linear_class(h)
        boxes = self.linear_bbox(h).sigmoid()

        masks = torch.zeros_like(caps)
        masks[torch.where(caps != 0)] = 1

        loss_class = self.criterion_class(logits, caps)
        loss_class = torch.sum(loss_class * masks) / torch.sum(caplens)

        bbox_masks = (1 - equals) * (1 - ignores) * masks

        bboxes_true = bboxes[bbox_masks]
        bboxes_pred = boxes[bbox_masks]

        bboxes_true = torchvision.ops.box_convert(bboxes_true, "cxcywh", "xyxy")
        bboxes_pred = torchvision.ops.box_convert(bboxes_pred, "cxcywh", "xyxy")

        loss_bbox = torchvision.ops.generalized_box_iou_loss(bboxes_true, bboxes_pred, reduction="mean")
        
        # return {'pred_logits': self.linear_class(h), 
        #         'pred_boxes': self.linear_bbox(h).sigmoid()}


        return loss_class, 