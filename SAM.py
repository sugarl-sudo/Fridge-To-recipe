import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# SAM
class SegmentAnything:
    def __init__(self, device, model_type, sam_checkpoint):
        print("init Segment Anything")
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)

    def generat_masks(self, image):
        print("generate masks by Segment Anything")
        # 検出するマスクの品質　（default: 0.88)
        # pred_iou_thresh = 0.96 640*480の時ちょうどよかった
        pred_iou_thresh = 0.98  #  960×720の時はこれぐらい
        mask_generator_2 = SamAutomaticMaskGenerator(
            model=self.sam, pred_iou_thresh=pred_iou_thresh
        )
        masks = mask_generator_2.generate(image)
        self.sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        self._length = len(masks)
        print("{} masks generated.".format(self._length))

    @property
    def length(self):
        return self._length

    def get(self, index):
        pixels = self.sorted_anns[index]["area"]
        mask = np.expand_dims(self.sorted_anns[index]["segmentation"], axis=-1)
        return mask, pixels