import cv2
import numpy as np
import random


class RandomPatchMask:

    def __init__(
        self,
        mask_ratio=0.5,
        patch_size=16,
        keep_center=True,
        center_ratio=0.75,
        fill_value=0
    ):

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.keep_center = keep_center
        self.center_ratio = center_ratio
        self.fill_value = fill_value


    def __call__(self, image, **kwargs):

        H, W, C = image.shape

        p = self.patch_size

        nh = H // p
        nw = W // p


        mask = np.ones(
            (nh,nw),
            dtype=np.float32
        )


        if self.keep_center:

            hs = int(
                nh*(1-self.center_ratio)/2
            )

            he = int(
                nh*(1+self.center_ratio)/2
            )

            ws = int(
                nw*(1-self.center_ratio)/2
            )

            we = int(
                nw*(1+self.center_ratio)/2
            )


            center = np.zeros_like(mask)

            center[
                hs:he,
                ws:we
            ] = 1


        else:

            center=np.zeros_like(mask)



        candidates=np.where(
            center.reshape(-1)==0
        )[0]


        num_mask=int(
            len(candidates)*self.mask_ratio
        )


        select=np.random.choice(
            candidates,
            num_mask,
            replace=False
        )


        flat=mask.reshape(-1)

        flat[select]=0


        mask=flat.reshape(nh,nw)


        mask=cv2.resize(
            mask,
            (nw*p,nh*p),
            interpolation=cv2.INTER_NEAREST
        )


        full=np.ones(
            (H,W),
            dtype=np.float32
        )

        full[:nh*p,:nw*p]=mask


        image=image*full[:,:,None]

        image=image.astype(np.uint8)

        return image