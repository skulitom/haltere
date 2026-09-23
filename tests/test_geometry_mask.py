import numpy as np
import pytest

from haltere.vision.geometry_mask import liftoff_geometry_mask


def test_moving_hud_marks_and_propeller_regions_never_supply_features():
    # Both a bright reticle away from its usual centre and a colored race cue.
    image = np.full((360, 640, 3), [40, 60, 100], np.uint8)
    image[110:116, 265:271] = 240
    image[205:211, 345:351] = [70, 220, 170]
    mask = liftoff_geometry_mask(image)
    assert not mask[110:116, 265:271].any()
    assert not mask[205:211, 345:351].any()
    assert not mask[250:280, 100:200].any()
    assert mask[130, 180] == 255  # ordinary dark scene region survives
    with pytest.raises(ValueError, match='RGB'):
        liftoff_geometry_mask(image.astype(float))
