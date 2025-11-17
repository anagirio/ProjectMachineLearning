"""
Configuration to tolerate truncated or corrupted images.

This module configures the PIL (Python Imaging Library) to attempt
to load images even if they are incomplete or partially corrupted.

To use, simply import this module at the top of any script
that works with images::

    import fix_images

This should be done BEFORE any image-loading operations.
"""

from PIL import ImageFile

# Allow loading truncated (incomplete) images
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("PIL configured to load truncated images")
