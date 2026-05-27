import os
import segmentation

segment = segmentation.segment_characters
segment_validation = segmentation.segmentation_is_valid


this_file = __file__
this_file_absolute = os.path.abspath(this_file)
script_dir = os.path.dirname(this_file_absolute)
data_dir = os.path.join(script_dir, 'data')