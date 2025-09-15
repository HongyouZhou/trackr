"""
Human Multi-Object Model for tracking and prediction.
This module provides the HumanMultiObjModel class which is an alias for HumanModel.
"""

from algo.tracking.human_model import HumanModel

class HumanMultiObjModel(HumanModel):
    """
    Human Multi-Object Model - alias for HumanModel.
    This class provides the same functionality as HumanModel but with a more descriptive name.
    """
    
    def __init__(self, cfg=None, max_ep_len=4096):
        super().__init__(max_ep_len=max_ep_len, cfg=cfg)
