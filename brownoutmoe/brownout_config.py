from dataclasses import dataclass
from typing import List

from brownoutmoe.scheduler import Scheduler

@dataclass
class BrownoutConfig:
    top_p:float = 1
    way:int = 2
    full_brownout_mode:bool=False
    united_experts_weight_dirctory:str=""
    use_fused_moe:bool=False
    scheduler:Scheduler=None

    
    
        
        
        
            

