# lora_callback.py

class LoRASelectionCallback:
    def __init__(self):
        self.selected_lora_classes = []
        self.enabled = True
    
    def record_lora_class(self, lora_class):
        if self.enabled:
            self.selected_lora_classes.append(lora_class)
    
    def reset(self):
        self.selected_lora_classes = []
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False

# 创建全局实例（可选）
global_callback = LoRASelectionCallback()