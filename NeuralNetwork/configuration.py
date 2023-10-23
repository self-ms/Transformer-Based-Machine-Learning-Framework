
class ModelConf:
    def __int__(self, num_cls, device):

        self.conf = {
            'device': device,
            'epoch': 200,
            'optimizer': {'lr': 0.1, 'momentum': 0.9},
            'accuracy': {'task': 'multiclass', 'num_classes': num_cls},
            'model_param': {
                'd_model': 256,
                'nhead': 8,
                'num_enc': 8,
                'd_feed': 512,
                'dropout': 0.1,
                'activation': 'gelu',
                'num_cls': num_cls,
            }
        }