import torchreid

# Mevcut modelleri listeler
torchreid.models.show_avai_models()

# Osnet_x1_0 modelini indirir
torchreid.utils.download_pretrained_weights('osnet_x1_0')
