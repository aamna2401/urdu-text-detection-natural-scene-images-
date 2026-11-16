class Config:
    model_weights = "checkpoints/mobilenet-model-21-mAp-0.08112358789599004.pth"
    image_path = "Dataset/test/16.jpg"
    gpu_id = 'cpu'
    num_classes = 12 + 1
    data_root_dir = "Dataset"
    test_folder = "/content/gdrive/MyDrive/Thesis/urdu_text_detection/FasterRCNN/test_folder"


test_cfg = Config()
